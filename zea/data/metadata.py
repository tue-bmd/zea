"""Selective metadata loading for :class:`~zea.data.dataloader.Dataloader`.

A zea file carries far more than its image data: scan parameters, probe
geometry, subject information, per-frame annotations, metrics.  Reading *all*
of it for every sample would dominate the time spent in the data pipeline, so
the dataloader asks for metadata by **dotted path** — the same path syntax used
by ``file_filter`` (see :func:`~zea.data.datasets.compile_file_filter`):

.. code-block:: python

    Dataloader(
        ...,
        return_metadata=[
            "scan.sampling_frequency",
            "probe.probe_geometry",
            "metadata.subject",  # a group: everything below it
        ],
    )

Each requested path is read straight off the HDF5 group, bypassing the eager
:attr:`~zea.data.file.File.scan` / :attr:`~zea.data.file.File.metadata`
properties (which read a whole group and re-run spec validation on every
access).  The result is a nested dict mirroring
:class:`~zea.data.spec.FileSpec`.

Fields whose leading dimension is ``n_frames`` in the spec (per-frame
annotations, per-frame metrics, map timestamps) are sliced with the same frame
selection used for the sample's images, so metadata stays aligned with the
frames it describes.
"""

from collections.abc import Iterable, Sequence

import h5py
import numpy as np

from zea.data.datasets import _resolve_dotted_path
from zea.data.file import File
from zea.data.spec import ROOT_SPECS, FileSpec, Spec

__all__ = ["has_per_frame_paths", "normalize_metadata_paths", "read_metadata", "slice_metadata"]


def _iter_shape_alternatives(shape) -> tuple[tuple, ...]:
    """Return the shape options of a SCHEMA entry as a tuple of shape tuples.

    A schema shape is either a single shape (``("n_frames", "n_tx")``) or a
    tuple of alternatives (``(("n_frames",), ())``).
    """
    if not isinstance(shape, tuple):
        return ()
    if shape and all(isinstance(alt, tuple) for alt in shape):
        return shape
    return (shape,)


def _collect_per_frame_paths() -> dict[str, tuple[tuple, ...]]:
    """Walk the spec tree and collect paths whose leading dimension is ``n_frames``.

    Returns a mapping of dotted path to the shape alternatives that start with
    ``n_frames``.  The alternatives are kept so that the runtime slicer can
    match on ``ndim`` — ``scan.time_to_next_transmit``, for instance, is
    per-frame only in its ``(n_frames, n_tx)`` form, not in its flat form.
    """
    out: dict[str, tuple[tuple, ...]] = {}

    def walk(spec_cls: type[Spec], prefix: str, seen: frozenset):
        if spec_cls in seen:
            return
        seen = seen | {spec_cls}
        for name, entry in getattr(spec_cls, "SCHEMA", {}).items():
            path = f"{prefix}{name}"
            sub = entry.get("spec")
            if sub is not None:
                walk(sub, f"{path}.", seen)
                continue
            alternatives = tuple(
                alt
                for alt in _iter_shape_alternatives(entry.get("shape"))
                if alt[:1] == ("n_frames",)
            )
            if alternatives:
                out[path] = alternatives

    walk(FileSpec, "", frozenset())
    for root, spec_cls in ROOT_SPECS.items():
        walk(spec_cls, f"{root}.", frozenset())
    return out


#: Dotted path -> shape alternatives that lead with ``n_frames``.
PER_FRAME_PATHS: dict[str, tuple[tuple, ...]] = _collect_per_frame_paths()


def normalize_metadata_paths(return_metadata) -> tuple[str, ...] | None:
    """Normalize the ``return_metadata`` argument into a tuple of dotted paths.

    Args:
        return_metadata: ``False``/``None`` to return arrays only, ``True`` for
            file identity only, or a string / iterable of dotted paths.

    Returns:
        ``None`` when no metadata should be returned, otherwise a tuple of
        dotted paths (possibly empty, meaning file identity only).
    """
    if return_metadata is None or return_metadata is False:
        return None
    if return_metadata is True:
        return ()
    if isinstance(return_metadata, str):
        return (return_metadata,)
    if isinstance(return_metadata, Iterable):
        paths = tuple(return_metadata)
        if not all(isinstance(path, str) for path in paths):
            raise TypeError(
                "return_metadata must be a bool, a dotted path string, or an iterable "
                f"of dotted path strings; got {return_metadata!r}."
            )
        return paths
    raise TypeError(
        "return_metadata must be a bool, a dotted path string, or an iterable of "
        f"dotted path strings; got {type(return_metadata).__name__}."
    )


def has_per_frame_paths(paths: Sequence[str]) -> bool:
    """Whether any of ``paths`` can resolve to a field with a leading ``n_frames`` axis.

    Lets the caller skip the per-frame slicing pass entirely when only static
    metadata (scan parameters, probe geometry, subject) was requested.
    """
    return any(
        per_frame == path or per_frame.startswith(f"{path}.")
        for path in paths
        for per_frame in PER_FRAME_PATHS
    )


def _read_dataset(dataset: "h5py.Dataset"):
    """Read an HDF5 dataset, decoding strings the same way as a group load."""
    if h5py.check_string_dtype(dataset.dtype) is not None:
        value = dataset.asstr()[()]
        if isinstance(value, np.ndarray) and value.dtype == object:
            value = value.astype(np.str_)
        return value
    return dataset[()]


def _read_path(file: File, path: str):
    """Read a single dotted path off an open :class:`~zea.data.file.File`.

    Tries, in order: the HDF5 object at ``path.replace(".", "/")`` (a group is
    loaded recursively into a dict), a root attribute, and finally plain
    attribute access on the ``File`` object (which covers derived properties
    such as ``probe_name`` and ``zea_version``).
    """
    try:
        obj = file[path.replace(".", "/")]
    except (KeyError, AttributeError):
        obj = None

    if isinstance(obj, h5py.Group):
        return file.load_group(obj)
    if obj is not None:
        return _read_dataset(obj)

    if "." not in path and path in file.attrs:
        return file.attrs[path]

    value = _resolve_dotted_path(file, path)
    if value is None:
        raise KeyError(
            f"Metadata path '{path}' not found in file '{file.path}'. Check the path "
            "against the zea file spec, or drop files that lack it with "
            f"file_filter={{'{path}': EXISTS}}."
        )
    return value


def _assign(tree: dict, parts: Sequence[str], value) -> None:
    """Insert ``value`` into the nested ``tree`` at the given path segments."""
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def read_metadata(file: File, paths: Sequence[str]) -> dict:
    """Read ``paths`` off an open file into a nested, ``FileSpec``-shaped dict.

    Args:
        file: An open :class:`~zea.data.file.File`.
        paths: Dotted paths to read. A path pointing at a group reads the whole
            group recursively.

    Returns:
        dict: Nested dict mirroring :class:`~zea.data.spec.FileSpec`.

    Raises:
        KeyError: If a requested path is absent from the file.
    """
    tree: dict = {}
    for path in paths:
        _assign(tree, path.split("."), _read_path(file, path))
    return tree


def _slice_value(path: str, value, frame_selection, n_frames: int | None):
    """Slice ``value`` along its leading axis when the spec marks it per-frame."""
    alternatives = PER_FRAME_PATHS.get(path)
    if not alternatives or n_frames is None or not isinstance(value, np.ndarray):
        return value
    if value.ndim == 0 or value.shape[0] != n_frames:
        # Broadcast scalar form (e.g. one annotation for the whole file), or a
        # shape that does not line up with the frame axis: leave it alone.
        return value
    if not any(len(alt) == value.ndim for alt in alternatives):
        return value
    return value[frame_selection]


def slice_metadata(tree: dict, frame_selection, n_frames: int | None, prefix: str = "") -> dict:
    """Return a copy of ``tree`` with per-frame fields sliced to the sample's frames.

    Args:
        tree: Nested metadata dict as returned by :func:`read_metadata`.
        frame_selection: The frame selector used to read the sample's images
            (a ``slice`` over the file's frame axis).
        n_frames: Total number of frames in the file, used to recognize which
            arrays actually carry a frame axis. ``None`` disables slicing.
        prefix: Dotted prefix of ``tree`` within the file spec (internal).

    Returns:
        dict: A new nested dict; unsliced values are shared, not copied.
    """
    out = {}
    for name, value in tree.items():
        path = f"{prefix}{name}"
        if isinstance(value, dict):
            out[name] = slice_metadata(value, frame_selection, n_frames, prefix=f"{path}.")
        else:
            out[name] = _slice_value(path, value, frame_selection, n_frames)
    return out
