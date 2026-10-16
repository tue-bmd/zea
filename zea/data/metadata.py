"""Selective metadata loading for :class:`~zea.data.dataloader.Dataloader`.

See the ``return_metadata`` argument of :class:`~zea.data.dataloader.Dataloader`
for the path syntax and the shape of the result.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import h5py
import numpy as np

from zea.data.datasets import _resolve_dotted_path
from zea.data.file import ChunkedDataset, File, _GroupProxy, _StringDataset
from zea.data.spec import (
    CONSISTENCY_DIMENSIONS,
    LOCAL_CONSISTENCY_DIMENSIONS,
    ROOT_SPECS,
    FileSpec,
    Spec,
    dim_names_for_key,
)

__all__ = [
    "has_per_frame_paths",
    "batch_leaf_shape",
    "metadata_signature",
    "missing_metadata_paths",
    "normalize_metadata_paths",
    "read_metadata",
    "select_metadata_axes",
    "selected_dimensions",
    "selected_leaf_shape",
    "slice_metadata",
]

#: Dimensions an ``axis_selections`` take also applies to metadata: the ones the spec
#: validates as file-wide consistent, minus those only consistent within one data product
#: (:data:`~zea.data.spec.LOCAL_CONSISTENCY_DIMENSIONS`, so an ``n_ch`` take on ``raw_data``
#: cannot cut a sibling product) and ``n_frames``, which :func:`slice_metadata` handles.
PROPAGATED_DIMENSIONS = CONSISTENCY_DIMENSIONS - LOCAL_CONSISTENCY_DIMENSIONS - {"n_frames"}


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


def _read_dataset(dataset: "h5py.Dataset | ChunkedDataset"):
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
    key = path.replace(".", "/")
    # Resolve with File.dataset rather than file[key]: the latter hands back the bare h5py
    # object, whose reads are serial, while File.dataset and load_group both go through the
    # concurrent chunk-read path. It raises the same exceptions file[key] does.
    try:
        obj = file.dataset(key)
    except (KeyError, AttributeError):
        obj = None

    if isinstance(obj, _GroupProxy):
        return file.load_group(key)
    if obj is not None:
        # _StringDataset already decodes bytes on read; _read_dataset covers the rest (and
        # still accepts a plain h5py.Dataset for other callers).
        return obj[()] if isinstance(obj, _StringDataset) else _read_dataset(obj)

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


def _path_exists(file: File, path: str) -> bool:
    """Whether ``path`` resolves in ``file``, without reading the value.

    Mirrors the resolution order of :func:`_read_path` -- HDF5 object, root
    attribute, then derived ``File`` property -- but stops at the handle wherever
    it can, so a presence check costs a name lookup rather than a read.  Keep the
    two in step: a path that resolves here must be readable there.
    """
    key = path.replace(".", "/")
    try:
        if file.dataset(key) is not None:
            return True
    except (KeyError, AttributeError):
        pass

    if "." not in path and path in file.attrs:
        return True

    # Derived properties (``probe_name``, ``zea_version``) are not HDF5 objects, so
    # they can only be confirmed by resolving them -- as _read_path does too.
    return _resolve_dotted_path(file, path) is not None


def missing_metadata_paths(file: File, paths: Sequence[str]) -> tuple[str, ...]:
    """Return the subset of ``paths`` that ``file`` cannot supply.

    Lets a caller check a whole dataset up front instead of discovering a missing
    path when :func:`read_metadata` reaches the file mid-epoch.

    Args:
        file: An open :class:`~zea.data.file.File`.
        paths: Dotted paths to check.

    Returns:
        tuple: The paths absent from ``file``, in the order given. Empty when the
        file can answer all of them.
    """
    return tuple(path for path in paths if not _path_exists(file, path))


def _collect_signature(file: File, path: str, out: dict) -> None:
    """Record the shape of every leaf at or below ``path`` into ``out``."""
    key = path.replace(".", "/")
    try:
        obj = file.dataset(key)
    except (KeyError, AttributeError):
        obj = None

    if isinstance(obj, _GroupProxy):
        # A requested group contributes one entry per leaf below it, since that is
        # what read_metadata returns and what batching stacks.
        for name in obj.keys():
            _collect_signature(file, f"{path}.{name}", out)
        return

    if obj is not None:
        # h5py reports shape off the handle, so no data is read here.
        out[path] = tuple(obj.shape)
        return

    if "." not in path and path in file.attrs:
        out[path] = tuple(np.shape(file.attrs[path]))
        return

    out[path] = tuple(np.shape(_resolve_dotted_path(file, path)))


def metadata_signature(file: File, paths: Sequence[str]) -> dict[str, tuple]:
    """Map every leaf reachable from ``paths`` to its shape in ``file``.

    The shapes are the raw stored ones: normalizing the frame axis needs the file's
    own frame count, which the caller holds. Dtypes are deliberately left out --
    stacking promotes them (``float32`` with ``float64``, ``<U4`` with ``<U9``), so a
    difference there is not a batching failure.

    Args:
        file: An open :class:`~zea.data.file.File`.
        paths: Dotted paths to describe. A path pointing at a group expands to one
            entry per leaf below it.

    Returns:
        dict: Leaf dotted path -> shape tuple. A path absent from the file maps to
        the shape of ``None``, i.e. ``()``; use :func:`missing_metadata_paths` to
        tell absence apart from a genuine scalar.
    """
    out: dict[str, tuple] = {}
    for path in paths:
        _collect_signature(file, path, out)
    return out


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


def _is_per_frame_shape(path: str, shape: tuple, n_frames: int | None) -> bool:
    """Whether a value of ``shape`` at ``path`` carries the file's frame axis.

    Decides on shape alone so that :func:`_slice_value` and :func:`batch_leaf_shape`
    cannot drift apart: what gets sliced is exactly what gets reported as per-frame.
    """
    alternatives = PER_FRAME_PATHS.get(path)
    if not alternatives or n_frames is None:
        return False
    if len(shape) == 0 or shape[0] != n_frames:
        # Broadcast scalar form (e.g. one annotation for the whole file), or a
        # shape that does not line up with the frame axis: leave it alone.
        return False
    return any(len(alt) == len(shape) for alt in alternatives)


def selected_dimensions(key: str, num_dims: int, axis_selections: dict) -> dict[str, Any]:
    """Resolve ``{axis: selection}`` on ``key`` into ``{dimension name: selection}``.

    Naming the dimension is what lets the same take reach metadata laid out differently:
    axis 1 of ``data/raw_data`` is ``n_tx``, which is axis 0 of ``scan.t0_delays``.
    Empty for a key the spec cannot name, and for dimensions the take may not travel
    along (:data:`PROPAGATED_DIMENSIONS`).
    """
    if not axis_selections:
        return {}
    dim_names = dim_names_for_key(key, num_dims)
    if dim_names is None:
        return {}
    return {
        dim_names[axis]: selection
        for axis, selection in axis_selections.items()
        if dim_names[axis] in PROPAGATED_DIMENSIONS
    }


def _leaf_axis_selections(
    path: str, shape: tuple, dim_selections: dict, dim_sizes: dict
) -> dict[int, Any]:
    """Map axis -> selection for the axes of ``path`` carrying a selected dimension.

    The non-frame counterpart of :func:`_is_per_frame_shape`, shared by
    :func:`_select_value` and :func:`selected_leaf_shape` so the take and the shape it
    yields cannot drift apart.  An axis is only taken from when its length is the file's
    full extent for that dimension (``dim_sizes``), which is what the selection indexes.
    """
    if not dim_selections or not shape:
        return {}
    dim_names = dim_names_for_key(path, len(shape))
    if dim_names is None:
        return {}
    return {
        axis: dim_selections[dim]
        for axis, dim in enumerate(dim_names)
        if dim in dim_selections and dim_sizes.get(dim) == shape[axis]
    }


def _selection_length(selection, size: int) -> int:
    """How many indices ``selection`` takes from an axis of length ``size``."""
    if isinstance(selection, slice):
        return len(range(*selection.indices(size)))
    return len(selection)


def selected_leaf_shape(path: str, shape: tuple, dim_selections: dict, dim_sizes: dict) -> tuple:
    """Return ``shape`` as it is after :func:`select_metadata_axes` narrows it."""
    selections = _leaf_axis_selections(path, shape, dim_selections, dim_sizes)
    if not selections:
        return tuple(shape)
    out = list(shape)
    for axis, selection in selections.items():
        out[axis] = _selection_length(selection, shape[axis])
    return tuple(out)


def batch_leaf_shape(
    path: str,
    shape: tuple,
    n_frames: int | None,
    dim_selections: dict | None = None,
    dim_sizes: dict | None = None,
) -> tuple:
    """Return ``shape`` as batching sees it: selected, with the frame axis a placeholder.

    Batching stacks metadata leaf by leaf, so the leaves of every file must line up.
    Both cuts a sample's metadata undergoes are applied here, since a leaf only has to
    match after them: the selection (:func:`selected_leaf_shape`), and the frame axis,
    sliced to the sample's frame count and so normalized rather than compared.
    Everything else must match exactly.
    """
    shape = selected_leaf_shape(path, shape, dim_selections or {}, dim_sizes or {})
    if _is_per_frame_shape(path, shape, n_frames):
        return ("n_frames",) + tuple(shape[1:])
    return tuple(shape)


def _slice_value(path: str, value, frame_selection, n_frames: int | None):
    """Slice ``value`` along its leading axis when the spec marks it per-frame."""
    if not isinstance(value, np.ndarray):
        return value
    if not _is_per_frame_shape(path, value.shape, n_frames):
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


def _select_value(path: str, value, dim_selections: dict, dim_sizes: dict):
    """Take the selected indices from every axis of ``value`` the spec names as shared."""
    if not isinstance(value, np.ndarray):
        return value
    for axis, selection in _leaf_axis_selections(
        path, value.shape, dim_selections, dim_sizes
    ).items():
        # One axis at a time: keeps it in place, and takes slices as well as index lists.
        value = value[(slice(None),) * axis + (selection,)]
    return value


def select_metadata_axes(
    tree: dict, dim_selections: dict, dim_sizes: dict, prefix: str = ""
) -> dict:
    """Return a copy of ``tree`` narrowed by the sample's ``axis_selections``.

    A selection means something about the acquisition -- "these 21 transmits" -- so a
    field carrying that dimension stops describing the sample it comes with unless it is
    cut the same way.  Unlike the frame axis of :func:`slice_metadata`, the cut is the
    same for every sample of a file, so this runs once per file rather than per sample.

    Args:
        tree: Nested metadata dict as returned by :func:`read_metadata`.
        dim_selections: Dimension name -> selection, from :func:`selected_dimensions`.
        dim_sizes: Dimension name -> that dimension's full extent in this file.
        prefix: Dotted prefix of ``tree`` within the file spec (internal).

    Returns:
        dict: A new nested dict; values with no selected axis are shared, not copied.
    """
    out = {}
    for name, value in tree.items():
        path = f"{prefix}{name}"
        if isinstance(value, dict):
            out[name] = select_metadata_axes(value, dim_selections, dim_sizes, prefix=f"{path}.")
        else:
            out[name] = _select_value(path, value, dim_selections, dim_sizes)
    return out
