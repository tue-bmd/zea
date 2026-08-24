"""H5 dataloader for loading images from zea datasets.

Example:
    .. code-block:: python

        import zea

        loader = zea.Dataloader(
            file_paths="/path/to/dataset",
            key="data/image/values",
            batch_size=16,
            image_range=(-60, 0),
            normalization_range=(0, 1),
            image_size=(256, 256),
            num_threads=16,
        )

        for batch in loader:
            # batch is a numpy array of shape (batch_size, 256, 256, 1)
            ...
"""

import re
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import grain
import keras
import numpy as np
from keras import ops

from zea import log
from zea.data.datasets import (
    FILE_HANDLE_CACHE_CAPACITY,
    Dataset,
    H5FileHandleCache,
    count_samples_per_directory,
)
from zea.data.layers import Resizer
from zea.data.metadata import (
    batch_leaf_shape,
    has_per_frame_paths,
    normalize_metadata_paths,
    read_metadata,
    slice_metadata,
)
from zea.data.spec import dim_names_for_key
from zea.func.tensor import translate
from zea.utils import canonicalize_axis, map_negative_indices

if TYPE_CHECKING:
    from zea.data.file import File

#: How many offending file names to name in a shape or metadata error before eliding.
_MAX_LISTED_FILES = 5

#: What a loader does with files it cannot serve: refuse to build, or drop them.
_FILE_POLICIES = ("error", "skip")


def _normalize_axis_selections(
    axis_selections: dict,
    num_dims: int,
    reserved_axes: set[int],
) -> dict[int, list[int] | slice]:
    """Validate and normalize ``axis_selections`` into a canonical form.

    Converts raw axis keys to non-negative indices, checks for conflicts with
    reserved axes (frame axis / additional_axes_iter), and validates that list
    selections are 1-D, non-empty, and strictly increasing (required by h5py).
    """
    normalized: dict[int, list[int] | slice] = {}
    for raw_axis, sel in axis_selections.items():
        axis = canonicalize_axis(int(raw_axis), num_dims)
        if axis in reserved_axes:
            raise ValueError(
                f"axis_selections axis {raw_axis} conflicts with the frame axis "
                "or additional_axes_iter"
            )
        if isinstance(sel, slice):
            normalized[axis] = sel
        else:
            arr = np.asarray(sel, dtype=np.intp)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(
                    f"axis_selections[{raw_axis}] must be a 1-D non-empty list of ints"
                )
            if np.any(np.diff(arr) <= 0):
                raise ValueError(
                    f"axis_selections[{raw_axis}] must be strictly increasing "
                    "(h5py requires sorted, unique indices)"
                )
            normalized[axis] = arr.tolist()
    return normalized


def generate_h5_indices(
    file_paths: List[str],
    file_shapes: list,
    n_frames: int | None,
    frame_index_stride: int,
    key: str = "data/image",
    source_frame_axis: int | None = 0,
    additional_axes_iter: List[int] | None = None,
    sort_files: bool = True,
    overlapping_blocks: bool = False,
    limit_n_frames: int | None = None,
    on_incomplete_blocks: str = "error",
    axis_selections: dict | None = None,
    offset_n_frames: int = 0,
):
    """Generate indices for h5 files.

    Generates a list of indices to extract images from hdf5 files. Length of this list
    is the length of the extracted dataset.

    Args:
        file_paths (list): List of file paths.
        file_shapes (list): List of file shapes.
        n_frames (int, optional): Number of frames per sample. ``None`` selects single
            frames with an integer index, so the frame axis is dropped from the result.
        frame_index_stride (int): Interval between frames to load.
        key (str, optional): Key of hdf5 dataset to grab data from. Defaults to "data/image".
        source_frame_axis (int, optional): Axis of the file's arrays that stores frames, or
            ``None`` when the data has no frame axis, in which case every file yields a
            single sample. Defaults to 0.
        additional_axes_iter (list, optional): Additional axes to iterate over in the dataset.
            Defaults to None.
        sort_files (bool, optional): Sort files by number. Defaults to True.
        overlapping_blocks (bool, optional): Will take n_frames from sequence, then move by 1.
            Defaults to False.
        limit_n_frames (int, optional): Maximum number of frames to load per file, counted from
            ``offset_n_frames``. Defaults to None (no limit).
        on_incomplete_blocks (str, optional): What to do with files holding too few frames to
            fill one block of ``n_frames * frame_index_stride``: ``"error"`` (default) raises
            and names them, ``"skip"`` drops them from the index table.
        axis_selections (dict, optional): Map of ``{axis: indices}`` applied at HDF5 read time to
            pre-filter non-frame axes. For example ``{1: [0, 2, 5]}`` loads only those indices
            along axis 1, avoiding reading unused data from disk. Defaults to None.
        offset_n_frames (int, optional): Frame index to start iteration from within each file.
            Combined with ``limit_n_frames`` this selects the half-open range
            ``[offset_n_frames, offset_n_frames + limit_n_frames)``. Defaults to 0.

    Returns:
        list: List of tuples with indices to extract images from hdf5 files.
            (file_name, key, indices) with indices being a tuple of slices.

    Example:
        .. code-block:: python

            [
                (
                    "/folder/path_to_file.hdf5",
                    "data/image",
                    (slice(0, 2, 1), slice(None, 256, None), slice(None, 256, None)),
                ),
                (
                    "/folder/path_to_file.hdf5",
                    "data/image",
                    (slice(2, 4, 1), slice(None, 256, None), slice(None, 256, None)),
                ),
                ...,
            ]

        With ``n_frames=None`` the frame entry is a plain int instead of a slice, so
        the frame axis never enters the loaded array.
    """
    if limit_n_frames is None:
        frame_limit: float = np.inf
    else:
        assert limit_n_frames > 0, f"limit_n_frames must be > 0, got {limit_n_frames}"
        frame_limit = float(limit_n_frames)

    assert len(file_paths) == len(file_shapes), "file_paths and file_shapes must have same length"

    if additional_axes_iter:
        assert source_frame_axis not in additional_axes_iter, (
            f"The frame axis (axis {source_frame_axis}) cannot be in additional_axes_iter. "
            "We are already iterating over that axis."
        )
    else:
        additional_axes_iter = []

    if sort_files:
        try:
            # this is like an np.argsort, returns the indices that would sort the array
            indices_sorting_file_paths = sorted(
                range(len(file_paths)),
                key=lambda i: int(re.findall(r"\d+", file_paths[i])[-2]),
            )
            file_paths = [file_paths[i] for i in indices_sorting_file_paths]
            file_shapes = [file_shapes[i] for i in indices_sorting_file_paths]
        except Exception:
            log.warning("Could not sort file_paths by number.")

    # Frames one sample consumes. With n_frames=None the selection is a single int, so a
    # sample spans exactly one frame however large the stride: there the stride only
    # spaces consecutive samples out, it does not widen them.
    block_size = 1 if n_frames is None else n_frames * frame_index_stride

    if not overlapping_blocks:
        block_step_size = frame_index_stride if n_frames is None else block_size
    else:
        # now blocks overlap by n_frames - 1
        block_step_size = 1

    def usable_frames(shape) -> int:
        """Frames of one file left for sampling, after ``offset``/``limit`` narrow it."""
        effective_end = int(min(shape[source_frame_axis], offset_n_frames + frame_limit))
        return max(0, effective_end - offset_n_frames)

    def frame_selections(shape):
        """Frame selections for one file, empty when it cannot fill a single block."""
        effective_end = offset_n_frames + usable_frames(shape)
        # An int rather than a slice when n_frames is None: h5py then drops the axis
        # on read, so single-frame samples never grow a length-1 frame dimension.
        return [
            i if n_frames is None else slice(i, i + block_size, frame_index_stride)
            for i in range(offset_n_frames, effective_end - block_size + 1, block_step_size)
        ]

    # The frame axis leads the product when there is one; without it (a field the spec
    # gives no n_frames axis) every file contributes exactly one sample.
    iter_axes = ([] if source_frame_axis is None else [source_frame_axis]) + list(
        additional_axes_iter
    )

    indices = []
    short_files: dict[str, int] = {}
    for file, shape in zip(file_paths, file_shapes):
        axis_indices = []
        if source_frame_axis is not None:
            selections = frame_selections(shape)
            # Files too small to fit a single block cannot be served: on_incomplete_blocks
            # below decides whether that is an error or a silent drop.
            if not selections:
                short_files[file] = usable_frames(shape)
                continue
            axis_indices.append(selections)

        if additional_axes_iter:
            axis_indices += [list(range(shape[axis])) for axis in additional_axes_iter]

        for axis_index in product(*axis_indices):
            full_indices = [slice(size) for size in shape]
            for axis, selection in zip(iter_axes, axis_index):
                full_indices[axis] = selection
            if axis_selections:
                for axis, sel in axis_selections.items():
                    full_indices[axis] = sel
            indices.append((file, key, tuple(full_indices)))

    if short_files:
        if on_incomplete_blocks == "error":
            raise _incomplete_blocks_error(
                short_files,
                len(file_paths),
                block_size,
                windowed=offset_n_frames > 0 or limit_n_frames is not None,
            )
        log.info(
            f"Skipping {len(short_files)} / {len(file_paths)} files "
            f"({len(short_files) / len(file_paths) * 100:.2f}% of the dataset) that hold "
            f"fewer than the {block_size} frames one sample needs."
        )

    return indices


def _resolve_source_frame_axis(key: str, num_dims: int) -> int | None:
    """Locate the axis that stores frames for ``key``, per the zea file spec.

    Returns the axis index, or ``None`` when the spec names every axis of the field
    and none of them is ``n_frames`` -- an array that simply has no frames, such as
    ``probe/probe_geometry``.  Data the spec cannot speak for (custom keys, a
    wildcard shape) falls back to axis 0, the convention everywhere in the spec.
    """
    if num_dims == 0:
        return 0
    dim_names = dim_names_for_key(key, num_dims)
    if dim_names is None:
        log.warning(
            f"Key '{key}' with {num_dims} dimensions does not match any field of the zea "
            "file spec, so the axis that stores frames is unknown. Assuming axis 0."
        )
        return 0
    if "n_frames" in dim_names:
        return dim_names.index("n_frames")
    return None


def _check_file_policy(name: str, value: str) -> str:
    """Validate one of the ``"error"`` / ``"skip"`` keywords, returning it unchanged."""
    if value not in _FILE_POLICIES:
        raise ValueError(
            f"{name} must be one of {_FILE_POLICIES}, got {value!r}. Use 'error' to refuse "
            "to build the loader and be told which files are at fault, or 'skip' to drop "
            "them from the dataset."
        )
    return value


def _name_files(files: Sequence[str]) -> str:
    """Name the first few of ``files`` by basename, counting the rest."""
    shown = ", ".join(f"'{Path(f).name}'" for f in files[:_MAX_LISTED_FILES])
    if len(files) > _MAX_LISTED_FILES:
        shown += f", +{len(files) - _MAX_LISTED_FILES} more"
    return shown


def _incomplete_blocks_error(
    short_files: dict, n_files: int, block_size: int, windowed: bool
) -> ValueError:
    """Build the error raised when files hold too few frames to fill one block.

    The counterpart of :func:`_missing_metadata_error`: both name files the loader
    cannot serve and point at the ``"skip"`` policy that drops them.  ``short_files``
    maps a file path to the frames it has available, which ``offset_n_frames`` and
    ``limit_n_frames`` can narrow -- ``windowed`` says whether they did.
    """
    listed = list(short_files.items())[:_MAX_LISTED_FILES]
    shown = ", ".join(f"'{Path(f).name}' ({n})" for f, n in listed)
    if len(short_files) > _MAX_LISTED_FILES:
        shown += f", +{len(short_files) - _MAX_LISTED_FILES} more"
    window = (
        " Those counts are what offset_n_frames and limit_n_frames leave of each file."
        if windowed
        else ""
    )
    return ValueError(
        f"{len(short_files)}/{n_files} files hold fewer than the {block_size} frames one "
        f"sample needs: {shown}.{window} Drop them with on_incomplete_blocks='skip', or "
        "lower n_frames / frame_index_stride until a block fits. Samples are stacked into "
        "batches, so a block short of n_frames cannot be served as it is; read those files "
        "directly if you need their frames."
    )


def _missing_metadata_error(metadata_gaps: dict, n_files: int) -> KeyError:
    """Build the error raised when files cannot supply a requested metadata path.

    ``metadata_gaps`` maps a file path to the paths that file lacks; this inverts it
    so the message is per path -- the unit the user acts on -- and names a few
    offending files rather than all of them.  Returns a :exc:`KeyError` to match the
    error :func:`~zea.data.metadata.read_metadata` raises for the same cause.
    """
    per_path: dict[str, list[str]] = defaultdict(list)
    for file_path, paths in metadata_gaps.items():
        for path in paths:
            per_path[path].append(file_path)

    parts = []
    for path, files in per_path.items():
        parts.append(
            f"return_metadata path '{path}' is missing from {len(files)}/{n_files} files "
            f"({_name_files(files)}); drop them with on_missing_metadata='skip', or up "
            f"front with file_filter={{'{path}': EXISTS}}"
        )
    # KeyError renders its argument with repr(), so keep the message to one line.
    return KeyError(
        " | ".join(parts) + ". Metadata is read for every sample, so the loader would "
        "otherwise fail partway through an epoch. Import EXISTS from zea.data.datasets, "
        "or drop the path from return_metadata."
    )


def _resized_shape(resizer: "Resizer", shape: tuple) -> tuple:
    """The shape ``resizer`` produces for ``shape``, asked of the resizer itself.

    Tries a symbolic call first, which traces the real layer without allocating or
    computing anything. Some keras layers cannot be traced that way (``RandomCrop``
    cropping down builds a slice from a tracer), so fall back to running the layer on
    one zero-filled sample -- still the real layer, just no longer free.
    """
    try:
        return tuple(resizer(keras.KerasTensor(shape, dtype="float32")).shape)
    except Exception:  # noqa: BLE001 -- any tracing failure just means "measure it instead"
        with keras.device("cpu"):
            return tuple(np.shape(resizer(np.zeros(shape, np.float32))))


def _sample_shape_error(sample_shapes: dict, batch_size: int) -> ValueError:
    """Build the error raised when samples cannot be stacked into a batch."""
    lines = [
        f"  - {shape} from {len(files)} file(s) (e.g. {', '.join(Path(f).name for f in files)})"
        for shape, files in sample_shapes.items()
    ]
    return ValueError(
        f"Samples differ in shape between files, so they cannot be stacked into a batch "
        f"of {batch_size}:\n"
        + "\n".join(lines)
        + "\nSet image_size (with resize_type) to bring them to a common shape, use "
        "batch_size=None to keep samples separate, or restrict the dataset to files that "
        "agree with file_filter."
    )


def _metadata_batch_conflicts(
    metadata_signatures: dict, file_n_frames: dict
) -> dict[str, dict[tuple, list[str]]]:
    """Find metadata leaves whose shape differs between files.

    Batching stacks metadata leaf by leaf, so a leaf that is a scalar in one file and
    an array in another (or absent altogether) cannot be stacked. Comparing the
    normalized shapes up front turns that into one clear error instead of a failure
    inside the batch op.

    Args:
        metadata_signatures: File path -> ``{leaf path: stored shape}``.
        file_n_frames: File path -> that file's frame count, for the files whose
            per-frame metadata is sliced. Pass ``{}`` when no slicing happens; a
            file missing from it keeps its stored shapes.

    Returns:
        dict: Leaf path -> ``{normalized shape: file paths}``, holding only leaves
        that resolved to more than one shape. Empty when the files agree.
    """
    per_leaf: dict[str, dict[tuple, list[str]]] = defaultdict(lambda: defaultdict(list))
    all_files = list(metadata_signatures)
    for file_path, signature in metadata_signatures.items():
        n_frames = file_n_frames.get(file_path)
        for leaf, shape in signature.items():
            per_leaf[leaf][batch_leaf_shape(leaf, shape, n_frames)].append(file_path)

    conflicts = {}
    for leaf, by_shape in per_leaf.items():
        # A leaf that some files do not have at all is a structural mismatch too: the
        # dicts being stacked would not share the same keys.
        seen = {f for files in by_shape.values() for f in files}
        if len(seen) != len(all_files):
            # Not a shape, but it reads correctly in the error: "absent in 2 file(s)".
            by_shape = {**by_shape, "absent": [f for f in all_files if f not in seen]}
        if len(by_shape) > 1:
            conflicts[leaf] = dict(by_shape)
    return conflicts


def _metadata_batch_error(conflicts: dict, batch_size: int) -> ValueError:
    """Build the error raised when metadata shapes cannot be stacked into a batch."""
    lines = []
    for leaf, by_shape in conflicts.items():
        variants = "; ".join(
            f"{shape} in {len(files)} file(s) (e.g. {Path(files[0]).name})"
            for shape, files in by_shape.items()
        )
        lines.append(f"  - '{leaf}': {variants}")
    return ValueError(
        "return_metadata fields differ in shape between files, so they cannot be stacked "
        f"into a batch of {batch_size}:\n"
        + "\n".join(lines)
        + "\nBatching stacks metadata leaf by leaf, so every file must supply the same "
        "leaves with the same shapes. Either use batch_size=None and batch the metadata "
        "yourself, narrow return_metadata to the fields that do line up, or restrict the "
        "dataset to files that agree with file_filter."
    )


class H5DataSource:
    """Thread-safe random-access data source for HDF5 files.

    Implements ``grain.RandomAccessDataSource`` protocol (``__getitem__``
    and ``__len__``) so it can be plugged directly into a
    ``grain.MapDataset`` pipeline.

    Each worker thread gets its own ``H5FileHandleCache`` via
    ``threading.local()`` so ``h5py`` file handles are never shared across
    threads.

    Args:
        file_paths: Path(s) to HDF5 directory(ies) or file(s).
        key: HDF5 dataset key, e.g. ``"data/image"``.
        n_frames: Number of consecutive frames per sample, or ``None`` (default) for
            single frames without a frame axis. See :class:`Dataloader`.
        frame_index_stride: Stride between frames.
        frame_axis: Axis the frame block is placed on in the output. Defaults to
            ``-1`` so frames land in the channel position for image data; see
            :class:`Dataloader`. Unused when ``n_frames is None``.
        additional_axes_iter: Extra axes to iterate over.
        sort_files: Sort files numerically.
        overlapping_blocks: Allow overlapping frame blocks.
        limit_n_examples: Cap the number of examples (dataset length).
        limit_n_frames: Cap frames loaded per file.
        return_metadata: Return a ``(sample, metadata)`` tuple. See :class:`Dataloader`.
        cache: Cache loaded samples to RAM.
        validate: Validate dataset against the zea format.
        revision: HuggingFace revision (branch, tag, or commit hash) for ``hf://`` paths.
        on_incomplete_blocks: ``"error"`` or ``"skip"`` for files too short to fill a
            block. See :class:`Dataloader`.
        on_missing_metadata: ``"error"`` or ``"skip"`` for files that cannot supply a
            requested ``return_metadata`` path. See :class:`Dataloader`.
        file_filter: Keep only files whose content matches a predicate. See
            :class:`Dataloader` for details. Defaults to ``None`` (no filtering).
    """

    def __init__(
        self,
        file_paths: List[str] | str,
        key: str = "data/image",
        n_frames: int | None = None,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        additional_axes_iter: tuple | None = None,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        limit_n_examples: int | None = None,
        limit_n_frames: int | None = None,
        offset_n_frames: int = 0,
        return_metadata: bool | str | Sequence[str] | None = None,
        cache: bool = False,
        validate: bool = False,
        revision: str | None = None,
        on_incomplete_blocks: str = "error",
        on_missing_metadata: str = "error",
        axis_selections: dict | None = None,
        file_filter: "Callable[[File], bool] | dict | None" = None,
        **kwargs,
    ):
        self.return_metadata = normalize_metadata_paths(return_metadata)
        self.returns_metadata = self.return_metadata is not None
        self.cache = cache
        self._data_cache = {}
        # Metadata is constant per file, so cache it per path rather than per sample.
        self._metadata_cache: OrderedDict[str, dict] = OrderedDict()
        self._metadata_lock = threading.Lock()
        self.on_incomplete_blocks = _check_file_policy("on_incomplete_blocks", on_incomplete_blocks)
        self.on_missing_metadata = _check_file_policy("on_missing_metadata", on_missing_metadata)

        self.key = key
        self.n_frames = None if n_frames is None else int(n_frames)
        self.frame_index_stride = int(frame_index_stride)
        self.frame_axis = int(frame_axis)

        assert self.frame_index_stride > 0, (
            f"`frame_index_stride` must be > 0, got {self.frame_index_stride}"
        )
        assert self.n_frames is None or self.n_frames > 0, (
            f"`n_frames` must be > 0 or None, got {self.n_frames}"
        )

        # Discover files and shapes (reuses Dataset machinery)
        lazy = kwargs.pop("lazy", False)
        if lazy:
            raise ValueError(
                "lazy=True is not supported in Dataloader / H5DataSource. "
                "All files must be downloaded before building the data pipeline. "
                "Use Dataset(..., lazy=True) directly for interactive use."
            )
        _dataset = Dataset(
            file_paths,
            validate=validate,
            revision=revision,
            file_filter=file_filter,
            _suggest_lazy=False,
            **kwargs,
        )
        self.file_paths = _dataset.file_paths
        # Requested metadata paths are checked in the same sweep that reads the shapes,
        # so a file that cannot answer surfaces here rather than mid-epoch.
        self.file_shapes, metadata_gaps, self._metadata_signatures = _dataset.load_file_shapes(
            key, self.return_metadata or ()
        )
        _dataset.close()

        num_dims = len(self.file_shapes[0]) if self.file_shapes else 0
        self.source_frame_axis = _resolve_source_frame_axis(self.key, num_dims)
        self.additional_axes_iter = map_negative_indices(list(additional_axes_iter or []), num_dims)

        if self.source_frame_axis is None:
            if self.n_frames is not None:
                raise ValueError(
                    f"'{key}' has no frame axis in the zea file spec (its dimensions are "
                    f"{dim_names_for_key(key, num_dims)}), so frames cannot be grouped into "
                    f"blocks of n_frames={self.n_frames}. Use n_frames=None to load one "
                    "sample per file."
                )
            # Unlike n_frames, a frame window is not fatal: every file still yields its one
            # sample. But it silently does nothing, so say so rather than let the caller
            # believe their data was narrowed.
            inert = [
                f"{name}={value}"
                for name, value in (
                    ("limit_n_frames", limit_n_frames),
                    ("offset_n_frames", offset_n_frames or None),
                )
                if value is not None
            ]
            if inert:
                log.warning(
                    f"Ignoring {' and '.join(inert)}: '{key}' has no frame axis in the zea "
                    f"file spec (its dimensions are {dim_names_for_key(key, num_dims)}), so "
                    "there are no frames to window and each file yields one whole sample. "
                    "To load only part of such an array, pick indices along one of the axes "
                    " using axis_selections, e.g. axis_selections={0: [0, 1]}."
                )

        self._file_n_frames = (
            {
                path: shape[self.source_frame_axis]
                for path, shape in zip(self.file_paths, self.file_shapes)
            }
            if self.source_frame_axis is not None
            else {}
        )
        self._slice_metadata_per_frame = bool(
            self.return_metadata
            and has_per_frame_paths(self.return_metadata)
            and self.source_frame_axis is not None
        )

        # Validate and normalize axis_selections
        reserved_axes = set(self.additional_axes_iter)
        if self.source_frame_axis is not None:
            reserved_axes.add(self.source_frame_axis)
        self.normalized_axis_selections = (
            _normalize_axis_selections(axis_selections, num_dims, reserved_axes)
            if axis_selections and num_dims > 0
            else {}
        )

        # Compute per-sample index table
        self.indices = generate_h5_indices(
            file_paths=self.file_paths,
            file_shapes=self.file_shapes,
            n_frames=self.n_frames,
            frame_index_stride=self.frame_index_stride,
            key=self.key,
            source_frame_axis=self.source_frame_axis,
            additional_axes_iter=self.additional_axes_iter,
            sort_files=sort_files,
            overlapping_blocks=overlapping_blocks,
            limit_n_frames=limit_n_frames,
            on_incomplete_blocks=self.on_incomplete_blocks,
            axis_selections=self.normalized_axis_selections or None,
            offset_n_frames=offset_n_frames,
        )

        # Only files that made it into the index table are ever read, so judge metadata
        # on those: a file already dropped for being too short must not fail the loader
        # over a path nothing will ever ask it for.
        contributing_files = {file_name for file_name, _key, _selection in self.indices}
        metadata_gaps = {
            file_path: paths
            for file_path, paths in metadata_gaps.items()
            if file_path in contributing_files
        }
        if metadata_gaps:
            if self.on_missing_metadata == "error":
                raise _missing_metadata_error(metadata_gaps, len(contributing_files))
            log.info(
                f"Skipping {len(metadata_gaps)} / {len(contributing_files)} files that cannot "
                "supply the requested return_metadata paths."
            )
            self.indices = [entry for entry in self.indices if entry[0] not in metadata_gaps]
            contributing_files -= set(metadata_gaps)
        self._metadata_signatures = {
            file_path: signature
            for file_path, signature in self._metadata_signatures.items()
            if file_path in contributing_files
        }

        # Left for the caller to act on: only a batched loader stacks metadata across
        # files, and this source does not know whether it feeds one.
        self.metadata_batch_conflicts = _metadata_batch_conflicts(
            self._metadata_signatures,
            self._file_n_frames if self._slice_metadata_per_frame else {},
        )

        # Last, so the cap counts samples the loader actually yields rather than ones a
        # policy above was about to drop.
        if limit_n_examples is not None:
            log.info(
                f"H5DataSource: Limiting to {limit_n_examples} / {len(self.indices)} examples."
            )
            self.indices = self.indices[:limit_n_examples]

        self.sample_shapes = self._collect_sample_shapes()

        # Thread-local file handle caches (one per thread)
        self._local = threading.local()
        self._all_caches: set[H5FileHandleCache] = set()
        self._all_caches_lock = threading.Lock()

    def _collect_sample_shapes(self) -> dict[tuple, list[str]]:
        """Map each distinct sample shape this source yields to the files producing it.

        Derived from the index table rather than by reading, so it costs no I/O. Every
        block holds exactly ``n_frames`` frames, so all samples of a file share one
        shape; more than one entry means the dataset is ragged -- the files disagree on
        an axis other than the frame axis.
        """
        shape_by_path = dict(zip(self.file_paths, self.file_shapes))
        shapes: dict[tuple, list[str]] = defaultdict(list)
        # Samples of one file differ only in where their block starts, and the axes
        # iterated alongside it are indexed away with an int -- so the sample shape
        # follows from the file shape alone. Deriving one shape per distinct file shape
        # keeps this off the per-sample path.
        shape_by_file_shape: dict[tuple, tuple] = {}
        for file_name, _key, selection in self.indices:
            file_shape = shape_by_path[file_name]
            shape = shape_by_file_shape.get(file_shape)
            if shape is None:
                shape = shape_by_file_shape[file_shape] = self._sample_shape(file_shape, selection)
            files = shapes[shape]
            if len(files) < _MAX_LISTED_FILES and file_name not in files:
                files.append(file_name)
        return dict(shapes)

    def _place_frames(self, images):
        """Move the loaded frame axis into the output position.

        Pure axis bookkeeping -- it reads only shapes, never values -- so
        :meth:`_sample_shape` can run it on a dummy to learn the output shape.
        """
        # With n_frames=None the read used an int index, so there is no frame axis to
        # place and nothing to pad -- the sample already has the file's own layout.
        if self.n_frames is None:
            return images

        # __init__ rejects n_frames without a frame axis, so this is never None here.
        assert self.source_frame_axis is not None
        source = self.source_frame_axis
        if self.additional_axes_iter:
            # Axes iterated with an int index are gone from the loaded array.
            source -= sum(ax < self.source_frame_axis for ax in self.additional_axes_iter)
        return np.moveaxis(images, source, self.frame_axis)

    def _sample_shape(self, file_shape: tuple, selection: tuple) -> tuple:
        """The shape :meth:`__getitem__` returns for a sample read with ``selection``.

        Runs the real transform on a zero-strided dummy: ``broadcast_to`` of a 0-d
        array is one byte however large the sample, and slicing and ``moveaxis`` keep
        it a view. Going through :meth:`_place_frames` rather than re-deriving its
        arithmetic here is what keeps the two from drifting apart.
        """
        dummy = np.broadcast_to(np.zeros((), np.uint8), file_shape)[selection]
        return tuple(np.shape(self._place_frames(dummy)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        """Return a single sample as a numpy array. Thread-safe."""
        if self.cache and index in self._data_cache:
            return self._data_cache[index]

        file_name, key, indices = self.indices[index]
        file_handle_cache = self._get_file_handle_cache()
        file = file_handle_cache.get_file(file_name)

        try:
            # ``file.dataset`` rather than ``file[key]``: reads then go through
            # zea's concurrent chunk reader instead of h5py's serial path.
            images = file.dataset(key)[indices]
        except (OSError, IOError):
            # Invalidate cache entry and retry once
            file_handle_cache.pop(file_name)
            file = file_handle_cache.get_file(file_name)
            images = file.dataset(key)[indices]

        images = self._place_frames(images)

        if self.returns_metadata:
            result = (images, self._build_metadata(file, file_name, indices))
        else:
            result = images

        if self.cache:
            self._data_cache[index] = result

        return result

    def __repr__(self) -> str:
        return (
            f"H5DataSource(n_samples={len(self)}, n_files={len(self.file_paths)}, key='{self.key}')"
        )

    def _build_metadata(self, file: "File", file_name: str, indices: tuple) -> dict:
        """Build the metadata dict returned alongside a sample."""
        metadata = {}
        if self.return_metadata:
            metadata = self._get_file_metadata(file, file_name)
            if self._slice_metadata_per_frame:
                # Only set when the key has a frame axis, see __init__.
                assert self.source_frame_axis is not None
                metadata = slice_metadata(
                    metadata,
                    indices[self.source_frame_axis],
                    self._file_n_frames.get(file_name),
                )
            else:
                metadata = dict(metadata)
        metadata["file"] = {
            # For streamed hf:// files ``filename`` is a placeholder for the
            # underlying file object, so prefer the original source path.
            "fullpath": getattr(file, "_source_name", None) or file.filename,
            "filename": file.stem,
            "indices": indices,
        }
        return metadata

    def _get_file_metadata(self, file: "File", file_name: str) -> dict:
        """Return the requested metadata for *file*, reading it at most once per file."""
        with self._metadata_lock:
            cached = self._metadata_cache.get(file_name)
            if cached is not None:
                self._metadata_cache.move_to_end(file_name)
                return cached

        metadata = read_metadata(file, self.return_metadata or ())

        with self._metadata_lock:
            self._metadata_cache[file_name] = metadata
            self._metadata_cache.move_to_end(file_name)
            while len(self._metadata_cache) > FILE_HANDLE_CACHE_CAPACITY:
                self._metadata_cache.popitem(last=False)
        return metadata

    def _get_file_handle_cache(self) -> H5FileHandleCache:
        """Return the file-handle cache for the current thread.

        Re-registered on every call: ``close()`` empties the registry but cannot reach
        another thread's thread-local, so each thread re-registers its own cache or the
        handles it reopens go missing from the next ``close()``.
        """
        if not hasattr(self._local, "cache"):
            self._local.cache = H5FileHandleCache()
        with self._all_caches_lock:
            self._all_caches.add(self._local.cache)
        return self._local.cache

    def close(self):
        """Close all file handles across all threads.

        Handles reopen lazily on the next read, so the source stays usable afterwards.
        """
        with self._all_caches_lock:
            for c in self._all_caches:
                c.close()
            self._all_caches.clear()


class Dataloader:
    """High-performance HDF5 dataloader built on `Grain <https://github.com/google/grain>`_.

    .. code-block:: text

        grain threads (N) → h5py (thread-local handles) → numpy -> cpu tensor → user

    The entire pipeline runs using numpy, and the resizing is done on the selected
    backend, all on cpu.

    Does the following in order to load a dataset:

    - Find all .hdf5 files in the director(ies)
    - Load the data from each file using the specified key
    - Apply the following transformations in order (if specified):

      - offset_n_frames / axis_selections (applied at HDF5 read time)
      - limit_n_frames
      - limit_n_examples
      - shuffle
      - shard
      - add channel dim
      - clip image range
      - assert image range
      - resize
      - repeat
      - batch
      - cast to ``dtype`` (if specified)
      - normalize
      - augmentation
      - convert_to_tensor


    Args:
        file_paths: Path(s) to directory(ies) and/or HDF5 file(s).
        key: HDF5 dataset key.
        batch_size: Batch size. Set to ``None`` to disable batching.
            Default is ``16``. Stacking two or more samples (incl. metadata) requires them to have
            the same shape. This is checked when the loader is built. Note that ``image_size`` can
            resolve differing sample shapes; for the rest, use
            ``batch_size=None`` (or ``1``, which stacks nothing) and batch it yourself.
        n_frames: Number of consecutive frames per sample, placed on ``frame_axis``.
            Default is ``None``, which loads single frames *without* a frame axis, so a
            sample keeps the file's own layout for one frame. Set an int to group
            consecutive frames into blocks -- including ``n_frames=1``, which gives a
            length-1 frame axis. Frames are read from whichever axis the zea file spec
            names ``n_frames`` for ``key``.
        shuffle: Shuffle dataset each epoch. Default is ``True``.
        return_metadata: Return a ``(sample, metadata)`` tuple instead of a bare
            sample. ``False`` (default) returns arrays only. ``True`` returns just
            the file identity. An iterable of dotted paths additionally loads those
            fields from the file, e.g.
            ``["scan.sampling_frequency", "metadata.subject"]``; a path pointing at a
            group loads everything below it. Paths use the same syntax as
            ``file_filter``. The returned dict mirrors :class:`~zea.data.spec.FileSpec`, with
            the loader's own provenance under a ``"file"`` key::

                {
                    "scan": {"sampling_frequency": 40e6},
                    "metadata": {"subject": {"age": 61}},
                    "file": {"fullpath": ..., "filename": ..., "indices": ...},
                }

            Fields whose leading dimension is ``n_frames`` in the spec are sliced to the
            sample's frames so they stay aligned with the returned images. During construction,
            the loader checks that all files can supply the requested paths and that
            they have the same shapes, raising :exc:`KeyError` or :exc:`ValueError` naming the
            offending files. They can be dropped with ``on_missing_metadata="skip"`` or
            ``file_filter``. Use ``batch_size=None`` for metadata that genuinely varies in shape
            between files.
        seed: Random seed used for dataloader (e.g. shuffling). Default is ``None``.
            If ``None`` a random seed is generated.
        limit_n_examples: Cap the total number of examples (== item before batching) the loader
            yields, across all files (useful for debugging). Default is ``None`` (no limit).
            Note that this happens before shuffle.
        limit_n_frames: Maximum number of frames to load per file, counted from
            ``offset_n_frames``. Default is ``None`` (no limit).
        offset_n_frames: Frame index to start iteration from within each file.
            Combined with ``limit_n_frames`` this selects the half-open range
            ``[offset_n_frames, offset_n_frames + limit_n_frames)``. Default is ``0``.
        drop_remainder: Drop the final incomplete batch. Default is ``False``.
        image_size: Target ``(height, width)``. Default is ``None`` (no resizing).
            Setting it is what lets files of differing image size be batched together,
            since they arrive at the batch op already sharing a shape.
        resize_type: Resize strategy. One of ``"resize"``, ``"center_crop"``,
            ``"random_crop"`` or ``"crop_or_pad"``. Default is ``None``,
            which resolves to ``"resize"`` when `image_size` is set.
        resize_axes: Axes to resize along, must have length 2 (height, width).
            Only needed when data has more than ``(h, w, c)`` dimensions.
            Axes are interpreted after frame-axis insertion/reordering.
            Default is ``None``.
        resize_kwargs: Extra keyword arguments passed to ``Resizer``.
            Default is ``None``.
        image_range: Source value range of images, e.g. ``(-60, 0)``.
            Used for clipping/asserting/normalization. Default is ``None``.
        normalization_range: Target value range, e.g. ``(0, 1)``.
            If set, ``image_range`` must also be set. Default is ``None``.
        clip_image_range: Clip values to ``image_range`` before normalization.
            Default is ``False``.
        assert_image_range: Assert values stay within ``image_range``.
            Default is ``True``.
        dtype: Cast samples to this dtype (e.g. ``"float32"``, ``np.float16``) after
            batching and before normalization, so it is also what picks the precision
            ``normalization_range`` normalizes in. Must be floating point whenever
            ``normalization_range`` is set. Default is ``None``, which keeps the dtype
            the files hold -- except that files holding integers are promoted to
            ``float32``, since normalizing has no integer-valued result.
        dataset_repetitions: Repeat dataset this many times. Repetition happens
            after sharding. Default is ``None`` (no repetition).
        cache: Cache loaded samples in RAM. Default is ``False``.
            Note that with ``overlapping_blocks=True``, the same frame can be part of multiple
            samples, so caching will consume more memory.
        additional_axes_iter: Additional axes to iterate over, on top of the frame axis.
            Each becomes an integer index, so those axes are dropped from the sample.
            Default is ``None``.
        sort_files: Sort files numerically before indexing. Default is ``True``.
        overlapping_blocks: If ``True``, frame blocks overlap by ``n_frames - 1``.
            Has no effect unless ``n_frames > 1``. Default is ``False``.
        on_incomplete_blocks: What to do with files holding too few frames to fill one
            block of ``n_frames`` (spaced by ``frame_index_stride``). ``"error"``
            (default) refuses to build the loader and names the offending files;
            ``"skip"`` drops them from the dataset.
        on_missing_metadata: What to do with files that cannot supply a path requested
            through ``return_metadata``. ``"error"`` (default) refuses to build the
            loader and names the offending files; ``"skip"`` drops them from the
            dataset. Default is ``"error"``.
        augmentation: Callable applied to each batch after normalization.
            Default is ``None``.
        frame_index_stride: Step between selected frames in a block.
            Default is ``1``.
        frame_axis: Axis the frame block is placed on in the output. Only applies when
            ``n_frames`` is set; with ``n_frames=None`` there is no frame axis to place.
            Default is ``-1``, which puts frames in the trailing, channel-like
            position: an image batch comes out as ``(batch, height, width, n_frames)``,
            the channels-last layout ``Resizer`` and Keras expect. That is why
            resizing without explicit ``resize_axes`` requires ``frame_axis=-1`` --
            with the frame axis elsewhere, the default resize axes ``(1, 2)`` would
            no longer be height and width.

            For raw channel data there is no channel axis to double as, and the
            trailing frame axis scrambles the ``(n_tx, n_ax, n_el, n_ch)`` layout the
            processing pipeline wants. Set ``frame_axis=0`` there, so blocks keep the
            file's own ``(n_frames, n_tx, n_ax, n_el, n_ch)`` order.
        validate: Validate discovered files against the zea format.
            Default is ``False``.
        revision: HuggingFace revision (branch, tag, or commit hash) for ``hf://`` paths.
            Defaults to ``None`` (uses the default branch, typically ``"main"``).
        prefetch: Enable Grain prefetching for iteration. Default is ``True``.
        shard_index: Shard index to select when ``num_shards > 1``.
            Must satisfy ``0 <= shard_index < num_shards``.
        num_shards: Total number of shards for distributed loading.
            Sharding happens before downstream transforms. Default is ``1``.
        num_threads: Number of Grain read threads (``0`` means main thread only).
            Default is ``16``.
        prefetch_buffer_size: Size of the Grain buffer for reading elements per Python
            process (not per thread). Useful when reading from a distributed file
            system. Default is ``500``.
        reshuffle_each_epoch: Whether to reshuffle the dataset after each epoch.
            Default is ``True``. For evaluation it might be useful to set this to
            ``False``. Or when you want to use a persistent iterator between epochs, using
            ``dataset_repetitions`` to specify the number of epochs.
        convert_to_tensor: Whether to convert the data to a tensor (on cpu). Default is ``True``.
        axis_selections: Map of ``{axis: indices}`` applied at HDF5 read time to pre-filter
            non-frame axes. For example ``{1: [0, 2, 5]}`` loads only those indices along axis 1,
            avoiding reading unused data chunks from disk. A selection confined to a few chunks
            saves both memory and time, while one spread across every chunk still saves memory but
            reads about as much as the full axis. Default is ``None``.
        file_filter: Keep only files whose content matches a predicate, discarding the rest
            before any frames are indexed. Either a callable ``File -> bool`` (a file is kept
            when it returns ``True``), or a declarative dotted-path dict mapping a path on the
            :class:`~zea.data.file.File` to a condition: the :func:`~zea.data.datasets.EXISTS`
            helper (field must be present), a plain value (equality), or a callable on the
            resolved value. All dict entries are ANDed. Files whose predicate raises
            (e.g. they have no ``metadata`` group) are excluded. Default is ``None`` (no filtering).

    Example:
        .. code-block:: python

            loader = Dataloader(
                file_paths="/data/camus",
                key="data/image/values",
                batch_size=32,
                image_range=(-60, 0),
                normalization_range=(0, 1),
                image_size=(256, 256),
            )
            for batch in loader:
                ...  # batch.shape == (32, 256, 256, 1)

    Filtering examples:
        .. testsetup::

            import os

            import numpy as np

            from zea import File

            n_frames, n_tx, n_el, n_ax, grid = 2, 2, 8, 64, 16

            def _make(path, fat, sex, center_frequency):
                data = {
                    "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32),
                    "image": {"values": np.zeros((n_frames, grid, grid), dtype=np.uint8)},
                }
                scan = {
                    "sampling_frequency": np.float32(40e6),
                    "center_frequency": np.float32(center_frequency),
                    "demodulation_frequency": np.float32(center_frequency),
                    "initial_times": np.zeros(n_tx, dtype=np.float32),
                    "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
                    "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
                    "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
                    "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
                    "polar_angles": np.zeros(n_tx, dtype=np.float32),
                }
                subject = {"sex": sex}
                if fat is not None:
                    subject["fat_percentage"] = np.float32(fat)
                File.create(
                    path,
                    data=data,
                    scan=scan,
                    probe={"name": "demo", "probe_geometry": np.zeros((n_el, 3), dtype=np.float32)},
                    metadata={"subject": subject},
                    overwrite=True,
                )

            os.makedirs("filter-demo-dataset", exist_ok=True)
            _make("filter-demo-dataset/a.hdf5", fat=17.5, sex="f", center_frequency=5e6)
            _make("filter-demo-dataset/b.hdf5", fat=None, sex="m", center_frequency=9e6)

        .. testcode::

            from zea import Dataloader, EXISTS

            # callable: keep only files that record a subject fat percentage
            loader = Dataloader(
                file_paths="filter-demo-dataset",
                key="data/image/values",
                file_filter=lambda f: f.metadata.subject is not None
                and f.metadata.subject.fat_percentage is not None,
            )

            # metadata: load selected fields alongside each sample
            loader = Dataloader(
                file_paths="filter-demo-dataset",
                key="data/image/values",
                batch_size=None,
                return_metadata=["scan.center_frequency", "metadata.subject"],
            )
            sample, meta = next(iter(loader))
            assert meta["scan"]["center_frequency"] in (5e6, 9e6)
            assert meta["metadata"]["subject"]["sex"] in ("f", "m")
            assert meta["file"]["filename"] in ("a", "b")

            # dict: presence + equality + a value-level predicate (all ANDed)
            loader = Dataloader(
                file_paths="filter-demo-dataset",
                key="data/image/values",
                file_filter={
                    "metadata.subject.fat_percentage": EXISTS,
                    "metadata.subject.sex": "f",
                    "scan.center_frequency": lambda v: 4e6 <= v <= 6e6,
                },
            )

        .. testcleanup::

            import shutil

            shutil.rmtree("filter-demo-dataset")
    """

    def __init__(
        self,
        file_paths: List[str] | str,
        key: str,
        batch_size: int | None = 16,
        n_frames: int | None = None,
        shuffle: bool = True,
        return_metadata: bool | str | Sequence[str] | None = None,
        seed: int | None = None,
        limit_n_examples: int | None = None,
        limit_n_frames: int | None = None,
        offset_n_frames: int = 0,
        drop_remainder: bool = False,
        image_size: tuple | None = None,
        resize_type: str | None = None,
        resize_axes: tuple | None = None,
        resize_kwargs: dict | None = None,
        image_range: tuple | None = None,
        normalization_range: tuple | None = None,
        clip_image_range: bool = False,
        assert_image_range: bool = True,
        dtype: "str | np.dtype | None" = None,
        dataset_repetitions: int | None = None,
        cache: bool = False,
        additional_axes_iter: tuple | None = None,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        augmentation: Callable | None = None,
        on_incomplete_blocks: str = "error",
        on_missing_metadata: str = "error",
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        validate: bool = False,
        revision: str | None = None,
        prefetch: bool = True,
        shard_index: int | None = None,
        num_shards: int = 1,
        num_threads: int = 16,
        prefetch_buffer_size: int = 500,
        reshuffle_each_epoch: bool = True,
        convert_to_tensor: bool = True,
        axis_selections: dict | None = None,
        file_filter: "Callable[[File], bool] | dict | None" = None,
        **kwargs,
    ):
        # ── Validation ────────────────────────────────────────────────
        if normalization_range is not None:
            assert image_range is not None, (
                "If normalization_range is set, image_range must be set too."
            )
            assert dtype is None or np.issubdtype(np.dtype(dtype), np.floating), (
                "If normalization_range is set, dtype must be a floating point dtype, "
                f"got {np.dtype(dtype)}. Normalized values cannot be held in an integer, "
                "so the cast would be undone."
            )
        if num_shards > 1:
            assert shard_index is not None, "shard_index must be specified"
            assert 0 <= shard_index < num_shards

        resize_kwargs = resize_kwargs or {}

        # ── Store config ──────────────────────────────────────────────
        self.batch_size = batch_size
        self.return_metadata = normalize_metadata_paths(return_metadata)
        self.returns_metadata = self.return_metadata is not None
        self.num_threads = num_threads
        self.prefetch_buffer_size = prefetch_buffer_size
        self.prefetch = prefetch
        self._shuffle = shuffle
        self.reshuffle_each_epoch = reshuffle_each_epoch

        # Grain requires a concrete seed for shuffle — generate one if needed
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # ── Data source ───────────────────────────────────────────────
        self.source = H5DataSource(
            file_paths=file_paths,
            key=key,
            n_frames=n_frames,
            frame_index_stride=frame_index_stride,
            frame_axis=frame_axis,
            additional_axes_iter=additional_axes_iter,
            sort_files=sort_files,
            overlapping_blocks=overlapping_blocks,
            limit_n_examples=limit_n_examples,
            limit_n_frames=limit_n_frames,
            offset_n_frames=offset_n_frames,
            return_metadata=self.return_metadata,
            cache=cache,
            validate=validate,
            revision=revision,
            on_incomplete_blocks=on_incomplete_blocks,
            on_missing_metadata=on_missing_metadata,
            axis_selections=axis_selections,
            file_filter=file_filter,
            **kwargs,
        )

        # Only a loader stacking two or more samples has to reconcile metadata across
        # files, so this is exactly the case batch_size=None exists to escape.
        if (
            self.batch_size is not None
            and self.batch_size > 1
            and self.source.metadata_batch_conflicts
        ):
            raise _metadata_batch_error(self.source.metadata_batch_conflicts, self.batch_size)

        # ── Store pipeline config for rebuilding per epoch ────────────
        self._pipeline_cfg: dict[str, Any] = dict(
            num_shards=num_shards,
            shard_index=shard_index,
            clip_image_range=clip_image_range,
            assert_image_range=assert_image_range,
            dtype=None if dtype is None else np.dtype(dtype),
            image_range=image_range,
            normalization_range=normalization_range,
            dataset_repetitions=dataset_repetitions,
            drop_remainder=drop_remainder,
            augmentation=augmentation,
            resizer=None,  # set later
            convert_to_tensor=convert_to_tensor,
        )

        # Pre-build the resizer (stateless, reusable across epochs)
        if image_size or resize_type:
            resize_type = resize_type or "resize"
            if frame_axis != -1:
                assert resize_axes is not None, (
                    "Resizing without `resize_axes` assumes axes (1, 2) are height and "
                    "width, which holds only when frames sit in the trailing channel "
                    f"position (frame_axis=-1), but frame_axis={frame_axis}. Either "
                    "use frame_axis=-1 or name the spatial axes via resize_axes."
                )
            assert image_size is not None, (
                "image_size must be provided when resizing (resize_type is set)."
            )
            self._pipeline_cfg["resizer"] = Resizer(
                image_size=image_size,
                resize_type=resize_type,
                resize_axes=resize_axes,
                seed=seed,
                **resize_kwargs,
            )

        # The resizer can bring differing files to a common shape, so this is only
        # decidable once it is built. Both steps run the real pipeline code on the
        # source shapes rather than re-deriving what it does; everything else in the
        # pipeline (clip, cast, normalize, augment) leaves a sample's shape alone.
        resizer = self._pipeline_cfg["resizer"]
        self._sample_shapes: dict[tuple, list[str]] = {}
        for source_shape, files in self.source.sample_shapes.items():
            dummy = np.broadcast_to(np.zeros((), np.uint8), source_shape)
            shape = tuple(np.shape(self._ensure_channel_dim(dummy)))
            if resizer is not None:
                shape = _resized_shape(resizer, shape)
            self._sample_shapes.setdefault(shape, []).extend(files)

        # A batch of one stacks a single sample, so raggedness is harmless there.
        if len(self._sample_shapes) > 1 and self.batch_size is not None and self.batch_size > 1:
            raise _sample_shape_error(self._sample_shapes, self.batch_size)

        self._map_dataset = self._build_pipeline(seed)

        if len(self._map_dataset) == 0:
            raise ValueError(
                "Dataloader produced no samples. Check that the dataset is non-empty "
                "and that the filters/transforms do not discard all items."
            )

        if len(self._sample_shapes) > 1:
            # Ragged and unbatched: there is no one shape to report.
            self._shape = None
        elif self.returns_metadata:
            self._shape = self._map_dataset[0][0].shape
        else:
            self._shape = self._map_dataset[0].shape

    def _build_pipeline(self, seed: int):
        """Build the Grain MapDataset pipeline with the given seed."""
        cfg = self._pipeline_cfg

        def _ds_map(ds, fn):
            def on_cpu(x, _fn=fn):
                with keras.device("cpu"):
                    return _fn(x)

            if self.returns_metadata:
                return ds.map(lambda item: (on_cpu(item[0]), item[1]))
            return ds.map(on_cpu)

        ds = grain.MapDataset.source(self.source)

        # Set the seed for the whole pipeline
        ds = ds.seed(seed)

        if self._shuffle:
            ds = ds.shuffle()

        if cfg["num_shards"] > 1:
            ds = ds[cfg["shard_index"] :: cfg["num_shards"]]

        ds = _ds_map(ds, self._ensure_channel_dim)

        if cfg["clip_image_range"] and cfg["image_range"] is not None:
            lo, hi = cfg["image_range"]
            ds = _ds_map(ds, lambda x, _lo=lo, _hi=hi: np.clip(x, _lo, _hi))

        if cfg["assert_image_range"] and cfg["image_range"] is not None:
            _ir = cfg["image_range"]
            ds = _ds_map(ds, lambda x, _r=_ir: Dataloader._assert_image_range(x, _r))

        if cfg["resizer"] is not None:
            ds = _ds_map(ds, cfg["resizer"])
            ds = _ds_map(ds, ops.convert_to_numpy)

        if cfg["dataset_repetitions"] is not None:
            ds = ds.repeat(num_epochs=cfg["dataset_repetitions"])

        if self.batch_size is not None:
            ds = ds.batch(batch_size=self.batch_size, drop_remainder=cfg["drop_remainder"])

        if cfg["dtype"] is not None:
            ds = _ds_map(ds, lambda x, _d=cfg["dtype"]: x.astype(_d))

        if cfg["normalization_range"] is not None:
            _ir, _nr = cfg["image_range"], cfg["normalization_range"]
            ds = _ds_map(ds, lambda x, _a=_ir, _b=_nr: Dataloader._normalize(x, _a, _b))

        if cfg["augmentation"] is not None:
            ds = _ds_map(ds, cfg["augmentation"])

        if cfg["convert_to_tensor"]:
            ds = _ds_map(ds, ops.convert_to_tensor)

        return ds

    @property
    def dataset(self):
        """The underlying ``grain.MapDataset``."""
        return self._map_dataset

    @property
    def shape(self):
        """Output shape of one batch (or sample if unbatched).

        Raises:
            ValueError: If the loader yields more than one shape, which only an
                unbatched loader can do -- there is no single shape to report.
                Inspect :attr:`sample_shapes` for the shapes it does yield.
        """
        if self._shape is None:
            raise ValueError(
                "This Dataloader has no single sample shape: its files yield "
                f"{len(self._sample_shapes)} different shapes "
                f"({', '.join(str(shape) for shape in self._sample_shapes)}). "
                "Set image_size (with resize_type) to bring them to a common shape, or "
                "read Dataloader.sample_shapes for the per-shape breakdown."
            )
        return self._shape

    @property
    def sample_shapes(self) -> dict[tuple, list[str]]:
        """Each shape this loader yields, mapped to a few files that produce it.

        One entry for a well-formed dataset; more than one means the files disagree
        and only ``batch_size=None`` can iterate them.
        """
        return self._sample_shapes

    def to_iter_dataset(self) -> grain.IterDataset:
        """Convert to a ``grain.IterDataset`` with prefetching.

        This is called automatically when you iterate, but you can call
        it explicitly if you want to hold onto the ``IterDataset`` object.
        """

        return self._map_dataset.to_iter_dataset(
            grain.ReadOptions(
                num_threads=self.num_threads,
                prefetch_buffer_size=self.prefetch_buffer_size if self.prefetch else 0,
            )
        )

    def shuffle(self, seed: int | None = None):
        """(Re-)shuffle the dataset. Rebuilds the pipeline with a fresh seed."""

        seed = seed or int(self._rng.integers(0, 2**31))
        self._map_dataset = self._build_pipeline(seed=seed)

    def __iter__(self):
        if self._shuffle and self.reshuffle_each_epoch:
            self.shuffle()

        return iter(self.to_iter_dataset())

    def __len__(self):
        """Number of batches (or samples if unbatched)."""
        return len(self._map_dataset)

    def __repr__(self):
        return (
            f"Dataloader(n_samples={len(self.source)}, "
            f"batch_size={self.batch_size}, "
            f"key='{self.source.key}', "
            f"threads={self.num_threads})"
        )

    @staticmethod
    def _ensure_channel_dim(image):
        """Ensure at least 3-D (H, W, C) so batching produces uniform shapes."""
        if len(np.shape(image)) < 3:
            return np.expand_dims(image, axis=-1)
        return image

    @staticmethod
    def _normalize(image, image_range, normalization_range):
        """Map ``image`` from ``image_range`` to ``normalization_range``."""
        # Promote integer samples to float32 so normalization doesn't wrap around
        if not np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)
        return translate(image, image_range, normalization_range)

    @staticmethod
    def _assert_image_range(image, image_range):
        """Assert that image values are within the specified range."""
        minval = float(np.min(image))
        maxval = float(np.max(image))
        if minval < image_range[0]:
            raise ValueError(
                f"Image min {minval} is below image_range lower bound {image_range[0]}"
            )
        if maxval > image_range[1]:
            raise ValueError(
                f"Image max {maxval} is above image_range upper bound {image_range[1]}"
            )
        return image

    def summary(self):
        """Print dataset statistics and per-directory breakdown."""
        src = self.source
        total_samples = len(src)
        file_names = [idx[0] for idx in src.indices]
        directories = sorted({str(Path(f).parent) for f in file_names})
        samples_per_dir = count_samples_per_directory(file_names, directories)

        parts = [f"Dataloader with {total_samples} total samples:"]
        for dir_path, count in samples_per_dir.items():
            pct = (count / total_samples) * 100 if total_samples else 0
            parts.append(f"  {dir_path}: {count} samples ({pct:.1f}%)")
        print("\n".join(parts))

    def close(self):
        """Release file handles."""
        self.source.close()

    def __enter__(self) -> "Dataloader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        # Worker threads pin their file-handle caches for the loader's whole lifetime,
        # so dropping it must release them even without a close(). Swallow errors:
        # at interpreter shutdown h5py may already be torn down.
        try:
            self.close()
        except Exception:
            pass
