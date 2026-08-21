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
    pad_incomplete_blocks: bool = False,
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
        pad_incomplete_blocks (bool, optional): Keep files that are too short to fill a full block
            by emitting a single partial block with the available frames. The loader zeropads these
            samples to n_frames. Defaults to False.
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

    def frame_selections(shape):
        """Frame selections for one file, empty when it cannot fill a single block."""
        total_frames_in_file = shape[source_frame_axis]
        effective_end = int(min(total_frames_in_file, offset_n_frames + frame_limit))
        # An int rather than a slice when n_frames is None: h5py then drops the axis
        # on read, so single-frame samples never grow a length-1 frame dimension.
        selections = [
            i if n_frames is None else slice(i, i + block_size, frame_index_stride)
            for i in range(offset_n_frames, effective_end - block_size + 1, block_step_size)
        ]
        if not selections and pad_incomplete_blocks and effective_end > offset_n_frames:
            selections = [slice(offset_n_frames, effective_end, frame_index_stride)]
        return selections

    # The frame axis leads the product when there is one; without it (a field the spec
    # gives no n_frames axis) every file contributes exactly one sample.
    iter_axes = ([] if source_frame_axis is None else [source_frame_axis]) + list(
        additional_axes_iter
    )

    indices = []
    skipped_files = 0
    for file, shape in zip(file_paths, file_shapes):
        axis_indices = []
        if source_frame_axis is not None:
            selections = frame_selections(shape)
            # drop files that are too small to fit a single block
            if not selections:
                skipped_files += 1
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

    if skipped_files > 0:
        log.warning(
            f"Skipping {skipped_files} files with not enough frames "
            f"which is about {skipped_files / len(file_paths) * 100:.2f}% of the "
            f"dataset. This can be fine if you expect set `n_frames` and "
            "`frame_index_stride` to be high. Minimum frames in a file needs to be at "
            f"least {block_size}. "
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
        shown = ", ".join(f"'{Path(f).name}'" for f in files[:_MAX_LISTED_FILES])
        if len(files) > _MAX_LISTED_FILES:
            shown += f", +{len(files) - _MAX_LISTED_FILES} more"
        parts.append(
            f"return_metadata path '{path}' is missing from {len(files)}/{n_files} files "
            f"({shown}); drop them with file_filter={{'{path}': EXISTS}}"
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
        pad_incomplete_blocks: bool = False,
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
        self.pad_incomplete_blocks = pad_incomplete_blocks

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
        assert not (pad_incomplete_blocks and self.n_frames is None), (
            "`pad_incomplete_blocks` pads samples up to `n_frames`, so it needs an "
            "`n_frames` to pad to. Set n_frames, or drop pad_incomplete_blocks."
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

        if self.source_frame_axis is None and self.n_frames is not None:
            raise ValueError(
                f"'{key}' has no frame axis in the zea file spec (its dimensions are "
                f"{dim_names_for_key(key, num_dims)}), so frames cannot be grouped into "
                f"blocks of n_frames={self.n_frames}. Use n_frames=None to load one "
                "sample per file."
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
            pad_incomplete_blocks=pad_incomplete_blocks,
            axis_selections=self.normalized_axis_selections or None,
            offset_n_frames=offset_n_frames,
        )

        if limit_n_examples is not None:
            log.info(
                f"H5DataSource: Limiting to {limit_n_examples} / {len(self.indices)} examples."
            )
            self.indices = self.indices[:limit_n_examples]

        # Only files that made it into the index table are ever read, so judge metadata
        # on those: a file dropped for being too short to fill a block must not fail the
        # loader over a path nothing will ever ask it for.
        contributing_files = {file_name for file_name, _key, _selection in self.indices}
        metadata_gaps = {
            file_path: paths
            for file_path, paths in metadata_gaps.items()
            if file_path in contributing_files
        }
        if metadata_gaps:
            raise _missing_metadata_error(metadata_gaps, len(contributing_files))
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

        self.sample_shapes = self._collect_sample_shapes()

        # Thread-local file handle caches (one per thread)
        self._local = threading.local()
        self._all_caches: set[H5FileHandleCache] = set()
        self._all_caches_lock = threading.Lock()

    def _collect_sample_shapes(self) -> dict[tuple, list[str]]:
        """Map each distinct sample shape this source yields to the files producing it.

        Derived from the index table rather than by reading, so it costs no I/O. More
        than one entry means the dataset is ragged: the files disagree on everything
        but the frame axis, or a file's last block is shorter than ``n_frames``.
        """
        shape_by_path = dict(zip(self.file_paths, self.file_shapes))
        shapes: dict[tuple, list[str]] = defaultdict(list)
        # A file's samples differ only in how many frames the block spans -- the other
        # iterated axes are indexed away with an int -- so one file has at most a full
        # block and a shorter tail. Deriving a shape per (file shape, block length)
        # rather than per sample also spares :meth:`_place_frames` from materializing a
        # padded dummy for every sample of a padded file.
        shape_by_block: dict[tuple, tuple] = {}
        for file_name, _key, selection in self.indices:
            file_shape = shape_by_path[file_name]
            block = (file_shape, self._block_length(file_shape, selection))
            shape = shape_by_block.get(block)
            if shape is None:
                shape = shape_by_block[block] = self._sample_shape(file_shape, selection)
            files = shapes[shape]
            if len(files) < _MAX_LISTED_FILES and file_name not in files:
                files.append(file_name)
        return dict(shapes)

    def _block_length(self, file_shape: tuple, selection: tuple) -> int | None:
        """Frames the sample read with ``selection`` spans, before any padding.

        ``None`` when the frame axis never reaches the sample: the data has no frame
        axis, or ``n_frames=None`` selected a single frame with an int index.
        """
        if self.source_frame_axis is None:
            return None
        frame_selection = selection[self.source_frame_axis]
        if not isinstance(frame_selection, slice):
            return None
        return len(range(*frame_selection.indices(file_shape[self.source_frame_axis])))

    def _place_frames(self, images):
        """Move the loaded frame axis into place and pad a block short of ``n_frames``.

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
        images = np.moveaxis(images, source, self.frame_axis)

        if self.pad_incomplete_blocks:
            n_loaded = images.shape[self.frame_axis]
            if n_loaded < self.n_frames:
                pad_width = [(0, 0)] * images.ndim
                pad_width[self.frame_axis] = (0, self.n_frames - n_loaded)
                images = np.pad(images, pad_width)
        return images

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
        """Return the file-handle cache for the current thread."""
        if not hasattr(self._local, "cache"):
            self._local.cache = H5FileHandleCache()
            with self._all_caches_lock:
                self._all_caches.add(self._local.cache)
        return self._local.cache

    def close(self):
        """Close all file handles across all threads."""
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
        key: HDF5 dataset key. Default is ``"data/image"``.
        batch_size: Batch size. Set to ``None`` to disable batching.
            Default is ``16``. Stacking two or more samples requires them to have the
            same shape, and metadata requested via ``return_metadata`` is stacked leaf
            by leaf just like the data, so every file must supply the same fields with
            the same shapes. Both are checked when the loader is built, which refuses
            to construct rather than failing on whichever batch first straddles two
            files. ``image_size`` resolves differing sample shapes; for the rest, use
            ``batch_size=None`` (or ``1``, which stacks nothing) and batch it yourself.
        n_frames: Number of consecutive frames per sample, placed on ``frame_axis``.
            Default is ``None``, which loads single frames *without* a frame axis, so a
            sample keeps the file's own layout for one frame. Set an int to group
            consecutive frames into blocks -- including ``n_frames=1``, which gives a
            length-1 frame axis. Frames are read from whichever axis the zea file spec
            names ``n_frames`` for ``key``; a field the spec gives no such axis is only
            loadable with ``n_frames=None``, one sample per file. Note that a 2-D sample
            still picks up a trailing channel axis on its way through the pipeline, so
            for plain images ``n_frames=None`` and ``n_frames=1`` batch identically.
        shuffle: Shuffle dataset each epoch. Default is ``True``.
        return_metadata: Return a ``(sample, metadata)`` tuple instead of a bare
            sample. ``False`` (default) returns arrays only. ``True`` returns just
            the file identity. An iterable of dotted paths additionally loads those
            fields from the file, e.g.
            ``["scan.sampling_frequency", "metadata.subject"]``; a path pointing at a
            group loads everything below it. Paths use the same syntax as
            ``file_filter`` and are read straight off the HDF5 groups, so only what
            you ask for is loaded. The returned dict mirrors
            :class:`~zea.data.spec.FileSpec`, with the loader's own provenance under
            a ``"file"`` key::

                {
                    "scan": {"sampling_frequency": 40e6},
                    "metadata": {"subject": {"age": 61}},
                    "file": {"fullpath": ..., "filename": ..., "indices": ...},
                }

            Metadata is read once per file, not once per sample. Fields whose
            leading dimension is ``n_frames`` in the spec (per-frame annotations,
            per-frame metrics) are sliced to the sample's frames so they stay
            aligned with the returned images. Because metadata is per file, a requested
            path is either present for every sample of a file or for none of them, so
            every file is checked for the requested paths when the loader is built:
            any that cannot answer raise :exc:`KeyError` there and then, naming them,
            rather than failing partway through an epoch. Drop them with e.g.
            ``file_filter={"metadata.subject.age": EXISTS}``. A per-frame field stored in
            a broadcast form the spec allows -- one value for the whole file, such as a
            single ``annotations.view`` string or a map grid with its leading frame axis
            omitted -- is returned in full with every sample rather than sliced.
            Batching stacks metadata leaf by leaf, so every file must supply the same
            leaves with the same shapes -- this too is checked when the loader is
            built, and raises :exc:`ValueError` naming the leaves that disagree. Files
            may still hold different numbers of frames: that axis is sliced away before
            stacking. Use ``batch_size=None`` for metadata that genuinely varies in
            shape between files. Per-frame metadata is not zero-padded along with the
            images when ``pad_incomplete_blocks=True``.
        seed: Random seed used for dataloader (e.g. shuffling). Default is ``None``.
            If ``None`` a random seed is generated.
        limit_n_examples: Cap the total number of examples the loader yields, across
            all files (useful for debugging). Default is ``None`` (no limit). An
            "example" is one item before batching -- a single frame, or a block of
            ``n_frames`` consecutive frames -- so this is neither a count of files nor,
            despite the ultrasound sense of the word, of axial samples. Contrast
            ``limit_n_frames``, which caps frames *per file*. Note that this happens
            before shuffle!
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
            batching and before normalization. Default is ``None``, which leaves the
            dtype the files provide untouched. Note that normalization of integer data
            needs a floating point ``dtype``, otherwise ``normalization_range`` is
            applied with integer arithmetic.
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
        pad_incomplete_blocks: If ``True``, keep files shorter than a full block and zeropad
            their samples up to ``n_frames``. Requires ``n_frames``. Default is ``False``.
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
        pad_incomplete_blocks: bool = False,
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
            pad_incomplete_blocks=pad_incomplete_blocks,
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
            ds = _ds_map(ds, lambda x, _a=_ir, _b=_nr: translate(x, _a, _b))

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
