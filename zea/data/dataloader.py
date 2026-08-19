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
import warnings
from collections import OrderedDict
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
    has_per_frame_paths,
    normalize_metadata_paths,
    read_metadata,
    slice_metadata,
)
from zea.func.tensor import translate
from zea.utils import canonicalize_axis, map_negative_indices

if TYPE_CHECKING:
    from zea.data.file import File

DEFAULT_NORMALIZATION_RANGE = (0, 1)


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
                f"axis_selections axis {raw_axis} conflicts with initial_frame_axis "
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
    n_frames: int,
    frame_index_stride: int,
    key: str = "data/image",
    initial_frame_axis: int = 0,
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
        n_frames (int): Number of frames to load from each hdf5 file.
        frame_index_stride (int): Interval between frames to load.
        key (str, optional): Key of hdf5 dataset to grab data from. Defaults to "data/image".
        initial_frame_axis (int, optional): Axis to iterate over. Defaults to 0.
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
                    (slice(0, 1, 1), slice(None, 256, None), slice(None, 256, None)),
                ),
                (
                    "/folder/path_to_file.hdf5",
                    "data/image",
                    (slice(1, 2, 1), slice(None, 256, None), slice(None, 256, None)),
                ),
                ...,
            ]
    """
    if limit_n_frames is None:
        frame_limit: float = np.inf
    else:
        assert limit_n_frames > 0, f"limit_n_frames must be > 0, got {limit_n_frames}"
        frame_limit = float(limit_n_frames)

    assert len(file_paths) == len(file_shapes), "file_paths and file_shapes must have same length"

    if additional_axes_iter:
        # cannot contain initial_frame_axis
        assert initial_frame_axis not in additional_axes_iter, (
            "initial_frame_axis cannot be in additional_axes_iter. "
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

    # block size with stride included
    block_size = n_frames * frame_index_stride

    if not overlapping_blocks:
        block_step_size = block_size
    else:
        # now blocks overlap by n_frames - 1
        block_step_size = 1

    def axis_indices_files():
        # For every file
        for shape in file_shapes:
            total_frames_in_file = shape[initial_frame_axis]
            effective_end = int(min(total_frames_in_file, offset_n_frames + frame_limit))
            indices = [
                slice(i, i + block_size, frame_index_stride)
                for i in range(offset_n_frames, effective_end - block_size + 1, block_step_size)
            ]
            if not indices and pad_incomplete_blocks and effective_end > offset_n_frames:
                indices = [slice(offset_n_frames, effective_end, frame_index_stride)]
            yield [indices]

    indices = []
    skipped_files = 0
    for file, shape, axis_indices in zip(file_paths, file_shapes, list(axis_indices_files())):
        # remove all the files that have empty list at initial_frame_axis
        # this can happen if the file is too small to fit a block
        if not axis_indices[0]:  # initial_frame_axis is the first entry in axis_indices
            skipped_files += 1
            continue

        if additional_axes_iter:
            axis_indices += [list(range(shape[axis])) for axis in additional_axes_iter]

        axis_indices = product(*axis_indices)

        for axis_index in axis_indices:
            full_indices = [slice(size) for size in shape]
            for i, axis in enumerate([initial_frame_axis] + list(additional_axes_iter)):
                full_indices[axis] = axis_index[i]
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
            f"least n_frames * frame_index_stride = {n_frames * frame_index_stride}. "
        )

    return indices


def _resolve_return_metadata(return_metadata, return_filename) -> tuple[str, ...] | None:
    """Reconcile ``return_metadata`` with the deprecated ``return_filename`` flag.

    Returns ``None`` when samples are plain arrays, otherwise a (possibly empty)
    tuple of dotted metadata paths to load alongside the file identity.
    """
    if return_filename is not None:
        warnings.warn(
            "`return_filename` is deprecated and will be removed in a future release; "
            "use `return_metadata` instead. `return_metadata=True` returns the same "
            "file identity under a 'file' key, and a list of dotted paths (e.g. "
            "`return_metadata=['scan.sampling_frequency']`) additionally loads metadata "
            "from the file.",
            DeprecationWarning,
            stacklevel=3,
        )
        if return_metadata is None:
            return_metadata = return_filename
    return normalize_metadata_paths(return_metadata)


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
        n_frames: Number of consecutive frames per sample.
        frame_index_stride: Stride between frames.
        frame_axis: Axis along which frames are stacked in the output. Defaults to
            ``-1`` so frames land in the channel position for image data; see
            :class:`Dataloader`.
        insert_frame_axis: Whether to insert a new axis for frames.
        initial_frame_axis: Source axis that stores frames in the file.
        additional_axes_iter: Extra axes to iterate over.
        sort_files: Sort files numerically.
        overlapping_blocks: Allow overlapping frame blocks.
        limit_n_examples: Cap the number of examples (dataset length).
        limit_n_frames: Cap frames loaded per file.
        return_metadata: Return a ``(sample, metadata)`` tuple. See :class:`Dataloader`.
        return_filename: Deprecated alias for ``return_metadata``.
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
        n_frames: int = 1,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        insert_frame_axis: bool = True,
        initial_frame_axis: int = 0,
        additional_axes_iter: tuple | None = None,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        limit_n_examples: int | None = None,
        limit_n_frames: int | None = None,
        offset_n_frames: int = 0,
        return_metadata: bool | str | Sequence[str] | None = None,
        return_filename: bool | None = None,
        cache: bool = False,
        validate: bool = False,
        revision: str | None = None,
        pad_incomplete_blocks: bool = False,
        axis_selections: dict | None = None,
        file_filter: "Callable[[File], bool] | dict | None" = None,
        **kwargs,
    ):
        self.return_metadata = _resolve_return_metadata(return_metadata, return_filename)
        self.returns_metadata = self.return_metadata is not None
        # Deprecated alias, kept so existing code reading the attribute keeps working.
        self.return_filename = self.returns_metadata
        self.cache = cache
        self._data_cache = {}
        # Metadata is constant per file, so cache it per path rather than per sample.
        self._metadata_cache: OrderedDict[str, dict] = OrderedDict()
        self._metadata_lock = threading.Lock()
        self.pad_incomplete_blocks = pad_incomplete_blocks

        self.key = key
        self.n_frames = int(n_frames)
        self.frame_index_stride = int(frame_index_stride)
        self.frame_axis = int(frame_axis)
        self.insert_frame_axis = insert_frame_axis

        assert self.frame_index_stride > 0, (
            f"`frame_index_stride` must be > 0, got {self.frame_index_stride}"
        )
        assert self.n_frames > 0, f"`n_frames` must be > 0, got {self.n_frames}"

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
        self.file_shapes = _dataset.load_file_shapes(key)
        _dataset.close()

        num_dims = len(self.file_shapes[0]) if self.file_shapes else 0
        self.initial_frame_axis = canonicalize_axis(int(initial_frame_axis), num_dims)
        self.additional_axes_iter = map_negative_indices(list(additional_axes_iter or []), num_dims)

        self._file_n_frames = {
            path: shape[self.initial_frame_axis]
            for path, shape in zip(self.file_paths, self.file_shapes)
        }
        self._slice_metadata_per_frame = bool(
            self.return_metadata and has_per_frame_paths(self.return_metadata)
        )

        # Validate and normalize axis_selections
        reserved_axes = {self.initial_frame_axis} | set(self.additional_axes_iter)
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
            initial_frame_axis=self.initial_frame_axis,
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

        # Thread-local file handle caches (one per thread)
        self._local = threading.local()
        self._all_caches: set[H5FileHandleCache] = set()
        self._all_caches_lock = threading.Lock()

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

        if self.insert_frame_axis:
            initial = self.initial_frame_axis
            if self.additional_axes_iter:
                initial -= sum(ax < self.initial_frame_axis for ax in self.additional_axes_iter)
            images = np.moveaxis(images, initial, self.frame_axis)
        else:
            images = np.concatenate(images, axis=self.frame_axis)

        if self.pad_incomplete_blocks:
            n_loaded = images.shape[self.frame_axis]
            if n_loaded < self.n_frames:
                pad_width = [(0, 0)] * images.ndim
                pad_width[self.frame_axis] = (0, self.n_frames - n_loaded)
                images = np.pad(images, pad_width)

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
                metadata = slice_metadata(
                    metadata,
                    indices[self.initial_frame_axis],
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
            Default is ``16``.
        n_frames: Number of consecutive frames per sample. Default is ``1``.
            When ``n_frames > 1``, frames are grouped into blocks.
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
            aligned with the returned images. A requested path that is missing from
            a file raises :exc:`KeyError` — drop those files with ``file_filter``.
            Note that batching stacks metadata leaf by leaf, so every file must
            supply the same structure and shapes; use ``batch_size=None`` for
            metadata that varies in shape between files. Per-frame metadata is not
            zero-padded along with the images when ``pad_incomplete_blocks=True``.
        return_filename: Deprecated. Use ``return_metadata`` instead.
            ``return_filename=True`` behaves like ``return_metadata=True``, except
            that the file identity used to be returned at the top level and is now
            under the ``"file"`` key.
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
        additional_axes_iter: Additional axes to iterate over in addition to
            ``initial_frame_axis``. Default is ``None``.
        sort_files: Sort files numerically before indexing. Default is ``True``.
        overlapping_blocks: If ``True``, frame blocks overlap by ``n_frames - 1``.
            Has no effect when ``n_frames == 1``. Default is ``False``.
        pad_incomplete_blocks: If ``True``, keep files shorter than a full block and zeropad
            their samples up to ``n_frames``. Default is ``False``.
        augmentation: Callable applied to each batch after normalization.
            Default is ``None``.
        initial_frame_axis: Axis in file data that represents frames.
            Default is ``0``.
        insert_frame_axis: If ``True``, keep per-frame samples and move/insert
            the frame dimension at ``frame_axis``. If ``False``, loaded frames
            are concatenated along ``frame_axis``. Default is ``True``.
        frame_index_stride: Step between selected frames in a block.
            Default is ``1``.
        frame_axis: Axis along which frames are stacked/placed in output.
            Default is ``-1``, which puts frames in the trailing, channel-like
            position: an image batch comes out as ``(batch, height, width, n_frames)``,
            the channels-last layout ``Resizer`` and Keras expect. That is why
            resizing without explicit ``resize_axes`` requires ``frame_axis=-1`` --
            with the frame axis elsewhere, the default resize axes ``(1, 2)`` would
            no longer be height and width.

            For raw channel data there is no channel axis to double as, and the
            trailing frame axis scrambles the ``(n_tx, n_ax, n_el, n_ch)`` layout the
            processing pipeline wants. Set ``frame_axis=0`` there, so blocks keep the
            file's own ``(n_frames, n_tx, n_ax, n_el, n_ch)`` order (or drop the dummy
            axis with ``sample[..., 0]`` when ``n_frames=1``).
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
            avoiding reading unused data from disk. Reads go through
            :mod:`zea.data.chunk_reader`, so the saving tracks the *chunks* the selection
            touches rather than the number of indices: a selection confined to a few chunks
            saves both memory and time, while one spread across every chunk still saves
            memory but reads about as much as the full axis. Default is ``None``.
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
        key: str = "data/image",
        batch_size: int | None = 16,
        n_frames: int = 1,
        shuffle: bool = True,
        return_metadata: bool | str | Sequence[str] | None = None,
        return_filename: bool | None = None,
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
        initial_frame_axis: int = 0,
        insert_frame_axis: bool = True,
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
        self.return_metadata = _resolve_return_metadata(return_metadata, return_filename)
        self.returns_metadata = self.return_metadata is not None
        # Deprecated alias, kept so existing code reading the attribute keeps working.
        self.return_filename = self.returns_metadata
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
            insert_frame_axis=insert_frame_axis,
            initial_frame_axis=initial_frame_axis,
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

        self._map_dataset = self._build_pipeline(seed)

        if len(self._map_dataset) == 0:
            raise ValueError(
                "Dataloader produced no samples. Check that the dataset is non-empty "
                "and that the filters/transforms do not discard all items."
            )

        if self.returns_metadata:
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
        """Output shape of one batch (or sample if unbatched)."""
        return self._shape

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
