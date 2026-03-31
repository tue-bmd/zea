"""H5 dataloader for loading images from zea datasets.

Example:
    .. code-block:: python

        from zea import Dataloader

        loader = Dataloader(
            file_paths="/path/to/dataset",
            key="data/image",
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
from itertools import product
from pathlib import Path
from typing import List

import grain
import keras
import numpy as np

from zea import log
from zea.data.datasets import Dataset, H5FileHandleCache, count_samples_per_directory
from zea.data.layers import Resizer
from zea.utils import canonicalize_axis, map_negative_indices

DEFAULT_NORMALIZATION_RANGE = (0, 1)


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
        limit_n_frames (int, optional): Limit the number of frames to load from each file. This
            means n_frames per data file will be used. These will be the first frames in the file.
            Defaults to None.

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
        limit_n_frames = np.inf
    else:
        assert limit_n_frames > 0, f"limit_n_frames must be > 0, got {limit_n_frames}"

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
            n_frames_in_file = shape[initial_frame_axis]
            # Optionally limit frames to load from each file
            n_frames_in_file = min(n_frames_in_file, limit_n_frames)
            indices = [
                slice(i, i + block_size, frame_index_stride)
                for i in range(0, n_frames_in_file - block_size + 1, block_step_size)
            ]
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
        frame_axis: Axis along which frames are stacked in the output.
        insert_frame_axis: Whether to insert a new axis for frames.
        initial_frame_axis: Source axis that stores frames in the file.
        additional_axes_iter: Extra axes to iterate over.
        sort_files: Sort files numerically.
        overlapping_blocks: Allow overlapping frame blocks.
        limit_n_samples: Cap the number of samples.
        limit_n_frames: Cap frames loaded per file.
        return_filename: Return filename metadata with each sample.
        cache: Cache loaded samples to RAM.
        validate: Validate dataset against the zea format.
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
        limit_n_samples: int | None = None,
        limit_n_frames: int | None = None,
        return_filename: bool = False,
        cache: bool = False,
        validate: bool = True,
        **kwargs,
    ):
        self.return_filename = return_filename
        self.cache = cache
        self._data_cache = {}

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
        _dataset = Dataset(file_paths, validate=validate, **kwargs)
        self.file_paths = _dataset.file_paths
        self.file_shapes = _dataset.load_file_shapes(key)
        _dataset.close()

        num_dims = len(self.file_shapes[0])
        self.initial_frame_axis = canonicalize_axis(int(initial_frame_axis), num_dims)
        self.additional_axes_iter = map_negative_indices(list(additional_axes_iter or []), num_dims)

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
        )

        if limit_n_samples is not None:
            log.info(f"H5DataSource: Limiting to {limit_n_samples} / {len(self.indices)} samples.")
            self.indices = self.indices[:limit_n_samples]

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
            images = file.load_data(key, indices)
        except (OSError, IOError):
            # Invalidate cache entry and retry once
            file_handle_cache.pop(file_name)
            file = file_handle_cache.get_file(file_name)
            images = file.load_data(key, indices)

        if self.insert_frame_axis:
            initial = self.initial_frame_axis
            if self.additional_axes_iter:
                initial -= sum(ax < self.initial_frame_axis for ax in self.additional_axes_iter)
            images = np.moveaxis(images, initial, self.frame_axis)
        else:
            images = np.concatenate(images, axis=self.frame_axis)

        if self.return_filename:
            file_data = {
                "fullpath": file.filename,  # same as file.path, but str type
                "filename": file.stem,
                "indices": indices,
            }
            result = (images, file_data)
        else:
            result = images

        if self.cache:
            self._data_cache[index] = result

        return result

    def __repr__(self) -> str:
        return (
            f"H5DataSource(n_samples={len(self)}, n_files={len(self.file_paths)}, key='{self.key}')"
        )

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

        grain threads (N) → h5py (thread-local handles) → numpy → user

    The entire pipeline runs in numpy — no framework dependency until
    you feed tensors to your model.

    Does the following in order to load a dataset:

        - Find all .hdf5 files in the director(ies)
        - Load the data from each file using the specified key
        - Apply the following transformations in order (if specified):

            - shuffle
            - shard
            - add channel dim
            - clip_image_range
            - assert_image_range
            - resize
            - repeat
            - batch
            - normalize
            - augmentation

    Args:
        file_paths: Path(s) to directory(ies) and/or HDF5 file(s).
        key: HDF5 dataset key. Default is ``"data/image"``.
        batch_size: Batch size. Set to ``None`` to disable batching.
            Default is ``16``.
        n_frames: Number of consecutive frames per sample. Default is ``1``.
            When ``n_frames > 1``, frames are grouped into blocks.
        shuffle: Shuffle dataset each epoch. Default is ``True``.
        return_filename: Return filename metadata together with each sample.
            Default is ``False``.
        seed: Random seed used for shuffling. Default is ``None``.
            If ``None`` and ``shuffle=True``, a random seed is generated.
        limit_n_samples: Limit total number of samples (useful for debugging).
            Default is ``None`` (no limit).
        limit_n_frames: Limit frames loaded per file to the first N frames.
            Default is ``None`` (no limit).
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
            Default is ``-1``.
        validate: Validate discovered files against the zea format.
            Default is ``True``.
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

    Example:
        .. code-block:: python

            loader = Dataloader(
                file_paths="/data/camus",
                key="data/image_sc",
                batch_size=32,
                image_range=(-60, 0),
                normalization_range=(0, 1),
                image_size=(256, 256),
            )
            for batch in loader:
                ...  # batch.shape == (32, 256, 256, 1)
    """

    def __init__(
        self,
        file_paths: List[str] | str,
        key: str = "data/image",
        batch_size: int | None = 16,
        n_frames: int = 1,
        shuffle: bool = True,
        return_filename: bool = False,
        seed: int | None = None,
        limit_n_samples: int | None = None,
        limit_n_frames: int | None = None,
        drop_remainder: bool = False,
        image_size: tuple | None = None,
        resize_type: str | None = None,
        resize_axes: tuple | None = None,
        resize_kwargs: dict | None = None,
        image_range: tuple | None = None,
        normalization_range: tuple | None = None,
        clip_image_range: bool = False,
        assert_image_range: bool = True,
        dataset_repetitions: int | None = None,
        cache: bool = False,
        additional_axes_iter: tuple | None = None,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        augmentation: callable = None,
        initial_frame_axis: int = 0,
        insert_frame_axis: bool = True,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        validate: bool = True,
        prefetch: bool = True,
        shard_index: int | None = None,
        num_shards: int = 1,
        num_threads: int = 16,
        prefetch_buffer_size: int = 500,
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
        self.return_filename = return_filename
        self.num_threads = num_threads
        self.prefetch_buffer_size = prefetch_buffer_size
        self.prefetch = prefetch
        self.shuffle = shuffle

        # Grain requires a concrete seed for shuffle — generate one if needed
        if seed is None and shuffle:
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
            limit_n_samples=limit_n_samples,
            limit_n_frames=limit_n_frames,
            return_filename=return_filename,
            cache=cache,
            validate=validate,
            **kwargs,
        )

        # ── Store pipeline config for rebuilding per epoch ────────────
        self._pipeline_cfg = dict(
            num_shards=num_shards,
            shard_index=shard_index,
            clip_image_range=clip_image_range,
            assert_image_range=assert_image_range,
            image_range=image_range,
            normalization_range=normalization_range,
            dataset_repetitions=dataset_repetitions,
            drop_remainder=drop_remainder,
            augmentation=augmentation,
            resizer=None,
        )

        # Pre-build the resizer (stateless, reusable across epochs)
        if image_size or resize_type:
            resize_type = resize_type or "resize"
            if frame_axis != -1:
                assert resize_axes is not None, (
                    "Resizing only works with frame_axis = -1. Alternatively, "
                    "you can specify resize_axes."
                )
            self._pipeline_cfg["resizer"] = Resizer(
                image_size=image_size,
                resize_type=resize_type,
                resize_axes=resize_axes,
                seed=seed,
                **resize_kwargs,
            )

        self._map_dataset = self._build_pipeline(seed)

    def _build_pipeline(self, seed: int):
        """Build the Grain MapDataset pipeline with the given shuffle seed."""
        cfg = self._pipeline_cfg

        def _ds_map(ds, fn):
            if self.return_filename:
                return ds.map(lambda item: (fn(item[0]), item[1]))
            return ds.map(fn)

        ds = grain.MapDataset.source(self.source)

        if self.shuffle:
            ds = ds.shuffle(seed=seed)

        if cfg["num_shards"] > 1:
            ds = ds[cfg["shard_index"] :: cfg["num_shards"]]

        ds = _ds_map(ds, self._ensure_channel_dim)

        if cfg["clip_image_range"] and cfg["image_range"] is not None:
            lo, hi = cfg["image_range"]
            ds = _ds_map(ds, lambda x, _lo=lo, _hi=hi: keras.ops.clip(x, _lo, _hi))

        if cfg["assert_image_range"] and cfg["image_range"] is not None:
            _ir = cfg["image_range"]
            ds = _ds_map(ds, lambda x, _r=_ir: Dataloader._assert_image_range(x, _r))

        if cfg["resizer"] is not None:
            ds = _ds_map(ds, cfg["resizer"])

        if cfg["dataset_repetitions"] is not None:
            ds = ds.repeat(num_epochs=cfg["dataset_repetitions"])

        if self.batch_size is not None:
            ds = ds.batch(batch_size=self.batch_size, drop_remainder=cfg["drop_remainder"])

        if cfg["normalization_range"] is not None:
            _ir, _nr = cfg["image_range"], cfg["normalization_range"]
            ds = _ds_map(ds, lambda x, _a=_ir, _b=_nr: Dataloader._normalize(x, _a, _b))

        if cfg["augmentation"] is not None:
            ds = _ds_map(ds, cfg["augmentation"])

        return ds

    @property
    def dataset(self):
        """The underlying ``grain.MapDataset``."""
        return self._map_dataset

    def to_iter_dataset(self):
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

    def __iter__(self):
        # Rebuild the pipeline with a fresh seed so each epoch sees a different order
        if self.shuffle:
            self._map_dataset = self._build_pipeline(seed=int(self._rng.integers(0, 2**31)))
        return iter(self.to_iter_dataset())

    def __len__(self):
        """Number of batches (or samples if unbatched)."""
        return len(self._map_dataset)

    def __repr__(self):
        return (
            f"<Dataloader: {len(self.source)} samples, "
            f"batch_size={self.batch_size}, "
            f"key='{self.source.key}', "
            f"threads={self.num_threads}>"
        )

    @staticmethod
    def _ensure_channel_dim(image):
        """Ensure at least 3-D (H, W, C) so batching produces uniform shapes."""
        if len(keras.ops.shape(image)) < 3:
            return keras.ops.expand_dims(image, axis=-1)
        return image

    @staticmethod
    def _assert_image_range(image, image_range):
        """Assert that image values are within the specified range."""
        minval = float(keras.ops.min(image))
        maxval = float(keras.ops.max(image))
        if minval < image_range[0]:
            raise ValueError(
                f"Image min {minval} is below image_range lower bound {image_range[0]}"
            )
        if maxval > image_range[1]:
            raise ValueError(
                f"Image max {maxval} is above image_range upper bound {image_range[1]}"
            )
        return image

    @staticmethod
    def _normalize(image, image_range, normalization_range):
        """Normalize image from image_range to normalization_range."""
        left_min, left_max = image_range
        right_min, right_max = normalization_range
        scale = (right_max - right_min) / (left_max - left_min)
        offset = right_min - scale * left_min
        return keras.ops.add(keras.ops.multiply(image, scale), offset)

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
