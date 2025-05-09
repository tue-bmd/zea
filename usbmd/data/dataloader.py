"""
H5 dataloader for loading images from USBMD datasets.

This module can be used with any backend.
"""

import copy
import math
import re
from itertools import product
from typing import List

import keras
import numpy as np
from keras import ops
from keras.src.utils import backend_utils

from usbmd.data.datasets import Dataset
from usbmd.data.file import File
from usbmd.data.layers import Resizer
from usbmd.utils import log, map_negative_indices, translate
from usbmd.utils.io_lib import retry_on_io_error

if keras.backend.backend() == "jax":
    from usbmd.backend.jax import on_device_jax as on_device
elif keras.backend.backend() == "tensorflow":
    from usbmd.backend.tensorflow import on_device_tf as on_device
elif keras.backend.backend() == "torch":
    from usbmd.backend.torch import on_device_torch as on_device
else:

    def on_device(func, *args, **kwargs):
        """Dummy function for non-backend specific code."""
        return func(*args, **kwargs)


DEFAULT_NORMALIZATION_RANGE = (0, 1)
MAX_RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 0.1


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
        file_paths (list): list of file paths.
        file_shapes (list): list of file shapes.
        n_frames (int): number of frames to load from each hdf5 file.
        frame_index_stride (int): interval between frames to load.
        key (str, optional): key of hdf5 dataset to grab data from. Defaults to "data/image".
        initial_frame_axis (int, optional): axis to iterate over. Defaults to 0.
        additional_axes_iter (list, optional): additional axes to iterate over in the dataset.
            Defaults to None.
        sort_files (bool, optional): sort files by number. Defaults to True.
        overlapping_blocks (bool, optional): will take n_frames from sequence, then move by 1.
            Defaults to False.
        limit_n_frames (int, optional): limit the number of frames to load from each file. This
            means n_frames per data file will be used. These will be the first frames in the file.
            Defaults to None

    Returns:
        list: list of tuples with indices to extract images from hdf5 files.
            (file_name, key, indices) with indices being a tuple of slices.
            example: [
                ('/folder/path_to_file.hdf5', 'data/image', [range(0, 1), slice(None, 256, None), slice(None, 256, None)]), # pylint: disable=line-too-long
                ('/folder/path_to_file.hdf5', 'data/image', [range(1, 2), slice(None, 256, None), slice(None, 256, None)]), # pylint: disable=line-too-long
                ...
            ]

    """
    if not limit_n_frames:
        limit_n_frames = np.inf

    assert len(file_paths) == len(
        file_shapes
    ), "file_paths and file_shapes must have same length"

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
        except:
            log.warning("H5Generator: Could not sort file_paths by number.")

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
                range(i, i + block_size, frame_index_stride)
                for i in range(0, n_frames_in_file - block_size + 1, block_step_size)
            ]
            yield [indices]

    indices = []
    skipped_files = 0
    for file, shape, axis_indices in zip(
        file_paths, file_shapes, list(axis_indices_files())
    ):
        # remove all the files that have empty list at initial_frame_axis
        # this can happen if the file is too small to fit a block
        if not axis_indices[0]:  # initial_frame_axis is the first entry in axis_indices
            skipped_files += 1
            continue

        if additional_axes_iter:
            axis_indices += [range(shape[axis]) for axis in additional_axes_iter]

        axis_indices = product(*axis_indices)

        for axis_index in axis_indices:
            full_indices = [slice(size) for size in shape]
            for i, axis in enumerate([initial_frame_axis] + list(additional_axes_iter)):
                full_indices[axis] = axis_index[i]
            indices.append((file, key, full_indices))

    if skipped_files > 0:
        log.warning(
            f"H5Generator: Skipping {skipped_files} files with not enough frames "
            f"which is about {skipped_files / len(file_paths) * 100:.2f}% of the "
            f"dataset. This can be fine if you expect set `n_frames` and "
            "`frame_index_stride` to be high. Minimum frames in a file needs to be at "
            f"least n_frames * frame_index_stride = {n_frames * frame_index_stride}. "
        )

    return indices


def _h5_reopen_on_io_error(
    dataloader_obj,
    indices,
    retry_count,
    **kwargs,  # pylint: disable=unused-argument
):
    """Reopen the file if an I/O error occurs.
    Also removes the file from the cache and try to close file.
    """
    file_name = indices[0]
    try:
        file = dataloader_obj._file_handle_cache.pop(file_name, None)
        if file is not None:
            file.close()
    except Exception:
        pass

    log.warning(
        f"H5Generator: I/O error occurred while reading file {file_name}. "
        f"Retry opening file. Retry count: {retry_count}."
    )


class H5Generator(Dataset, keras.utils.PyDataset):
    """Generator from h5 file using provided indices."""

    def __init__(
        self,
        file_paths: List[str],
        key: str = "data/image",
        n_frames: int = 1,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        insert_frame_axis: bool = True,
        initial_frame_axis: int = 0,
        return_filename: bool = False,
        shuffle: bool = True,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        limit_n_samples: int | None = None,
        limit_n_frames: int | None = None,
        seed: int | None = None,
        batch_size: int = 1,
        as_tensor: bool = True,
        additional_axes_iter: tuple | None = None,
        drop_remainder: bool = False,
        caching: bool = False,
        **kwargs,
    ):
        super().__init__(file_paths, key, **kwargs)

        self.n_frames = int(n_frames)
        self.frame_index_stride = int(frame_index_stride)
        self.frame_axis = int(frame_axis)
        self.insert_frame_axis = insert_frame_axis
        self.initial_frame_axis = int(initial_frame_axis)
        self.return_filename = return_filename
        self.shuffle = shuffle
        self.sort_files = sort_files
        self.overlapping_blocks = overlapping_blocks
        self.limit_n_samples = limit_n_samples
        self.limit_n_frames = limit_n_frames
        self.seed = seed
        self.batch_size = batch_size
        self.as_tensor = as_tensor
        self.additional_axes_iter = additional_axes_iter or []
        self.drop_remainder = drop_remainder
        self.caching = caching

        self.maybe_tensor = ops.convert_to_tensor if self.as_tensor else lambda x: x

        assert (
            self.frame_index_stride > 0
        ), f"`frame_index_stride` must be greater than 0, got {self.frame_index_stride}"
        assert (
            self.n_frames > 0
        ), f"`n_frames` must be greater than 0, got {self.n_frames}"

        # Extract some general information about the dataset
        image_shapes = np.array(self.file_shapes)
        image_shapes = np.delete(
            image_shapes, (self.initial_frame_axis, *self.additional_axes_iter), axis=1
        )
        n_dims = len(image_shapes[0])

        self.equal_file_shapes = np.all(image_shapes == image_shapes[0])
        if not self.equal_file_shapes:
            log.warning(
                "H5Generator: Not all files have the same shape. "
                "This can lead to issues when resizing images later...."
            )
            self.shape = np.array([None] * n_dims)
        else:
            self.shape = np.array(image_shapes[0])

        if insert_frame_axis:
            _frame_axis = map_negative_indices([frame_axis], len(self.shape) + 1)
            self.shape = np.insert(self.shape, _frame_axis, 1)
        if self.shape[frame_axis]:
            self.shape[frame_axis] = self.shape[frame_axis] * n_frames

        # Set random number generator
        self.rng = np.random.default_rng(self.seed)

        self.indices = generate_h5_indices(
            file_paths=self.file_paths,
            file_shapes=self.file_shapes,
            n_frames=self.n_frames,
            frame_index_stride=self.frame_index_stride,
            key=self.key,
            initial_frame_axis=self.initial_frame_axis,
            additional_axes_iter=self.additional_axes_iter,
            sort_files=self.sort_files,
            overlapping_blocks=self.overlapping_blocks,
            limit_n_frames=self.limit_n_frames,
        )

        if not self.shuffle:
            log.warning("H5Generator: Not shuffling data.")

        if limit_n_samples:
            log.warning(
                f"H5Generator: Limiting number of samples to {limit_n_samples} "
                f"out of {len(self.indices)}"
            )
            self.indices = self.indices[:limit_n_samples]

        self.shuffled_items = range(len(self.indices))

        # Retry count for I/O errors
        self.retry_count = 0

        # Create a cache for the data
        self.caching = caching
        self._data_cache = {}

    def __repr__(self):
        return f"{self.__class__.__name__} containing {len(self)} batches."

    def _get_single_item(self, idx):
        # Check if the item is already in the cache
        if self.caching and idx in self._data_cache:
            return self._data_cache[idx]

        # Get the data
        file_name, key, indices = self.indices[idx]
        file = self.get_file(file_name)
        image = self.load(file, key, indices)
        file_data = {
            "fullpath": file.filename,
            "filename": file.stem,
            "indices": indices,
        }

        if self.caching:
            # Store the image and file data in the cache
            self._data_cache[idx] = [image, file_data]
        return image, file_data

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.indices))
        shuffled_items_list = self.shuffled_items[low:high]

        images = []
        filenames = []
        for idx in shuffled_items_list:
            image, file_data = self._get_single_item(idx)
            images.append(image)
            filenames.append(file_data)

        if self.batch_size == 1:
            images = images[0]
        else:
            images = np.stack(images)

        if self.batch_size == 1:
            filenames = filenames[0]

        if self.return_filename:
            return self.maybe_tensor(images), filenames
        else:
            return self.maybe_tensor(images)

    @retry_on_io_error(
        max_retries=MAX_RETRY_ATTEMPTS,
        initial_delay=INITIAL_RETRY_DELAY,
        retry_action=_h5_reopen_on_io_error,
    )
    def load(self, file: File, key: str, indices: tuple | str):
        """Extract data from hdf5 file.
        Args:
            file_name (str): name of the file to extract image from.
            key (str): key of the hdf5 dataset to grab data from.
            indices (tuple): indices to extract image from (tuple of slices)
        Returns:
            np.ndarray: image extracted from hdf5 file and indexed by indices.
        """
        try:
            images = file.load_data(key, indices)
        except (OSError, IOError):
            # Let the decorator handle I/O errors
            raise
        except Exception as exc:
            # For non-I/O errors, provide detailed context
            raise ValueError(
                f"Could not load image at index {indices} "
                f"and file {file.name} of shape {file[key].shape}"
            ) from exc

        # stack frames along frame_axis
        if self.insert_frame_axis:
            # move frames axis to self.frame_axis
            initial_frame_axis = self.initial_frame_axis
            if self.additional_axes_iter:
                # offset initial_frame_axis if we have additional axes that are before
                # the initial_frame_axis
                additional_axes_before = sum(
                    axis < self.initial_frame_axis for axis in self.additional_axes_iter
                )
                initial_frame_axis = initial_frame_axis - additional_axes_before

            images = np.moveaxis(images, initial_frame_axis, self.frame_axis)
        else:
            # append frames to existing axis
            images = np.concatenate(images, axis=self.frame_axis)

        return images

    def _shuffle(self):
        self.rng.shuffle(self.shuffled_items)
        log.info("H5Generator: Shuffled data.")

    def __len__(self):
        if self.drop_remainder:
            return math.floor(len(self.indices) / self.batch_size)
        else:
            return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        """
        Generator that yields images from the hdf5 files.
        """
        if self.shuffle:
            self._shuffle()
        for idx in range(len(self)):
            yield self[idx]


class Dataloader(H5Generator):
    """Dataloader for h5 files."""

    # TODO: implement prefetch & shard
    # TODO: sort the args and kwargs to be more readable

    def __init__(
        self,
        file_paths: List[str],
        key: str = "data/image",
        resize_type: str = "center_crop",
        image_size: tuple | None = None,
        image_range: tuple | None = None,
        normalization_range: tuple | None = None,
        resize_axes: tuple | None = None,
        resize_kwargs: dict | None = None,
        map_fns: list | None = None,
        augmentation: callable = None,
        dataset_repetitions: int = 1,
        assert_image_range: bool = True,
        clip_image_range: bool = False,
        backend: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        """Initialize the dataloader.

        Args:
            file_paths (str or list): Path to the folder(s) containing the HDF5 files or a single
                HDF5 file path.
            key (str): The key to access the HDF5 dataset.
            batch_size (int, optional): batch the dataset. Defaults to None.
            image_size (tuple, optional): resize images to image_size. Should
                be of length two (height, width). Defaults to None.
            shuffle (bool, optional): shuffle dataset.
            seed (int, optional): random seed of shuffle.
            limit_n_samples (int, optional): take only a subset of samples.
                Useful for debuging. Defaults to None.
            limit_n_frames (int, optional): limit the number of frames to load from each file. This
                means n_frames per data file will be used. These will be the first frames in
                the file. Defaults to None
            resize_type (str, optional): resize type. Defaults to 'center_crop'.
                can be 'center_crop', 'random_crop' or 'resize'.
            resize_axes (tuple, optional): axes to resize along. Should be of length 2
                (height, width) as resizing function only supports 2D resizing / cropping.
                Should only be set when your data is more than (h, w, c). Defaults to None.
            resize_kwargs (dict, optional): kwargs for the resize function.
            image_range (tuple, optional): image range. Defaults to (0, 255).
                will always translate from specified image range to normalization range.
                if image_range is set to None, no normalization will be done. Note that it does not
                clip to the image range, so values outside the image range will be outside the
                normalization range!
            normalization_range (tuple, optional): normalization range. Defaults to (0, 1).
                See image_range for more info!
            augmentation (keras.Sequential, optional): keras augmentation layer.
            dataset_repetitions (int, optional): repeat dataset. Note that this happens
                after sharding, so the shard will be repeated. Defaults to None.
            n_frames (int, optional): number of frames to load from each hdf5 file.
                Defaults to 1. These frames are stacked along the last axis (channel).
            insert_frame_axis (bool, optional): if True, new dimension to stack
                frames along will be created. Defaults to False. In that case
                frames will be stacked along existing dimension (frame_axis).
            frame_axis (int, optional): dimension to stack frames along.
                Defaults to -1. If insert_frame_axis is True, this will be the
                new dimension to stack frames along.
            initial_frame_axis (int, optional): axis where in the files the frames are stored.
                Defaults to 0.
            frame_index_stride (int, optional): interval between frames to load.
                Defaults to 1. If n_frames > 1, a lower frame rate can be simulated.
            additional_axes_iter (tuple, optional): additional axes to iterate over
                in the dataset. Defaults to None, in that case we only iterate over
                the first axis (we assume those contain the frames).
            overlapping_blocks (bool, optional): if True, blocks overlap by n_frames - 1.
                Defaults to False. Has no effect if n_frames = 1.
            return_filename (bool, optional): return file name with image. Defaults to False.
            shard_index (int, optional): index which part of the dataset should be selected.
                Can only be used if num_shards is specified. Defaults to None.
                See for info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard
            num_shards (int, optional): this is used to divide the dataset into `num_shards` parts.
                Sharding happens before all other operations. Defaults to 1.
                See for info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard
            search_file_tree_kwargs (dict, optional): kwargs for search_file_tree.
            drop_remainder (bool, optional): representing whether the last batch should be dropped
                in the case it has fewer than batch_size elements. Defaults to False.
            cache (bool or str, optional): cache dataset. If a string is provided, caching will
                be done to disk with that filename. Defaults to False.
            prefetch (bool, optional): prefetch elements from dataset. Defaults to True.
            wrap_in_keras (bool, optional): wrap dataset in TFDatasetToKeras. Defaults to True.
                If True, will convert the dataset that returns backend tensors.
        """
        super().__init__(file_paths, key, **kwargs)
        self.resize_type = resize_type
        self.image_size = image_size
        self.image_range = image_range
        if normalization_range is not None:
            assert (
                self.image_range is not None
            ), "If normalization_range is set, image_range must be set as well."
        self.normalization_range = normalization_range
        self.resize_kwargs = resize_kwargs or {}
        self.resizer = Resizer(
            resize_type=resize_type,
            image_size=image_size,
            resize_axes=resize_axes,
            seed=self.seed,
            **self.resize_kwargs,
        )
        self.map_fns = map_fns or []
        if augmentation is not None:
            self.map_fns.append(augmentation)
        self.assert_image_range = assert_image_range
        self.clip_image_range = clip_image_range
        if backend is None:
            backend = keras.backend.backend()
        self.backend = backend_utils.DynamicBackend(backend)
        self.device = device
        self.dataset_repetitions = dataset_repetitions

    def map(self, fn):
        """Add a mapping function to the dataloader.

        Args:
            fn (callable): Function to map over the images.

        Example usage:
            dataloader = dataloader.map(lambda x: x / 255)

        """
        dl = copy.deepcopy(self)
        dl.map_fns.append(fn)
        return dl

    def preprocess(self, out):
        """Preprocess the images such as resizing and normalizing them."""
        if self.return_filename:
            images, filenames = out
        else:
            images = out

        # add channel dim
        if len(self.shape) != 3:
            images = self.backend.numpy.expand_dims(images, axis=-1)

        # Check if there are outliers in the image range
        if self.assert_image_range and self.image_range is not None:
            minval = self.backend.numpy.min(images)
            maxval = self.backend.numpy.max(images)
            assert self.image_range[0] <= minval and maxval <= self.image_range[1], (
                f"Image range {self.image_range} is not in the range of the data "
                f"{minval} - {maxval}"
            )

        # Clip to image range
        if self.clip_image_range and self.image_range is not None:
            images = self.backend.numpy.clip(images, *self.image_range[0])

        # Translate to normalization range
        if self.normalization_range is not None:
            images = translate(images, self.image_range, self.normalization_range)

        # resize
        if self.image_size is not None:
            images = self.resizer(images)

        for map_fn in self.map_fns:
            images = map_fn(images)

        if self.return_filename:
            return images, filenames
        else:
            return images

    def __len__(self):
        """Return the total number of batches, accounting for repetitions."""
        return super().__len__() * self.dataset_repetitions

    def __getitem__(self, index):
        # Repeat the dataset by wrapping the index if dataset_repetitions > 1
        if index >= len(self) or index < 0:
            raise IndexError("Index out of range for repeated dataset.")
        out = super().__getitem__(index % super().__len__())

        if self.device is None:
            return self.preprocess(out)
        else:
            return on_device(self.preprocess, out, self.device)
