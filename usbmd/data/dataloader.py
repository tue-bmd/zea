"""
H5 dataloader for loading images from USBMD datasets.

This module can be used with any backend.
"""

import json
import math
import re
from collections import OrderedDict
from itertools import product
from pathlib import Path
from typing import List

import h5py
import keras
import numpy as np
from keras import ops
from keras.src.utils.backend_utils import DynamicBackend

from usbmd.utils import log, translate
from usbmd.utils.io_lib import _get_shape_hdf5_file, search_file_tree

FILE_TYPES = [".hdf5", ".h5"]
FILE_HANDLE_CACHE_CAPACITY = 128
DEFAULT_IMAGE_RANGE = (0, 255)
DEFAULT_NORMALIZATION_RANGE = (0, 1)


def json_dumps(obj):
    """Used to serialize objects that contain range and slice objects.
    Args:
        obj: object to serialize (most likely a dictionary).
    Returns:
        str: serialized object (json string).
    """
    return json.dumps(obj, cls=USBMDJSONEncoder)


def json_loads(obj):
    """Used to deserialize objects that contain range and slice objects.
    Args:
        obj: object to deserialize (most likely a json string).
    Returns:
        object: deserialized object (dictionary).
    """
    return json.loads(obj, object_hook=_usbmd_datasets_json_decoder)


def decode_file_info(file_info):
    """Decode file info from a json string.
    A batch of H5Generator can return a list of file_info that are json strings.
    This function decodes the json strings and returns a list of dictionaries
    with the information, namely:
    - full_path: full path to the file
    - file_name: file name
    - indices: indices used to extract the image from the file
    """

    if file_info.ndim == 0:
        file_info = [file_info]

    decoded_info = []
    for info in file_info:
        info = ops.convert_to_numpy(info)[()].decode("utf-8")
        decoded_info.append(json_loads(info))
    return decoded_info


def _map_negative_indices(indices: list, length: int):
    """Maps negative indices for array indexing to positive indices.
    Example:
        >>> _map_negative_indices([-1, -2], 5)
        [4, 3]
    """
    return [i if i >= 0 else length + i for i in indices]


def _usbmd_datasets_json_decoder(dct):
    """Wrapper for json.loads to decode range and slice objects."""
    if "__type__" in dct:
        if dct["__type__"] == "range":
            return range(dct["start"], dct["stop"], dct["step"])
        if dct["__type__"] == "slice":
            return slice(dct["start"], dct["stop"], dct["step"])
    return dct


def generate_h5_indices(
    file_names: List[str],
    file_shapes: list,
    n_frames: int,
    frame_index_stride: int,
    key: str = "data/image",
    initial_frame_axis: int = 0,
    additional_axes_iter: List[int] | None = None,
    sort_files: bool = True,
    overlapping_blocks: bool = False,
):
    """Generate indices for h5 files.

    Generates a list of indices to extract images from hdf5 files. Length of this list
    is the length of the extracted dataset.

    Args:
        file_names (list): list of file names.
        file_shapes (list): list of file shapes.
        n_frames (int): number of frames to load from each hdf5 file.
        frame_index_stride (int): interval between frames to load.
        key (str, optional): key of hdf5 dataset to grab data from. Defaults to "data/image".
        initial_frame_axis (int, optional): axis to iterate over. Defaults to 0.
        additional_axes_iter (list, optional): additional axes to iterate over in the dataset.
            Defaults to None.
        sort_files (bool, optional): sort files by number. Defaults to True.

    Returns:
        list: list of tuples with indices to extract images from hdf5 files.
            (file_name, key, indices) with indices being a tuple of slices.
            example: [
                ('/folder/path_to_file.hdf5', 'data/image', [range(0, 1), slice(None, 256, None), slice(None, 256, None)]), # pylint: disable=line-too-long
                ('/folder/path_to_file.hdf5', 'data/image', [range(1, 2), slice(None, 256, None), slice(None, 256, None)]), # pylint: disable=line-too-long
                ...
            ]

    """

    assert len(file_names) == len(
        file_shapes
    ), "file_names and file_shapes must have same length"

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
            indices_sorting_file_names = sorted(
                range(len(file_names)),
                key=lambda i: int(re.findall(r"\d+", file_names[i])[-2]),
            )
            file_names = [file_names[i] for i in indices_sorting_file_names]
            file_shapes = [file_shapes[i] for i in indices_sorting_file_names]
        except:
            log.warning("H5Generator: Could not sort file_names by number.")

    block_size = n_frames * frame_index_stride

    if not overlapping_blocks:
        block_step_size = block_size
    else:
        # now blocks overlap by n_frames - 1
        block_step_size = 1

    axis_indices_files = [
        [
            [
                range(i, i + block_size, frame_index_stride)
                for i in range(
                    0, shape[initial_frame_axis] - block_size + 1, block_step_size
                )
            ]
        ]
        for shape in file_shapes
    ]

    indices = []
    skipped_files = 0
    for file, shape, axis_indices in zip(file_names, file_shapes, axis_indices_files):
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
            f"which is about {skipped_files / len(file_names) * 100:.2f}% of the "
            f"dataset. This can be fine if you expect set `n_frames` and "
            "`frame_index_stride` to be high. Minimum frames in a file needs to be at "
            f"least n_frames * frame_index_stride = {n_frames * frame_index_stride}. "
        )

    return indices


def _find_h5_files_from_directory(
    directory,
    key: str,
    search_file_tree_kwargs: dict | None = None,
    additional_axes_iter: tuple | None = None,
):
    """
    Find HDF5 files from a directory or list of directories and retrieve their shapes.

    Args:
        directory (str or list): A single directory path, a list of directory paths,
            or a single HDF5 file path.
        key (str): The key to access the HDF5 dataset.
        search_file_tree_kwargs (dict, optional): Additional keyword arguments for the
            search_file_tree function. Defaults to None.
        additional_axes_iter (tuple, optional): Additional axes to iterate over if dataset_info
            contains file_shapes. Defaults to None.

    Returns:
        - file_names (list): List of file paths to the HDF5 files.
        - file_shapes (list): List of shapes of the HDF5 datasets.
    """

    file_names = []
    file_shapes = []

    if search_file_tree_kwargs is None:
        search_file_tree_kwargs = {}

    # 'directory' is actually just a single hdf5 file
    if not isinstance(directory, list) and Path(directory).is_file():
        filename = directory
        file_shapes = [_get_shape_hdf5_file(filename, key)]
        file_names = [str(filename)]

    # 'directory' points to a directory or list of directories
    else:
        if not isinstance(directory, list):
            directory = [directory]

        for _dir in directory:
            dataset_info = search_file_tree(
                _dir,
                filetypes=FILE_TYPES,
                hdf5_key_for_length=key,
                **search_file_tree_kwargs,
            )
            file_paths = dataset_info["file_paths"]
            file_paths = [str(Path(_dir) / file_path) for file_path in file_paths]
            file_names.extend(file_paths)
            if "file_shapes" not in dataset_info:
                assert additional_axes_iter is None, (
                    "additional_axes_iter is only supported if the dataset_info "
                    "contains file_shapes. please remove dataset_info.yaml files and rerun."
                )
                # since in this case we only need to iterate over the first axis it is
                # okay we only have the lengths of the files (and not the full shape)
                file_shapes.extend(
                    [[length] for length in dataset_info["file_lengths"]]
                )
            else:
                file_shapes.extend(dataset_info["file_shapes"])

    return file_names, file_shapes


class Resizer:
    """
    Resize layer for resizing images. Can deal with N-dimensional images.
    Can do resize, center_crop and random_crop.

    Can be used in tf.data pipelines, but you must set backend = "tensorflow".
    """

    def __init__(
        self,
        image_size: tuple,
        resize_type: str,
        resize_axes: tuple | None = None,
        seed: int | None = None,
        backend: str | None = None,
        **resize_kwargs,
    ):
        """
        Initializes the data loader with the specified parameters.

        Args:
            image_size (tuple): The target size of the images.
            resize_type (str): The type of resizing to apply.
                Supported types are 'center_crop', 'random_crop', 'resize'.
            resize_axes (tuple | None, optional): The axes along which to resize.
                Must be of length 2. Defaults to None. In that case, can only process
                default tensors of shape (batch, height, width, channels), where the
                resize axes are (1, 2), i.e. height and width. If processing higher
                dimensional tensors, you must specify the resize axes.
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.
            backend (str | None, optional): The backend to use for image processing.
            **resize_kwargs: Additional keyword arguments for the resizing operation.

        Raises:
            ValueError: If an unsupported resize type is provided.
            AssertionError: If resize_axes is not of length 2.
        """
        super().__init__()
        self.image_size = image_size

        if backend == "tensorflow":
            import tensorflow as tf  # pylint: disable=import-outside-toplevel

            layers = tf.keras.layers  # pylint: disable=no-member
        else:
            layers = keras.layers

        self.backend = DynamicBackend(backend)

        if image_size is not None:
            if resize_type == "resize":
                self.resizer = layers.Resizing(*image_size, **resize_kwargs)
            elif resize_type == "center_crop":
                self.resizer = layers.CenterCrop(*image_size, **resize_kwargs)
            elif resize_type == "random_crop":
                self.resizer = layers.RandomCrop(
                    *image_size, seed=seed, **resize_kwargs
                )
            else:
                raise ValueError(
                    f"Unsupported resize type: {resize_type}. "
                    "Supported types are 'center_crop', 'random_crop', 'resize'."
                )
        else:
            self.resizer = None

        self.resize_axes = resize_axes
        if resize_axes is not None:
            assert len(resize_axes) == 2, "resize_axes must be of length 2"

    def _permute_before_resize(self, x, ndim, resize_axes):
        """Permutes tensor to put resize axes in correct position before resizing."""
        # Create permutation that moves resize axes to second to last dimensions
        # Keeping channel axis as last dimension
        perm = list(range(ndim))
        perm.remove(resize_axes[0])
        perm.remove(resize_axes[1])
        perm.insert(-1, resize_axes[0])
        perm.insert(-1, resize_axes[1])

        # Apply permutation
        x = self.backend.numpy.transpose(x, perm)
        perm_shape = self.backend.core.shape(x)

        # Reshape to collapse all leading dimensions
        flattened_shape = [-1, perm_shape[-3], perm_shape[-2], perm_shape[-1]]
        x = self.backend.numpy.reshape(x, flattened_shape)

        return x, perm, perm_shape

    def _permute_after_resize(self, x, perm, perm_shape, ndim):
        """Restores original tensor shape and axes order after resizing."""
        # Restore original shape with new resized dimensions
        # Get all dimensions except the resized ones and channel dim
        shape_prefix = perm_shape[:-3]
        # Create new shape list starting with original prefix dims, then resize dims, then channel
        new_shape = list(shape_prefix) + list(self.image_size) + [perm_shape[-1]]
        x = self.backend.numpy.reshape(x, new_shape)

        # Transpose back to original axis order
        inverse_perm = list(range(ndim))
        for i, p in enumerate(perm):
            inverse_perm[p] = i
        x = self.backend.numpy.transpose(x, inverse_perm)

        return x

    def __call__(self, x):
        """
        Resize the input tensor.
        """
        if self.resizer is None:
            return x

        ndim = self.backend.numpy.ndim(x)

        if self.resize_axes is None:
            assert ndim == 4, (
                f"`resize_axes` must be specified for when ndim != 4, got {ndim}. "
                "For ndim == 4, the resize axes are default to (1, 2)."
            )
            x = self.resizer(x)
            return x

        assert ndim >= 4, f"We expect at least 4 dimensions for Resizer, got {ndim}."

        resize_axes = _map_negative_indices(self.resize_axes, ndim)

        # Prepare tensor for resizing
        x, perm, perm_shape = self._permute_before_resize(x, ndim, resize_axes)

        # Apply resize
        x = self.resizer(x)

        # Restore original shape and order
        x = self._permute_after_resize(x, perm, perm_shape, ndim)

        return x


class H5Generator(keras.utils.PyDataset):
    """Generator from h5 file using provided indices."""

    def __init__(
        self,
        directory: str = None,
        file_names: List[str] = None,
        file_shapes: List[tuple] = None,
        n_frames: int = 1,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        insert_frame_axis: bool = True,
        initial_frame_axis: int = 0,
        return_filename: bool = False,
        additional_axes_iter: tuple = None,
        key: str = "data/image",
        shuffle: bool = True,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        limit_n_samples: int | None = None,
        seed: int | None = None,
        batch_size: int = 1,
        as_tensor: bool = True,
        search_file_tree_kwargs: dict | None = None,
        file_handle_cache_capacity: int = FILE_HANDLE_CACHE_CAPACITY,
        **kwargs,
    ):
        assert (directory is not None) ^ (
            file_names is not None and file_shapes is not None
        ), "Either `directory` or `file_names` and `file_shapes` must be provided."

        super().__init__(**kwargs)
        self.directory = directory
        self.n_frames = int(n_frames)
        self.frame_index_stride = int(frame_index_stride)
        self.frame_axis = int(frame_axis)
        self.insert_frame_axis = insert_frame_axis
        self.initial_frame_axis = int(initial_frame_axis)
        self.return_filename = return_filename
        if additional_axes_iter is None:
            self.additional_axes_iter = []
        else:
            self.additional_axes_iter = additional_axes_iter
        self.key = key
        self.shuffle = shuffle
        self.sort_files = sort_files
        self.overlapping_blocks = overlapping_blocks
        self.limit_n_samples = limit_n_samples
        self.seed = seed
        self.batch_size = batch_size
        self.as_tensor = as_tensor
        self.search_file_tree_kwargs = search_file_tree_kwargs

        self.maybe_tensor = ops.convert_to_tensor if self.as_tensor else lambda x: x

        if self.directory is not None:
            file_names, file_shapes = _find_h5_files_from_directory(
                self.directory,
                self.key,
                self.search_file_tree_kwargs,
                self.additional_axes_iter,
            )

        assert len(file_names) > 0, f"No files in directories:\n{directory}"

        assert (
            self.frame_index_stride > 0
        ), f"`frame_index_stride` must be greater than 0, got {self.frame_index_stride}"
        assert (
            self.n_frames > 0
        ), f"`n_frames` must be greater than 0, got {self.n_frames}"

        # Extract some general information about the dataset
        image_shapes = np.array(file_shapes)
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
            _frame_axis = _map_negative_indices([frame_axis], len(self.shape) + 1)
            self.shape = np.insert(self.shape, _frame_axis, 1)
        self.shape[frame_axis] = self.shape[frame_axis] * n_frames

        # Set random number generator
        self.rng = np.random.default_rng(self.seed)

        self.indices = generate_h5_indices(
            file_names=file_names,
            file_shapes=file_shapes,
            n_frames=self.n_frames,
            frame_index_stride=self.frame_index_stride,
            key=self.key,
            initial_frame_axis=self.initial_frame_axis,
            additional_axes_iter=self.additional_axes_iter,
            sort_files=self.sort_files,
            overlapping_blocks=self.overlapping_blocks,
        )

        if not self.shuffle:
            log.warning("H5Generator: Not shuffling data.")

        if limit_n_samples:
            log.warning(
                f"H5Generator: Limiting number of samples to {limit_n_samples} "
                f"out of {len(self.indices)}"
            )
            self.indices = self.indices[:limit_n_samples]

        # LRU cache for file handles
        self._file_handle_cache = OrderedDict()
        self.file_handle_cache_capacity = file_handle_cache_capacity

    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            self._shuffle()

        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.indices))
        indices_list = self.indices[low:high]

        images = []
        for indices in indices_list:
            images.append(self.extract_image(indices))

        if self.batch_size == 1:
            images = images[0]
        else:
            images = np.stack(images)

        fileinfo = [
            json_dumps(
                {
                    "fullpath": filename,
                    "filename": Path(filename).stem,
                    "indices": indices,
                }
            )
            for filename, _, indices in indices_list
        ]

        if self.batch_size == 1:
            fileinfo = fileinfo[0]

        if self.return_filename:
            return self.maybe_tensor(images), fileinfo
        else:
            return self.maybe_tensor(images)

    def __iter__(self):
        """
        Generator that yields images from the hdf5 files.
        """
        for idx in range(len(self)):
            yield self[idx]

    def __call__(self):
        return iter(self)

    def _get_file(self, file_name):
        """Open an HDF5 file and cache it."""
        # If file is already in cache, return it and move it to the end
        if file_name in self._file_handle_cache:
            self._file_handle_cache.move_to_end(file_name)
            return self._file_handle_cache[file_name]
        # If file is not in cache, open it and add it to the cache
        else:
            # If cache is full, close the least recently used file
            if len(self._file_handle_cache) >= self.file_handle_cache_capacity:
                _, close_file = self._file_handle_cache.popitem(last=False)
                close_file.close()
            file = h5py.File(file_name, "r", locking=False)
            self._file_handle_cache[file_name] = file
            return file

    def extract_image(self, indices):
        """Extract image from hdf5 file.
        Args:
            indices (tuple): indices to extract image from.
                (file_name, key, indices) with indices being a tuple of slices.
        Returns:
            np.ndarray: image extracted from hdf5 file and indexed by indices.
        """
        file_name, key, indices = indices
        file = self._get_file(file_name)

        # Convert any range objects in indices to lists
        processed_indices = tuple(
            list(idx) if isinstance(idx, range) else idx for idx in indices
        )
        try:
            images = file[key][processed_indices]
        except Exception as exc:
            raise ValueError(
                f"Could not load image at index {processed_indices} "
                f"and file {file_name} of shape {file[key].shape}"
            ) from exc

        # stack frames along frame_axis, and default to last axis
        frame_axis = self.frame_axis

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

            images = np.moveaxis(images, initial_frame_axis, frame_axis)
        else:
            # append frames to existing axis
            images = np.concatenate(images, axis=frame_axis)

        return images

    def _shuffle(self):
        self.rng.shuffle(self.indices)
        log.info("H5Generator: Shuffled data.")

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __del__(self):
        """Ensure cached files are closed."""
        for _, file in self._file_handle_cache.items():
            file.close()
        self._file_handle_cache = OrderedDict()


class H5Dataloader(H5Generator):
    """Dataloader for h5 files. Can resize images and normalize them."""

    def __init__(
        self,
        resize_type: str = "center_crop",
        image_size: tuple | None = None,
        image_range: tuple = DEFAULT_IMAGE_RANGE,
        normalization_range: tuple = DEFAULT_NORMALIZATION_RANGE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resize_type = resize_type
        self.image_size = image_size
        self.image_range = image_range
        self.normalization_range = normalization_range
        self.resizer = Resizer(
            resize_type=resize_type,
            image_size=image_size,
            seed=self.seed,
        )

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if self.return_filename:
            images, filenames = out
        else:
            images = out

        # add channel dim
        if len(self.shape) != 3:
            images = ops.expand_dims(images, axis=-1)

        # normalize
        images = translate(images, self.image_range, self.normalization_range)

        # resize
        if self.image_size is not None:
            images = self.resizer(images)

        if self.return_filename:
            return images, filenames
        else:
            return images


class USBMDJSONEncoder(json.JSONEncoder):
    """Wrapper for json.dumps to encode range and slice objects.

    Example:
        >>> json.dumps(range(10), cls=USBMDJSONEncoder)
        '{"__type__": "range", "start": 0, "stop": 10, "step": 1}'

    Note:
        Probably you would use the `usbmd.data.dataloader.json_dumps()`
        function instead of using this class directly.
    """

    def default(self, obj):
        if isinstance(obj, range):
            return {
                "__type__": "range",
                "start": obj.start,
                "stop": obj.stop,
                "step": obj.step,
            }
        if isinstance(obj, slice):
            return {
                "__type__": "slice",
                "start": obj.start,
                "stop": obj.stop,
                "step": obj.step,
            }
        return super().default(obj)
