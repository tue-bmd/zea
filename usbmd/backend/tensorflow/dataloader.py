"""H5 dataloader.

Convenient way of loading data from hdf5 files in a ML pipeline.

Allows for flexible indexing and stacking of dimensions. There are a few important parameters:

- `directory` can be a single file or a list of directories. If it is a directory, all hdf5 files
    in the directory will be loaded. If it is a list of directories, all hdf5 files in all
    directories will be loaded.
- `key` is the key of the dataset in the hdf5 file that will be used. For example `data/image`.
    It is assumed that the dataset contains a multidimensional array.
- `n_frames`: One of the axis of the multidimensional array is designated as the `frame_axis`.
    We will iterate over this axis and stack `n_frames` frames together.
- `frame_index_stride`: additionally this parameter can be used to skip frames at a regular
    interval. This can be useful to simulate a higher frame rates.
- `additional_axes_iter`: additional axes to iterate over in the dataset. This can be useful
    if you have additional dimensions in the dataset that you want to iterate over. For example,
    if you have a 3D dataset but want to train on 2D slices.
- `insert_frame_axis`: if True, a new dimension to stack frames along will be created. If False,
    frames will be stacked along an existing dimension.
- `frame_axis`: dimension to stack frames along. If `insert_frame_axis` is True, this will be
    the new dimension to stack frames along. Else, this will be the existing dimension to
    stack frames along.

- **Author(s)**     : Tristan Stevens
- **Date**          : Thu Nov 18 2021
"""

import re
from itertools import product
from pathlib import Path
from typing import List

import h5py
import keras
import numpy as np
import tensorflow as tf

from usbmd.utils import log, translate
from usbmd.utils.io_lib import _get_shape_hdf5_file, search_file_tree


class H5Generator:
    """Generator from h5 file using provided indices."""

    def __init__(
        self,
        file_names: list,
        file_shapes: list,
        n_frames: int = 1,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        insert_frame_axis: bool = False,
        initial_frame_axis: int = 0,
        return_filename: bool = False,
        additional_axes_iter: tuple = None,
        key: str = "data/image",
        shuffle: bool = True,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        limit_n_samples: int | None = None,
        seed: int | None = None,
    ):
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

        if limit_n_samples:
            log.warning(
                f"H5Generator: Limiting number of samples to {limit_n_samples} "
                f"out of {len(self.indices)}"
            )
            self.indices = self.indices[:limit_n_samples]

        if self.shuffle:
            self._shuffle()
        else:
            log.warning("H5Generator: Not shuffling data.")

    @property
    def tensorflow_dtype(self):
        """
        Extracts one image from the dataset to get the dtype. Converts it to a tensorflow dtype.
        """
        out = next(self())
        if self.return_filename:
            out = out[0]
        dtype = out.dtype
        if "float" in str(dtype):
            dtype = tf.float32
        elif "complex" in str(dtype):
            dtype = tf.complex64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dtype

    @property
    def output_signature(self):
        """
        Get the output signature of the generator as a tensorflow `TensorSpec`.
        This is useful for creating a `tf.data.Dataset` from the generator.
        """
        output_signature = tf.TensorSpec(shape=self.shape, dtype=self.tensorflow_dtype)
        if self.return_filename:
            output_signature = (
                output_signature,
                tf.TensorSpec(shape=(), dtype=tf.string),
            )
        return output_signature

    def __call__(self):
        """
        Generator that yields images from the hdf5 files.
        """
        for i, indices in enumerate(self.indices):
            images = self.extract_image(indices)

            file_name, _, indices = indices

            # shuffle if we reached end
            if i == self.__len__() - 1:
                if self.shuffle:
                    self._shuffle()

            if self.return_filename:
                file_name = (
                    Path(str(file_name)).stem + "_" + "_".join(map(str, indices))
                )
                yield images, file_name
            else:
                yield images

    def extract_image(self, indices):
        """Extract image from hdf5 file.
        Args:
            indices (tuple): indices to extract image from.
                (file_name, key, indices) with indices being a tuple of slices.
        Returns:
            np.ndarray: image extracted from hdf5 file and indexed by indices.
        """
        file_name, key, indices = indices
        with h5py.File(file_name, "r") as file:
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
        log.info("H5Generator: Shuffling data.")
        self.rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)


def generate_h5_indices(
    file_names: list,
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
                filetypes=[".hdf5", ".h5"],
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


def recursive_map_fn(func, data, num_maps):
    """
    Recursively apply tf.map_fn.
    """
    # Base case: when no more maps are needed
    if num_maps == 0:
        return func(data)

    # Apply tf.map_fn and recursively reduce num_maps
    return tf.map_fn(lambda x: recursive_map_fn(func, x, num_maps - 1), data)


class Resizer:
    """
    Resize layer for resizing images. Can deal with N-dimensional images.
    Can do resize, center_crop and random_crop.
    """

    def __init__(
        self,
        image_size: tuple,
        resize_type: str,
        resize_axes: tuple | None = None,
        seed: int | None = None,
        **resize_kwargs,
    ):
        """
        Get a resize layer based on the resize type.
        """
        super().__init__()
        self.image_size = image_size

        if image_size is not None:
            if resize_type == "resize":
                self.resizer = tf.keras.layers.Resizing(  # pylint: disable=no-member
                    *image_size, **resize_kwargs
                )
            elif resize_type == "center_crop":
                self.resizer = tf.keras.layers.CenterCrop(  # pylint: disable=no-member
                    *image_size, **resize_kwargs
                )  # pylint: disable=no-member
            elif resize_type == "random_crop":
                self.resizer = tf.keras.layers.RandomCrop(  # pylint: disable=no-member
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
        x = tf.transpose(x, perm)
        perm_shape = tf.shape(x)

        # Reshape to collapse all leading dimensions
        flattened_shape = [-1, perm_shape[-3], perm_shape[-2], perm_shape[-1]]
        x = tf.reshape(x, flattened_shape)

        return x, perm, perm_shape

    def _permute_after_resize(self, x, perm, perm_shape, ndim):
        """Restores original tensor shape and axes order after resizing."""
        # Restore original shape with new resized dimensions
        # Get all dimensions except the resized ones and channel dim
        shape_prefix = perm_shape[:-3]
        # Create new shape list starting with original prefix dims, then resize dims, then channel
        new_shape = tf.concat([shape_prefix, self.image_size, [perm_shape[-1]]], axis=0)
        x = tf.reshape(x, new_shape)

        # Transpose back to original axis order
        inverse_perm = list(range(ndim))
        for i, p in enumerate(perm):
            inverse_perm[p] = i
        x = tf.transpose(x, inverse_perm)

        return x

    def __call__(self, x):
        """
        Resize the input tensor.
        """
        if self.resizer is None:
            return x

        ndim = tf.experimental.numpy.ndim(x)

        if ndim > 4:
            assert (
                self.resize_axes is not None
            ), "resize_axes must be specified when ndim > 4"
            resize_axes = _map_negative_indices(self.resize_axes, ndim)

            # Prepare tensor for resizing
            x, perm, perm_shape = self._permute_before_resize(x, ndim, resize_axes)

            # Apply resize
            x = self.resizer(x)

            # Restore original shape and order
            x = self._permute_after_resize(x, perm, perm_shape, ndim)
        else:
            assert self.resize_axes is None, "resize_axes must be None when ndim <= 4"
            x = self.resizer(x)

        return x


def h5_dataset_from_directory(
    directory,
    key: str,
    batch_size: int | None = None,
    image_size: tuple | None = None,
    shuffle: bool = None,
    seed: int | None = None,
    limit_n_samples: int | None = None,
    resize_type: str = "crop",
    resize_axes: tuple | None = None,
    image_range: tuple = (0, 255),
    normalization_range: tuple = (0, 1),
    augmentation: keras.Sequential | None = None,
    dataset_repetitions: int | None = None,
    n_frames: int = 1,
    insert_frame_axis: bool = True,
    frame_axis: int = -1,
    initial_frame_axis: int = 0,
    frame_index_stride: int = 1,
    additional_axes_iter: tuple | None = None,
    overlapping_blocks: bool = False,
    return_filename: bool = False,
    shard_index: int | None = None,
    num_shards: int = 1,
    search_file_tree_kwargs: dict | None = None,
    drop_remainder: bool = False,
    cache: bool | str = False,
    prefetch: bool = True,
):
    """Creates a `tf.data.Dataset` from .hdf5 files in a directory.

    Mimicks the native TF function `tf.keras.utils.image_dataset_from_directory`
    but for .hdf5 files.

    Saves a dataset_info.yaml file in the directory with information about the dataset.
    This file is used to load the dataset later on, which speeds up the initial loading
    of the dataset for very large datasets.

    Does the following in order to load a dataset:
    - Find all .hdf5 files in the directory
    - Load the dataset from each file using the specified key
    - Apply the following transformations in order (if specified):
        - limit_n_samples
        - shuffle (if not cached)
        - shard
        - add channel dim
        - cache
        - shuffle (if cached)
        - resize
        - repeat
        - batch
        - normalize
        - augmentation
        - prefetch

    Args:
        directory (str or list): Directory where the data is located.
            can also be a list of directories. Works recursively.
        key (str): key of hdf5 dataset to grab data from.
        batch_size (int, optional): batch the dataset. Defaults to None.
        image_size (tuple, optional): resize images to image_size. Should
            be of length two (height, width). Defaults to None.
        shuffle (bool, optional): shuffle dataset.
        seed (int, optional): random seed of shuffle.
        limit_n_samples (int, optional): take only a subset of samples.
            Useful for debuging. Defaults to None.
        resize_type (str, optional): resize type. Defaults to 'crop'.
            can be 'crop' or 'resize'.
        resize_axes (tuple, optional): axes to resize along. Should be of length 2
            (height, width) as resizing function only supports 2D resizing / cropping. Should only
            be set when your data is more than (h, w, c). Defaults to None.
        image_range (tuple, optional): image range. Defaults to (0, 255).
            will always normalize from specified image range to normalization range.
            if image_range is set to None, no normalization will be done.
        normalization_range (tuple, optional): normalization range. Defaults to (0, 1).
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

    Returns:
        tf.data.Dataset: dataset
    """
    tf_data_shuffle = shuffle and cache  # shuffle after caching
    generator_shuffle = shuffle and not cache  # shuffle on the generator level

    if tf_data_shuffle:
        log.warning("Will shuffle on the image level, this can be slower.")

    file_names, file_shapes = _find_h5_files_from_directory(
        directory, key, search_file_tree_kwargs, additional_axes_iter
    )

    assert len(file_names) > 0, f"No files in directories:\n{directory}"

    image_extractor = H5Generator(
        file_names,
        file_shapes,
        key=key,
        n_frames=n_frames,
        frame_index_stride=frame_index_stride,
        frame_axis=frame_axis,
        insert_frame_axis=insert_frame_axis,
        initial_frame_axis=initial_frame_axis,
        return_filename=return_filename,
        additional_axes_iter=additional_axes_iter,
        overlapping_blocks=overlapping_blocks,
        limit_n_samples=limit_n_samples,
        sort_files=True,
        shuffle=generator_shuffle,
        seed=seed,
    )

    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        image_extractor, output_signature=image_extractor.output_signature
    )

    # Assert cardinality
    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(len(image_extractor))
    )

    # Shard dataset
    if num_shards > 1:
        assert shard_index is not None, "shard_index must be specified"
        assert shard_index < num_shards, "shard_index must be less than num_shards"
        assert shard_index >= 0, "shard_index must be greater than or equal to 0"
        dataset = dataset.shard(num_shards, shard_index)

    # Define helper function to apply map function to dataset
    def dataset_map(dataset, func):
        """Does not apply func to filename."""
        if return_filename:
            return dataset.map(
                lambda x, filename: (func(x), filename),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            return dataset.map(func, num_parallel_calls=tf.data.AUTOTUNE)

    # add channel dim
    if len(image_extractor.shape) != 3:
        dataset = dataset_map(dataset, lambda x: tf.expand_dims(x, axis=-1))

    # If cache is a string, caching will be done to disk with that filename
    # Otherwise caching will be done in memory
    if cache:
        filename = cache if isinstance(cache, str) else ""
        dataset = dataset.cache(filename)

    if num_shards > 1 and shuffle:
        log.warning("Shuffling after sharding, so the shard will be shuffled.")

    # Shuffle after caching for random order every epoch
    if tf_data_shuffle:
        buffer_size = len(image_extractor) if len(image_extractor) < 1000 else 1000
        dataset = dataset.shuffle(buffer_size, seed=seed)

    if image_size is not None:
        assert (
            len(image_size) == 2
        ), f"image_size must be of length 2 (height, width), got {image_size}"

        resizer = Resizer(image_size, resize_type, resize_axes, seed=seed)
        dataset = dataset_map(dataset, resizer)

    # repeat dataset if needed (used for smaller datasets)
    if dataset_repetitions:
        dataset = dataset.repeat(dataset_repetitions)

    # batch
    if batch_size:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # normalize
    if image_range is not None:
        dataset = dataset_map(
            dataset, lambda x: translate(x, image_range, normalization_range)
        )

    # augmentation
    if augmentation is not None:
        dataset = dataset_map(dataset, augmentation)

    # prefetch
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _map_negative_indices(indices: list, length: int):
    """Maps negative indices for array indexing to positive indices.
    Example:
        >>> _map_negative_indices([-1, -2], 5)
        [4, 3]
    """
    return [i if i >= 0 else length + i for i in indices]
