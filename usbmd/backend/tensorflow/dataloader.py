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
        frame_axis: int | None = None,
        insert_frame_axis: bool = False,
        initial_frame_axis: int = 0,
        return_filename: bool = False,
        additional_axes_iter: tuple = None,
        key: str = "data/image",
        shuffle: bool = True,
        sort_files: bool = True,
        limit_n_samples: int | None = None,
        seed: int | None = None,
    ):
        self.n_frames = n_frames
        self.frame_index_stride = frame_index_stride
        self.frame_axis = frame_axis
        self.insert_frame_axis = insert_frame_axis
        self.initial_frame_axis = initial_frame_axis
        self.return_filename = return_filename
        self.additional_axes_iter = additional_axes_iter
        self.key = key
        self.shuffle = shuffle
        self.sort_files = sort_files
        self.limit_n_samples = limit_n_samples
        self.seed = seed

        self.indices = generate_h5_indices(
            file_names=file_names,
            file_shapes=file_shapes,
            n_frames=self.n_frames,
            frame_index_stride=self.frame_index_stride,
            key=self.key,
            initial_frame_axis=self.initial_frame_axis,
            additional_axes_iter=self.additional_axes_iter,
            sort_files=self.sort_files,
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

    def __call__(self):
        for i, indices in enumerate(self.indices):
            images = self.extract_image(indices)
            file_name, _, indices = indices

            # shuffle if we reached end
            if i == self.__len__() - 1:
                if self.shuffle:
                    log.info("H5Generator: Shuffling data.")
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
            if self.frame_axis is None:
                frame_axis = len(images.shape) - 1

            images = np.moveaxis(images, initial_frame_axis, frame_axis)
        else:
            if self.frame_axis is None:
                frame_axis = len(images.shape) - 2
            # append frames to existing axis
            images = np.concatenate(images, axis=frame_axis)

        return images

    def _shuffle(self):
        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.indices)

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
    axis_indices_files = [
        [
            [
                range(
                    i,
                    i + block_size,
                    frame_index_stride,
                )
                for i in range(
                    0, shape[initial_frame_axis] - block_size + 1, block_size
                )
            ]
        ]
        for shape in file_shapes
    ]

    # remove all the files that have empty list at initial_frame_axis
    # this can happen if the file is too small to fit a block
    remove_indices = []
    for i, axis_indices in enumerate(axis_indices_files):
        if not axis_indices[0]:
            remove_indices.append(i)

    if remove_indices:
        # skip_files = [file_names[i] for i in remove_indices]

        log.warning(
            f"H5Generator: Skipping {len(remove_indices)} files with not enough frames "
            f"which is about {len(remove_indices) / len(file_names) * 100:.2f}% of the "
            f"dataset. This can be fine if you expect set `n_frames` and "
            "`frame_index_stride` to be high. Minimum frames in a file needs to be at "
            f"least n_frames * frame_index_stride = {n_frames * frame_index_stride}. "
        )

    indices = []
    for file, shape, axis_indices in zip(file_names, file_shapes, axis_indices_files):

        if additional_axes_iter:
            axis_indices += [range(shape[axis]) for axis in additional_axes_iter]

        axis_indices = product(*axis_indices)

        for axis_index in axis_indices:
            full_indices = [slice(size) for size in shape]
            for i, axis in enumerate([initial_frame_axis] + list(additional_axes_iter)):
                full_indices[axis] = axis_index[i]
            indices.append((file, key, full_indices))

    return indices


def h5_dataset_from_directory(
    directory,
    key: str,
    batch_size: int | None = None,
    image_size: tuple | None = None,
    shuffle: bool = None,
    seed: int | None = None,
    limit_n_samples: int | None = None,
    resize_type: str = "crop",
    image_range: tuple = (0, 255),
    normalization_range: tuple = (0, 1),
    augmentation: keras.Sequential | None = None,
    dataset_repetitions: int | None = None,
    n_frames: int = 1,
    insert_frame_axis: bool = True,
    frame_axis: int | None = None,
    initial_frame_axis: int = 0,
    frame_index_stride: int = 1,
    additional_axes_iter: tuple | None = None,
    return_filename: bool = False,
    shard_index: int | None = None,
    num_shards: int = 1,
    search_file_tree_kwargs: dict | None = None,
    drop_remainder: bool = False,
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
        - shuffle
        - add channel dim
        - cache
        - repeat
        - limit_n_samples
        - batch
        - resize
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
        frame_index_stride (int, optional): interval between frames to load.
            Defaults to 1. If n_frames > 1, a lower frame rate can be simulated.
        additional_axes_iter (tuple, optional): additional axes to iterate over
            in the dataset. Defaults to None, in that case we only iterate over
            the first axis (we assume those contain the frames).
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

    Returns:
        tf.data.Dataset: dataset
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

    # Extract some general information about the dataset
    image_shapes = np.array(file_shapes)
    # depending on which axis we iterate over and frame dimension
    # we need to adjust the image shapes
    if additional_axes_iter:
        # we iterate over these axis so we don't care about their size
        # remove that dim
        for axis in additional_axes_iter:
            image_shapes = np.delete(image_shapes, axis, axis=1)
    # also don't care about frame axis
    image_shapes = np.delete(image_shapes, 0, axis=1)

    equal_file_shapes = np.all(image_shapes == image_shapes[0])
    if not equal_file_shapes:
        log.warning(
            "H5Generator: Not all files have the same shape. "
            "This can lead to issues when resizing images."
        )
        assert (
            resize_type != "crop"
        ), "Currently unsupported to crop images with different shapes."

    assert len(file_names) > 0, f"No files in directories:\n{directory}"
    if image_size is not None:
        assert resize_type in [
            "crop",
            "resize",
        ], 'resize_type must be "crop" or "resize"'

    image_extractor = H5Generator(
        file_names,
        file_shapes,
        n_frames=n_frames,
        frame_index_stride=frame_index_stride,
        frame_axis=frame_axis,
        insert_frame_axis=insert_frame_axis,
        initial_frame_axis=initial_frame_axis,
        return_filename=return_filename,
        additional_axes_iter=additional_axes_iter,
        key=key,
        limit_n_samples=limit_n_samples,
        sort_files=True,
        shuffle=shuffle,
        seed=seed,
    )

    image = next(image_extractor())

    dtype = (
        tf.float32 if image.dtype not in ["complex64", "complex128"] else tf.complex64
    )

    n_axis = len(image.shape)

    assert n_axis in [
        2,
        3,
    ], f"Currently only supports 2D and 3D dimensions, got {n_axis}. "

    if not equal_file_shapes:
        shape = [None] * n_axis
        # channels need to be defined
        if n_axis == 3:
            shape[-1] = image.shape[-1]
    else:
        shape = image.shape

    dataset = tf.data.Dataset.from_generator(
        image_extractor,
        output_signature=(
            (
                tf.TensorSpec(shape=shape, dtype=dtype),
                tf.TensorSpec(shape=(), dtype=tf.string),
            )
            if return_filename
            else tf.TensorSpec(shape=shape, dtype=dtype)
        ),
    ).apply(tf.data.experimental.assert_cardinality(len(image_extractor)))

    if num_shards > 1:
        assert shard_index is not None, "shard_index must be specified"
        assert shard_index < num_shards, "shard_index must be less than num_shards"
        assert shard_index >= 0, "shard_index must be greater than or equal to 0"
        dataset = dataset.shard(num_shards, shard_index)

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
    if n_axis != 3:
        dataset = dataset_map(dataset, lambda x: tf.expand_dims(x, axis=-1))

    # cache samples in dataset
    dataset = dataset.cache()

    # repeat dataset if needed (used for smaller datasets)
    if dataset_repetitions:
        dataset = dataset.repeat(dataset_repetitions)

    # resize before batching to deal with different image sizes
    if image_size:
        assert (
            len(image_size) == 2
        ), f"image_size must be of length 2 (height, width), got {image_size}"

        if resize_type == "resize":
            dims = len(dataset.element_spec.shape)
            if dims > 4:
                log.warning(
                    f"Resizing not supported with dimensions > 4, got {dims} dims which "
                    "could be result of `insert_frame_axis=True`."
                )

            resize_layer = keras.layers.Resizing(*image_size)
            dataset = dataset_map(dataset, resize_layer)
        else:
            crop_layer = keras.layers.RandomCrop(*image_size)
            dataset = dataset_map(dataset, crop_layer)

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
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
