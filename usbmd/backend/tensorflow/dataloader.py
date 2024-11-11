"""

- **Author(s)**     : Tristan Stevens
- **Date**          : Thu Nov 18 2021
"""

import re
from itertools import product
from pathlib import Path
from typing import Generator, Tuple, Union

import h5py
import keras
import numpy as np
import tensorflow as tf

from usbmd.utils import log, translate
from usbmd.utils.io_lib import _get_shape_hdf5_file, search_file_tree


class H5Generator:
    """Generator from h5 file."""

    def __init__(
        self,
        n_frames: int = 1,
        frames_dim: int = -1,
        new_frames_dim: bool = False,
        frame_index_stride: int = 1,
        additional_axes_iter: tuple = None,
        return_filename: bool = False,
    ):
        """Initialize H5Generator.
        Args:
            n_frames (int, optional): number of frames to load from file.
                Defaults to 1. Frames are read in sequence and stacked along
                the last axis (channel).
            frames_dim (int, optional): dimension to stack frames along.
                Defaults to -1. If new_frames_dim is True, this will be the
                new dimension to stack frames along.
            new_frames_dim (bool, optional): if True, new dimension to stack
                frames along will be created. Defaults to False. In that case
                frames will be stacked along existing dimension (frames_dim).
            frame_index_stride (int, optional): interval between frames to load.
                Defaults to 1. If n_frames > 1, a lower frame rate can be simulated.
            additional_axes_iter (tuple, optional): additional axes to iterate over
                in the dataset. Defaults to None, in that case we only iterate over
                the first axis (we assume those contain the frames).
            return_filename (bool, optional): return file name with image.
                will return a string with the file name and frame number as follows:
                <file_name>_<frame_number>. In case multiple frames are returned,
                only the first frame_number string is returned. This is enough as
                frames are always consecutive. Defaults to False.
        """
        self.n_frames = n_frames
        self.frames_dim = frames_dim
        self.new_frames_dim = new_frames_dim
        self.frame_index_stride = frame_index_stride
        self.return_filename = return_filename
        self.additional_axes_iter = additional_axes_iter

        assert self.n_frames >= 1, "n_frames must be greater or equal to 1."
        self._axis_indices_cache = {}

    def __call__(
        self, file_name: str, key: str
    ) -> Generator[Union[np.ndarray, Tuple[np.ndarray, str]], None, None]:
        """Yields image from h5 file using key.

        Args:
            file_name (str): file name of h5 file.
            key (str): key of dataset in h5 file.

        Yields:
            np.ndarray: image of shape image_shape + (n_channels * n_frames,).
        """
        if file_name not in self._axis_indices_cache:
            with h5py.File(file_name, "r") as file:
                self._axis_indices_cache[file_name] = list(
                    self._get_iteration_indices(file, key)
                )

        axis_indices = self._axis_indices_cache[file_name]

        for indices in axis_indices:
            try:
                with h5py.File(file_name, "r") as file:
                    if self.additional_axes_iter:
                        data_shape = file[key].shape
                        full_indices = [slice(shape) for shape in data_shape]
                        # now populate full_indices with indices at the correct position
                        # indicated by additional_axes_iter
                        indices = list(indices)
                        for i, axis in enumerate([0] + list(self.additional_axes_iter)):
                            full_indices[axis] = indices[i]
                        indices = tuple(full_indices)

                    images = file[key][tuple(indices)]
            except Exception as exc:
                raise ValueError(
                    f"Could not load image at index {tuple(indices)} "
                    f"and file {file_name} of length {len(file[key])}"
                ) from exc

            if self.new_frames_dim:
                # move frames axis to self.frames_dim
                images = np.moveaxis(images, 0, self.frames_dim)
            else:
                # append frames to existing axis
                images = np.concatenate(images, axis=self.frames_dim)

            if self.return_filename:
                filename = Path(str(file_name)).stem + "_" + "_".join(map(str, indices))
                yield images, filename
            else:
                yield images

    def _length(self, file: h5py.File, key: str) -> int:
        """
        Returns the length of the iteration indices for a given file and key.

        Args:
            file (object): The file object.
            key (str): The key to access the data in the file.
        Returns:
            int: The length of the iteration indices.
        """
        return len(list(self._get_iteration_indices(file, key)))

    def _get_iteration_indices(self, file: h5py.File, key: str) -> iter:
        """
        Get the iteration indices for the given file and key.

        Args:
            file (object): The file object.
            key (str): The key to access the data in the file.

        Returns:
            tuple: A tuple of iteration indices.

        """
        data_shape = file[key].shape
        block_size = self.n_frames * self.frame_index_stride
        axis_indices = [
            [
                range(
                    i,
                    i + block_size,
                    self.frame_index_stride,
                )
                for i in range(0, data_shape[0] - block_size + 1, block_size)
            ]
        ]
        if self.additional_axes_iter:
            axis_indices += [
                range(data_shape[axis]) for axis in self.additional_axes_iter
            ]
        return product(*axis_indices)

    def length(self, file_name: str, key: str) -> int:
        """Return length (number of elements) in h5 file.
        Args:
            file_name (str): file name of h5 file.
            key (str): key of dataset in h5 file.
        Returns:
            int: length of dataset (number of frames).
                if n_frames > 1, the length will be reduced by n_frames - 1.
                this is because when indexing always n_frames are read at the
                same time, effectively reducing the length of the dataset.
                same holds for frame_index_stride > 1.
        """
        with h5py.File(file_name, "r") as file:
            return self._length(file, key)


def h5_dataset_from_directory(
    directory,
    key: str,
    batch_size: int | None = None,
    image_size: tuple | None = None,
    shuffle: bool = None,
    seed: int | None = None,
    cycle_length: int | None = None,
    block_length: int | None = None,
    limit_n_samples: int | None = None,
    resize_type: str = "crop",
    image_range: tuple = (0, 255),
    normalization_range: tuple = (0, 1),
    augmentation: keras.Sequential | None = None,
    dataset_repetitions: int | None = None,
    n_frames: int = 1,
    new_frames_dim: bool = False,
    frames_dim: int = -1,
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
        - add channel dim
        - cache
        - repeat
        - shuffle
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
        cycle_length (int, optional): see tf.data.Dataset.interleave.
            Defaults to None.
        block_length (int, optional): see tf.data.Dataset.interleave.
            Defaults to None.
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
        new_frames_dim (bool, optional): if True, new dimension to stack
            frames along will be created. Defaults to False. In that case
            frames will be stacked along existing dimension (frames_dim).
        frames_dim (int, optional): dimension to stack frames along.
            Defaults to -1. If new_frames_dim is True, this will be the
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
    filenames = []
    file_shapes = []

    if search_file_tree_kwargs is None:
        search_file_tree_kwargs = {}

    # 'directory' is actually just a single hdf5 file
    if not isinstance(directory, list) and Path(directory).is_file():
        filename = directory
        file_shapes = [_get_shape_hdf5_file(filename, key)]
        filenames = [str(filename)]

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
            filenames.extend(file_paths)
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
    image_shapes = np.array(file_shapes)[:, -2:]
    equal_file_shapes = np.all(image_shapes == image_shapes[0])
    if not equal_file_shapes:
        log.warning(
            "H5Generator: Not all files have the same shape. "
            "This can lead to issues when resizing images."
        )
        assert (
            resize_type != "crop"
        ), "Currently unsupported to crop images with different shapes."

    assert len(filenames) > 0, f"No files in directories:\n{directory}"
    if image_size is not None:
        assert resize_type in [
            "crop",
            "resize",
        ], 'resize_type must be "crop" or "resize"'

    try:
        # this is like an np.argsort, returns the indices that would sort the array
        indices_sorting_filenames = sorted(
            range(len(filenames)),
            key=lambda i: int(re.findall(r"\d+", filenames[i])[-2]),
        )
        filenames = [filenames[i] for i in indices_sorting_filenames]
        file_shapes = [file_shapes[i] for i in indices_sorting_filenames]
    except:
        log.warning("H5Generator: Could not sort filenames by number.")

    def _compute_length_dataset(
        filenames, file_shapes, n_frames, additional_axes_iter=None
    ):
        assert len(filenames) == len(
            file_shapes
        ), "filenames and file_shapes must have same length"

        axis_indices_files = [
            [
                [
                    range(i, i + n_frames * frame_index_stride, frame_index_stride)
                    for i in range(0, shape[0], n_frames * frame_index_stride)
                ]
            ]
            for shape in file_shapes
        ]
        if additional_axes_iter:
            # now add the additional axes to iterate over to each file
            axis_indices_files_out = []
            for axis_indices, shape in zip(axis_indices_files, file_shapes):
                axis_indices += [range(shape[axis]) for axis in additional_axes_iter]
                axis_indices_files_out.append(axis_indices)
            axis_indices_files = axis_indices_files_out

        n_samples = []
        for axis_indices in axis_indices_files:
            n_samples.append(len(list(product(*axis_indices))))

        # Check for files that don't have any samples (i.e. less than n_frames samples)
        if 0 in n_samples:
            _skip_files_idx = np.where(np.array(n_samples) == 0)[0]
            skip_files = [filenames[i] for i in _skip_files_idx]
            filenames = [
                filenames[i] for i in range(len(filenames)) if i not in _skip_files_idx
            ]
            log.warning(
                f"Skipping {len(skip_files)} files with not enough frames which is about "
                f"{len(skip_files) / len(filenames) * 100:.2f}% of the dataset. "
                f"This can be fine if you expect set `n_frames` and `frame_index_stride` "
                "to be high. Minimum frames in a file needs to be at "
                "least n_frames * frame_index_stride = "
                f"{n_frames * frame_index_stride}. "
            )
        return sum(n_samples), filenames

    n_samples, filenames = _compute_length_dataset(
        filenames,
        file_shapes,
        n_frames,
        additional_axes_iter=additional_axes_iter,
    )

    if not shuffle:
        cycle_length = 1

    generator = H5Generator(
        n_frames,
        frames_dim,
        new_frames_dim,
        frame_index_stride,
        additional_axes_iter,
    )
    image = next(generator(filenames[0], key))
    dtype = (
        tf.float32 if image.dtype not in ["complex64", "complex128"] else tf.complex64
    )
    n_axis = len(image.shape)

    dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(filenames, dtype=tf.string)
    )

    # create output signature
    output_shape = [None] * n_axis
    output_signature = tf.TensorSpec(shape=output_shape, dtype=dtype)
    if return_filename:
        output_signature = (
            output_signature,
            tf.TensorSpec(shape=(), dtype=tf.string),
        )

    dataset = dataset.interleave(
        lambda filename: tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
            args=(filename, key),
        ),
        cycle_length=cycle_length,
        block_length=block_length,
    )

    dataset = dataset.apply(tf.data.experimental.assert_cardinality(n_samples))

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

    # shuffle
    if shuffle:
        if len(dataset) > 100:
            buffer_size = 100
        else:
            buffer_size = len(dataset)
        dataset = dataset.shuffle(buffer_size, seed=seed)

    # limit number of samples
    if limit_n_samples:
        dataset = dataset.take(limit_n_samples)

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
                    "could be result of `new_frames_dim=True`."
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
