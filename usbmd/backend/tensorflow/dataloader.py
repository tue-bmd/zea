"""

- **Author(s)**     : Tristan Stevens
- **Date**          : Thu Nov 18 2021
"""

import re
from pathlib import Path

import h5py
import keras
import numpy as np
import tensorflow as tf

from usbmd.utils import log, translate
from usbmd.utils.io_lib import _get_length_hdf5_file, search_file_tree


class H5Generator:
    """Generator from h5 file."""

    def __init__(
        self,
        n_frames: int = 1,
        frames_dim: int = -1,
        new_frames_dim: bool = False,
        frame_index_stride: int = 1,
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

    def __call__(self, file_name, key):
        """Yields image from h5 file using key.

        Args:
            file_name (str): file name of h5 file.
            key (str): key of dataset in h5 file.

        Yields:
            np.ndarray: image of shape image_shape + (n_channels * n_frames,).
        """
        with h5py.File(file_name, "r") as file:
            for i in range(self._length(file, key) // self.frame_index_stride):
                images = []
                for j in range(
                    0,
                    self.n_frames * self.frame_index_stride,
                    self.frame_index_stride,
                ):
                    try:
                        image = file[key][i * self.n_frames + j]
                    except Exception as exc:
                        raise ValueError(
                            f"Could not load image at index {i * self.n_frames + j} "
                            f"and file {file_name} of length {len(file[key])}"
                        ) from exc
                    images.append(image)
                if self.new_frames_dim:
                    images = np.stack(images, axis=self.frames_dim)
                else:
                    images = np.concatenate(images, axis=self.frames_dim)
                if self.return_filename:
                    filename = Path(str(file_name)).stem + "_" + str(i)
                    yield images, filename
                else:
                    yield images

    def _length(self, file, key):
        return len(file[key]) // self.n_frames

    def length(self, file_name, key):
        """Return length (number of elements) in h5 file.
        Args:
            file_name (str): file name of h5 file.
            key (str): key of dataset in h5 file.
        Returns:
            int: length of dataset (number of frames).
                if n_frames > 1, the length will be reduced by n_frames - 1.
                this is because when indexing always n_frames are read at the
                same time, effectively reducing the length of the dataset.
        """
        with h5py.File(file_name, "r") as file:
            return self._length(file, key)


def h5_dataset_from_directory(
    directory,
    key: str,
    batch_size: int = None,
    image_size: tuple = None,
    color_mode: bool = None,
    shuffle: bool = None,
    seed: int = None,
    cycle_length: int = None,
    block_length: int = None,
    limit_n_samples: int = None,
    resize_type: str = "crop",
    image_range: tuple = (0, 255),
    normalization_range: tuple = (0, 1),
    augmentation: keras.Sequential = None,
    dataset_repetitions: int = None,
    n_frames: int = 1,
    new_frames_dim: bool = False,
    frames_dim: int = -1,
    frame_index_stride: int = 1,
    return_filename: bool = False,
    shard_index: int = None,
    num_shards: int = 1,
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
        total_n_samples (int, optional): the total number of frames in a dataset
            This is used to compute the cardinality of the dataset more efficiently.
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
        frames_dim (int, optional): dimension to stack frames along.
            Defaults to -1. If new_frames_dim is True, this will be the
            new dimension to stack frames along.
        new_frames_dim (bool, optional): if True, new dimension to stack
            frames along will be created. Defaults to False. In that case
            frames will be stacked along existing dimension (frames_dim).
        frame_index_stride (int, optional): interval between frames to load.
            Defaults to 1. If n_frames > 1, a lower frame rate can be simulated.
        save_file_paths (bool, optional): save file paths to file. Defaults to False.
            Can be useful to check which files are being loaded.
        return_filename (bool, optional): return file name with image. Defaults to False.
        shard_index (int, optional): index which part of the dataset should be selected.
            Can only be used if num_shards is specified. Defaults to None.
            See for info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard
        num_shards (int, optional): this is used to divide the dataset into `num_shards` parts.
            Sharding happens before all other operations. Defaults to 1.
            See for info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard

    Returns:
        tf.data.Dataset: dataset
    """
    filenames = []
    file_lengths = []

    # 'directory' is actually just a single hdf5 file
    if not isinstance(directory, list) and Path(directory).is_file():
        filename = directory
        file_lengths = [_get_length_hdf5_file(filename, key)]
        filenames = [str(filename)]
    # 'directory' points to a directory or list of directories
    else:
        if not isinstance(directory, list):
            directory = [directory]

        for _dir in directory:
            dataset_info = search_file_tree(
                _dir, filetypes=[".hdf5", ".h5"], hdf5_key_for_length=key
            )
            file_paths = dataset_info["file_paths"]
            file_paths = [str(Path(_dir) / file_path) for file_path in file_paths]
            filenames.extend(file_paths)
            file_lengths.extend(dataset_info["file_lengths"])

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
        file_lengths = [file_lengths[i] for i in indices_sorting_filenames]
    except:
        print("H5Generator: Could not sort filenames by number.")

    # n_frames=1 here to get true image shape
    generator = H5Generator(n_frames=1)
    image = next(generator(filenames[0], key))
    image_shape = image.shape
    dtype = (
        tf.float32 if image.dtype not in ["complex64", "complex128"] else tf.complex64
    )

    # infer total number of samples in dataset from number of samples in each file
    # not necessarily the same if n_frames > 1
    def _compute_length_dataset(filenames, file_lengths, n_frames):
        assert len(filenames) == len(
            file_lengths
        ), "filenames and file_lengths must have same length"

        # file lengths are effectively reduced if frame_index_stride > 1
        file_lengths = [length // frame_index_stride for length in file_lengths]

        # number of samples in each file is reduced if n_frames > 1
        n_samples = [length // n_frames for length in file_lengths]

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
        file_lengths,
        n_frames,
    )

    if not shuffle:
        cycle_length = 1

    generator = H5Generator(n_frames, frames_dim, new_frames_dim, frame_index_stride)
    image = next(generator(filenames[0], key))

    # n_frames are stacked along the last axis (channel)
    image_shape = list(image_shape)
    if new_frames_dim:
        image_shape.insert(
            frames_dim if frames_dim >= 0 else len(image_shape) + (frames_dim + 1), 1
        )

    image_shape[frames_dim] *= n_frames

    dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(filenames, dtype=tf.string)
    )

    if return_filename:
        output_signature = (
            tf.TensorSpec(shape=image_shape, dtype=dtype),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )
    else:
        output_signature = tf.TensorSpec(shape=image_shape, dtype=dtype)

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

    # add channel dim
    if len(image_shape) != 3:
        if return_filename:
            dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
        else:
            dataset = dataset.map(lambda x: tf.expand_dims(x, axis=-1))
        if color_mode == "grayscale":
            image_shape = [*image_shape, 1]

    # cache samples in dataset
    dataset = dataset.cache()

    # repeat dataset if needed (used for smaller datasets)
    if dataset_repetitions:
        dataset = dataset.repeat(dataset_repetitions)

    # shuffle
    if shuffle:
        if len(dataset) > 1000:
            buffer_size = 1000
        else:
            buffer_size = len(dataset)
        dataset = dataset.shuffle(buffer_size, seed=seed)

    # limit number of samples
    if limit_n_samples:
        dataset = dataset.take(limit_n_samples)

    # batch
    if batch_size:
        dataset = dataset.batch(batch_size)

    # resize
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
            if return_filename:
                resize_layer = lambda x, y: (resize_layer(x), y)
            dataset = dataset.map(resize_layer, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            crop_layer = keras.layers.RandomCrop(*image_size)
            if return_filename:
                crop_layer = lambda x, y: (crop_layer(x), y)
            dataset = dataset.map(crop_layer, num_parallel_calls=tf.data.AUTOTUNE)

    # normalize
    if image_range is not None:
        translate_fn = lambda x: translate(x, image_range, normalization_range)
        if return_filename:
            translate_fn = lambda x, y: (translate_fn(x), y)
        dataset = dataset.map(
            translate_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    # augmentation
    if augmentation is not None:
        if return_filename:
            augmentation = lambda x, y: (augmentation(x), y)
        dataset = dataset.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    # prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
