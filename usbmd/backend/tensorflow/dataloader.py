"""

- **Author(s)**     : Tristan Stevens
- **Date**          : Thu Nov 18 2021
"""

import itertools
import re
from pathlib import Path

import cv2
import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from usbmd.utils import log, translate
from usbmd.utils.io_lib import search_file_tree


class H5Generator:
    """Generator from h5 file."""

    def __init__(self, n_frames=1, frames_dim=-1, new_frames_dim=False):
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
        """
        self.file_name = None
        self.n_frames = n_frames
        self.frames_dim = frames_dim
        self.new_frames_dim = new_frames_dim

    def __call__(self, file_name, key):
        """Yields image from h5 file using key.

        Args:
            file_name (str): file name of h5 file.
            key (str): key of dataset in h5 file.

        Yields:
            np.ndarray: image of shape [n_frames, shape].
                where shape is image_shape + (n_channels * n_frames,).
        """
        with h5py.File(file_name, "r") as file:
            for i in range(self._length(file, key)):
                images = []
                for j in range(self.n_frames):
                    image = file[key][i * self.n_frames + j]
                    images.append(image)
                if self.new_frames_dim:
                    images = np.stack(images, axis=self.frames_dim)
                else:
                    images = np.concatenate(images, axis=self.frames_dim)

                self.file_name = Path(str(file_name)).stem + "_" + str(i)
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


def _infer_n_samples(filenames, key, n_frames, frames_dim, new_frames_dim):
    # generator here to get length of dataset
    generator = H5Generator(n_frames, frames_dim, new_frames_dim)
    # sum up number of samples in each file
    n_samples = [generator.length(filename, key) for filename in filenames]

    # Check for files that don't have any samples (i.e. less than n_frames samples)
    if 0 in n_samples:
        _skip_files = np.where(np.array(n_samples) == 0)[0]
        _skip_filenames = [Path(filenames[i]).name for i in _skip_files]
        log.warning(
            f"Skipping files with 0 samples (or that do not contain `n_frames`={n_frames}): "
            f"{_skip_filenames}"
        )
    return sum(n_samples)


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
    n_samples_per_file: int = None,
    resize_type: str = "crop",
    image_range: tuple = (0, 255),
    normalization_range: tuple = (0, 1),
    augmentation: keras.Sequential = None,
    dataset_repetitions: int = None,
    n_frames: int = 1,
    new_frames_dim: bool = False,
    frames_dim: int = -1,
    save_file_paths: bool = False,
    shard_index: int = None,
    num_shards: int = 1,
):
    """Creates a `tf.data.Dataset` from .hdf5 files in a directory.

    Mimicks the native TF function `tf.keras.utils.image_dataset_from_directory`
    but for .hdf5 files.

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
        n_samples_per_file (int, optional): the number of samples contained
            in each file. It is assumed that this number doesn't change
            across files, i.e. that each file has the same number of samples.
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
        save_file_paths (bool, optional): save file paths to file. Defaults to False.
            Can be useful to check which files are being loaded.
        shard_index (int, optional): index which part of the dataset should be selected.
            Can only be used if num_shards is specified. Defaults to None.
            See for info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard
        num_shards (int, optional): this is used to divide the dataset into `num_shards` parts.
            Sharding happens before all other operations. Defaults to 1.
            See for info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard

    Returns:
        tf.data.Dataset: dataset
    """
    filenames = None
    if not isinstance(directory, list):
        if Path(directory).is_file():
            filenames = [str(directory)]
        else:
            directory = [directory]

    if filenames is None:
        filenames = [list(Path(dir).rglob("*.hdf5")) for dir in directory]
        filenames += [list(Path(dir).rglob("*.h5")) for dir in directory]
        filenames = list(itertools.chain(*filenames))
        filenames = [str(s) for s in filenames]

    assert len(filenames) > 0, f"No files in directories:\n{directory}"
    if image_size is not None:
        assert resize_type in [
            "crop",
            "resize",
        ], 'resize_type must be "crop" or "resize"'

    try:
        filenames = sorted(filenames, key=lambda x: int(re.findall(r"\d+", x)[-2]))
    except:
        print("H5Generator: Could not sort filenames by number.")

    # n_frames=1 here to get true image shape
    generator = H5Generator(n_frames=1)
    image = next(generator(filenames[0], key))
    image_shape = image.shape
    dtype = (
        tf.float32 if image.dtype not in ["complex64", "complex128"] else tf.complex64
    )

    # infer total number of samples in dataset unless n_samples_per_file is known
    if n_samples_per_file is not None:
        n_samples = len(filenames) * n_samples_per_file
    else:
        n_samples = _infer_n_samples(
            filenames, key, n_frames, frames_dim, new_frames_dim
        )

    if not shuffle:
        cycle_length = 1

    generator = H5Generator(n_frames, frames_dim, new_frames_dim)

    # save paths to file
    if save_file_paths:
        file_path = Path("file_paths.txt")

        if file_path.exists():
            file_path.unlink()
        with open(file_path, "w", encoding="utf-8") as f:
            for filename in filenames:
                for i in range(generator.length(filename, key)):
                    f.write(f"{Path(filename).stem}_{i}\n")

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

    dataset = dataset.interleave(
        lambda filename: tf.data.Dataset.from_generator(
            generator,
            output_signature=(tf.TensorSpec(shape=image_shape, dtype=dtype)),
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
        assert len(image_size) == 2, "image_size must be of length 2 (height, width)"

        if resize_type == "resize":
            resize_layer = layers.Resizing(*image_size)
            dataset = dataset.map(resize_layer, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            crop_layer = layers.RandomCrop(*image_size)
            dataset = dataset.map(crop_layer, num_parallel_calls=tf.data.AUTOTUNE)

    # normalize
    if image_range is not None:
        dataset = dataset.map(
            lambda x: translate(x, image_range, normalization_range),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # augmentation
    if augmentation is not None:
        dataset = dataset.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    # prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


class ImageLoader(keras.utils.Sequence):
    """Class for loading ultrasound dataset for training.

    Make sure file names in x_directory and y_directory match

    """

    def __init__(
        self,
        x_directory,
        y_directory,
        batch_size=32,
        image_shape=(32, 32),
        n_channels=1,
        normalization=(0, 1),
        shuffle=True,
    ):
        """Initialization"""
        self.x_dataset_directory = Path(x_directory)
        self.y_dataset_directory = Path(y_directory)

        self.image_shape = image_shape
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.normalization = normalization
        self.shuffle = shuffle

        self.file_paths = search_file_tree(self.x_dataset_directory)

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of file paths
        file_paths_batch = [self.file_paths[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(file_paths_batch)

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_paths_batch):
        """Generates a batch of data containing batch_size imagess

        Args:
            file_paths_batch (list): list of file paths to images for batch.

        Returns:
            tuple: X, Y each of size [n_samples, *image_shape, n_channels].

        """
        X = np.empty((self.batch_size, *self.image_shape, self.n_channels))
        Y = np.empty((self.batch_size, *self.image_shape, self.n_channels))

        # Load data
        for i, file_path in enumerate(file_paths_batch):
            x_file_path = str(file_path)

            # find y file path with exact name of x but in y directory
            y_file_path = str(self.y_dataset_directory / Path(file_path).name)

            images = []
            for file_path in [x_file_path, y_file_path]:
                # Store sample
                if self.n_channels == 1:
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(file_path)

                # always check for None
                if image is None:
                    raise ValueError(f"Unable to load image at {file_path}")

                image = cv2.resize(image, self.image_shape[::-1])

                if self.n_channels == 1:
                    image = np.expand_dims(image, axis=-1)

                image = translate(image, (0, 255), self.normalization)

                images.append(image)

            X[i,] = images[0]
            Y[i,] = images[1]

        return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)
