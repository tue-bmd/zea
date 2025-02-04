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

import keras
import tensorflow as tf
from keras.src.trainers.data_adapters import TFDatasetAdapter

from usbmd.data.dataloader import H5Generator, Resizer
from usbmd.utils import find_methods_with_return_type, log, translate

METHODS_THAT_RETURN_DATASET = find_methods_with_return_type(
    tf.data.Dataset, "DatasetV2"
)


class H5GeneratorTF(H5Generator):
    """Adds a tensorflow dtype property and output_signature to the H5Generator class."""

    @property
    def tensorflow_dtype(self):
        """
        Extracts one image from the dataset to get the dtype. Converts it to a tensorflow dtype.
        """
        out = next(iter(self))
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


class TFDatasetToKeras(TFDatasetAdapter):
    """
    This class wraps a tf.data.Dataset object and allows it to be used with Keras backends.
    """

    def __init__(self, dataset):
        super().__init__(dataset)

    def __iter__(self):
        backend = keras.backend.backend()
        if backend == "tensorflow":
            return iter(self.get_tf_dataset())
        elif backend == "jax":
            return self.get_jax_iterator()
        elif backend == "torch":
            return iter(self.get_torch_dataloader())
        elif backend == "numpy":
            return self.get_numpy_iterator()
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                "Please use one of the following: 'tensorflow', 'jax', 'torch', 'numpy'."
            )

    def __len__(self):
        return self.num_batches

    def __getattr__(self, name):
        # Delegate all calls to self._dataset, and wraps the result in TFDatasetToKeras
        if name in METHODS_THAT_RETURN_DATASET:

            def method(*args, **kwargs):
                result = getattr(self._dataset, name)(*args, **kwargs)
                return TFDatasetToKeras(result)

            return method
        else:
            return getattr(self._dataset, name)


def h5_dataset_from_directory(
    directory,
    key: str,
    batch_size: int | None = None,
    image_size: tuple | None = None,
    shuffle: bool = None,
    seed: int | None = None,
    limit_n_samples: int | None = None,
    resize_type: str = "center_crop",
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
    wrap_in_keras: bool = True,
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
        resize_type (str, optional): resize type. Defaults to 'center_crop'.
            can be 'center_crop', 'random_crop' or 'resize'.
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
        wrap_in_keras (bool, optional): wrap dataset in TFDatasetToKeras. Defaults to True.
            If True, will convert the dataset that returns backend tensors.

    Returns:
        tf.data.Dataset: dataset
    """
    tf_data_shuffle = shuffle and cache  # shuffle after caching
    generator_shuffle = shuffle  # shuffle on the generator level

    if tf_data_shuffle:
        log.warning("Will shuffle on the image level, this can be slower.")

    image_extractor = H5GeneratorTF(
        directory,
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
        as_tensor=False,
        search_file_tree_kwargs=search_file_tree_kwargs,
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

        resizer = Resizer(
            image_size, resize_type, resize_axes, seed=seed, backend="tensorflow"
        )
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

    if wrap_in_keras:
        dataset = TFDatasetToKeras(dataset)

    return dataset
