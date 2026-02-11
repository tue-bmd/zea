"""HDF5 Tensorflow dataloader.

Convenient way of loading data from hdf5 files in a ML pipeline.
"""

from functools import partial
from typing import List

import keras
import tensorflow as tf
from keras.src.trainers.data_adapters import TFDatasetAdapter

from zea.data.dataloader import H5Generator
from zea.data.layers import Resizer
from zea.func.tensor import translate
from zea.internal.utils import find_methods_with_return_type

METHODS_THAT_RETURN_DATASET = find_methods_with_return_type(tf.data.Dataset, "DatasetV2")


class TFDatasetToKeras(TFDatasetAdapter):
    """Tensorflow Dataset to Keras Dataset.

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


class H5GeneratorTF(H5Generator):
    """Adds a tensorflow dtype property and output_signature to the H5Generator class."""

    @property
    def tensorflow_dtype(self):
        """
        Extracts one image from the dataset to get the dtype. Converts it to a tensorflow dtype.
        """
        out = next(self.iterator())
        if self.return_filename:
            out = out[0]
        dtype = out.dtype
        if "float" in str(dtype):
            dtype = tf.float32
        elif "complex" in str(dtype):
            dtype = tf.complex64
        elif "uint8" in str(dtype):
            dtype = tf.uint8
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


def _assert_image_range(images, image_range):
    # Check if there are outliers in the image range
    minval = tf.reduce_min(images)
    maxval = tf.reduce_max(images)
    _msg = f"Image range {image_range} is not in the range of the data {minval} - {maxval}"
    tf.debugging.assert_greater_equal(
        minval,
        tf.cast(image_range[0], minval.dtype),
        message=_msg,
    )
    tf.debugging.assert_less_equal(
        maxval,
        tf.cast(image_range[1], maxval.dtype),
        message=_msg,
    )
    return images


def make_dataloader(
    file_paths: List[str],
    batch_size: int,
    key: str = "data/image",
    n_frames: int = 1,
    shuffle: bool = True,
    return_filename: bool = False,
    limit_n_samples: int | None = None,
    limit_n_frames: int | None = None,
    seed: int | None = None,
    drop_remainder: bool = False,
    resize_type: str | None = None,
    resize_axes: tuple | None = None,
    resize_kwargs: dict | None = None,
    image_size: tuple | None = None,
    image_range: tuple | None = None,
    normalization_range: tuple | None = None,
    dataset_repetitions: int | None = None,
    cache: bool = False,
    additional_axes_iter: tuple | None = None,
    sort_files: bool = True,
    overlapping_blocks: bool = False,
    augmentation: callable = None,
    assert_image_range: bool = True,
    clip_image_range: bool = False,
    initial_frame_axis: int = 0,
    insert_frame_axis: bool = True,
    frame_index_stride: int = 1,
    frame_axis: int = -1,
    validate: bool = True,
    prefetch: bool = True,
    shard_index: int | None = None,
    num_shards: int = 1,
    wrap_in_keras: bool = True,
    **kwargs,
) -> tf.data.Dataset:
    """Creates a ``tf.data.Dataset`` from .hdf5 files in the specified directory or directories.

    Mimics the native TF function ``tf.keras.utils.image_dataset_from_directory``
    but for .hdf5 files.

    Does the following in order to load a dataset:

        - Find all .hdf5 files in the director(ies)
        - Load the data from each file using the specified key
        - Apply the following transformations in order (if specified):

            - limit_n_samples
            - cache
            - shuffle
            - shard
            - add channel dim
            - assert_image_range
            - clip_image_range
            - resize
            - repeat
            - batch
            - normalize
            - augmentation
            - prefetch
            - tf -> keras tensor

    Args:
        file_paths (str or list): Path(s) to the folder(s) or h5 file(s) to load.
        batch_size (int): Batch the dataset.
        key (str): The key to access the HDF5 dataset.
        n_frames (int, optional): Number of frames to load from each hdf5 file.
            Defaults to 1. These frames are stacked along the last axis (channel).
        shuffle (bool, optional): Shuffle dataset.
        return_filename (bool, optional): Return file name with image. Defaults to False.
        limit_n_samples (int, optional): Take only a subset of samples.
            Useful for debugging. Defaults to None.
        limit_n_frames (int, optional): Limit the number of frames to load from each file.
            This means n_frames per data file will be used. These will be the first frames in
            the file. Defaults to None.
        seed (int, optional): Random seed of shuffle.
        drop_remainder (bool, optional): Whether the last batch should be dropped.
        resize_type (str, optional): Resize type. Defaults to 'center_crop'.
            Can be 'center_crop', 'random_crop' or 'resize'.
        resize_axes (tuple, optional): Axes to resize along. Should be of length 2
            (height, width) as resizing function only supports 2D resizing / cropping.
            Should only be set when your data is more than (h, w, c). Defaults to None.
            Note that it considers the axes after inserting the frame axis.
        resize_kwargs (dict, optional): Kwargs for the resize function.
        image_size (tuple, optional): Resize images to image_size. Should
            be of length two (height, width). Defaults to None.
        image_range (tuple, optional): Image range. Defaults to (0, 255).
            Will always translate from specified image range to normalization range.
            If image_range is set to None, no normalization will be done. Note that it does not
            clip to the image range, so values outside the image range will be outside the
            normalization range!
        normalization_range (tuple, optional): Normalization range. Defaults to (0, 1).
            See image_range for more info!
        dataset_repetitions (int, optional): Repeat dataset. Note that this happens
            after sharding, so the shard will be repeated. Defaults to None.
        cache (bool, optional): Cache dataset to RAM.
        additional_axes_iter (tuple, optional): Additional axes to iterate over
            in the dataset. Defaults to None, in that case we only iterate over
            the first axis (we assume those contain the frames).
        sort_files (bool, optional): Sort files by number. Defaults to True.
        overlapping_blocks (bool, optional): If True, blocks overlap by n_frames - 1.
            Defaults to False. Has no effect if n_frames = 1.
        augmentation (keras.Sequential, optional): Keras augmentation layer.
        assert_image_range (bool, optional): Assert that the image range is
            within the specified image range. Defaults to True.
        clip_image_range (bool, optional): Clip the image range to the specified
            image range. Defaults to False.
        initial_frame_axis (int, optional): Axis where in the files the frames are stored.
            Defaults to 0.
        insert_frame_axis (bool, optional): If True, new dimension to stack
            frames along will be created. Defaults to True. In that case
            frames will be stacked along existing dimension (frame_axis).
        frame_index_stride (int, optional): Interval between frames to load.
            Defaults to 1. If n_frames > 1, a lower frame rate can be simulated.
        frame_axis (int, optional): Dimension to stack frames along.
            Defaults to -1. If insert_frame_axis is True, this will be the
            new dimension to stack frames along.
        validate (bool, optional): Validate if the dataset adheres to the zea format.
            Defaults to True.
        prefetch (bool, optional): Prefetch the dataset. Defaults to True.
        shard_index (int, optional): Index which part of the dataset should be selected.
            Can only be used if num_shards is specified. Defaults to None.
            See for info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard
        num_shards (int, optional): This is used to divide the dataset into ``num_shards`` parts.
            Sharding happens before all other operations. Defaults to 1.
            See for info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard
        wrap_in_keras (bool, optional): Wrap dataset in TFDatasetToKeras. Defaults to True.
            If True, will convert the dataset that returns backend tensors.

    Returns:
        tf.data.Dataset: The constructed dataset.

    """
    # Setup
    if normalization_range is not None:
        assert image_range is not None, (
            "If normalization_range is set, image_range must be set as well."
        )

    resize_kwargs = resize_kwargs or {}

    if num_shards > 1:
        assert shard_index is not None, "shard_index must be specified"
        assert shard_index < num_shards, "shard_index must be less than num_shards"
        assert shard_index >= 0, "shard_index must be greater than or equal to 0"

    image_extractor = H5GeneratorTF(
        file_paths,
        key,
        n_frames=n_frames,
        frame_index_stride=frame_index_stride,
        frame_axis=frame_axis,
        insert_frame_axis=insert_frame_axis,
        initial_frame_axis=initial_frame_axis,
        return_filename=return_filename,
        shuffle=shuffle,
        sort_files=sort_files,
        overlapping_blocks=overlapping_blocks,
        limit_n_samples=limit_n_samples,
        limit_n_frames=limit_n_frames,
        seed=seed,
        additional_axes_iter=additional_axes_iter,
        cache=cache,
        validate=validate,
        **kwargs,
    )

    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        image_extractor, output_signature=image_extractor.output_signature
    )

    # Assert cardinality
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(image_extractor)))

    # Shard dataset
    if num_shards > 1:
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
    if len(image_extractor.shape) < 3:
        dataset = dataset_map(dataset, lambda x: tf.expand_dims(x, axis=-1))

    # Clip to image range
    if clip_image_range and image_range is not None:
        dataset = dataset_map(
            dataset,
            partial(
                tf.clip_by_value,
                clip_value_min=image_range[0],
                clip_value_max=image_range[1],
            ),
        )

    # Check if there are outliers in the image range
    if assert_image_range and image_range is not None:
        dataset = dataset_map(dataset, partial(_assert_image_range, image_range=image_range))

    if image_size or resize_type:
        if frame_axis != -1:
            assert resize_axes is not None, (
                "Resizing only works with frame_axis = -1. Alternatively, "
                "you can specify resize_axes."
            )

        # Let resizer handle the assertions.
        resizer = Resizer(
            image_size=image_size,
            resize_type=resize_type,
            resize_axes=resize_axes,
            seed=seed,
            **resize_kwargs,
        )
        dataset = dataset_map(dataset, resizer)

    # repeat dataset if needed (used for smaller datasets)
    if dataset_repetitions:
        dataset = dataset.repeat(dataset_repetitions)

    # batch
    if batch_size:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # normalize
    if normalization_range is not None:
        dataset = dataset_map(
            dataset,
            lambda x: translate(x, image_range, normalization_range),
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
