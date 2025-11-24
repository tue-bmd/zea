"""Keras layers for data preprocessing."""

import keras
import numpy as np
from keras.src.layers.preprocessing.data_layer import DataLayer

from zea.ops import Pad as PadOp
from zea.utils import map_negative_indices


class Pad(PadOp):
    """Pad layer for padding tensors to a specified shape which can be used in tf.data pipelines."""

    __call__ = DataLayer.__call__

    def call(self, inputs):
        """
        Pad the input tensor.
        """
        return super().call(data=inputs)["data"]


class Resizer(DataLayer):
    """
    Resize layer for resizing images. Can deal with N-dimensional images.
    Can do resize, center_crop, random_crop and crop_or_pad.

    Can be used in `tf.data` pipelines.
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
        Initializes the data loader with the specified parameters.

        Args:
            image_size (tuple): The target size of the images.
            resize_type (str): The type of resizing to apply. Supported types are
                ['center_crop'](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/center_crop/),
                ['random_crop'](https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_crop/),
                ['resize'](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/resizing/),
                'crop_or_pad': resizes an image to a target width and height by either centrally
                cropping the image, padding it evenly with zeros or a combination of both.
            resize_axes (tuple | None, optional): The axes along which to resize.
                Must be of length 2. Defaults to None. In that case, can only process
                default tensors of shape (batch, height, width, channels), where the
                resize axes are (1, 2), i.e. height and width. If processing higher
                dimensional tensors, you must specify the resize axes.
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.
            **resize_kwargs: Additional keyword arguments for the resizing operation.

        Raises:
            ValueError: If an unsupported resize type is provided.
            AssertionError: If resize_axes is not of length 2.
        """
        super().__init__()

        assert isinstance(image_size, (tuple, list, np.ndarray)) and len(image_size) == 2, (
            f"image_size must be of length 2, got: {image_size}"
        )
        assert isinstance(resize_type, str), f"resize_type must be a string, got: {resize_type}"

        self.image_size = image_size

        if resize_type == "resize":
            self.resizer = keras.layers.Resizing(*image_size, **resize_kwargs)
        elif resize_type == "center_crop":
            self.resizer = keras.layers.CenterCrop(*image_size, **resize_kwargs)
        elif resize_type == "random_crop":
            self.resizer = keras.layers.RandomCrop(*image_size, seed=seed, **resize_kwargs)
        elif resize_type == "crop_or_pad":
            pad_kwargs = {}
            if "constant_values" in resize_kwargs:
                pad_kwargs["constant_values"] = resize_kwargs.pop("constant_values")
            if "mode" in resize_kwargs:
                pad_kwargs["mode"] = resize_kwargs.pop("mode")
            self.resizer = keras.layers.Pipeline(
                [
                    Pad(
                        image_size,
                        axis=(-3, -2),
                        uniform=True,
                        fail_on_bigger_shape=False,
                        **pad_kwargs,
                    ),
                    keras.layers.CenterCrop(*image_size, **resize_kwargs),
                ]
            )
        else:
            raise ValueError(
                f"Unsupported resize type: {resize_type}. "
                "Supported types are 'center_crop', 'random_crop', 'resize'."
            )

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

    def call(self, inputs):
        """
        Resize the input tensor.
        """
        ndim = self.backend.numpy.ndim(inputs)

        if self.resize_axes is None:
            assert ndim in [3, 4], (
                f"`resize_axes` must be specified for when ndim not in [3, 4], got {ndim}. "
                "For ndim == 3 or 4, the resize axes are default to (1, 2)."
            )
            return self.resizer(inputs)

        assert ndim >= 4, f"We expect at least 4 dimensions for Resizer, got {ndim}."

        resize_axes = map_negative_indices(self.resize_axes, ndim)

        # Prepare tensor for resizing
        inputs, perm, perm_shape = self._permute_before_resize(inputs, ndim, resize_axes)

        # Apply resize
        out = self.resizer(inputs)

        # Restore original shape and order
        return self._permute_after_resize(out, perm, perm_shape, ndim)
