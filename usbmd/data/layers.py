"""Keras layers for data preprocessing.
- **Author(s)**     : Wessel van Nierop
- **Date**          : 12/02/2025
"""

from typing import List, Union

import keras
import numpy as np
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer

from usbmd.utils.utils import map_negative_indices

# pylint: disable=arguments-differ


def _prep_to_shape(shape_array, pad_to_shape, axis):
    if isinstance(axis, int):
        axis = [axis]
    assert len(axis) == len(
        pad_to_shape
    ), "The length of axis must be equal to the length of pad_to_shape."
    axis = map_negative_indices(axis, len(shape_array))

    pad_to_shape = [
        pad_to_shape[axis.index(i)] if i in axis else shape_array[i]
        for i in range(len(shape_array))
    ]
    return pad_to_shape


class Pad(TFDataLayer):
    """Pad layer for padding tensors to a specified shape."""

    def __init__(
        self,
        pad_to_shape: list | tuple,
        uniform: bool = True,
        axis: Union[int, List[int]] = None,
        fail_on_bigger_shape: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._pad_to_shape = pad_to_shape
        self.uniform = uniform
        self.axis = axis
        self.kwargs = kwargs
        self.fail_on_bigger_shape = fail_on_bigger_shape

    def pad_to_shape(
        self,
        z,
        pad_to_shape: list | tuple,
        uniform: bool = True,
        axis: Union[int, List[int]] = None,
        fail_on_bigger_shape: bool = True,
        **kwargs,
    ):
        """
        Pads the input tensor `z` to the specified shape.

        Parameters:
            z (tensor): The input tensor to be padded.
            pad_to_shape (list or tuple): The target shape to pad the tensor to.
            uniform (bool, optional): If True, ensures that padding is uniform (even on both sides).
                Default is False.
            axis (int or list of int, optional): The axis or axes along which `pad_to_shape` was
                specified. If None, `len(pad_to_shape) == `len(ops.shape(z))` must hold.
                Default is None.
            fail_on_bigger_shape (bool, optional): If True, raises an error if the target shape is
                bigger than the input shape. If False, will pad to match the target shape wherever
                needed. Default is True.
            kwargs: Additional keyword arguments to pass to the padding function.

        Returns:
            tensor: The padded tensor with the specified shape.
        """
        shape_array = self.backend.shape(z)

        # When axis is provided, convert pad_to_shape
        if axis is not None:
            pad_to_shape = _prep_to_shape(shape_array, pad_to_shape, axis)

        if not fail_on_bigger_shape:
            pad_to_shape = [
                max(pad_to_shape[i], shape_array[i]) for i in range(len(shape_array))
            ]

        # Compute the padding required for each dimension
        pad_shape = np.array(pad_to_shape) - shape_array

        # Create the paddings array
        if uniform:
            # if odd, pad more on the left, same as:
            # https://keras.io/api/layers/preprocessing_layers/image_preprocessing/center_crop/
            right_pad = pad_shape // 2
            left_pad = pad_shape - right_pad
            paddings = np.stack([right_pad, left_pad], axis=1)
        else:
            paddings = np.stack([np.zeros_like(pad_shape), pad_shape], axis=1)

        return self.backend.numpy.pad(z, paddings, **kwargs)

    def call(self, inputs):
        return self.pad_to_shape(
            inputs,
            self._pad_to_shape,
            self.uniform,
            self.axis,
            self.fail_on_bigger_shape,
            **self.kwargs,
        )


class Resizer(TFDataLayer):
    """
    Resize layer for resizing images. Can deal with N-dimensional images.
    Can do resize, center_crop and random_crop.

    Can be used in tf.data pipelines.
    """

    def __init__(
        self,
        image_size: tuple,
        resize_type: str,
        resize_axes: tuple | None = None,
        seed: int | None = None,
        **resize_kwargs,
    ):
        # pylint: disable=line-too-long
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
        # pylint enable=line-too-long
        super().__init__()
        self.image_size = image_size

        if image_size is not None:
            if resize_type == "resize":
                self.resizer = keras.layers.Resizing(*image_size, **resize_kwargs)
            elif resize_type == "center_crop":
                self.resizer = keras.layers.CenterCrop(*image_size, **resize_kwargs)
            elif resize_type == "random_crop":
                self.resizer = keras.layers.RandomCrop(
                    *image_size, seed=seed, **resize_kwargs
                )
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

    def call(self, inputs):
        """
        Resize the input tensor.
        """
        if self.resizer is None:
            return inputs

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
        inputs, perm, perm_shape = self._permute_before_resize(
            inputs, ndim, resize_axes
        )

        # Apply resize
        out = self.resizer(inputs)

        # Restore original shape and order
        return self._permute_after_resize(out, perm, perm_shape, ndim)
