from typing import List, Tuple, Union

import numpy as np
from keras import ops
from keras.src.layers.preprocessing.data_layer import DataLayer

from zea.func import normalize
from zea.func.tensor import gaussian_filter
from zea.internal.registry import ops_registry
from zea.ops.base import Filter, Operation
from zea.utils import map_negative_indices


@ops_registry("gaussian_blur")
class GaussianBlur(Filter):
    """
    GaussianBlur is an operation that applies a Gaussian blur to an input image.
    Uses scipy.ndimage.gaussian_filter to create a kernel.
    """

    def __init__(
        self,
        sigma: float,
        order: int | Tuple[int] = 0,
        mode: str = "symmetric",
        cval: float | None = None,
        truncate: float = 4.0,
        axes: Tuple[int] = (-3, -2),
        **kwargs,
    ):
        """
        Args:
            sigma (float or tuple): Standard deviation for Gaussian kernel. The standard deviations
                of the Gaussian filter are given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes.
            order (int or Tuple[int]): The order of the filter along each axis is given as a
                sequence of integers, or as a single number. An order of 0 corresponds to
                convolution with a Gaussian kernel. A positive order corresponds to convolution
                with that derivative of a Gaussian. Default is 0.
            mode (str, optional): Padding mode for the input image. Default is 'symmetric'.
                See [keras docs](https://www.tensorflow.org/api_docs/python/tf/keras/ops/pad) for
                all options and [tensorflow docs](https://www.tensorflow.org/api_docs/python/tf/pad)
                for some examples. Note that the naming differs from scipy.ndimage.gaussian_filter!
            cval (float, optional): Value to fill past edges of input if mode is 'constant'.
                Default is None.
            truncate (float, optional): Truncate the filter at this many standard deviations.
                Default is 4.0.
            axes (Tuple[int], optional): If None, input is filtered along all axes. Otherwise, input
                is filtered along the specified axes. When axes is specified, any tuples used for
                sigma, order, mode and/or radius must match the length of axes. The ith entry in
                any of these tuples corresponds to the ith entry in axes. Default is (-3, -2),
                which corresponds to the height and width dimensions of a
                (..., height, width, channels) tensor.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.order = order
        self.mode = mode
        self.cval = cval
        self.truncate = truncate
        self.axes = axes

    def call(self, **kwargs):
        """Apply a Gaussian filter to the input data.

        Args:
            data (ops.Tensor): Input image data of shape (height, width, channels) with
                optional batch dimension if ``self.with_batch_dim``.
        """
        data = kwargs[self.key]
        axes = self._resolve_filter_axes(data, self.axes)

        out = gaussian_filter(
            data, self.sigma, self.order, self.mode, self.cval, self.truncate, axes
        )

        return {self.output_key: out}


@ops_registry("normalize")
class Normalize(Operation):
    """Normalize data to a given range."""

    def __init__(self, output_range=None, input_range=None, **kwargs):
        super().__init__(additional_output_keys=["minval", "maxval"], **kwargs)
        if output_range is None:
            output_range = (0, 1)
        self.output_range = self.to_float32(output_range)
        self.input_range = self.to_float32(input_range)

        if len(self.output_range) != 2:
            raise ValueError(
                f"output_range must have exactly 2 elements, got {len(self.output_range)}"
            )
        if self.input_range is not None and len(self.input_range) != 2:
            raise ValueError(
                f"input_range must have exactly 2 elements, got {len(self.input_range)}"
            )

    @staticmethod
    def to_float32(data):
        """Converts an iterable to float32 and leaves None values as is."""
        return (
            [np.float32(x) if x is not None else None for x in data] if data is not None else None
        )

    @property
    def valid_keys(self):
        if self.input_range is None:
            return super().valid_keys.union({"maxval", "minval"})
        else:
            return super().valid_keys

    def call(self, **kwargs):
        """Normalize data to a given range.

        Args:
            output_range (tuple, optional): Range to which data should be mapped.
                Defaults to (0, 1).
            input_range (tuple, optional): Range of input data. If None, the range
                of the input data will be computed. Defaults to None.

        Returns:
            dict: Dictionary containing normalized data, along with the computed
                  or provided input range (minval and maxval).
        """
        data = kwargs[self.key]

        # If input_range is not provided, try to get it from kwargs
        # This allows you to normalize based on the first frame in a sequence and avoid flicker
        if self.input_range is None:
            maxval = kwargs.get("maxval", None)
            minval = kwargs.get("minval", None)
        # If input_range is provided, use it
        else:
            minval, maxval = self.input_range

        # If input_range is still not provided, compute it from the data
        if minval is None:
            minval = ops.min(data)
        if maxval is None:
            maxval = ops.max(data)

        normalized_data = normalize(
            data, output_range=self.output_range, input_range=(minval, maxval)
        )

        return {self.output_key: normalized_data, "minval": minval, "maxval": maxval}


@ops_registry("pad")
class Pad(Operation, DataLayer):
    """Pad layer for padding tensors to a specified shape."""

    def __init__(
        self,
        target_shape: list | tuple,
        uniform: bool = True,
        axis: Union[int, List[int]] = None,
        fail_on_bigger_shape: bool = True,
        pad_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_shape = target_shape
        self.uniform = uniform
        self.axis = axis
        self.pad_kwargs = pad_kwargs or {}
        self.fail_on_bigger_shape = fail_on_bigger_shape

    @staticmethod
    def _format_target_shape(shape_array, target_shape, axis):
        if isinstance(axis, int):
            axis = [axis]
        assert len(axis) == len(target_shape), (
            "The length of axis must be equal to the length of target_shape."
        )
        axis = map_negative_indices(axis, len(shape_array))

        target_shape = [
            target_shape[axis.index(i)] if i in axis else shape_array[i]
            for i in range(len(shape_array))
        ]
        return target_shape

    def pad(
        self,
        z,
        target_shape: list | tuple,
        uniform: bool = True,
        axis: Union[int, List[int]] = None,
        fail_on_bigger_shape: bool = True,
        **kwargs,
    ):
        """
        Pads the input tensor `z` to the specified shape.

        Parameters:
            z (tensor): The input tensor to be padded.
            target_shape (list or tuple): The target shape to pad the tensor to.
            uniform (bool, optional): If True, ensures that padding is uniform (even on both sides).
                Default is False.
            axis (int or list of int, optional): The axis or axes along which `target_shape` was
                specified. If None, `len(target_shape) == `len(ops.shape(z))` must hold.
                Default is None.
            fail_on_bigger_shape (bool, optional): If True (default), raises an error if any target
                dimension is smaller than the input shape; if False, pads only where the
                target shape exceeds the input shape and leaves other dimensions unchanged.
            kwargs: Additional keyword arguments to pass to the padding function.

        Returns:
            tensor: The padded tensor with the specified shape.
        """
        shape_array = self.backend.shape(z)

        # When axis is provided, convert target_shape
        if axis is not None:
            target_shape = self._format_target_shape(shape_array, target_shape, axis)

        if not fail_on_bigger_shape:
            target_shape = [max(target_shape[i], shape_array[i]) for i in range(len(shape_array))]

        # Compute the padding required for each dimension
        pad_shape = np.array(target_shape) - shape_array

        # Create the paddings array
        if uniform:
            # if odd, pad more on the left, same as:
            # https://keras.io/api/layers/preprocessing_layers/image_preprocessing/center_crop/
            right_pad = pad_shape // 2
            left_pad = pad_shape - right_pad
            paddings = np.stack([right_pad, left_pad], axis=1)
        else:
            paddings = np.stack([np.zeros_like(pad_shape), pad_shape], axis=1)

        if np.any(paddings < 0):
            raise ValueError(
                f"Target shape {target_shape} must be greater than or equal "
                f"to the input shape {shape_array}."
            )

        return self.backend.numpy.pad(z, paddings, **kwargs)

    def call(self, **kwargs):
        data = kwargs[self.key]
        padded_data = self.pad(
            data,
            self.target_shape,
            self.uniform,
            self.axis,
            self.fail_on_bigger_shape,
            **self.pad_kwargs,
        )
        return {self.output_key: padded_data}


@ops_registry("threshold")
class Threshold(Operation):
    """Threshold an array, setting values below/above a threshold to a fill value."""

    def __init__(
        self,
        threshold_type="hard",
        below_threshold=True,
        fill_value="min",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if threshold_type not in ("hard", "soft"):
            raise ValueError("threshold_type must be 'hard' or 'soft'")
        self.threshold_type = threshold_type
        self.below_threshold = below_threshold
        self._fill_value_type = fill_value

        # Define threshold function at init
        if threshold_type == "hard":
            if below_threshold:
                self._threshold_func = lambda data, threshold, fill: ops.where(
                    data < threshold, fill, data
                )
            else:
                self._threshold_func = lambda data, threshold, fill: ops.where(
                    data > threshold, fill, data
                )
        else:  # soft
            if below_threshold:
                self._threshold_func = lambda data, threshold, fill: (
                    ops.maximum(data - threshold, 0) + fill
                )
            else:
                self._threshold_func = lambda data, threshold, fill: (
                    ops.minimum(data - threshold, 0) + fill
                )

    def _resolve_fill_value(self, data, threshold):
        """Get the fill value based on the fill_value_type."""
        fv = self._fill_value_type
        if isinstance(fv, (int, float)):
            return ops.convert_to_tensor(fv, dtype=data.dtype)
        elif fv == "min":
            return ops.min(data)
        elif fv == "max":
            return ops.max(data)
        elif fv == "threshold":
            return threshold
        else:
            raise ValueError("Unknown fill_value")

    def call(
        self,
        threshold=None,
        percentile=None,
        **kwargs,
    ):
        """Threshold the input data.

        Args:
            threshold: Numeric threshold.
            percentile: Percentile to derive threshold from.
        Returns:
            Tensor with thresholding applied.
        """
        data = kwargs[self.key]
        if (threshold is None) == (percentile is None):
            raise ValueError("Pass either threshold or percentile, not both or neither.")

        if percentile is not None:
            # Convert percentile to quantile value (0-1 range)
            threshold = ops.quantile(data, percentile / 100.0)

        fill_value = self._resolve_fill_value(data, threshold)
        result = self._threshold_func(data, threshold, fill_value)
        return {self.output_key: result}
