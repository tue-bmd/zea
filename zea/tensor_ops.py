"""Basic tensor operations implemented with the multi-backend ``keras.ops``."""

from typing import Tuple, Union

import keras
import numpy as np
from keras import ops
from scipy.ndimage import _ni_support
from scipy.ndimage._filters import _gaussian_kernel1d

from zea import log
from zea.utils import map_negative_indices


def split_seed(seed, n):
    """Split a seed into n seeds for reproducible random ops.

    Supports `keras.random.SeedGenerator <https://keras.io/api/random/#seedgenerator-class>`_
    and `JAX random keys <https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.PRNGKey>`_.

    Args:
        seed: None, jax.Array, or keras.random.SeedGenerator.
        n (int): Number of seeds to generate.

    Returns:
        list: List of n seeds (JAX keys, SeedGenerator, or None).
    """
    # If seed is None, return a list of None
    if seed is None:
        return [None for _ in range(n)]

    # If seed is a JAX key, split it into n keys
    if keras.backend.backend() == "jax":
        import jax

        return jax.random.split(seed, n)

    # For other backends, we have to use Keras SeedGenerator
    else:
        assert isinstance(seed, keras.random.SeedGenerator), (
            "seed must be a SeedGenerator when not using JAX."
        )

        # Just duplicate the SeedGenerator
        return [seed for _ in range(n)]


def is_jax_prng_key(x):
    """Distinguish between jax.random.PRNGKey() and jax.random.key()"""
    if keras.backend.backend() == "jax":
        import jax

        return isinstance(x, jax.Array) and x.shape == (2,) and x.dtype == jax.numpy.uint32
    else:
        return False


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob=None, seed=None):
    """Adds salt and pepper noise to the input image.

    Args:
        image (ndarray): The input image, must be of type float32 and normalized between 0 and 1.
        salt_prob (float): The probability of adding salt noise to each pixel.
        pepper_prob (float, optional): The probability of adding pepper noise to each pixel.
            If not provided, it will be set to the same value as `salt_prob`.
        seed: A Python integer or instance of
            `keras.random.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.random.SeedGenerator`.

    Returns:
        ndarray: The noisy image with salt and pepper noise added.
    """
    if pepper_prob is None:
        pepper_prob = salt_prob

    if salt_prob == 0.0 and pepper_prob == 0.0:
        return image

    assert ops.dtype(image) == "float32", "Image should be of type float32."

    noisy_image = ops.copy(image)

    # Add salt noise
    salt_mask = keras.random.uniform(ops.shape(image), seed=seed) < salt_prob
    noisy_image = ops.where(salt_mask, 1.0, noisy_image)

    # Add pepper noise
    pepper_mask = keras.random.uniform(ops.shape(image), seed=seed) < pepper_prob
    noisy_image = ops.where(pepper_mask, 0.0, noisy_image)

    return noisy_image


def extend_n_dims(arr, axis, n_dims):
    """Extend the number of dimensions of an array.

    Inserts 'n_dims' ones at the specified axis.

    Args:
        arr: The input array.
        axis: The axis at which to insert the new dimensions.
        n_dims: The number of dimensions to insert.

    Returns:
        The array with the extended number of dimensions.

    Raises:
        AssertionError: If the axis is out of range.
    """
    assert axis <= ops.ndim(arr), (
        "Axis must be less than or equal to the number of dimensions in the array"
    )
    assert axis >= -ops.ndim(arr) - 1, (
        "Axis must be greater than or equal to the negative number of dimensions minus 1"
    )
    axis = ops.ndim(arr) + axis + 1 if axis < 0 else axis

    # Get the current shape of the array
    shape = ops.shape(arr)

    # Create the new shape, inserting 'n_dims' ones at the specified axis
    new_shape = shape[:axis] + (1,) * n_dims + shape[axis:]

    # Reshape the array to the new shape
    return ops.reshape(arr, new_shape)


def func_with_one_batch_dim(
    func,
    tensor,
    n_batch_dims: int,
    batch_size: int | None = None,
    func_axis: int | None = None,
    **kwargs,
):
    """Wraps a function to apply it to an input tensor with one or more batch dimensions.

    The function will be executed in parallel on all batch elements.

    Args:
        func (function): The function to apply to the image.
            Will take the `func_axis` output from the function.
        tensor (Tensor): The input tensor.
        n_batch_dims (int): The number of batch dimensions in the input tensor.
            Expects the input to start with n_batch_dims batch dimensions. Defaults to 2.
        batch_size (int, optional): Integer specifying the size of the batch for
            each step to execute in parallel. Defaults to None, in which case the function
            will run everything in parallel.
        func_axis (int, optional): If `func` returns mulitple outputs, this axis will be returned.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The output tensor with the same batch dimensions as the input tensor.

    Raises:
        ValueError: If the number of batch dimensions is greater than the rank of the input tensor.
    """
    # Extract the shape of the batch dimensions from the input tensor
    batch_dims = ops.shape(tensor)[:n_batch_dims]

    # Extract the shape of the remaining (non-batch) dimensions
    other_dims = ops.shape(tensor)[n_batch_dims:]

    # Reshape the input tensor to merge all batch dimensions into one
    reshaped_input = ops.reshape(tensor, [-1, *other_dims])

    # Apply the given function to the reshaped input tensor
    if batch_size is None:
        reshaped_output = func(reshaped_input, **kwargs)
    else:
        reshaped_output = batched_map(func, reshaped_input, batch_size=batch_size)

    # If the function returns multiple outputs, select the one corresponding to `func_axis`
    if isinstance(reshaped_output, (tuple, list)):
        if func_axis is None:
            raise ValueError(
                "func_axis must be specified when the function returns multiple outputs."
            )
        reshaped_output = reshaped_output[func_axis]

    # Extract the shape of the output tensor after applying the function (excluding the batch dim)
    output_other_dims = ops.shape(reshaped_output)[1:]

    # Reshape the output tensor to restore the original batch dimensions
    return ops.reshape(reshaped_output, [*batch_dims, *output_other_dims])


def matrix_power(matrix, power):
    """Compute the power of a square matrix.

    Should match the
    [numpy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html)
    implementation.

    Parameters:
        matrix (array-like): A square matrix to be raised to a power.
        power (int): The exponent to which the matrix is to be raised.
                    Must be a non-negative integer.
    Returns:
        array-like: The resulting matrix after raising the input matrix to the specified power.

    """
    if power == 0:
        return ops.eye(matrix.shape[0])
    if power == 1:
        return matrix
    if power % 2 == 0:
        half_power = matrix_power(matrix, power // 2)
        return ops.matmul(half_power, half_power)
    return ops.matmul(matrix, matrix_power(matrix, power - 1))


def boolean_mask(tensor, mask, size=None):
    """Apply a boolean mask to a tensor.

    Args:
        tensor (Tensor): The input tensor.
        mask (Tensor): The boolean mask to apply.
        size (int, optional): The size of the output tensor. Only used for Jax backend if you
            want to trace the function. Defaults to None.

    Returns:
        Tensor: The masked tensor.
    """
    if keras.backend.backend() == "jax" and size is not None:
        import jax.numpy as jnp

        indices = jnp.where(mask, size=size)  # Fixed size allows Jax tracing
        return tensor[indices]
    elif keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        return tf.boolean_mask(tensor, mask)
    else:
        return tensor[mask]


if keras.backend.backend() == "jax":
    import jax.numpy as jnp

    def nonzero(x, size=None, fill_value=None):
        """Return the indices of the elements that are non-zero.

        Args:
            x (Tensor): Input tensor.
            size (int, optional): optional static integer specifying the number of nonzero
                entries to return. If there are more nonzero elements than the specified size,
                then indices will be truncated at the end. If there are fewer nonzero elements
                than the specified size, then indices will be padded with fill_value.
            fill_value (int, optional): Value to fill in case there are not enough
                non-zero elements. Defaults to None.
        """
        return jnp.nonzero(x, size=size, fill_value=fill_value)

else:

    def nonzero(x, size=None, fill_value=None):
        """Return the indices of the elements that are non-zero."""
        return ops.nonzero(x)


def flatten(tensor, start_dim=0, end_dim=-1):
    """Should be similar to: https://pytorch.org/docs/stable/generated/torch.flatten.html"""
    # Get the shape of the input tensor
    old_shape = ops.shape(tensor)

    # Adjust end_dim if it's negative
    end_dim = ops.ndim(tensor) + end_dim if end_dim < 0 else end_dim

    # Create a new shape with -1 in the flattened dimensions
    new_shape = [*old_shape[:start_dim], -1, *old_shape[end_dim + 1 :]]

    # Reshape the tensor
    return ops.reshape(tensor, new_shape)


def batch_cov(x, rowvar=True, bias=False, ddof=None):
    """Compute the batch covariance matrices of the input tensor.

    Args:
        x (Tensor): Input tensor of shape (..., m, n) where m is the number of features
            and n is the number of observations.
        rowvar (bool, optional): If True, each row represents a variable,
            while each column represents an observation. If False, each column represents
            a variable, while each row represents an observation. Defaults to True.
        bias (bool, optional): If True, the biased estimator of the covariance is computed.
            If False, the unbiased estimator is computed. Defaults to False.
        ddof (int, optional): Delta degrees of freedom. The divisor used in the calculation
            is (num_obs - ddof), where num_obs is the number of observations.
            If ddof is not specified, it is set to 0 if bias is True, and 1 if bias is False.
            Defaults to None.

    Returns:
        Tensor: Batch covariance matrices of shape (..., m, m) if rowvar=True,
                or (..., n, n) if rowvar=False.
    """
    # Ensure the input has at least 3 dimensions
    if ops.ndim(x) == 2:
        x = x[None]

    if not rowvar:
        x = ops.moveaxis(x, -1, -2)

    num_obs = x.shape[-1]

    if ddof is None:
        ddof = 0 if bias else 1

    # Subtract the mean from each observation
    mean_x = ops.mean(x, axis=-1, keepdims=True)
    x_centered = x - mean_x

    # Compute the covariance using einsum
    cov_matrices = ops.einsum("...ik,...jk->...ij", x_centered, x_centered) / (num_obs - ddof)
    return cov_matrices


def patched_map(f, xs, patches: int, jit=True, **batch_kwargs):
    """Wrapper around `batched_map` for patching.

    Allows you to specify the number of patches rather than the batch size.
    """
    assert patches > 0, "Number of patches must be greater than 0."

    if patches == 1:
        return f(xs)
    else:
        length = ops.shape(xs)[0]
        batch_size = np.ceil(length / patches).astype(int)
        return batched_map(f, xs, batch_size, jit, **batch_kwargs)


def batched_map(f, xs, batch_size=None, jit=True, **batch_kwargs):
    """Map a function over leading array axes.

    Args:
        f (callable): Function to apply element-wise over the first axis.
        xs (Tensor): Values over which to map along the leading axis.
        batch_size (int, optional): Size of the batch for each step. Defaults to None,
            in which case the function will be equivalent to `ops.map`, and thus map over
            the leading axis.
        jit (bool, optional): If True, use a jitted version of the function for
            faster batched mapping. Else, loop over the data with the original function.
        batch_kwargs (dict, optional): Additional keyword arguments (tensors) to
            batch along with xs. Must have the same first dimension size as xs.

    Returns:
        The mapped tensor(s).

    Idea taken from: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.map.html
    """
    if batch_kwargs is None:
        batch_kwargs = {}

    # Ensure all batch kwargs have the same leading dimension as xs.
    if batch_kwargs:
        assert all(
            ops.shape(xs)[0] == ops.shape(v)[0] for v in batch_kwargs.values() if v is not None
        ), "All batch kwargs must have the same first dimension size as xs."

    total = ops.shape(xs)[0]
    # TODO: could be rewritten with ops.cond such that it also works for jit=True.
    if not jit and batch_size is not None and total <= batch_size:
        return f(xs, **batch_kwargs)

    ## Non-jitted version: simply iterate over batches.
    if not jit:
        bs = batch_size or 1  # Default batch size to 1 if not specified.
        outputs = []
        for i in range(0, total, bs):
            idx = slice(i, i + bs)
            current_kwargs = {k: v[idx] for k, v in batch_kwargs.items()}
            outputs.append(f(xs[idx], **current_kwargs))
        return ops.concatenate(outputs, axis=0)

    ## Jitted version.

    # Helper to create the batched function for use with ops.map.
    def create_batched_f(kw_keys):
        def batched_f(inputs):
            x, *kw_values = inputs
            kw = dict(zip(kw_keys, kw_values))
            return f(x, **kw)

        return batched_f

    if batch_size is None:
        batched_f = create_batched_f(list(batch_kwargs.keys()))
        return ops.map(batched_f, (xs, *batch_kwargs.values()))

    # Pad and reshape primary tensor.
    xs_padded = pad_array_to_divisible(xs, batch_size, axis=0)
    new_shape = (-1, batch_size) + ops.shape(xs_padded)[1:]
    xs_reshaped = ops.reshape(xs_padded, new_shape)

    # Pad and reshape batch_kwargs similarly.
    reshaped_kwargs = {}
    for k, v in batch_kwargs.items():
        if v is None:
            reshaped_kwargs[k] = None
        else:
            v_padded = pad_array_to_divisible(v, batch_size, axis=0)
            reshaped_kwargs[k] = ops.reshape(v_padded, (-1, batch_size) + ops.shape(v_padded)[1:])

    batched_f = create_batched_f(list(reshaped_kwargs.keys()))
    out = ops.map(batched_f, (xs_reshaped, *reshaped_kwargs.values()))
    out_reshaped = ops.reshape(out, (-1,) + ops.shape(out)[2:])
    return out_reshaped[:total]  # Remove any padding added.


if keras.backend.backend() == "jax":
    # For jit purposes
    def _get_padding(N, remainder):
        return N - remainder if remainder != 0 else 0

else:

    def _get_padding(N, remainder):
        return ops.where(remainder != 0, N - remainder, 0)


def pad_array_to_divisible(arr, N, axis=0, mode="constant", pad_value=None):
    """Pad an array to be divisible by N along the specified axis.

    Args:
        arr (Tensor): The input array to pad.
        N (int): The number to which the length of the specified axis should be divisible.
        axis (int, optional): The axis along which to pad the array. Defaults to 0.
        mode (str, optional): The padding mode to use. Defaults to 'constant'.
            One of `"constant"`, `"edge"`, `"linear_ramp"`,
            `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
            `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`,
            `"circular"`. Defaults to `"constant"`.
        pad_value (float, optional): The value to use for padding when mode='constant'.
            Defaults to None. If mode is not `constant`, this value should be None.

    Returns:
        Tensor: The padded array.
    """
    # Get the length of the specified axis
    length = ops.shape(arr)[axis]

    # Calculate how much padding is needed for the specified axis
    remainder = length % N
    padding = _get_padding(N, remainder)

    # Create a tuple with (before, after) padding for each axis
    pad_width = [(0, 0)] * ops.ndim(arr)  # No padding for other axes
    pad_width[axis] = (0, padding)  # Padding for the specified axis

    # Pad the array
    padded_array = ops.pad(arr, pad_width, mode=mode, constant_values=pad_value)

    return padded_array


def interpolate_data(subsampled_data, mask, order=1, axis=-1):
    """Interpolate subsampled data along a specified axis using `map_coordinates`.

    Args:
        subsampled_data (ndarray): The data subsampled along the specified axis.
            Its shape matches `mask` except along the subsampled axis.
        mask (ndarray): Boolean array with the same shape as the full data.
            `True` where data is known.
        order (int, optional): The order of the spline interpolation. Default is `1`.
        axis (int, optional): The axis along which the data is subsampled. Default is `-1`.

    Returns:
        ndarray: The data interpolated back to the original grid.

    ValueError: If `mask` does not indicate any missing data or if `mask` has `False`
        values along multiple axes.
    """
    mask = ops.cast(mask, "bool")
    # Check that mask indicates subsampled data along the specified axis
    if ops.sum(mask) == 0:
        raise ValueError("Mask does not indicate any known data.")

    if ops.sum(mask) == ops.prod(mask.shape):
        raise ValueError("Mask does not indicate any missing data.")

    # make sure subsampled data corresponds with number of 1s in the mask
    assert len(ops.where(mask)[0]) == ops.prod(subsampled_data.shape), (
        "Subsampled data does not match the number of 1s in the mask."
    )

    assert subsampled_data.ndim == 1, "Subsampled data should be a flattened 1D array"

    assert mask.ndim == 2, "Currently only 2D interpolation supported"

    # Get the indices of the known data points
    known_indices = ops.stack(ops.where(mask), axis=-1)

    # Get the indices of the unknown data points
    unknown_indices = ops.stack(ops.where(~mask), axis=-1)

    # map the unknown indices to the new coordinate system
    # which basically is range(0, mask.shape[axis]) for each axis
    # but with the gaps removed
    interp_coords = []
    subsampled_shape = []
    axis = axis if axis >= 0 else mask.ndim + axis

    for _axis in range(mask.ndim):
        length_axis = mask.shape[_axis]
        if _axis == axis:
            indices = ops.where(
                ops.any(~mask, axis=tuple(i for i in range(mask.ndim) if i != _axis))
            )[0]
            # unknown indices
            indices = map_indices_for_interpolation(indices)
            subsampled_shape.append(length_axis - len(indices))
        else:
            # broadcast indices
            indices = ops.arange(length_axis, dtype="float32")
            subsampled_shape.append(length_axis)

        interp_coords.append(indices)

    # create the grid of coordinates for the interpolation
    interp_coords = ops.meshgrid(*interp_coords, indexing="ij")
    # should be of shape (mask.ndim, -1)

    subsampled_data = ops.reshape(subsampled_data, subsampled_shape)

    # Use map_coordinates to interpolate the data
    interpolated_data = ops.image.map_coordinates(
        subsampled_data,
        interp_coords,
        order=order,
    )

    interpolated_data = ops.reshape(interpolated_data, -1)

    # now distirubute the interpolated data back to the original grid
    output_data = ops.zeros_like(mask, dtype=subsampled_data.dtype)

    output_data = ops.scatter_update(output_data, unknown_indices, interpolated_data)

    # Get the values at the known data points
    known_values = ops.reshape(subsampled_data, (-1,))
    output_data = ops.scatter_update(
        output_data,
        known_indices,
        known_values,
    )

    return output_data


def is_monotonic(array, increasing=True):
    """Checks if a given 1D array is monotonic.

    Either entirely non-decreasing or non-increasing.

    Args:
        array (ndarray): A 1D numpy array.

    Returns:
        bool: True if the array is monotonic, False otherwise.
    """
    # Convert to numpy array to handle general cases
    array = ops.array(array)

    # Check if the array is non-decreasing or non-increasing
    if increasing:
        return ops.all(array[1:] >= array[:-1])
    return ops.all(array[1:] <= array[:-1])


def map_indices_for_interpolation(indices):
    """Interpolates a 1D array of indices with gaps.

    Maps a 1D array of indices with gaps to a 1D array where gaps
    would be between the integers.

    Used in the `interpolate_data` function.

    Args:
        (indices): A 1D array of indices with gaps.
    Returns:
        (indices): mapped to a 1D array where gaps would be between
        the integers

    There are two segments here of length 4 and 2

    Example:
        >>> indices = [5, 6, 7, 8, 12, 13, 19]
        >>> mapped_indices = [5, 5.25, 5.5, 5.75, 8, 8.5, 12.5]

    """
    indices = ops.array(indices, dtype="int32")

    assert is_monotonic(indices, increasing=True), "Indices should be monotonically increasing"

    gap_starts = ops.where(indices[1:] - indices[:-1] > 1)[0]
    gap_starts = ops.concatenate([ops.array([0]), gap_starts + 1], axis=0)

    gap_lengths = ops.concatenate(
        [gap_starts[1:] - gap_starts[:-1], ops.array([len(indices) - gap_starts[-1]])],
        axis=0,
    )

    cumul_gap_lengths = ops.cumsum(gap_lengths)
    cumul_gap_lengths = ops.concatenate([ops.array([0]), cumul_gap_lengths], axis=0)

    gap_start_values = ops.take(indices, gap_starts)
    mapped_starts = gap_start_values - cumul_gap_lengths[:-1]
    mapped_starts = ops.cast(mapped_starts, "float32")
    gap_lengths = ops.cast(gap_lengths, "float32")

    spacing = 1 / (gap_lengths + 1)

    # Vectorized creation of gap_length entries between the start and end
    mapped_indices = ops.concatenate(
        [
            (mapped_starts[i] + spacing[i]) + spacing[i] * ops.arange(gap_lengths[i])
            for i in range(len(gap_lengths))
        ],
        axis=0,
    )
    mapped_indices -= 1

    return mapped_indices


def stack_volume_data_along_axis(data, batch_axis: int, stack_axis: int, number: int):
    """Stacks tensor data along a specified stack axis.

    Stack tensor data along a specified stack axis by splitting it into blocks along the batch axis.

    Args:
        data (Tensor): Input tensor to be stacked.
        batch_axis (int): Axis along which to split the data into blocks.
        stack_axis (int): Axis along which to stack the blocks.
        number (int): Number of slices per stack.

    Returns:
        Tensor: Reshaped tensor with data stacked along stack_axis.

    Example:
        .. code-block:: python

            import keras

            data = keras.random.uniform((10, 20, 30))
            # stacking along 1st axis with 2 frames per block
            stacked_data = stack_volume_data_along_axis(data, 0, 1, 2)
            stacked_data.shape
    """
    blocks = int(ops.ceil(data.shape[batch_axis] / number))
    data = pad_array_to_divisible(data, axis=batch_axis, N=blocks, mode="reflect")
    data = ops.split(data, blocks, axis=batch_axis)
    data = ops.stack(data, axis=batch_axis)
    # put batch_axis in front
    data = ops.transpose(
        data,
        (
            batch_axis + 1,
            *range(batch_axis + 1),
            *range(batch_axis + 2, data.ndim),
        ),
    )
    data = ops.concatenate(list(data), axis=stack_axis)
    return data


def split_volume_data_from_axis(data, batch_axis: int, stack_axis: int, number: int, padding: int):
    """Splits previously stacked tensor data back to its original shape.

    This function reverses the operation performed by `stack_volume_data_along_axis`.

    Args:
        data (Tensor): Input tensor to be split.
        batch_axis (int): Axis along which to restore the blocks.
        stack_axis (int): Axis from which to split the stacked data.
        number (int): Number of slices per stack.
        padding (int): Amount of padding to remove from the result.

    Returns:
        Tensor: Reshaped tensor with data split back to original format.

    Example:
        .. code-block:: python

            import keras

            data = keras.random.uniform((20, 10, 30))
            split_data = split_volume_data_from_axis(data, 0, 1, 2, 2)
            split_data.shape
    """
    if data.shape[stack_axis] == 1:
        # in this case it was a broadcasted axis which does not need to be split
        return data
    data = ops.split(data, number, axis=stack_axis)
    data = ops.stack(data, axis=batch_axis + 1)
    # combine the unstacked axes (dim 1 and 2)
    total_block_size = data.shape[batch_axis] * data.shape[batch_axis + 1]
    data = ops.reshape(
        data,
        (*data.shape[:batch_axis], total_block_size, *data.shape[batch_axis + 2 :]),
    )

    # cut off padding
    if padding > 0:
        indices = ops.arange(data.shape[batch_axis] - padding + 1)
        data = ops.take(data, indices, axis=batch_axis)

    return data


def compute_required_patch_overlap(image_shape, patch_shape):
    """Compute required overlap between patches to cover the entire image.

    Args:
        image_shape: Tuple of (height, width)
        patch_shape: Tuple of (patch_height, patch_width)

    Returns:
        Tuple of (overlap_y, overlap_x)

        Or None if there is no overlap that will result in integer number of patches
        given the image and patch shapes.
    """
    assert len(image_shape) == 2, "image_shape must be a tuple of (height, width)"
    assert len(patch_shape) == 2, "patch_shape must be a tuple of (patch_height, patch_width)"

    assert all(image_shape[i] >= patch_shape[i] for i in range(2)), (
        "patch_shape must be equal or smaller than image_shape"
    )

    image_y, image_x = image_shape
    patch_y, patch_x = patch_shape

    # Calculate number of patches needed in each dimension
    n_patch_y = max(1, int(ops.ceil(image_y / patch_y)))
    n_patch_x = max(1, int(ops.ceil(image_x / patch_x)))

    # Calculate new overlap only if we have more than one patch
    new_overlap = (
        ((patch_y * n_patch_y - image_y) / (n_patch_y - 1) if n_patch_y > 1 else 0),
        ((patch_x * n_patch_x - image_x) / (n_patch_x - 1) if n_patch_x > 1 else 0),
    )

    # check if can be integer
    if not all(ops.isclose(new_overlap, ops.round(new_overlap))):
        return

    new_overlap = tuple(map(int, new_overlap))

    return new_overlap


def compute_required_patch_shape(image_shape, patch_shape, overlap):
    """Compute required patch shape to cover the entire image.

    Compute patch_shape closest to the original patch_shape that will result
    in integer number of patches given the image and overlap.

    Args:
        image_shape: Tuple of (height, width)
        patch_shape: Tuple of (patch_height, patch_width)
        overlap: Tuple of (overlap_y, overlap_x)

    Returns:
        Tuple of (patch_shape_y, patch_shape_x)

        or None if there is no patch_shape that will result in integer number of patches
        given the image and overlap.
    """
    image_y, image_x = image_shape
    overlap_y, overlap_x = overlap
    patch_y, patch_x = patch_shape

    def compute_patch_size(image_size, patch_size, overlap):
        n_patches = (image_size - overlap) // (patch_size - overlap)
        new_patch_size = (image_size + (n_patches - 1) * overlap) / n_patches
        return int(new_patch_size)

    new_patch_y = compute_patch_size(image_y, patch_y, overlap_y)
    new_patch_x = compute_patch_size(image_x, patch_x, overlap_x)

    if (image_y - new_patch_y) % (new_patch_y - overlap_y) != 0 or (image_x - new_patch_x) % (
        new_patch_x - overlap_x
    ) != 0:
        return None

    return new_patch_y, new_patch_x


def check_patches_fit(
    image_shape: tuple, patch_shape: tuple, overlap: Union[int, Tuple[int, int]]
) -> tuple:
    """Checks if patches with overlap fit an integer amount in the original image.

    Args:
        image_shape: A tuple representing the shape of the original image.
        patch_size: A tuple representing the shape of the patches.
        overlap: A float representing the overlap between patches.

    Returns:
        A tuple containing a boolean indicating if the patches fit an integer amount
        in the original image and the new image shape if the patches do not fit.

    Example:
        .. code-block:: python

            image_shape = (10, 10)
            patch_shape = (4, 4)
            overlap = (2, 2)
            patches_fit, new_shape = check_patches_fit(image_shape, patch_shape, overlap)
            patches_fit
            new_shape
    """
    if overlap:
        stride = (np.array(patch_shape) - np.array(overlap)).astype(int)
    else:
        stride = (np.array(patch_shape)).astype(int)
        overlap = (0, 0)

    stride_y, stride_x = stride
    patch_y, patch_x = patch_shape
    image_y, image_x = image_shape

    if (image_y - patch_y) % stride_y != 0 or (image_x - patch_x) % stride_x != 0:
        new_shape = (
            (image_y - patch_y) // stride_y * stride_y + patch_y,
            (image_x - patch_x) // stride_x * stride_x + patch_x,
        )
        # new_patch_shape = tuple(map(int, new_patch_shape))
        new_patch_shape = compute_required_patch_shape(image_shape, patch_shape, overlap)

        # Calculate new overlap only if we have more than one patch
        new_overlap = compute_required_patch_overlap(image_shape, patch_shape)

        msg = (
            "patches with overlap do not fit an integer amount in the original image. "
            f"Cropping image to closest dimensions that work: {new_shape}. "
        )

        if new_patch_shape is not None:
            msg += f"Alternatively, change patch shape to: {new_patch_shape} "

        if new_overlap is not None:
            msg += f"or change overlap to: {new_overlap}"

        log.warning(msg)

        return False, new_shape
    return True, image_shape


def images_to_patches(
    images: keras.KerasTensor,
    patch_shape: Union[int, Tuple[int, int]],
    overlap: Union[int, Tuple[int, int]] = None,
) -> keras.KerasTensor:
    """Creates patches from images.

    Args:
        images (Tensor): input images [batch, height, width, channels].
        patch_shape (int or tuple, optional): Height and width of patch. Defaults to 4.
        overlap (int or tuple, optional): Overlap between patches in px. Defaults to None.

    Returns:
        patches (Tensor): batch of patches of size:
            [batch, #patch_y, #patch_x, patch_size_y, patch_size_x, #channels].

    Example:
        .. code-block:: python

            import keras

            images = keras.random.uniform((2, 8, 8, 3))
            patches = images_to_patches(images, patch_shape=(4, 4), overlap=(2, 2))
            patches.shape
    """
    assert len(images.shape) == 4, (
        f"input array should have 4 dimensions, but has {len(images.shape)} dimensions"
    )
    assert isinstance(patch_shape, int) or len(patch_shape) == 2, (
        f"patch_shape should be an integer or a tuple of length 2, but is {patch_shape}"
    )
    assert isinstance(overlap, (int, type(None))) or len(overlap) == 2, (
        f"overlap should be an integer or a tuple of length 2, but is {overlap}"
    )

    batch_size, *image_shape, n_channels = images.shape

    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, patch_shape)
    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    patch_size_y, patch_size_x = patch_shape

    patches_fit, image_shape = check_patches_fit(image_shape, patch_shape, overlap)
    if not patches_fit:
        images = images[:, : image_shape[0], : image_shape[1], :]

    if overlap:
        stride = (np.array(patch_shape) - np.array(overlap)).astype(int)
    else:
        stride = np.array(patch_shape).astype(int)

    # assert that stride is never smaller than 0 or larger than patch_shape
    stride = np.maximum(stride, 1)
    stride = np.minimum(stride, patch_shape)
    assert np.all(stride <= patch_shape), "Stride should be smaller than patch shape"
    assert np.all(stride >= 0), "Stride should be larger than 0"

    ## create patches using ops (this operation is too memory intensive)
    # patches = ops.image.extract_patches(
    #     images, size=patch_shape, strides=list(stride), padding="valid"
    # )

    ## manual solution instead
    patches_list = []
    for i in range(0, image_shape[0] - patch_size_y + 1, stride[0]):
        row_patches = []
        for j in range(0, image_shape[1] - patch_size_x + 1, stride[1]):
            patch = images[:, i : i + patch_size_y, j : j + patch_size_x, :]
            row_patches.append(patch)
        patches_list.append(ops.stack(row_patches, axis=1))
    patches = ops.stack(patches_list, axis=1)

    _, n_patch_y, n_patch_x, *_ = patches.shape

    shape = [batch_size, n_patch_y, n_patch_x, patch_size_y, patch_size_x, n_channels]
    patches = ops.reshape(patches, shape)
    return patches


def patches_to_images(
    patches: keras.KerasTensor,
    image_shape: tuple,
    overlap: Union[int, Tuple[int, int]] = None,
    window_type="average",
) -> keras.KerasTensor:
    """Reconstructs images from patches.

    Args:
        patches (Tensor): Array with batch of patches to convert to batch of images.
            [batch_size, #patch_y, #patch_x, patch_size_y, patch_size_x, n_channels]
        image_shape (Tuple): Shape of output image. (height, width, channels)
        overlap (int or tuple, optional): Overlap between patches in px. Defaults to None.
        window_type (str, optional): Type of stitching to use. Defaults to 'average'.
            Options: 'average', 'replace'.

    Returns:
        images (Tensor): Reconstructed batch of images from batch of patches.

    Example:
        .. code-block:: python

            import keras

            patches = keras.random.uniform((2, 3, 3, 4, 4, 3))
            images = patches_to_images(patches, image_shape=(8, 8, 3), overlap=(2, 2))
            images.shape
    """
    # Input validation
    assert len(image_shape) == 3, "image_shape must have 3 dimensions: (height, width, channels)."
    assert len(patches.shape) == 6, (
        "patches must have 6 dimensions: [batch_size, n_patch_y, n_patch_x, "
        "patch_size_y, patch_size_x, n_channels]."
    )
    assert window_type in [
        "average",
        "replace",
    ], "window_type must be one of 'average', or 'replace'."

    # Extract dimensions
    batch_size, n_patches_y, n_patches_x, patch_size_y, patch_size_x, _ = patches.shape
    dtype = patches.dtype

    if isinstance(overlap, int):
        overlap = (overlap, overlap)
    if overlap is None:
        overlap = (0, 0)

    stride_y, stride_x = np.array([patch_size_y, patch_size_x]) - np.array(overlap)

    # Initialize the output tensor (image) and mask
    images = keras.ops.zeros((batch_size, *image_shape), dtype=dtype)
    mask = keras.ops.zeros((batch_size, *image_shape), dtype=dtype)

    # Loop through each patch
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            start_y = i * stride_y
            start_x = j * stride_x
            patch = patches[:, i, j]

            if window_type == "replace":
                # Replace pixels directly with the current patch
                images = keras.ops.slice_update(images, [0, start_y, start_x, 0], patch)
            else:
                # Add the current patch to the image
                images = keras.ops.slice_update(
                    images,
                    [0, start_y, start_x, 0],
                    images[
                        :,
                        start_y : start_y + patch_size_y,
                        start_x : start_x + patch_size_x,
                        :,
                    ]
                    + patch,
                )
                # Update the mask for averaging
                mask = keras.ops.slice_update(
                    mask,
                    [0, start_y, start_x, 0],
                    mask[
                        :,
                        start_y : start_y + patch_size_y,
                        start_x : start_x + patch_size_x,
                        :,
                    ]
                    + 1,
                )

    if window_type == "average":
        # Normalize overlapping regions if needed
        images = keras.ops.where(mask > 0, images / mask, images)

    return images


def reshape_axis(data, newshape: tuple, axis: int):
    """Reshape data along axis.

    Args:
        data (tensor): input data.
        newshape (tuple): new shape of data along axis.
        axis (int): axis to reshape.

    Example:
        .. code-block:: python

            import keras

            data = keras.random.uniform((3, 4, 5))
            newshape = (2, 2)
            reshaped_data = reshape_axis(data, newshape, axis=1)
            reshaped_data.shape
    """
    axis = map_negative_indices([axis], data.ndim)[0]
    shape = list(ops.shape(data))  # list
    shape = shape[:axis] + list(newshape) + shape[axis + 1 :]
    return ops.reshape(data, shape)


def _gaussian_filter1d(array, kernel, radius, cval=None, axis=-1, mode="symmetric"):
    if keras.backend.backend() == "torch":
        assert mode == "constant", (
            "Only constant padding is for sure correct in torch."
            "Symmetric padding produces different results in torch compared to tensorflow..."
        )

    # Pad input along the specified axis.
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (radius, radius)
    padded = ops.pad(array, pad_width, mode=mode, constant_values=cval)

    # Move the convolution axis to the last axis.
    moved = ops.moveaxis(padded, axis, -1)  # shape: (..., length)
    orig_shape = moved.shape
    length = orig_shape[-1]

    # Collapse all non-convolution dimensions into the batch.
    reshaped = ops.reshape(moved, (-1, length, 1))  # shape: (batch, length, in_channels=1)

    # Reshape kernel for convolution: expected shape (kernel_size, in_channels, out_channels)
    kernel_size = kernel.shape[0]
    kernel_reshaped = ops.reshape(kernel, (kernel_size, 1, 1))

    # Run the convolution using 'VALID' padding.
    conv_result = ops.depthwise_conv(
        reshaped,
        kernel_reshaped,
        padding="valid",
        data_format="channels_last",
    )

    # Reshape the convolved result back to the padded shape.
    new_length = conv_result.shape[1]
    conv_result = ops.reshape(conv_result, (*orig_shape[:-1], new_length))

    # Move the convolution axis back to its original position.
    result = ops.moveaxis(conv_result, -1, axis)
    return result


def gaussian_filter1d(array, sigma, axis=-1, order=0, mode="symmetric", truncate=4.0, cval=None):
    """1-D Gaussian filter.

    Args:
        array (Tensor): The input array.
        sigma (float or tuple): Standard deviation for Gaussian kernel. The standard deviations
            of the Gaussian filter are given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes.
        order (int or Tuple[int]): The order of the filter along each axis is given as a sequence of
            integers, or as a single number. An order of 0 corresponds to convolution with a
            Gaussian kernel. A positive order corresponds to convolution with that derivative
            of a Gaussian. Default is 0.
        mode (str, optional): Padding mode for the input image. Default is 'symmetric'.
            See [keras docs](https://www.tensorflow.org/api_docs/python/tf/keras/ops/pad) for
            all options and [tensoflow docs](https://www.tensorflow.org/api_docs/python/tf/pad)
            for some examples. Note that the naming differs from scipy.ndimage.gaussian_filter!
        cval (float, optional): Value to fill past edges of input if mode is 'constant'.
            Default is None.
        truncate (float, optional): Truncate the filter at this many standard deviations.
            Default is 4.0.
        axes (Tuple[int], optional): If None, input is filtered along all axes. Otherwise, input
            is filtered along the specified axes. When axes is specified, any tuples used for
            sigma, order, mode and/or radius must match the length of axes. The ith entry in
            any of these tuples corresponds to the ith entry in axes.
    """
    # Determine the effective kernel radius and generate the Gaussian kernel
    radius = int(round(truncate * sigma))
    kernel = _gaussian_kernel1d(sigma, order, radius).astype(
        ops.dtype(array)
    )  # shape: (kernel_size,)

    # Reverse the kernel for odd orders to mimic correlation (SciPy behavior)
    if order % 2:
        kernel = kernel[::-1]

    return _gaussian_filter1d(array, kernel, radius, cval, axis, mode)


def gaussian_filter(
    array,
    sigma,
    order: int | Tuple[int] = 0,
    mode: str = "symmetric",
    cval: float | None = None,
    truncate: float = 4.0,
    axes: Tuple[int] = None,
):
    """Multidimensional Gaussian filter.

    If you want to use this function with jax.jit, you can set:
    `static_argnames=("truncate", "sigma")`

    Args:
        array (Tensor): The input array.
        sigma (float or tuple): Standard deviation for Gaussian kernel. The standard deviations
            of the Gaussian filter are given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes.
        order (int or Tuple[int]): The order of the filter along each axis is given as a sequence of
            integers, or as a single number. An order of 0 corresponds to convolution with a
            Gaussian kernel. A positive order corresponds to convolution with that derivative
            of a Gaussian. Default is 0.
        mode (str, optional): Padding mode for the input image. Default is 'symmetric'.
            See [keras docs](https://www.tensorflow.org/api_docs/python/tf/keras/ops/pad) for
            all options and [tensoflow docs](https://www.tensorflow.org/api_docs/python/tf/pad)
            for some examples. Note that the naming differs from scipy.ndimage.gaussian_filter!
        cval (float, optional): Value to fill past edges of input if mode is 'constant'.
            Default is None.
        truncate (float, optional): Truncate the filter at this many standard deviations.
            Default is 4.0.
        axes (Tuple[int], optional): If None, input is filtered along all axes. Otherwise, input
            is filtered along the specified axes. When axes is specified, any tuples used for
            sigma, order, mode and/or radius must match the length of axes. The ith entry in
            any of these tuples corresponds to the ith entry in axes.
    """
    axes = _ni_support._check_axes(axes, array.ndim)
    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii]) for ii in range(num_axes)]
    if len(axes) > 0:
        for (
            axis,
            sigma,
            order,
            mode,
        ) in axes:
            output = gaussian_filter1d(array, sigma, axis, order, mode, truncate, cval)
            array = output
    else:
        output = array
    return output


def resample(x, n_samples, axis=-2, order=1):
    """Resample tensor along axis.

    Similar to scipy.signal.resample.

    Args:
        x: input tensor.
        n_samples: number of samples after resampling.
        axis: axis to resample along.
        order: interpolation order (1=linear).

    Returns:
        Resampled tensor.
    """
    shape = ops.shape(x)
    rank = len(shape)

    # Move axis-to-resample to position 1
    perm = list(range(rank))
    perm_axis1 = perm.copy()
    perm_axis1[axis], perm_axis1[1] = perm_axis1[1], perm_axis1[axis]
    x = ops.transpose(x, perm_axis1)

    # Shape after transpose
    shape = ops.shape(x)
    batch_size = shape[0]
    old_n = shape[1]
    other_dims = shape[2:]

    # Create sampling grid
    batch_coords = ops.arange(batch_size, dtype="float32")  # (batch_size,)
    new_coords = ops.linspace(0.0, ops.cast(old_n - 1, dtype="float32"), n_samples)  # (n_samples,)
    other_coords = [ops.arange(d, dtype="float32") for d in other_dims]

    # Meshgrid
    grid = ops.meshgrid(
        batch_coords, new_coords, *other_coords, indexing="ij"
    )  # list of (batch_size, n_samples, ...)
    coord_grid = ops.stack(grid, axis=0)  # shape: (rank, batch_size, n_samples, ...)

    # Interpolate
    resampled = ops.image.map_coordinates(x, coord_grid, order=order)

    # Inverse transpose to restore original axis order
    inv_perm = [perm_axis1.index(i) for i in range(rank)]
    resampled = ops.transpose(resampled, inv_perm)

    return resampled


def fori_loop(lower, upper, body_fun, init_val, disable_jit=False):
    """For loop allowing for non-jitted for loop with same signature as jax.

    Args:
        lower (int): Lower bound of the loop.
        upper (int): Upper bound of the loop.
        body_fun (function): Function to be executed in the loop.
        init_val (any): Initial value for the loop.
        disable_jit (bool, optional): If True, disables JIT compilation. Defaults to False.
    """
    if not disable_jit:
        return ops.fori_loop(lower, upper, body_fun, init_val)

    # Fallback to a non-jitted for loop
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def L2(x):
    """L2 norm of a tensor.

    Implementation of L2 norm: https://mathworld.wolfram.com/L2-Norm.html
    """
    return ops.sqrt(ops.sum(x**2))


def linear_sum_assignment(cost):
    """Greedy linear sum assignment.

    Args:
        cost (Tensor): Cost matrix of shape (n, n).
    Returns:
        Tuple: Row indices and column indices for assignment.

    Returns row indices and column indices for assignment.
    """
    n = ops.shape(cost)[0]
    assigned_true = ops.zeros((n,), dtype="bool")
    row_ind = []
    col_ind = []
    for i in range(n):
        mask = 1.0 - ops.cast(assigned_true, "float32")
        masked_cost = cost[i] + (1.0 - mask) * 1e6
        idx = int(ops.argmin(masked_cost))
        row_ind.append(i)
        col_ind.append(idx)
        assigned_true = keras.ops.scatter_update(assigned_true, [[idx]], [True])
    return np.array(row_ind), np.array(col_ind)


def sinc(x, eps=keras.config.epsilon()):
    """Sinc function."""
    return ops.sin(x + eps) / (x + eps)


if keras.backend.backend() == "tensorflow":

    def safe_vectorize(
        pyfunc,
        excluded=None,
        signature=None,
    ):
        """Just a wrapper around ops.vectorize.

        Because tensorflow does not support multiple arguments to ops.vectorize(func)(...)
        We will just map the function manually.
        """

        def _map(*args):
            outputs = []
            for i in range(ops.shape(args[0])[0]):
                outputs.append(pyfunc(*[arg[i] for arg in args]))
            return ops.stack(outputs)

        return _map

else:

    def safe_vectorize(pyfunc, excluded=None, signature=None):
        """Just a wrapper around ops.vectorize."""
        return ops.vectorize(pyfunc, excluded=excluded, signature=signature)
