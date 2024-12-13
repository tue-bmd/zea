"""Basic tensor operations implemented with the multi-backend `keras.ops`."""

import os

import keras
from keras import ops


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob=None, seed=None):
    """
    Adds salt and pepper noise to the input image.

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
    """
    Extend the number of dimensions of an array by inserting 'n_dims' ones at the specified axis.

    Args:
        arr: The input array.
        axis: The axis at which to insert the new dimensions.
        n_dims: The number of dimensions to insert.

    Returns:
        The array with the extended number of dimensions.

    Raises:
        AssertionError: If the axis is out of range.
    """
    assert axis <= ops.ndim(
        arr
    ), "Axis must be less than or equal to the number of dimensions in the array"
    assert (
        axis >= -ops.ndim(arr) - 1
    ), "Axis must be greater than or equal to the negative number of dimensions minus 1"
    axis = ops.ndim(arr) + axis + 1 if axis < 0 else axis

    # Get the current shape of the array
    shape = ops.shape(arr)

    # Create the new shape, inserting 'n_dims' ones at the specified axis
    new_shape = shape[:axis] + (1,) * n_dims + shape[axis:]

    # Reshape the array to the new shape
    return ops.reshape(arr, new_shape)


def func_with_one_batch_dim(
    func, tensor, n_batch_dims: int = 2, func_axis: int | None = None, **kwargs
):
    """
    Applies a function to an input tensor with one or more batch dimensions. The function will
    be executed in parallel on all batch elements.

    Args:
        func (function): The function to apply to the image.
            Will take the `func_axis` output from the function.
        tensor (Tensor): The input tensor.
        n_batch_dims (int): The number of batch dimensions in the input tensor.
            Expects the input to start with n_batch_dims batch dimensions. Defaults to 2.
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
    reshaped_output = func(reshaped_input, **kwargs)

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
    """
    Compute the power of a square matrix.
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


def boolean_mask(tensor, mask):
    """
    Apply a boolean mask to a tensor.

    Args:
        tensor (Tensor): The input tensor.
        mask (Tensor): The boolean mask to apply.

    Returns:
        Tensor: The masked tensor.
    """
    if os.environ.get("KERAS_BACKEND") != "tensorflow":
        return tensor[mask]
    else:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        return tf.boolean_mask(tensor, mask)


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
    """
    Compute the batch covariance matrices of the input tensor.

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
    cov_matrices = ops.einsum("...ik,...jk->...ij", x_centered, x_centered) / (
        num_obs - ddof
    )
    return cov_matrices


def batched_map(f, xs, batch_size=None):
    """
    Map a function over leading array axes.

    Args:
        f (callable): Function to apply element-wise over the first axis.
        xs (Tensor): Values over which to map along the leading axis.
        batch_size (int, optional): Integer specifying the size of the batch for
            each step to execute in parallel.

    Idea taken from: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.map.html
    """
    if batch_size == 1:
        return f(xs)

    if batch_size is None:
        out = ops.map(f, xs)
    else:
        length = ops.shape(xs)[0]
        xs = pad_array_to_divisible(xs, batch_size, axis=0)
        xs = ops.reshape(xs, (-1, batch_size) + ops.shape(xs)[1:])
        out = ops.map(f, xs)
        out = ops.reshape(out, (-1,) + ops.shape(out)[2:])
        out = out[:length]  # remove padding

    return out


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
    padding = N - remainder if remainder != 0 else 0

    # Create a tuple with (before, after) padding for each axis
    pad_width = [(0, 0)] * ops.ndim(arr)  # No padding for other axes
    pad_width[axis] = (0, padding)  # Padding for the specified axis

    # Pad the array
    padded_array = ops.pad(arr, pad_width, mode=mode, constant_values=pad_value)

    return padded_array


def interpolate_data(subsampled_data, mask, order=1, axis=-1):
    """
    Interpolate subsampled data along a specified axis using `map_coordinates`.

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
    assert len(ops.where(mask)[0]) == ops.prod(
        subsampled_data.shape
    ), "Subsampled data does not match the number of 1s in the mask."

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
    """
    Checks if a given 1D array is monotonic (either entirely non-decreasing or non-increasing).

    Args:
        array (ndarray): A 1D numpy array.
    Returns:
        bool: True if the array is monotonic, False otherwise.
    """
    # Convert to numpy array to handle general cases
    array = ops.array(array)

    # Check if the array is non-decreasing or non-increasing
    if increasing:
        return ops.all(array[1:] <= array[:-1])
    return ops.all(array[1:] >= array[:-1])


def map_indices_for_interpolation(indices):
    """Map a 1D array of indices with gaps to a 1D array where gaps
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

    assert is_monotonic(
        indices, increasing=True
    ), "Indices should be monotonically increasing"

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
    """
    Stack tensor data along a specified stack axis by splitting it into blocks along the batch axis.

    Args:
        data (Tensor): Input tensor to be stacked.
        batch_axis (int): Axis along which to split the data into blocks.
        stack_axis (int): Axis along which to stack the blocks.
        number (int): Number of slices per stack.

    Returns:
        Tensor: Reshaped tensor with data stacked along stack_axis.

    Example:
        ```python
        >>> keras.random.uniform((10, 20, 30))
        >>> # stacking along 1st axis with 2 frames per block
        >>> stacked_data = stack_volume_data_along_axis(data, 0, 1, 2)
        >>> stacked_data.shape
        (20, 10, 30)

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


def split_volume_data_from_axis(
    data, batch_axis: int, stack_axis: int, number: int, padding: int
):
    """
    Split previously stacked tensor data back to its original shape.
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
        >>> data = keras.random.uniform((20, 10, 30))
        >>> # splitting along 1st axis with 2 frames per block and padding of 2
        >>> split_data = split_volume_data_from_axis(data, 0, 1, 2, 2)
        >>> split_data.shape
        (10, 20, 30)

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
