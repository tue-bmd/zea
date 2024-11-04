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
