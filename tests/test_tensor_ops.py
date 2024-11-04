"""
Tests for the `tensor_ops` module.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pytest
import torch
from keras import ops

from tests.helpers import equality_libs_processing
from usbmd import tensor_ops


@pytest.mark.parametrize(
    "array, start_dim, end_dim",
    [
        [np.random.normal(size=(5, 10)), 0, 1],
        [np.random.normal(size=(5, 10, 15, 20)), 1, -1],
        [np.random.normal(size=(5, 10, 15, 20)), 2, 3],
        [np.random.normal(size=(5, 10, 15, 20, 25)), 0, 2],
    ],
)
@equality_libs_processing()
def test_flatten(array, start_dim, end_dim):
    """Test the `flatten` function to `torch.flatten`."""
    out = tensor_ops.flatten(array, start_dim, end_dim)
    torch_out = torch.flatten(
        torch.from_numpy(array), start_dim=start_dim, end_dim=end_dim
    ).numpy()

    # Test if the output is equal to the torch.flatten implementation
    np.testing.assert_almost_equal(torch_out, out)

    return out  # Return the output for the equality_libs_processing decorator


def recursive_cov(data, *args, **kwargs):
    """
    Helper function to test `batch_cov` to `np.cov` with multiple batch dimensions.
    """
    if data.ndim == 2:
        return np.cov(data, *args, **kwargs)
    else:
        return np.stack([recursive_cov(sub_data, *args, **kwargs) for sub_data in data])


_DEFAULT_BATCH_COV_KWARGS = {"rowvar": True, "bias": False, "ddof": None}


@pytest.mark.parametrize(
    "data, rowvar, bias, ddof",
    [
        [np.random.normal(size=(5, 30, 10, 20)), *_DEFAULT_BATCH_COV_KWARGS.values()],
        [np.random.normal(size=(5, 30, 10, 20)), False, False, None],
        [np.random.normal(size=(2, 1, 5, 8)), True, True, 0],
        [np.random.normal(size=(1, 4, 3, 3)), False, True, 1],
    ],
)
@equality_libs_processing()
def test_batch_cov(data, rowvar, bias, ddof):
    """
    Test the `batch_cov` function to `np.cov` with multiple batch dimensions.

    Args:
        data (np.array): [*batch_dims, num_obs, num_features]
    """
    out = tensor_ops.batch_cov(data, rowvar=rowvar, bias=bias, ddof=ddof)

    # Assert that is is equal to the numpy implementation
    np.testing.assert_allclose(
        out,
        recursive_cov(data, rowvar=rowvar, bias=bias, ddof=ddof),
    )

    return out  # Return the output for the equality_libs_processing decorator


def test_add_salt_and_pepper_noise():
    """Tests if add_salt_and_pepper_noise runs."""
    image = ops.zeros((28, 28), "float32")
    tensor_ops.add_salt_and_pepper_noise(image, 0.1, 0.1)


def test_extend_n_dims():
    """Tests if extend_n_dims runs."""
    tensor = ops.zeros((28, 28), "float32")
    out = tensor_ops.extend_n_dims(tensor, axis=1, n_dims=2)
    assert ops.ndim(out) == 4
    assert ops.shape(out) == (28, 1, 1, 28)


@pytest.mark.parametrize(
    "array, n",
    [
        [np.random.normal(size=(3, 5, 5)), 3],
        [np.random.normal(size=(3, 5, 5)), 5],
    ],
)
@equality_libs_processing()
def test_matrix_power(array, n):
    """Test matrix_power to np.linalg.matrix_power."""

    out = tensor_ops.matrix_power(array, n)

    # Test if the output is equal to the np.linalg.matrix_power implementation
    np.testing.assert_almost_equal(np.linalg.matrix_power(array, n), out)

    return out  # Return the output for the equality_libs_processing decorator


@pytest.mark.parametrize(
    "array, mask",
    [
        [np.zeros((28, 28)), np.random.uniform(size=(28, 28)) > 0.5],
        [np.random.normal(size=(2, 28, 28)), np.random.uniform(size=(2, 28, 28)) > 0.5],
    ],
)
@equality_libs_processing()
def test_boolean_mask(array, mask):
    """Tests if boolean_mask runs."""
    out = tensor_ops.boolean_mask(array, mask)
    assert ops.prod(ops.shape(out)) == ops.sum(mask), "Output shape is incorrect."
    return out  # Return the output for the equality_libs_processing decorator


@pytest.mark.parametrize(
    "func, tensor, n_batch_dims, func_axis",
    [
        [
            ops.image.rgb_to_grayscale,
            np.zeros((2, 3, 4, 28, 28, 3), np.float32),  # 3 batch dims
            3,
            None,
        ],
    ],
)
@equality_libs_processing()
def test_func_with_one_batch_dim(func, tensor, n_batch_dims, func_axis):
    """Tests if func_with_one_batch_dim runs."""

    out = tensor_ops.func_with_one_batch_dim(func, tensor, n_batch_dims, func_axis)
    assert ops.shape(out) == (*tensor.shape[:-1], 1), "Output shape is incorrect."
    return out  # Return the output for the equality_libs_processing decorator
