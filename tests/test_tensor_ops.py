"""Tests for the `tensor_ops` module."""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pytest
import torch

from usbmd import tensor_ops


@pytest.mark.parametrize(
    "array",
    [
        np.random.normal(size=(10, 20)),
        # TODO: add more test cases
    ],
)
def test_flatten(array):
    np.testing.assert_almost_equal(
        torch.flatten(torch.from_numpy(array)).numpy(), tensor_ops.flatten(array)
    )


def recursive_cov(data, *args, **kwargs):
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
def test_batch_cov(data, rowvar, bias, ddof):
    """
    Args:
        data (np.array): [*batch_dims, num_obs, num_features]
    """
    np.testing.assert_allclose(
        tensor_ops.batch_cov(data, rowvar=rowvar, bias=bias, ddof=ddof),
        recursive_cov(data, rowvar=rowvar, bias=bias, ddof=ddof),
    )


# TODO add tests for:
# - `tensor_ops.add_salt_and_pepper_noise`
# - `tensor_ops.extend_n_dims`
# - `tensor_ops.func_with_one_batch_dim`
# - `tensor_ops.matrix_power`
# - `tensor_ops.boolean_mask`
