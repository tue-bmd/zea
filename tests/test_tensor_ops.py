"""
Tests for the `tensor_ops` module.
"""

import numpy as np
import pytest
import torch
from keras import ops
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter

from zea import tensor_ops

from . import backend_equality_check


@pytest.mark.parametrize(
    "array, start_dim, end_dim",
    [
        [np.random.normal(size=(5, 10)), 0, 1],
        [np.random.normal(size=(5, 10, 15, 20)), 1, -1],
        [np.random.normal(size=(5, 10, 15, 20)), 2, 3],
        [np.random.normal(size=(5, 10, 15, 20, 25)), 0, 2],
    ],
)
@backend_equality_check()
def test_flatten(array, start_dim, end_dim):
    """Test the `flatten` function to `torch.flatten`."""
    from zea import tensor_ops

    out = tensor_ops.flatten(array, start_dim, end_dim)
    torch_out = torch.flatten(torch.from_numpy(array), start_dim=start_dim, end_dim=end_dim).numpy()

    # Test if the output is equal to the torch.flatten implementation
    np.testing.assert_almost_equal(torch_out, out)

    return out


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
@backend_equality_check()
def test_batch_cov(data, rowvar, bias, ddof):
    """
    Test the `batch_cov` function to `np.cov` with multiple batch dimensions.

    Args:
        data (np.array): [*batch_dims, num_obs, num_features]
    """
    from keras import ops

    from zea import tensor_ops

    data = ops.convert_to_tensor(data)

    out = tensor_ops.batch_cov(data, rowvar=rowvar, bias=bias, ddof=ddof)

    # Assert that is is equal to the numpy implementation
    np.testing.assert_allclose(
        out,
        recursive_cov(data, rowvar=rowvar, bias=bias, ddof=ddof),
        rtol=1e-5,
        atol=1e-5,
    )

    return out


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
@backend_equality_check()
def test_matrix_power(array, n):
    """Test matrix_power to np.linalg.matrix_power."""
    from zea import tensor_ops

    out = tensor_ops.matrix_power(array, n)

    # Test if the output is equal to the np.linalg.matrix_power implementation
    np.testing.assert_almost_equal(
        np.linalg.matrix_power(array, n),
        out,
        decimal=4,
        err_msg="`tensor_ops.matrix_power` is not equal to `np.linalg.matrix_power`.",
    )

    return out


@pytest.mark.parametrize(
    "array, mask",
    [
        [np.zeros((28, 28)), np.random.uniform(size=(28, 28)) > 0.5],
        [np.random.normal(size=(2, 28, 28)), np.random.uniform(size=(2, 28, 28)) > 0.5],
    ],
)
@backend_equality_check()
def test_boolean_mask(array, mask):
    """Tests if boolean_mask runs."""
    from keras import ops

    from zea import tensor_ops

    out = tensor_ops.boolean_mask(array, mask)

    out = ops.convert_to_numpy(out)
    assert ops.prod(ops.shape(out)) == ops.sum(mask), "Output shape is incorrect."
    return out


@pytest.mark.parametrize(
    "func, tensor, n_batch_dims, func_axis",
    [
        [
            "rgb_to_grayscale",
            np.zeros((2, 3, 4, 28, 28, 3), np.float32),  # 3 batch dims
            3,
            None,
        ],
    ],
)
@backend_equality_check()
def test_func_with_one_batch_dim(func, tensor, n_batch_dims, func_axis):
    """Tests if func_with_one_batch_dim runs."""

    from keras import ops

    from zea import tensor_ops

    if func == "rgb_to_grayscale":
        func = ops.image.rgb_to_grayscale

    out = tensor_ops.func_with_one_batch_dim(func, tensor, n_batch_dims, func_axis)
    assert ops.shape(out) == (*tensor.shape[:-1], 1), "Output shape is incorrect."
    return out


@pytest.mark.parametrize(
    "shape, batch_axis, stack_axis, n_frames",
    [
        [(10, 20, 30), 0, 1, 2],  # Simple 3D case
        [(8, 16, 24, 32), 1, 2, 4],  # 4D case
        [(5, 10, 15, 20, 25), 2, 3, 5],  # 5D case
        [(10, 20, 30), 0, 2, 1],
    ],
)
@backend_equality_check(backends=["tensorflow", "jax"])
def test_stack_and_split_volume_data(shape, batch_axis, stack_axis, n_frames):
    """Test that stack_volume_data_along_axis and split_volume_data_from_axis
    are inverse operations.

    TODO: does not work for torch...
    """
    from zea import tensor_ops

    # Create random test data (gradient)
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    # First stack the data
    stacked = tensor_ops.stack_volume_data_along_axis(data, batch_axis, stack_axis, n_frames)

    # Calculate padding that was added (if any)
    original_size = data.shape[batch_axis]
    blocks = int(np.ceil(original_size / n_frames))
    padded_size = blocks * n_frames
    padding = padded_size - original_size

    # Then split it back
    restored = tensor_ops.split_volume_data_from_axis(
        stacked, batch_axis, stack_axis, n_frames, padding
    )

    # Verify shapes match
    assert restored.shape == data.shape, "Shapes don't match after stack/split operations"

    # Verify contents match
    np.testing.assert_allclose(restored, data, rtol=1e-5, atol=1e-5)

    return restored


@pytest.fixture
def _test_function():
    """Fixture for testing batched_map function."""

    def _test(tensor, extra_tensor=None):
        if extra_tensor is not None:
            return tensor + extra_tensor
        return tensor

    return _test


@pytest.mark.parametrize(
    "array, batch_size, batched_kwargs",
    [
        [np.random.normal(size=(2, 3, 4, 5)), 2, {}],
        [
            np.random.normal(size=(3, 4, 5, 6)),
            1,
            {"extra_tensor": np.random.normal(size=(3, 4, 5, 6))},
        ],
    ],
)
@backend_equality_check()
def test_batched_map(_test_function, array, batch_size, batched_kwargs):
    """Test the batched_map function using _test_function fixture."""
    from keras import ops

    from zea import tensor_ops

    array = ops.convert_to_tensor(array)
    # Convert any numpy arrays in batched_kwargs to tensors.
    batched_kwargs = {
        k: ops.convert_to_tensor(v) if isinstance(v, np.ndarray) else v
        for k, v in batched_kwargs.items()
    }

    out_jit = tensor_ops.batched_map(
        _test_function,
        array,
        batch_size,
        jit=True,
        **batched_kwargs,
    )
    out_no_jit = tensor_ops.batched_map(
        _test_function,
        array,
        batch_size,
        jit=False,
        **batched_kwargs,
    )

    # Check against python's map function
    # this does not do batching, but the output should be the same
    expected = np.stack(list(map(_test_function, array, *batched_kwargs.values())))

    np.testing.assert_allclose(out_jit, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_no_jit, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_jit, out_no_jit, rtol=1e-5, atol=1e-5)
    return out_jit


@pytest.mark.parametrize(
    "array, patches, batched_kwargs",
    [
        [np.random.normal(size=(2, 3, 4, 5)), 2, {}],
        [
            np.random.normal(size=(3, 4, 5, 6)),
            1,
            {"extra_tensor": np.random.normal(size=(3, 4, 5, 6))},
        ],
    ],
)
@backend_equality_check()
def test_patched_map(_test_function, array, patches, batched_kwargs):
    """Test the patched_map function using _test_function fixture."""
    from keras import ops

    from zea import tensor_ops

    array = ops.convert_to_tensor(array)
    # Convert any numpy arrays in batched_kwargs to tensors.
    batched_kwargs = {
        k: ops.convert_to_tensor(v) if isinstance(v, np.ndarray) else v
        for k, v in batched_kwargs.items()
    }

    out_jit = tensor_ops.patched_map(
        _test_function,
        array,
        patches,
        jit=True,
        **batched_kwargs,
    )
    out_no_jit = tensor_ops.patched_map(
        _test_function,
        array,
        patches,
        jit=False,
        **batched_kwargs,
    )

    # Check against python's map function
    # this does not do batching, but the output should be the same
    expected = np.stack(list(map(_test_function, array, *batched_kwargs.values())))

    np.testing.assert_allclose(out_jit, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_no_jit, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_jit, out_no_jit, rtol=1e-5, atol=1e-5)
    return out_jit


@pytest.mark.parametrize(
    "array, divisor, axis",
    [
        [np.random.normal(size=(10, 15)), 8, -1],
        [np.random.normal(size=(7, 9, 11)), 4, 1],
        [np.random.normal(size=(5, 6, 7, 8)), 2, 0],
    ],
)
@backend_equality_check()
def test_pad_array_to_divisible(array, divisor, axis):
    """Test the pad_array_to_divisible function."""
    from keras import ops

    from zea import tensor_ops

    array = ops.convert_to_tensor(array)

    padded = tensor_ops.pad_array_to_divisible(array, divisor, axis=axis)

    # Check that output shape is divisible by divisor only on specified axis
    assert padded.shape[axis] % divisor == 0, (
        "Output dimension not divisible by divisor on specified axis"
    )

    # Check that the original array is preserved in the first part
    np.testing.assert_array_equal(padded[tuple(slice(0, s) for s in array.shape)], array)

    # Check that padding size is minimal on specified axis
    axis_dim = padded.shape[axis]
    orig_dim = array.shape[axis]
    assert axis_dim >= orig_dim and axis_dim - orig_dim < divisor, "Padding is not minimal"

    if axis < 0:  # deal with negative axis
        axis = array.ndim + axis
    # Check other dimensions remain unchanged
    for i, (p_dim, o_dim) in enumerate(zip(padded.shape, array.shape)):
        if i != axis:
            assert p_dim == o_dim, "Dimensions not matching axis should remain unchanged"

    return padded


@pytest.mark.parametrize(
    "image, patch_size, overlap",
    [
        [np.random.normal(size=(1, 28, 28, 3)), (7, 7), (0, 0)],
        [np.random.normal(size=(2, 32, 32, 3)), (8, 8), (4, 4)],
        [np.random.normal(size=(1, 28, 28, 1)), (4, 4), (2, 2)],
        [np.random.normal(size=(1, 28, 28, 3)), (6, 6), (2, 2)],
    ],
)
@backend_equality_check()
def test_images_to_patches(image, patch_size, overlap):
    """Test the images_to_patches function."""
    from zea import tensor_ops

    patches = tensor_ops.images_to_patches(image, patch_size, overlap)
    assert patches.shape[0] == image.shape[0]
    assert patches.shape[3] == patch_size[0]
    assert patches.shape[4] == patch_size[1]
    assert patches.shape[5] == image.shape[-1]
    return patches


@pytest.mark.parametrize(
    "patches, image_shape, overlap, window_type",
    [
        [np.random.normal(size=(1, 4, 4, 7, 7, 3)), (28, 28, 3), (0, 0), "average"],
        [np.random.normal(size=(2, 3, 3, 8, 8, 3)), (32, 32, 3), (4, 4), "replace"],
        [np.random.normal(size=(1, 7, 7, 4, 4, 1)), (28, 28, 1), (2, 2), "average"],
    ],
)
@backend_equality_check()
def test_patches_to_images(patches, image_shape, overlap, window_type):
    """Test the patches_to_images function."""
    from zea import tensor_ops

    image = tensor_ops.patches_to_images(patches, image_shape, overlap, window_type)
    assert image.shape[1:] == image_shape
    return image


@pytest.mark.parametrize(
    "image, patch_size, overlap, window_type",
    [
        [np.random.normal(size=(1, 28, 28, 3)), (7, 7), (0, 0), "average"],
        [np.random.normal(size=(2, 32, 32, 3)), (8, 8), (4, 4), "replace"],
        [np.random.normal(size=(1, 28, 28, 1)), (4, 4), (2, 2), "average"],
    ],
)
@backend_equality_check()
def test_images_to_patches_and_back(image, patch_size, overlap, window_type):
    """Test images_to_patches and patches_to_images together."""
    from zea import tensor_ops

    patches = tensor_ops.images_to_patches(image, patch_size, overlap)
    reconstructed_image = tensor_ops.patches_to_images(
        patches,
        image.shape[1:],
        overlap,
        window_type,
    )
    np.testing.assert_allclose(image, reconstructed_image, rtol=1e-5, atol=1e-5)
    return reconstructed_image


@pytest.mark.parametrize(
    "array, sigma, order, truncate",
    [
        [default_rng(seed=1).normal(size=(32, 32)), 0.5, 0, 4.0],
        [default_rng(seed=2).normal(size=(32, 32)), 1.0, 0, 5.0],
        [default_rng(seed=3).normal(size=(32, 32)), 1.5, (0, 1), 4.0],
        [default_rng(seed=4).normal(size=(32, 32)), (1.0, 2.0), (1, 0), 4.0],
    ],
)
@backend_equality_check(backends=["jax", "tensorflow"])
def test_gaussian_filter(array, sigma, order, truncate):
    """
    Test `tensor_ops.gaussian_filter against scipy.ndimage.gaussian_filter.`
    `GaussianBlur` with default args should be equivalent to scipy.
    """
    from keras import ops

    from zea import tensor_ops

    array = array.astype(np.float32)

    blurred_scipy = gaussian_filter(array, sigma=sigma, order=order, truncate=truncate)

    tensor = ops.convert_to_tensor(array)
    blurred_zea = tensor_ops.gaussian_filter(tensor, sigma=sigma, order=order, truncate=truncate)
    blurred_zea = ops.convert_to_numpy(blurred_zea)

    np.testing.assert_allclose(blurred_scipy, blurred_zea, rtol=1e-5, atol=1e-5)
    return blurred_zea


def test_linear_sum_assignment_greedy():
    """Test the custom greedy linear_sum_assignment function."""
    from zea import tensor_ops

    # Simple cost matrix: diagonal is optimal
    cost = np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1]], dtype=np.float32)
    row_ind, col_ind = tensor_ops.linear_sum_assignment(cost)
    # Should assign 0->0, 1->1, 2->2
    assert np.all(row_ind == np.array([0, 1, 2]))
    assert np.all(col_ind == np.array([0, 1, 2]))
