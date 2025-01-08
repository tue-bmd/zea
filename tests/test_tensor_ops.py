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


@pytest.mark.parametrize(
    "shape, batch_axis, stack_axis, n_frames",
    [
        [(10, 20, 30), 0, 1, 2],  # Simple 3D case
        [(8, 16, 24, 32), 1, 2, 4],  # 4D case
        [(5, 10, 15, 20, 25), 2, 3, 5],  # 5D case
        [(10, 20, 30), 0, 2, 1],
    ],
)
@equality_libs_processing()
def test_stack_and_split_volume_data(shape, batch_axis, stack_axis, n_frames):
    """Test that stack_volume_data_along_axis and split_volume_data_from_axis
    are inverse operations.
    """
    # Create random test data (gradient)
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    # First stack the data
    stacked = tensor_ops.stack_volume_data_along_axis(
        data, batch_axis, stack_axis, n_frames
    )

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
    assert (
        restored.shape == data.shape
    ), "Shapes don't match after stack/split operations"

    # Verify contents match
    np.testing.assert_allclose(restored, data, rtol=1e-5, atol=1e-5)

    return restored  # Return for equality_libs_processing decorator


@pytest.mark.parametrize(
    "array, batch_dims, func",
    [
        [np.random.normal(size=(2, 3, 4, 5)), 2, ops.square],
        [np.random.normal(size=(3, 4, 5, 6)), 1, lambda x: x * 2],
        [np.random.normal(size=(2, 2, 3, 4)), 3, ops.abs],
    ],
)
@equality_libs_processing()
def test_batched_map(array, batch_dims, func):
    """Test the batched_map function against manual batch processing."""
    array = ops.convert_to_tensor(array)
    out = tensor_ops.batched_map(func, array, batch_dims)

    # Compute expected result manually
    expected = []
    batch_shape = array.shape[:batch_dims]
    for idx in np.ndindex(*batch_shape):
        slicing = tuple(
            slice(None) if i >= len(idx) else idx[i] for i in range(array.ndim)
        )
        expected.append(func(array[slicing]))

    expected = ops.stack(expected, axis=0)
    expected = ops.reshape(expected, batch_shape + array.shape[batch_dims:])

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    return out


@pytest.mark.parametrize(
    "array, divisor, axis",
    [
        [np.random.normal(size=(10, 15)), 8, -1],
        [np.random.normal(size=(7, 9, 11)), 4, 1],
        [np.random.normal(size=(5, 6, 7, 8)), 2, 0],
    ],
)
@equality_libs_processing()
def test_pad_array_to_divisible(array, divisor, axis):
    """Test the pad_array_to_divisible function."""
    padded = tensor_ops.pad_array_to_divisible(array, divisor, axis=axis)

    # Check that output shape is divisible by divisor only on specified axis
    assert (
        padded.shape[axis] % divisor == 0
    ), "Output dimension not divisible by divisor on specified axis"

    # Check that the original array is preserved in the first part
    np.testing.assert_array_equal(
        padded[tuple(slice(0, s) for s in array.shape)], array
    )

    # Check that padding size is minimal on specified axis
    axis_dim = padded.shape[axis]
    orig_dim = array.shape[axis]
    assert (
        axis_dim >= orig_dim and axis_dim - orig_dim < divisor
    ), "Padding is not minimal"

    if axis < 0:  # deal with negative axis
        axis = array.ndim + axis
    # Check other dimensions remain unchanged
    for i, (p_dim, o_dim) in enumerate(zip(padded.shape, array.shape)):
        if i != axis:
            assert (
                p_dim == o_dim
            ), "Dimensions not matching axis should remain unchanged"

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
@equality_libs_processing()
def test_images_to_patches(image, patch_size, overlap):
    """Test the images_to_patches function."""
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
@equality_libs_processing()
def test_patches_to_images(patches, image_shape, overlap, window_type):
    """Test the patches_to_images function."""
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
@equality_libs_processing()
def test_images_to_patches_and_back(image, patch_size, overlap, window_type):
    """Test images_to_patches and patches_to_images together."""
    patches = tensor_ops.images_to_patches(image, patch_size, overlap)
    reconstructed_image = tensor_ops.patches_to_images(
        patches,
        image.shape[1:],
        overlap,
        window_type,
    )
    np.testing.assert_allclose(image, reconstructed_image, rtol=1e-5, atol=1e-5)
    return reconstructed_image
