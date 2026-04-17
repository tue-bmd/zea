"""
Tests for the `tensor_ops` module.
"""

import numpy as np
import pytest
import torch
from keras import ops
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter

import zea

from . import DEFAULT_TEST_SEED, backend_equality_check


@pytest.mark.parametrize(
    "array, start_dim, end_dim",
    [
        [default_rng(DEFAULT_TEST_SEED).normal(size=(5, 10)), 0, 1],
        [default_rng(DEFAULT_TEST_SEED + 1).normal(size=(5, 10, 15, 20)), 1, -1],
        [default_rng(DEFAULT_TEST_SEED + 2).normal(size=(5, 10, 15, 20)), 2, 3],
        [default_rng(DEFAULT_TEST_SEED + 3).normal(size=(5, 10, 15, 20, 25)), 0, 2],
    ],
)
@backend_equality_check()
def test_flatten(array, start_dim, end_dim):
    """Test the `flatten` function to `torch.flatten`."""
    import zea

    out = zea.func.flatten(array, start_dim, end_dim)
    torch_out = torch.flatten(
        torch.from_numpy(array),
        start_dim=start_dim,
        end_dim=end_dim,
    ).numpy()

    # Test if the output is equal to the torch.flatten implementation
    np.testing.assert_almost_equal(torch_out, out, decimal=6)

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
        [
            default_rng(DEFAULT_TEST_SEED).normal(size=(5, 30, 10, 20)),
            *_DEFAULT_BATCH_COV_KWARGS.values(),
        ],
        [default_rng(DEFAULT_TEST_SEED + 1).normal(size=(5, 30, 10, 20)), False, False, None],
        [default_rng(DEFAULT_TEST_SEED + 2).normal(size=(2, 1, 5, 8)), True, True, 0],
        [default_rng(DEFAULT_TEST_SEED + 3).normal(size=(1, 4, 3, 3)), False, True, 1],
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

    import zea

    data = ops.convert_to_tensor(data)

    out = zea.func.batch_cov(data, rowvar=rowvar, bias=bias, ddof=ddof)

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
    zea.func.add_salt_and_pepper_noise(image, 0.1, 0.1)


def test_extend_n_dims():
    """Tests if extend_n_dims runs."""
    tensor = ops.zeros((28, 28), "float32")
    out = zea.func.extend_n_dims(tensor, axis=1, n_dims=2)
    assert ops.ndim(out) == 4
    assert ops.shape(out) == (28, 1, 1, 28)


@pytest.mark.parametrize(
    "array, n",
    [
        [default_rng(DEFAULT_TEST_SEED).normal(size=(3, 5, 5)), 3],
        [default_rng(DEFAULT_TEST_SEED + 1).normal(size=(3, 5, 5)), 5],
    ],
)
@backend_equality_check()
def test_matrix_power(array, n):
    """Test matrix_power to np.linalg.matrix_power."""

    out = zea.func.matrix_power(array, n)

    # Test if the output is equal to the np.linalg.matrix_power implementation
    np.testing.assert_almost_equal(
        np.linalg.matrix_power(array, n),
        out,
        decimal=3,
        err_msg="`tensor_ops.matrix_power` is not equal to `np.linalg.matrix_power`.",
    )

    return out


@pytest.mark.parametrize(
    "array, mask",
    [
        [np.zeros((28, 28)), default_rng(DEFAULT_TEST_SEED).uniform(size=(28, 28)) > 0.5],
        [
            default_rng(DEFAULT_TEST_SEED + 1).normal(size=(2, 28, 28)),
            default_rng(DEFAULT_TEST_SEED + 2).uniform(size=(2, 28, 28)) > 0.5,
        ],
    ],
)
@backend_equality_check()
def test_boolean_mask(array, mask):
    """Tests if boolean_mask runs."""
    from keras import ops

    import zea

    out = zea.func.boolean_mask(array, mask)

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

    import zea

    if func == "rgb_to_grayscale":
        func = ops.image.rgb_to_grayscale

    out = zea.func.func_with_one_batch_dim(func, tensor, n_batch_dims, func_axis=func_axis)
    out2 = zea.func.func_with_one_batch_dim(
        func, tensor, n_batch_dims, batch_size=2, func_axis=func_axis
    )
    assert ops.shape(out) == (*tensor.shape[:-1], 1), "Output shape is incorrect."
    assert np.allclose(ops.convert_to_numpy(out), ops.convert_to_numpy(out2)), (
        "Outputs with and without batch_size do not match."
    )
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
    import zea

    # Create random test data (gradient)
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    # First stack the data
    stacked = zea.func.stack_volume_data_along_axis(data, batch_axis, stack_axis, n_frames)

    # Calculate padding that was added (if any)
    original_size = data.shape[batch_axis]
    blocks = int(np.ceil(original_size / n_frames))
    padded_size = blocks * n_frames
    padding = padded_size - original_size

    # Then split it back
    restored = zea.func.split_volume_data_from_axis(
        stacked, batch_axis, stack_axis, n_frames, padding
    )

    # Verify shapes match
    assert restored.shape == data.shape, "Shapes don't match after stack/split operations"

    # Verify contents match
    np.testing.assert_allclose(restored, data, rtol=1e-5, atol=1e-5)

    return restored


@pytest.mark.parametrize(
    "array, divisor, axis",
    [
        [default_rng(DEFAULT_TEST_SEED).normal(size=(10, 15)), 8, -1],
        [default_rng(DEFAULT_TEST_SEED + 1).normal(size=(7, 9, 11)), 4, 1],
        [default_rng(DEFAULT_TEST_SEED + 2).normal(size=(5, 6, 7, 8)), 2, 0],
    ],
)
@backend_equality_check()
def test_pad_array_to_divisible(array, divisor, axis):
    """Test the pad_array_to_divisible function."""
    from keras import ops

    import zea

    array = ops.convert_to_tensor(array)

    padded = zea.func.pad_array_to_divisible(array, divisor, axis=axis)

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
        [default_rng(DEFAULT_TEST_SEED).normal(size=(1, 28, 28, 3)), (7, 7), (0, 0)],
        [default_rng(DEFAULT_TEST_SEED + 1).normal(size=(2, 32, 32, 3)), (8, 8), (4, 4)],
        [default_rng(DEFAULT_TEST_SEED + 2).normal(size=(1, 28, 28, 1)), (4, 4), (2, 2)],
        [default_rng(DEFAULT_TEST_SEED + 3).normal(size=(1, 28, 28, 3)), (6, 6), (2, 2)],
    ],
)
@backend_equality_check()
def test_images_to_patches(image, patch_size, overlap):
    """Test the images_to_patches function."""
    import zea

    patches = zea.func.images_to_patches(image, patch_size, overlap)
    assert patches.shape[0] == image.shape[0]
    assert patches.shape[3] == patch_size[0]
    assert patches.shape[4] == patch_size[1]
    assert patches.shape[5] == image.shape[-1]
    return patches


@pytest.mark.parametrize(
    "patches, image_shape, overlap, window_type",
    [
        [
            default_rng(DEFAULT_TEST_SEED).normal(size=(1, 4, 4, 7, 7, 3)),
            (28, 28, 3),
            (0, 0),
            "average",
        ],
        [
            default_rng(DEFAULT_TEST_SEED + 1).normal(size=(2, 3, 3, 8, 8, 3)),
            (32, 32, 3),
            (4, 4),
            "replace",
        ],
        [
            default_rng(DEFAULT_TEST_SEED + 2).normal(size=(1, 7, 7, 4, 4, 1)),
            (28, 28, 1),
            (2, 2),
            "average",
        ],
    ],
)
@backend_equality_check()
def test_patches_to_images(patches, image_shape, overlap, window_type):
    """Test the patches_to_images function."""
    import zea

    image = zea.func.patches_to_images(patches, image_shape, overlap, window_type)
    assert image.shape[1:] == image_shape
    return image


@pytest.mark.parametrize(
    "image, patch_size, overlap, window_type",
    [
        [default_rng(DEFAULT_TEST_SEED).normal(size=(1, 28, 28, 3)), (7, 7), (0, 0), "average"],
        [default_rng(DEFAULT_TEST_SEED + 1).normal(size=(2, 32, 32, 3)), (8, 8), (4, 4), "replace"],
        [default_rng(DEFAULT_TEST_SEED + 2).normal(size=(1, 28, 28, 1)), (4, 4), (2, 2), "average"],
    ],
)
@backend_equality_check()
def test_images_to_patches_and_back(image, patch_size, overlap, window_type):
    """Test images_to_patches and patches_to_images together."""
    import zea

    patches = zea.func.images_to_patches(image, patch_size, overlap)
    reconstructed_image = zea.func.patches_to_images(
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
        [default_rng(DEFAULT_TEST_SEED + 1).normal(size=(32, 32)), 0.5, 0, 4.0],
        [default_rng(DEFAULT_TEST_SEED + 2).normal(size=(32, 32)), 1.0, 0, 5.0],
        [default_rng(DEFAULT_TEST_SEED + 3).normal(size=(32, 32)), 1.5, (0, 1), 4.0],
        [default_rng(DEFAULT_TEST_SEED + 4).normal(size=(32, 32)), (1.0, 2.0), (1, 0), 4.0],
    ],
)
@backend_equality_check(backends=["jax", "tensorflow"])
def test_gaussian_filter(array, sigma, order, truncate):
    """
    Test `tensor_ops.gaussian_filter against scipy.ndimage.gaussian_filter.`
    `GaussianBlur` with default args should be equivalent to scipy.
    """
    from keras import ops

    import zea

    array = array.astype(np.float32)

    blurred_scipy = gaussian_filter(array, sigma=sigma, order=order, truncate=truncate)

    tensor = ops.convert_to_tensor(array)
    blurred_zea = zea.func.gaussian_filter(tensor, sigma=sigma, order=order, truncate=truncate)
    blurred_zea = ops.convert_to_numpy(blurred_zea)

    np.testing.assert_allclose(blurred_scipy, blurred_zea, rtol=1e-5, atol=1e-5)
    return blurred_zea


def test_linear_sum_assignment_greedy():
    """Test the custom greedy linear_sum_assignment function."""
    import zea

    # Simple cost matrix: diagonal is optimal
    cost = np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1]], dtype=np.float32)
    row_ind, col_ind = zea.func.linear_sum_assignment(cost)
    # Should assign 0->0, 1->1, 2->2
    assert np.all(row_ind == np.array([0, 1, 2]))
    assert np.all(col_ind == np.array([0, 1, 2]))


@pytest.mark.parametrize(
    "array, axis, fn",
    [
        [default_rng(DEFAULT_TEST_SEED + 1).normal(size=(2, 3)), 0, "sum"],
        [default_rng(DEFAULT_TEST_SEED + 2).normal(size=(2, 3, 4)), 1, "argmax"],
        [default_rng(DEFAULT_TEST_SEED + 3).normal(size=(2, 3, 4, 5)), 2, "var"],
        [default_rng(DEFAULT_TEST_SEED + 4).normal(size=(9, 268, 8, 1)), 1, "correlate"],
    ],
)
@backend_equality_check()
def test_apply_along_axis(array, axis, fn):
    """Test the apply_along_axis function."""
    from keras import ops

    import zea

    if fn == "sum":
        fn = ops.sum
        np_fn = np.sum
    elif fn == "var":
        fn = ops.var
        np_fn = np.var
    elif fn == "argmax":
        fn = ops.argmax
        np_fn = np.argmax
    elif fn == "correlate":
        fn = lambda x: zea.func.correlate(x, ops.ones(10), mode="valid")
        np_fn = lambda x: np.correlate(x, np.ones(10), mode="valid")
    else:
        raise ValueError(f"Function {fn} not recognized.")

    # Simple test: sum along axis
    array = array.astype(np.float32)
    result = zea.func.apply_along_axis(fn, axis, array)
    expected = np.apply_along_axis(np_fn, axis, array)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    return result


@pytest.mark.parametrize("mode", ["valid", "same", "full"])
@backend_equality_check()
def test_correlate(mode):
    """Test the correlate function with random complex vectors against np.correlate."""
    import zea

    # Set random seed for reproducibility
    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    # Test with real vectors
    a_real = rng.standard_normal(10).astype(np.float32)
    v_real = rng.standard_normal(7).astype(np.float32)

    result_real = zea.func.correlate(a_real, v_real, mode=mode)
    expected_real = np.correlate(a_real, v_real, mode=mode)

    np.testing.assert_allclose(result_real, expected_real, rtol=1e-5, atol=1e-5)

    # Test with complex vectors
    a_complex = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).astype(np.complex64)
    v_complex = (rng.standard_normal(5) + 1j * rng.standard_normal(5)).astype(np.complex64)

    result_complex = zea.func.correlate(a_complex, v_complex, mode=mode)
    expected_complex = np.correlate(a_complex, v_complex, mode=mode)

    np.testing.assert_allclose(result_complex, expected_complex, rtol=1e-5, atol=1e-5)

    # Test edge case: different lengths
    a_short = rng.standard_normal(3).astype(np.float32)
    v_long = rng.standard_normal(12).astype(np.float32)

    result_edge = zea.func.correlate(a_short, v_long, mode=mode)
    expected_edge = np.correlate(a_short, v_long, mode=mode)

    np.testing.assert_allclose(result_edge, expected_edge, rtol=1e-5, atol=1e-5)

    # Return one of the results for backend_equality_check
    return result_complex


@pytest.mark.parametrize(
    "func, in_axes, out_axes, batch_size, chunks, fn_supports_batch",
    [
        ["multiply", (0, 0), 0, None, None, False],  # vmap
        ["multiply", (0, None), 0, None, None, False],  # vmap
        ["mean", (0, 1), 1, None, None, False],  # vmap
        ["multiply", 0, 0, None, None, True],  # doesn't have to vmap
        ["multiply", 0, 0, None, 1, True],  # doesn't have to vmap (chunks==1)
        ["multiply", (0, 0), 0, 2, None, False],  # batched map
        ["multiply", (0, None), 0, 3, None, False],  # batched map
        ["dummy", (0, None), 0, 3, None, True],  # batched map
        ["dummy", (0, 1), 0, None, 5, True],  # chunked map
        ["multiple_out", (1, None), (1, None), None, 4, False],  # chunked map with multiple outputs
        [
            "multiple_out",
            (1, None),
            (1, None),
            None,
            None,
            False,
        ],  # vmap with multiple outputs
    ],
)
@backend_equality_check(backends=["tensorflow", "torch"])
def test_vmap(func, in_axes, out_axes, batch_size, chunks, fn_supports_batch):
    """Test the `zea` `vmap` function against `jax.vmap`."""
    import jax
    from keras import ops

    import zea

    shape = (10, 10, 3, 2)
    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    if isinstance(in_axes, int):
        _in_axes = (in_axes, in_axes)
    else:
        _in_axes = in_axes

    if chunks is not None:
        total_length = shape[_in_axes[0]]
        _batch_size = np.ceil(total_length / chunks).astype(int)
    else:
        _batch_size = batch_size

    def _assert_batch_size(array, axis):
        # When fn_supports_batch is True, the function can handle batches internally,
        # so we can check if the batch size is correct.
        if _batch_size is not None and axis is not None and fn_supports_batch:
            assert array.shape[axis] == _batch_size
        # When fn_supports_batch is False, the function cannot handle batches internally,
        # so it will vmap over the batch dimension and the batch dimension gone.
        elif _batch_size is not None and not fn_supports_batch:
            expected_shape = list(shape)
            if axis is not None:
                expected_shape.pop(axis)
            assert array.shape == tuple(expected_shape)

    if func == "multiply":

        def func(a, b):
            _assert_batch_size(a, _in_axes[0])
            _assert_batch_size(b, _in_axes[1])
            return a * b

        def jax_func(a, b):
            return a * b
    elif func == "mean":

        def func(a, b):
            _assert_batch_size(a, _in_axes[0])
            _assert_batch_size(b, _in_axes[1])
            return ops.mean(a * b, axis=(-1, -2))

        def jax_func(a, b):
            return jax.numpy.mean(a * b, axis=(-1, -2))
    elif func == "dummy":

        def func(a, b):
            _assert_batch_size(a, _in_axes[0])
            _assert_batch_size(b, _in_axes[1])
            return a

        def jax_func(a, b):
            return a
    elif func == "multiple_out":

        def func(a, b):
            _assert_batch_size(a, _in_axes[0])
            _assert_batch_size(b, _in_axes[1])
            return a, b

        def jax_func(a, b):
            return a, b

    # Create batched data
    x = rng.standard_normal(size=shape).astype(np.float32)
    y = rng.standard_normal(size=shape).astype(np.float32)
    x_tensor = ops.convert_to_tensor(x)
    y_tensor = ops.convert_to_tensor(y)

    # Apply vmap
    expected = jax.vmap(jax_func, in_axes, out_axes)(x, y)
    result = zea.func.vmap(
        func,
        in_axes,
        out_axes,
        batch_size=batch_size,
        chunks=chunks,
        fn_supports_batch=fn_supports_batch,
    )(x_tensor, y_tensor)
    no_jit_result = zea.func.vmap(
        func,
        in_axes,
        out_axes,
        batch_size=batch_size,
        chunks=chunks,
        fn_supports_batch=fn_supports_batch,
        disable_jit=True,
    )(x_tensor, y_tensor)
    np.testing.assert_allclose(no_jit_result, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    if func == "multiple_out":
        return result[0]  # only return one output for backend_equality_check

    return result


@backend_equality_check()
def test_vmap_none_arg():
    """Test the `zea` `vmap` function with `None` argument."""
    from keras import ops

    import zea

    shape = (10, 10, 3, 2)

    def func(a, b):
        assert b is None
        return a + 1

    # Create batched data
    rng = default_rng(DEFAULT_TEST_SEED)
    x = rng.standard_normal(shape).astype(np.float32)
    x_tensor = ops.convert_to_tensor(x)

    # Apply map
    result = zea.func.vmap(func)(x_tensor, None)
    expected = x + 1
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    return result


@backend_equality_check()
def test_simple_map_one_input():
    """Test the `zea` `simple_map` function against `keras.ops.map`."""
    from keras import ops

    import zea

    # One input
    def func_one_input(x):
        return x * 2

    rng = default_rng(DEFAULT_TEST_SEED)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    x_tensor = ops.convert_to_tensor(x)
    expected_one_input = ops.map(func_one_input, x_tensor)
    result_one_input = zea.func.simple_map(func_one_input, x_tensor)
    np.testing.assert_allclose(result_one_input, expected_one_input, rtol=1e-5, atol=1e-5)

    return result_one_input


@backend_equality_check()
def test_simple_map_multiple_inputs():
    """Test the `zea` `simple_map` function against `keras.ops.map`."""
    from keras import ops

    import zea

    # Multiple inputs
    def func_multiple_inputs(inputs):
        x, y = inputs
        return x + y

    rng = default_rng(DEFAULT_TEST_SEED)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    y = rng.standard_normal((10, 5)).astype(np.float32)
    x_tensor = ops.convert_to_tensor(x)
    y_tensor = ops.convert_to_tensor(y)
    expected_multiple_inputs = ops.map(func_multiple_inputs, [x_tensor, y_tensor])
    result_multiple_inputs = zea.func.simple_map(func_multiple_inputs, [x_tensor, y_tensor])
    np.testing.assert_allclose(
        result_multiple_inputs, expected_multiple_inputs, rtol=1e-5, atol=1e-5
    )

    return result_multiple_inputs


@pytest.mark.parametrize(
    "mask_shape, blob_center, blob_radius",
    [
        [(50, 50), (25, 25), 10],
        [(100, 100), (30, 70), 15],
        [(64, 64), (32, 32), 8],
    ],
)
@backend_equality_check()
def test_find_contour(mask_shape, blob_center, blob_radius):
    """Test the find_contour function."""
    import zea

    # Create a binary mask with a circular blob
    mask = ops.zeros(mask_shape, dtype="float32")
    y, x = np.ogrid[: mask_shape[0], : mask_shape[1]]
    circle_mask = (y - blob_center[0]) ** 2 + (x - blob_center[1]) ** 2 <= blob_radius**2
    mask = ops.convert_to_tensor(circle_mask)

    contour = zea.func.find_contour(mask)

    # Check output shape and type
    assert ops.ndim(contour) == 2
    assert ops.shape(contour)[1] == 2
    assert ops.dtype(contour) == "float32"

    # Should find some contour points
    assert ops.shape(contour)[0] > 0

    # All contour points should be on the boundary
    contour_np = ops.convert_to_numpy(contour)
    assert np.all(contour_np[:, 0] >= 0) and np.all(contour_np[:, 0] < mask_shape[0])
    assert np.all(contour_np[:, 1] >= 0) and np.all(contour_np[:, 1] < mask_shape[1])

    return contour


def test_find_contours_empty_mask():
    """Test find_contours with empty mask."""
    import zea

    mask = ops.zeros((50, 50), dtype="float32")
    contour = zea.func.find_contour(mask)

    # Should return empty contours
    assert ops.shape(contour) == (0, 2)


@pytest.mark.parametrize(
    "range_from, range_to",
    [((0, 100), (2, 5)), ((-60, 0), (0, 255))],
)
@backend_equality_check()
def test_translate(range_from, range_to):
    """Tests the translate function by providing a test array with its range_from and
    a range to."""
    import zea

    rng = default_rng(DEFAULT_TEST_SEED)
    arr = rng.integers(low=range_from[0] + 1, high=range_from[1] - 2, size=10)
    right_min, right_max = range_to
    result = zea.func.translate(arr, range_from, range_to)
    assert right_min <= np.min(result), "Minimum value is too small"
    assert np.max(result) <= right_max, "Maximum value is too large"

    return result


@pytest.mark.parametrize(
    "num_taps, f1, f2, sampling_frequency",
    [
        (127, 2e6, 4e6, 20e6),  # Standard case
        (41, 1e6, 5e6, 25e6),  # Different parameters
        (21, 0.5e6, 4.5e6, 10e6),  # Close to but below Nyquist
        (128, 2e6, 4e6, 20e6),  # Even num_taps (not at Nyquist)
        (11, 1e6, 1.5e6, 10e6),  # Small num_taps
        (127, 0.1e6, 0.5e6, 20e6),  # Very low frequencies
        (2, 2e6, 4e6, 20e6),  # Minimal num_taps
    ],
)
def test_get_band_pass_filter(num_taps, f1, f2, sampling_frequency):
    """Tests if get_band_pass_filter is equivalent to scipy.signal.firwin."""
    import scipy.signal

    import zea

    b1 = zea.func.get_band_pass_filter(num_taps, sampling_frequency, f1, f2)
    b2 = scipy.signal.firwin(num_taps, [f1, f2], pass_zero=False, fs=sampling_frequency)

    np.testing.assert_allclose(b1, b2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "num_taps, f1, f2, sampling_frequency",
    [
        (127, 0, 4e6, 20e6),  # f1 == 0
        (127, 2e6, 10e6, 20e6),  # f2 == Nyquist (fs/2)
        (21, 0.5e6, 5e6, 10e6),  # f2 == Nyquist (fs/2)
        (127, 5e6, 2e6, 20e6),  # f1 > f2 (reversed)
        (127, 2e6, 2e6, 20e6),  # f1 == f2 (zero bandwidth)
        (127, -1e6, 4e6, 20e6),  # Negative f1
        (127, 2e6, 25e6, 20e6),  # f2 > Nyquist
    ],
)
def test_get_band_pass_filter_invalid_inputs(num_taps, f1, f2, sampling_frequency):
    """Tests that get_band_pass_filter raises appropriate errors for invalid inputs."""
    import scipy.signal

    import zea

    # Verify scipy also raises an error for these cases
    with pytest.raises(ValueError):
        scipy.signal.firwin(num_taps, [f1, f2], pass_zero=False, fs=sampling_frequency)

    with pytest.raises(ValueError):
        zea.func.get_band_pass_filter(num_taps, sampling_frequency, f1, f2)
