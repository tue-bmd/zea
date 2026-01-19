"""Tests for the display module."""

import numpy as np
import pytest

from zea import display

from . import DEFAULT_TEST_SEED, backend_equality_check


@pytest.mark.parametrize(
    "size, resolution, order",
    [
        ((128, 32), None, 1),
        ((512, 512), 0.1, 1),
        ((40, 20, 20), None, 1),
        ((40, 20, 20), 0.5, 1),
        ((112, 112), None, 3),  # will use scipy ndimage for order > 1
    ],
)
@backend_equality_check(decimal=[0, 2, 0], backends=["torch", "jax", "tensorflow"])
def test_scan_conversion(size, resolution, order):
    """Tests the scan_conversion function with random data."""
    import keras
    from keras import ops

    from zea import display

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.standard_normal(size).astype(np.float32)

    rho_range = (0, 100)
    theta_range = (-45, 45)
    theta_range = np.deg2rad(theta_range)

    if len(size) == 3:
        phi_range = (-20, 20)
        phi_range = np.deg2rad(phi_range)
        out, params = display.scan_convert_3d(
            data,
            rho_range,
            theta_range,
            phi_range,
            resolution=resolution,
            order=order,
        )
    else:
        out, params = display.scan_convert_2d(
            data,
            rho_range,
            theta_range,
            resolution=resolution,
            order=order,
        )

    assert isinstance(params, dict), "params is not a dict"

    # Check that dtype was not changed
    assert ops.dtype(out) == ops.dtype(data), "output dtype is not the same as input dtype"

    out = ops.convert_to_numpy(out)

    # make sure outputs are not all nans or zeros
    assert not np.all(np.isnan(out)), "scan conversion is all nans"
    assert not np.all(out == 0), (
        f"scan conversion is all zeros for backend {keras.backend.backend()}"
    )
    out = np.nan_to_num(out, nan=0)
    return out


def create_radial_pattern(size):
    """Creates a radial pattern for testing scan conversion."""
    x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
    r = np.sqrt(x**2 + y**2)
    image = np.exp(-(r**2))
    return image.astype("float32")


def create_concentric_rings(size):
    """Creates a ring pattern for testing scan conversion."""
    x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
    r = np.sqrt(x**2 + y**2)
    image = np.sin(10 * r) ** 2
    return image.astype("float32")


@pytest.mark.parametrize(
    "size, pattern_creator, allowed_error, angle",
    [
        ((200, 200), "create_radial_pattern", 0.001, None),
        ((100, 333), "create_radial_pattern", 0.001, None),
        ((200, 200), "create_concentric_rings", 0.1, None),
        ((100, 333), "create_concentric_rings", 0.1, None),
        ((200, 200), "create_radial_pattern", 0.001, np.deg2rad(30)),
    ],
)
@backend_equality_check(decimal=2)
def test_scan_conversion_and_inverse(size, pattern_creator, allowed_error, angle):
    """Tests the scan_conversion function with structured test patterns and
    inverts the data with inverse_scan_convert_2d.

    Note:
        The allowed_error is set to 0.1 for concentric rings because the MSE is
        expected to be higher due to the nature of the pattern.
    """
    from keras import ops

    from zea import display

    # data range is [0, 1] and type is float32
    if pattern_creator == "create_radial_pattern":
        polar_data = create_radial_pattern(size)
    elif pattern_creator == "create_concentric_rings":
        polar_data = create_concentric_rings(size)
    else:
        raise ValueError("Unknown pattern creator")

    rho_range = (0, 100)

    if angle is None:
        theta_range = np.deg2rad((-45, 45))
    else:
        theta_range = (-angle, angle)

    # Scan convert
    cartesian_data, _ = display.scan_convert_2d(polar_data, rho_range, theta_range)

    # Inverse scan convert
    cartesian_data_inv = display.inverse_scan_convert_2d(
        cartesian_data, output_size=polar_data.shape, find_scan_cone=False, angle=angle
    )
    cartesian_data_inv = ops.convert_to_numpy(cartesian_data_inv)

    mean_squared_error = ((polar_data - cartesian_data_inv) ** 2).mean()

    assert mean_squared_error < allowed_error, f"MSE is too high: {mean_squared_error:.4f}"

    return cartesian_data_inv


@pytest.mark.parametrize(
    "size, pattern_creator, allowed_error",
    [
        ((200, 200), "create_radial_pattern", 0.0015),
        ((100, 333), "create_radial_pattern", 0.0015),
        ((200, 200), "create_concentric_rings", 0.1),
        ((100, 333), "create_concentric_rings", 0.1),
    ],
)
@backend_equality_check(decimal=2)
def test_scan_conversion_and_inverse_padded(size, pattern_creator, allowed_error):
    """Tests the scan_conversion function with structured test patterns and
    inverts the data with inverse_scan_convert_2d. In this case, the scan cone is
    padded such that it is no longer centered and cropped. find_scan_cone=True is
    used to automatically crop and center the scan cone.
    """
    from keras import ops

    from zea import display

    if pattern_creator == "create_radial_pattern":
        polar_data = create_radial_pattern(size)
    elif pattern_creator == "create_concentric_rings":
        polar_data = create_concentric_rings(size)
    else:
        raise ValueError("Unknown pattern creator")

    rho_range = (0, 100)
    theta_range = np.deg2rad((-45, 45))

    cartesian_data, _ = display.scan_convert_2d(polar_data, rho_range, theta_range)
    cartesian_data = ops.convert_to_numpy(cartesian_data)

    # now pad the cartesian image and test with find_scan_cone=True
    left_padding = ops.zeros((ops.shape(cartesian_data)[0], 20))
    cartesian_data_padded = ops.concatenate([left_padding, cartesian_data], axis=1)
    top_padding = ops.zeros((20, ops.shape(cartesian_data_padded)[1]))
    cartesian_data_padded = ops.concatenate([top_padding, cartesian_data_padded], axis=0)
    cartesian_data_inv = display.inverse_scan_convert_2d(
        cartesian_data_padded, output_size=polar_data.shape, find_scan_cone=True, image_range=(0, 1)
    )
    cartesian_data_inv = ops.convert_to_numpy(cartesian_data_inv)
    mean_squared_error = ((polar_data - cartesian_data_inv) ** 2).mean()

    assert mean_squared_error < allowed_error, f"MSE is too high: {mean_squared_error:.4f}"

    return cartesian_data_inv


@pytest.mark.parametrize(
    "size, dynamic_range",
    [
        ((2, 1, 128, 32), (-30, -5)),
        ((512, 512), None),
        ((1, 128, 32), None),
    ],
)
def test_converting_to_image(size, dynamic_range):
    """Test converting to image functions"""
    # create random data between dynamic range
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    if dynamic_range is None:
        _dynamic_range = (-60, 0)
    else:
        _dynamic_range = dynamic_range

    data = rng.standard_normal(size) * (_dynamic_range[1] - _dynamic_range[0]) + _dynamic_range[0]
    _data = display.to_8bit(data, dynamic_range, pillow=False)
    assert np.all(np.logical_and(_data >= 0, _data <= 255))
    assert _data.dtype == "uint8"


@pytest.mark.parametrize(
    "dtype, order",
    [
        ("float16", 0),
        ("float16", 1),
        ("float16", 2),
        ("float32", 0),
        ("float32", 1),
        ("float32", 2),
    ],
)
def test_map_coordinates_dtype(dtype, order):
    """Test map_coordinates with different data types and interpolation orders.

    This test verifies that map_coordinates works correctly with float16 and float32
    inputs across different interpolation orders.
    """
    from keras import ops

    from zea import display

    # Create a simple 2D test image
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image = rng.random((32, 32)).astype(dtype)

    # Create simple coordinates for interpolation
    # Sample points at fractional positions to test interpolation
    coords = np.array(
        [
            [15.5, 16.2, 20.1, 10.3],  # y coordinates
            [15.5, 14.8, 18.7, 12.9],  # x coordinates
        ],
        dtype="float32",
    )

    # Convert to ops tensors
    image_tensor = ops.convert_to_tensor(image)
    coords_tensor = ops.convert_to_tensor(coords)

    # Perform map_coordinates
    result = display.map_coordinates(
        image_tensor, coords_tensor, order=order, fill_mode="constant", fill_value=0.0
    )

    # Convert result to numpy for assertions
    result_np = ops.convert_to_numpy(result)

    # Basic sanity checks
    assert result_np.shape == (4,), f"Expected shape (4,), got {result_np.shape}"
    assert not np.any(np.isnan(result_np)), "Result contains NaN values"
    assert not np.all(result_np == 0), "Result is all zeros (likely failed)"

    # The output dtype should always match the input dtype
    assert result_np.dtype == np.dtype(dtype), (
        f"Expected output dtype {dtype}, got {result_np.dtype}"
    )

    # Verify interpolated values are within reasonable range
    assert np.all(result_np >= 0) and np.all(result_np <= 1), (
        f"Interpolated values out of expected range [0, 1]: {result_np}"
    )
