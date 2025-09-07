"""Tests for the display module."""

import numpy as np
import pytest

from zea import display

from . import backend_equality_check


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

    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float32)

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
    "size, pattern_creator, allowed_error",
    [
        ((200, 200), "create_radial_pattern", 0.001),
        ((100, 333), "create_radial_pattern", 0.001),
        ((200, 200), "create_concentric_rings", 0.1),
        ((100, 333), "create_concentric_rings", 0.1),
    ],
)
@backend_equality_check(decimal=2)
def test_scan_conversion_and_inverse(size, pattern_creator, allowed_error):
    """Tests the scan_conversion function with structured test patterns and
    inverts the data with inverse_scan_convert_2d.

    Note:
        The allowed_error is set to 0.1 for concentric rings because the MSE is
        expected to be higher due to the nature of the pattern.
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
    cartesian_data_inv = display.inverse_scan_convert_2d(
        cartesian_data, output_size=polar_data.shape, find_scan_cone=False
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
        cartesian_data_padded, output_size=polar_data.shape, find_scan_cone=True
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
    if dynamic_range is None:
        _dynamic_range = (-60, 0)
    else:
        _dynamic_range = dynamic_range

    data = np.random.random(size) * (_dynamic_range[1] - _dynamic_range[0]) + _dynamic_range[0]
    _data = display.to_8bit(data, dynamic_range, pillow=False)
    assert np.all(np.logical_and(_data >= 0, _data <= 255))
    assert _data.dtype == "uint8"
