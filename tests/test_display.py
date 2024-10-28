"""Tests for the display module."""

import numpy as np
import pytest

from usbmd import display
from usbmd.setup_usbmd import set_backend

from .helpers import equality_libs_processing


@pytest.mark.parametrize(
    "size, resolution",
    [
        ((128, 32), None),
        ((512, 512), 0.1),
        ((40, 20, 20), None),
        ((40, 20, 20), 0.5),
    ],
)
@equality_libs_processing(decimal=3, backends=["torch", "jax"])
def test_scan_conversion(size, resolution):
    """
    Tests the scan_conversion function with random data.

    TODO: This test fails for tensorflow on cpu because of `keras.ops.image.map_coordinates`.
    Therefore tensorflow is not included in the backends. Maybe in the future we can check
    if the error is fixed with a new keras or tensorflow version.
    """
    data = np.random.random(size)
    from keras import ops  # pylint: disable=reimported,import-outside-toplevel

    from usbmd import display  # pylint: disable=reimported,import-outside-toplevel

    rho_range = (0, 100)
    theta_range = (-45, 45)
    theta_range = np.deg2rad(theta_range)

    if len(size) == 3:
        phi_range = (-20, 20)
        phi_range = np.deg2rad(phi_range)
        out = display.scan_convert_3d(
            data,
            rho_range,
            theta_range,
            phi_range,
            resolution=resolution,
        )
    else:
        out = display.scan_convert_2d(
            data,
            rho_range,
            theta_range,
            resolution=resolution,
        )

    out = ops.convert_to_numpy(out)

    # make sure outputs are not all nans or zeros
    assert not np.all(np.isnan(out)), "scan conversion is all nans"
    assert not np.all(out == 0), "scan conversion is all zeros"
    out = np.nan_to_num(out, nan=0)
    return out


@pytest.mark.parametrize(
    "size, random_data_type",
    [
        ((200, 200), "gaussian"),
        ((100, 333), "gaussian"),
        ((200, 200), "radial"),
        ((100, 333), "radial"),
    ],
)
def test_scan_conversion_and_inverse(size, random_data_type):
    """Tests the scan_conversion function with random data and
    invert the data with transform_sc_image_to_polar.
    For gaussian data, the mean squared error is around 0.09.
    For radial data, the mean squared error is around 0.0002.
    """
    set_backend("numpy")
    from keras import ops  # pylint: disable=reimported,import-outside-toplevel

    from usbmd import display  # pylint: disable=reimported,import-outside-toplevel

    if random_data_type == "gaussian":
        polar_data = np.random.random(size)
        # random data allow large error since interpolation is hard
        allowed_error = 0.2
    elif random_data_type == "radial":
        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        r = np.sqrt(x**2 + y**2)
        polar_data = np.exp(-(r**2))
        allowed_error = 0.001
    else:
        raise NotImplementedError

    rho_range = (0, 100)
    theta_range = (-45, 45)
    theta_range = np.deg2rad(theta_range)

    cartesian_data = display.scan_convert_2d(
        polar_data,
        rho_range,
        theta_range,
    )
    cartesian_data = ops.where(ops.isnan(cartesian_data), 0, cartesian_data)
    cartesian_data = ops.convert_to_numpy(cartesian_data)
    cartesian_data_inv = display.transform_sc_image_to_polar(
        cartesian_data, output_size=polar_data.shape
    )
    mean_squared_error = ((polar_data - cartesian_data_inv) ** 2).mean()

    assert (
        mean_squared_error < allowed_error
    ), f"MSE is too high: {mean_squared_error:.4f} > {allowed_error:.4f}"


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

    data = (
        np.random.random(size) * (_dynamic_range[1] - _dynamic_range[0])
        + _dynamic_range[0]
    )
    _data = display.to_8bit(data, dynamic_range, pillow=False)
    assert np.all(np.logical_and(_data >= 0, _data <= 255))
    assert _data.dtype == "uint8"
