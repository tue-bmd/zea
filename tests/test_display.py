"""Tests for the display module."""

import numpy as np
import pytest

from usbmd.display import (
    project_to_cartesian_grid,
    scan_convert,
    to_8bit,
    transform_sc_image_to_polar,
)


@pytest.mark.parametrize(
    "size",
    [
        (2, 1, 128, 32),
        (512, 512),
        (1, 128, 32),
    ],
)
def test_scan_conversion(size):
    """Tests the scan_conversion function with random data"""
    data = np.random.random(size)
    x_axis = np.linspace(-50, 50, 100)  # angles
    z_axis = np.linspace(0, 100, 2000)
    scan_convert(data, x_axis, z_axis, n_pixels=500, spline_order=1, fill_value=0)


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

    x_axis = np.linspace(-45, 45, 100)  # angles
    z_axis = np.linspace(0, 100, 2000)
    data_sc = scan_convert(
        polar_data, x_axis, z_axis, n_pixels=500, spline_order=1, fill_value=0
    )
    data_sc_inv = transform_sc_image_to_polar(data_sc, output_size=polar_data.shape)
    mean_squared_error = ((polar_data - data_sc_inv) ** 2).mean()

    assert (
        mean_squared_error < allowed_error
    ), f"MSE is too high: {mean_squared_error:.4f} > {allowed_error:.4f}"


@pytest.mark.parametrize(
    "size",
    [
        (128, 32),
        (512, 512),
    ],
)
def test_grid_conversion(size):
    """Tests the grid conversion function with random 2d data"""
    data = np.random.random(size)
    x_grid_points = np.linspace(-45, 45, 100)  # angles
    z_grid_points = np.linspace(0, 100, 100)

    x_sample_points = np.deg2rad(x_grid_points) + np.pi / 2
    z_sample_points = z_grid_points

    project_to_cartesian_grid(
        data,
        (x_sample_points, z_sample_points),
        (x_grid_points, z_grid_points),
        spline_order=1,
        fill_value=0,
    )


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
    _data = to_8bit(data, dynamic_range, pillow=False)
    assert np.all(np.logical_and(_data >= 0, _data <= 255))
    assert _data.dtype == "uint8"
