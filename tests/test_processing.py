"""Tests for the processing module."""
import random

import decorator
import numpy as np
import pytest
import tensorflow as tf
import torch

from usbmd import processing
from usbmd.probes import get_probe
from usbmd.processing import (
    channels_to_complex,
    complex_to_channels,
    demodulate,
    downsample,
    normalize,
    project_to_cartesian_grid,
    scan_convert,
    to_8bit,
    to_image,
    transform_sc_image_to_polar,
    upmix,
)
from usbmd.pytorch_ultrasound import processing as processing_torch
from usbmd.scan import PlaneWaveScan
from usbmd.tensorflow_ultrasound import processing as processing_tf
from usbmd.utils.simulator import UltrasoundSimulator


def set_random_seed(seed=None):
    """Set random seed to all random generators."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    return seed


def equality_libs_processing(test_func):
    """Test the processing functions of different libraries

    Check if np / tf / torch processing funcs produce equal output.

    Example:
        ```python
            @pytest.mark.parametrize('some_keys', [some_values])
            @equality_libs_processing # <-- add as inner most decorator
            def test_my_processing_func(some_arguments):
                # Do some processing
                output = my_processing_func(some_arguments)
                return output
        ```
        To make it work with the @equality_libs_processing decorator,
        the name of the processing function should reappear in the
        torch / tensorflow modules:
        my_processing_func ->
            - my_processing_func_torch
            - my_processing_func_tf
            - test_my_processing_func
    """

    # @functools.wraps(test_func)
    def wrapper(test_func, *args, **kwargs):
        # Set random seed
        seed = np.random.randint(0, 1000)
        set_random_seed(seed)

        # Extract function name from test function
        func_name = test_func.__name__.split("test_", 1)[-1]

        # Run the test function with the original processing module
        original_output = test_func(*args, **kwargs)

        # Get the original processing function
        original_processing_func = getattr(processing, func_name)

        # Run the test function with processing_tf module
        tf_output = None
        if hasattr(processing_tf, func_name + "_tf"):
            processing_func_tf = getattr(processing_tf, func_name + "_tf")
            setattr(processing, func_name, processing_func_tf)
            set_random_seed(seed)
            tf_output = np.array(test_func(*args, **kwargs))

        # Run the test function with processing_torch module
        torch_output = None
        if hasattr(processing_torch, func_name + "_torch"):
            processing_func_torch = getattr(
                processing_torch, func_name + "_torch")
            setattr(processing, func_name, processing_func_torch)
            set_random_seed(seed)
            torch_output = np.array(test_func(*args, **kwargs))

        # Check if the outputs from the individual test functions are equal
        if tf_output is not None:
            np.testing.assert_almost_equal(
                original_output,
                tf_output,
                decimal=6,
                err_msg=f"Function {func_name} failed with tensorflow processing.",
            )
            print(f"Function {func_name} passed with tensorflow output.")
        if torch_output is not None:
            np.testing.assert_almost_equal(
                original_output,
                torch_output,
                decimal=6,
                err_msg=f"Function {func_name} failed with pytorch processing.",
            )
            print(f"Function {func_name} passed with pytorch output.")
        if tf_output is not None and torch_output is not None:
            np.testing.assert_almost_equal(
                tf_output,
                torch_output,
                decimal=6,
                err_msg=f"Function {func_name} failed, tensorflow "
                "and pytorch output not the same.",
            )

        # Reset the processing function to the original implementation
        setattr(processing, func_name, original_processing_func)

    return decorator.decorator(wrapper, test_func)


@pytest.mark.parametrize(
    "comp_type, size",
    [
        ("a", (2, 1, 128, 32)),
        ("a", (512, 512)),
        ("mu", (2, 1, 128, 32)),
        ("mu", (512, 512)),
    ],
)
@equality_libs_processing
def test_companding(comp_type, size):
    """Test companding function"""
    signal = np.clip((np.random.random(size) - 0.5) * 2, -1, 1)
    signal = signal.astype(np.float32)

    signal_out = processing.companding(
        signal, expand=False, comp_type=comp_type)
    assert np.any(
        np.not_equal(signal, signal_out)
    ), "Companding failed, arrays should not be equal"
    signal_out = processing.companding(
        signal_out, expand=True, comp_type=comp_type)

    np.testing.assert_almost_equal(signal, signal_out, decimal=6)
    return signal_out


@pytest.mark.parametrize(
    "size, dynamic_range, input_range",
    [
        ((2, 1, 128, 32), (-30, -5), None),
        ((512, 512), None, (0, 1)),
        ((1, 128, 32), None, None),
    ],
)
def test_converting_to_image(size, dynamic_range, input_range):
    """Test converting to image functions"""
    data = np.random.random(size)
    _data = to_image(data, dynamic_range, input_range)
    _data = to_8bit(data, dynamic_range)
    assert _data.dtype == "uint8"


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

    assert mean_squared_error < allowed_error, \
        f"MSE is too high: {mean_squared_error:.4f} > {allowed_error:.4f}"


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
    "size, output_range, input_range",
    [
        ((2, 1, 128, 32), (-30, -5), (0, 1)),
        ((512, 512), (-2, -1), (-3, 50)),
        ((1, 128, 32), (50, 51), (-2.2, 3.0)),
    ],
)
def test_normalize(size, output_range, input_range):
    """Test normalize function"""
    data = np.random.random(size)
    _data = normalize(data, output_range, input_range)
    input_range, output_range = output_range, input_range
    _data = normalize(_data, output_range, input_range)
    # test if default args work too
    _ = normalize(data)
    np.testing.assert_almost_equal(data, _data)


@pytest.mark.parametrize(
    "size, axis",
    [
        ((2, 1, 128, 32), (2)),
        ((512, 512), (-1)),
        ((1, 128, 32), (0)),
    ],
)
def test_complex_to_channels(size, axis):
    """Test complex to channels and back"""
    data = np.random.random(size) + 1j * np.random.random(size)
    _data = complex_to_channels(data, axis=axis)
    __data = channels_to_complex(_data, axis=axis)
    np.testing.assert_almost_equal(data, __data)
    return _data


@pytest.mark.parametrize(
    "size, axis",
    [
        ((222, 1, 2, 32), (2)),
        ((512, 512, 2), (-1)),
        ((2, 1, 128, 32), (0)),
    ],
)
@equality_libs_processing
def test_channels_to_complex(size, axis):
    """Test channels to complex and back"""
    data = np.random.random(size)
    _data = channels_to_complex(data, axis=axis)
    __data = complex_to_channels(_data, axis=axis)
    np.testing.assert_almost_equal(data, __data)
    return _data


@pytest.mark.parametrize(
    "factor, batch_size",
    [
        (1, 2),
        (6, 1),
        (2, 3),
    ],
)
def test_up_and_down_conversion(factor, batch_size):
    """Test rf2iq and iq2rf in sequence"""
    probe = get_probe("verasonics_l11_4v")
    probe_parameters = probe.get_default_scan_parameters()
    fs = probe_parameters["fs"]
    fc = probe_parameters["fc"]
    scan = PlaneWaveScan(
        n_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        n_ax=2048,
        fs=fs,
        fc=fc,
        angles=np.array(
            [
                0,
            ]
        ),
    )

    simulator = UltrasoundSimulator(probe, scan, batch_size=batch_size)

    # Generate pseudorandom input tensor
    data = simulator.generate(points=200)
    data = np.squeeze(data[0])

    # slice data such that decimation fits exactly
    idx = data.shape[-2] % factor
    if idx > 0:
        data = data[..., :-idx, :]

    _data = demodulate(data, fs=fs, fc=fc, bandwidth=None, filter_coeff=None)
    _data = downsample(_data, factor=factor, axis=-2)
    _data = upmix(_data, fs=fs / factor, fc=fc, upsampling_rate=factor)
    # TODO: add check if equal. due to filtering / decimation hard to do.
    # np.testing.assert_almost_equal(data, _data)
