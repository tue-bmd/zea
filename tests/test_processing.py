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
    to_image,
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
            processing_func_torch = getattr(processing_torch, func_name + "_torch")
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
                err_msg=(
                    f"Function {func_name} failed, tensorflow "
                    "and pytorch output not the same."
                ),
            )

        # Reset the processing function to the original implementation
        setattr(processing, func_name, original_processing_func)

    return decorator.decorator(wrapper, test_func)


@pytest.mark.parametrize(
    "comp_type, size, parameter_value_range",
    [
        ("a", (2, 1, 128, 32), (50, 200)),
        ("a", (512, 512), (50, 200)),
        ("mu", (2, 1, 128, 32), (50, 300)),
        ("mu", (512, 512), (50, 300)),
    ],
)
@equality_libs_processing
def test_companding(comp_type, size, parameter_value_range):
    """Test companding function"""

    for parameter_value in np.linspace(*parameter_value_range, 10):
        A = parameter_value if comp_type == "a" else 0
        mu = parameter_value if comp_type == "mu" else 0

        signal = np.clip((np.random.random(size) - 0.5) * 2, -1, 1)
        signal = signal.astype(np.float32)

        signal_out = processing.companding(
            signal, expand=False, comp_type=comp_type, A=A, mu=mu
        )
        assert np.any(
            np.not_equal(signal, signal_out)
        ), "Companding failed, arrays should not be equal"
        signal_out = processing.companding(
            signal_out, expand=True, comp_type=comp_type, A=A, mu=mu
        )

        np.testing.assert_almost_equal(signal, signal_out, decimal=6)
    return signal_out


@pytest.mark.parametrize(
    "size, dynamic_range, input_range",
    [
        ((2, 1, 128, 32), (-30, -5), None),
        ((512, 512), None, (0, 1)),
        ((512, 600), None, (-10, 300)),
        ((1, 128, 32), None, None),
    ],
)
def test_converting_to_image(size, dynamic_range, input_range):
    """Test converting to image functions"""
    if dynamic_range is None:
        _dynamic_range = (-60, 0)
    else:
        _dynamic_range = dynamic_range
    if input_range is None:
        _input_range = (0, 1)
    else:
        _input_range = input_range

    data = (
        np.random.random(size) * (_input_range[1] - _input_range[0]) + _input_range[0]
    )
    _data = to_image(data, dynamic_range, input_range)
    # data should be in dynamic range
    assert np.all(
        np.logical_and(_data >= _dynamic_range[0], _data <= _dynamic_range[1])
    ), f"Data is not in dynamic range after converting to image {_dynamic_range}"


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
    # create random data between input range
    data = np.random.random(size) * (input_range[1] - input_range[0]) + input_range[0]
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
    fs = probe_parameters["sampling_frequency"]
    fc = probe_parameters["center_frequency"]
    scan = PlaneWaveScan(
        probe.probe_geometry,
        n_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        n_ax=2048,
        sampling_frequency=fs,
        center_frequency=fc,
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
