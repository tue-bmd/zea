"""Tests for the processing module."""

import random
from math import e

import decorator
import numpy as np
import pytest
import tensorflow as tf
import torch

from usbmd.ops import (
    Companding,
    Demodulate,
    Downsample,
    EnvelopeDetect,
    LogCompress,
    Normalize,
    UpMix,
    channels_to_complex,
    complex_to_channels,
    hilbert,
)
from usbmd.probes import get_probe
from usbmd.processing import Process
from usbmd.scan import PlaneWaveScan
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
    decimal = 5

    # @functools.wraps(test_func)
    def wrapper(test_func, *args, **kwargs):
        # Set random seed
        seed = np.random.randint(0, 1000)
        set_random_seed(seed)

        # Extract function name from test function
        func_name = test_func.__name__.split("test_", 1)[-1]

        assert args[-1] == "numpy", (
            "Last argument should be 'numpy' for equality libs processing decorator."
            f"got {args[-1]} instead."
        )
        args = list(args)

        # Run the test function with the original processing module
        args[-1] = "numpy"
        original_output = test_func(*args, **kwargs)

        tf_output = None
        set_random_seed(seed)
        args[-1] = "tensorflow"
        tf_output = np.array(test_func(*args, **kwargs))

        # Run the test function with processing_torch module
        torch_output = None
        set_random_seed(seed)
        args[-1] = "torch"
        torch_output = np.array(test_func(*args, **kwargs))

        # Check if the outputs from the individual test functions are equal
        if tf_output is not None:
            np.testing.assert_almost_equal(
                original_output,
                tf_output,
                decimal=decimal,
                err_msg=f"Function {func_name} failed with tensorflow processing.",
            )
            print(f"Function {func_name} passed with tensorflow output.")
        if torch_output is not None:
            np.testing.assert_almost_equal(
                original_output,
                torch_output,
                decimal=decimal,
                err_msg=f"Function {func_name} failed with pytorch processing.",
            )
            print(f"Function {func_name} passed with pytorch output.")
        if tf_output is not None and torch_output is not None:
            np.testing.assert_almost_equal(
                tf_output,
                torch_output,
                decimal=decimal,
                err_msg=(
                    f"Function {func_name} failed, tensorflow "
                    "and pytorch output not the same."
                ),
            )

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
def test_companding(comp_type, size, parameter_value_range, ops="numpy"):
    """Test companding function"""

    for parameter_value in np.linspace(*parameter_value_range, 10):
        A = parameter_value if comp_type == "a" else 0
        mu = parameter_value if comp_type == "mu" else 0

        companding = Companding(comp_type=comp_type, A=A, mu=mu, expand=False, ops=ops)

        signal = np.clip((np.random.random(size) - 0.5) * 2, -1, 1)
        signal = signal.astype("float32")

        signal = companding.prepare_tensor(signal)
        signal_out = companding.process(signal)
        assert np.any(
            np.not_equal(np.array(signal), np.array(signal_out))
        ), "Companding failed, arrays should not be equal"
        companding.expand = True
        signal_out = companding.process(signal_out)

        signal = np.array(signal)
        signal_out = np.array(signal_out)
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
@equality_libs_processing
def test_converting_to_image(size, dynamic_range, input_range, ops="numpy"):
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
    output_range = (0, 1)
    normalize = Normalize(output_range, input_range, ops=ops)
    log_compress = LogCompress(dynamic_range, ops=ops)

    data = normalize.prepare_tensor(data)
    _data = log_compress(normalize(data))
    _data = np.array(_data)
    # data should be in dynamic range
    assert np.all(
        np.logical_and(_data >= _dynamic_range[0], _data <= _dynamic_range[1])
    ), f"Data is not in dynamic range after converting to image {_dynamic_range}"
    return _data


@pytest.mark.parametrize(
    "size, output_range, input_range",
    [
        ((2, 1, 128, 32), (-30, -5), (0, 1)),
        ((512, 512), (-2, -1), (-3, 50)),
        ((1, 128, 32), (50, 51), (-2.2, 3.0)),
    ],
)
@equality_libs_processing
def test_normalize(size, output_range, input_range, ops="numpy"):
    """Test normalize function"""
    normalize = Normalize(output_range, input_range, ops=ops)

    _input_range = output_range
    _output_range = input_range
    normalize_back = Normalize(_output_range, _input_range, ops=ops)

    # create random data between input range
    data = np.random.random(size) * (input_range[1] - input_range[0]) + input_range[0]
    data = normalize.prepare_tensor(data)
    _data = normalize(data)
    input_range, output_range = output_range, input_range
    _data = normalize_back(_data)
    # test if default args work too
    normalize = Normalize(ops=ops)
    _ = normalize(data)

    np.testing.assert_almost_equal(np.array(data), np.array(_data))
    return _data


@pytest.mark.parametrize(
    "size, axis",
    [
        ((2, 1, 128, 32), (-1)),
        ((512, 512), (-1)),
        ((1, 128, 32), (-1)),
    ],
)
def test_complex_to_channels(size, axis):
    """Test complex to channels and back"""
    data = np.random.random(size) + 1j * np.random.random(size)
    _data = complex_to_channels(data, axis=axis)
    __data = channels_to_complex(_data)
    np.testing.assert_almost_equal(data, __data)
    return _data


@pytest.mark.parametrize(
    "size, axis",
    [
        ((222, 1, 2, 2), (-1)),
        ((512, 512, 2), (-1)),
        ((2, 20, 128, 2), (-1)),
    ],
)
def test_channels_to_complex(size, axis):
    """Test channels to complex and back"""
    data = np.random.random(size)
    _data = channels_to_complex(data)
    __data = complex_to_channels(_data, axis=axis)
    np.testing.assert_almost_equal(data, __data)
    return _data


@pytest.mark.parametrize(
    "factor, batch_size",
    [
        (1, 2),
        (4, 1),
        (2, 3),
    ],
)
def test_up_and_down_conversion(factor, batch_size, ops="numpy"):
    """Test rf2iq and iq2rf in sequence"""
    probe = get_probe("verasonics_l11_4v")
    probe_parameters = probe.get_parameters()
    fs = probe_parameters["sampling_frequency"]
    fc = probe_parameters["center_frequency"]
    scan = PlaneWaveScan(
        probe.probe_geometry,
        n_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        n_ax=2094,
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
    data = np.expand_dims(data[0], axis=-1)

    # slice data such that decimation fits exactly
    idx = data.shape[-2] % factor
    if idx > 0:
        data = data[..., :-idx, :]

    downsample = Downsample(factor=factor, axis=-3, ops=ops)
    demodulate = Demodulate(fs=fs, fc=fc, bandwidth=None, filter_coeff=None, ops=ops)
    upmix = UpMix(fs=fs, fc=fc, upsampling_rate=factor, ops=ops)

    # cut n_ax data so it is divisible by factor
    data = data[:, : (data.shape[1] // factor) * factor]

    _data = demodulate(data)
    _data = downsample(_data)
    _data = upmix(_data)

    # TODO: make this test more tight
    assert (
        np.mean(np.abs((data - _data) ** 2)) < 10
    ), "Data is not equal after up and down conversion."


@equality_libs_processing
def test_hilbert_transform(ops="numpy"):
    """Test hilbert transform"""
    # create some dummy sinusoidal data of size (2, 500, 128, 1)
    # sinusoids on axis 1
    data = np.sin(np.linspace(0, 2 * e * np.pi, 500))
    data = np.expand_dims(data, axis=-1)
    data = np.expand_dims(data, axis=0)
    data = np.tile(data, (2, 1, 128, 1))

    data = data + np.random.random(data.shape) * 0.1

    # just getting this operation for the utils
    envelope_detect = EnvelopeDetect(axis=-3, ops=ops)
    ops = envelope_detect.ops

    data = envelope_detect.prepare_tensor(data)
    data_iq = hilbert(data, axis=-3, ops=ops)
    assert data_iq.dtype in [
        ops.complex64,
        ops.complex128,
    ], f"Data type should be complex, got {data_iq.dtype} instead."

    data_iq = np.array(data_iq)
    return data_iq


@equality_libs_processing
def test_processing_class(ops="numpy"):
    """Test the processing class"""
    operation_chain = [
        {
            "name": "multi_bandpass_filter",
            "params": {
                "params": {
                    "freqs": [-0.2e6, 0.0e6, 0.2e6],
                    "bandwidths": [1.2e6, 1.4e6, 1.0e6],
                    "num_taps": 81,
                },
                "modtype": "iq",
                "fs": 40e6,
                "fc": 5e6,
            },
        },  # this bandpass filters the data three times and returns a list
        {"name": "demodulate", "params": {"fs": 40e6, "fc": 5e6}},
        {"name": "envelope_detect"},
        {"name": "downsample", "params": {"factor": 4}},
        {"name": "normalize", "params": {"output_range": (0, 1)}},
        {"name": "log_compress", "params": {"dynamic_range": (-60, 0)}},
        {
            "name": "stack",
            "params": {"axis": 0},
        },  # stack the data back together
        {
            "name": "mean",
            "params": {"axis": 0},
        },  # take the mean of the stack
    ]

    process = Process()
    process.set_pipeline(
        operation_chain=operation_chain,
        device="cpu",
        ml_library=ops,
    )

    beamformed_data = np.random.random((2, 500, 128, 2))
    process.run(beamformed_data)

    process.set_pipeline(
        dtype="beamformed_data",
        to_dtype="image",
        device="cpu",
        ml_library=ops,
    )

    beamformed_data = np.random.random((2, 500, 128, 2))
    image = process.run(beamformed_data)

    return image
