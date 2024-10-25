"""Tests for the processing module."""

import math

import decorator
import jax
import numpy as np
import pytest
from keras import ops as kops
from scipy.signal import hilbert

from usbmd import ops
from usbmd.probes import get_probe
from usbmd.processing import Process
from usbmd.scan import PlaneWaveScan
from usbmd.setup_usbmd import set_backend
from usbmd.utils.simulator import UltrasoundSimulator


def equality_libs_processing(decimal=4):
    """Test the processing functions of different libraries

    Check if numpy, tensorflow, torch and jax processing funcs produce equal output.

    Example:
        ```python
            @pytest.mark.parametrize('some_keys', [some_values])
            @equality_libs_processing(decimal=4) # <-- add as inner most decorator
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

    BACKENDS = ["numpy", "tensorflow", "torch", "jax"]

    def wrapper(test_func, *args, **kwargs):
        # Set random seed
        seed = np.random.randint(0, 1000)

        # Extract function name from test function
        func_name = test_func.__name__.split("test_", 1)[-1]

        output = {}
        for backend in BACKENDS:
            print(f"Running {func_name} in {backend}")
            set_backend(backend)
            import keras  # pylint: disable=import-outside-toplevel

            keras.utils.set_random_seed(seed)
            with jax.disable_jit():
                output[backend] = np.array(test_func(*args, **kwargs))

        # Check if the outputs from the individual test functions are equal
        for backend in BACKENDS[1:]:
            np.testing.assert_almost_equal(
                output["numpy"],
                output[backend],
                decimal=decimal,
                err_msg=f"Function {func_name} failed with {backend} processing.",
            )
            print(f"Function {func_name} passed with {backend} output.")

    return decorator.decorator(wrapper)


@pytest.mark.parametrize(
    "comp_type, size, parameter_value_range",
    [
        ("a", (2, 1, 128, 32), (50, 200)),
        ("a", (512, 512), (50, 200)),
        ("mu", (2, 1, 128, 32), (50, 300)),
        ("mu", (512, 512), (50, 300)),
    ],
)
@equality_libs_processing(decimal=4)
def test_companding(comp_type, size, parameter_value_range):
    """Test companding function"""

    for parameter_value in np.linspace(*parameter_value_range, 10):
        A = parameter_value if comp_type == "a" else 0
        mu = parameter_value if comp_type == "mu" else 0

        companding = ops.Companding(comp_type=comp_type, A=A, mu=mu, expand=False)

        signal = np.clip((np.random.random(size) - 0.5) * 2, -1, 1)
        signal = signal.astype("float32")

        signal = companding.prepare_tensor(signal)
        signal_out = companding.process(signal)
        assert np.any(
            np.not_equal(companding.to_numpy(signal), companding.to_numpy(signal_out))
        ), "Companding failed, arrays should not be equal"
        companding.expand = True
        signal_out = companding.process(signal_out)

        signal = companding.to_numpy(signal)
        signal_out = companding.to_numpy(signal_out)
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
@equality_libs_processing(decimal=4)
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
    output_range = (0, 1)
    normalize = ops.Normalize(output_range, input_range)
    log_compress = ops.LogCompress(dynamic_range)

    data = normalize.prepare_tensor(data)
    _data = log_compress(normalize(data))
    _data = log_compress.to_numpy(_data)
    # data should be in dynamic range
    assert np.all(
        np.logical_and(_data >= _dynamic_range[0], _data <= _dynamic_range[1]),
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
@equality_libs_processing(decimal=4)
def test_normalize(size, output_range, input_range):
    """Test normalize function"""
    normalize = ops.Normalize(output_range, input_range)

    _input_range = output_range
    _output_range = input_range
    normalize_back = ops.Normalize(_output_range, _input_range)

    # create random data between input range
    data = np.random.random(size) * (input_range[1] - input_range[0]) + input_range[0]
    data = normalize.prepare_tensor(data)
    _data = normalize(data)
    input_range, output_range = output_range, input_range
    _data = normalize_back(_data)
    # test if default args work too
    normalize = ops.Normalize()
    _ = normalize(data)

    np.testing.assert_almost_equal(
        normalize.to_numpy(data), normalize.to_numpy(_data), decimal=4
    )
    return normalize.to_numpy(_data)


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
    _data = ops.complex_to_channels(data, axis=axis)
    __data = ops.channels_to_complex(_data)
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
    _data = ops.channels_to_complex(data)
    __data = ops.complex_to_channels(_data, axis=axis)
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
def test_up_and_down_conversion(factor, batch_size):
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

    downsample = ops.Downsample(factor=factor, axis=-3)
    demodulate = ops.Demodulate(fs=fs, fc=fc, bandwidth=None, filter_coeff=None)
    upmix = ops.UpMix(fs=fs, fc=fc, upsampling_rate=factor)

    # cut n_ax data so it is divisible by factor
    data = data[:, : (data.shape[1] // factor) * factor]

    _data = demodulate(data)
    _data = downsample(_data)
    _data = upmix(_data)

    # TODO: make this test more tight
    assert (
        np.mean(np.abs((data - _data) ** 2)) < 10
    ), "Data is not equal after up and down conversion."


@equality_libs_processing(decimal=4)
def test_hilbert_transform():
    """Test hilbert transform"""
    # create some dummy sinusoidal data of size (2, 500, 128, 1)
    # sinusoids on axis 1
    data = np.sin(np.linspace(0, 2 * math.e * np.pi, 500))
    data = data[np.newaxis, :, np.newaxis, np.newaxis]
    data = np.tile(data, (2, 1, 128, 1))

    data = data + np.random.random(data.shape) * 0.1

    # just getting this operation for the utils
    envelope_detect = ops.EnvelopeDetect(axis=-3)

    data_prepared = envelope_detect.prepare_tensor(data)
    data_iq = ops.hilbert(data_prepared, axis=-3)
    assert kops.dtype(data_iq) in [
        "complex64",
        "complex128",
    ], f"Data type should be complex, got {kops.dtype(data_iq)} instead."

    data_iq = envelope_detect.to_numpy(data_iq)

    reference_data_iq = hilbert(data, axis=-3)
    np.testing.assert_almost_equal(reference_data_iq, data_iq, decimal=4)

    return data_iq


@equality_libs_processing(decimal=4)
def test_processing_class():
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
    )

    beamformed_data = np.random.random((2, 500, 128, 2))
    process.run(beamformed_data)

    process.set_pipeline(
        dtype="beamformed_data",
        to_dtype="image",
        device="cpu",
    )

    beamformed_data = np.random.random((2, 500, 128, 2))
    image = process.run(beamformed_data)

    return image
