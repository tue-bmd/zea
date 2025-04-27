"""Tests for the processing module."""

# pylint: disable=import-outside-toplevel
# pylint: disable=reimported

import math

import numpy as np
import pytest
from scipy.signal import hilbert

import usbmd.ops_v2 as ops
from usbmd.ops_v2 import Pipeline, Simulate
from usbmd.probes import Probe
from usbmd.scan import Scan

from . import backend_equality_check


@pytest.mark.parametrize(
    "comp_type, size, parameter_value_range",
    [
        ("a", (2, 1, 128, 32), (50, 200)),
        ("a", (512, 512), (50, 200)),
        ("mu", (2, 1, 128, 32), (50, 300)),
        ("mu", (512, 512), (50, 300)),
    ],
)
@backend_equality_check(decimal=4)
def test_companding(comp_type, size, parameter_value_range):
    """Test companding function"""

    from usbmd import ops_v2 as ops

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
@backend_equality_check(decimal=4)
def test_converting_to_image(size, dynamic_range, input_range):
    """Test converting to image functions"""

    from usbmd import ops_v2 as ops

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
@backend_equality_check(decimal=4)
def test_normalize(size, output_range, input_range):
    """Test normalize function"""

    from usbmd import ops_v2 as ops

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
    n_el = 128
    n_scat = 3
    n_tx = 2
    n_ax = 512

    aperture = 30e-3

    tx_apodizations = np.ones((n_tx, n_el))
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )

    t0_delays = np.stack(
        [
            np.linspace(0, 1e-6, n_el),
            np.linspace(1e-6, 0, n_el),
        ]
    )

    scan = Scan(
        n_tx=n_tx,
        n_ax=n_ax,
        n_el=n_el,
        center_frequency=3.125e6,
        sampling_frequency=12.5e6,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        apply_lens_correction=True,
        lens_sound_speed=1440.0,
        lens_thickness=1e-3,
        initial_times=np.zeros((n_tx,)),
        attenuation_coef=0.7,
        n_ch=1,
        selected_transmits="all",
    )
    probe = Probe(
        probe_geometry=probe_geometry,
        center_frequency=3.125e6,
        sampling_frequency=12.5e6,
    )

    # use pipeline here so it is easy to propagate the scan parameters
    simulator_pipeline = Pipeline([Simulate()])
    parameters = simulator_pipeline.prepare_parameters(probe, scan)

    data = []
    for _ in range(batch_size):

        # Define scatterers with random variation
        scat_x_base, scat_z_base = np.meshgrid(
            np.linspace(-10e-3, 10e-3, 5),
            np.linspace(5e-3, 30e-3, 5),
            indexing="ij",
        )
        # Add random perturbations
        scat_x = np.ravel(scat_x_base) + np.random.uniform(-1e-3, 1e-3, 25)
        scat_z = np.ravel(scat_z_base) + np.random.uniform(-1e-3, 1e-3, 25)
        n_scat = len(scat_x)
        # Select random subset of scatterers
        idx = np.random.choice(n_scat, n_scat, replace=False)[:n_scat]
        scat_positions = np.stack(
            [
                scat_x[idx],
                np.zeros_like(scat_x[idx]),
                scat_z[idx],
            ],
            axis=1,
        )

        output = simulator_pipeline(
            **parameters,
            scatterer_positions=scat_positions.astype(np.float32),
            scatterer_magnitudes=np.ones(n_scat, dtype=np.float32),
        )

        data.append(output["data"])
    data = np.concatenate(data)

    # slice data such that decimation fits exactly
    idx = data.shape[-3] % factor
    if idx > 0:
        data = data[..., :-idx, :, :]

    downsample = ops.Downsample(factor=factor, axis=-3)
    demodulate = ops.Demodulate(
        sampling_frequency=scan.sampling_frequency,
        center_frequency=scan.center_frequency,
        bandwidth=None,
        filter_coeff=None,
    )
    upmix = ops.UpMix(
        sampling_frequency=scan.sampling_frequency,
        center_frequency=scan.center_frequency,
        upsampling_rate=factor,
    )

    # cut n_ax data so it is divisible by factor
    data = data[:, :, : (data.shape[2] // factor) * factor]

    _data = demodulate(data)
    _data = downsample(_data)
    _data = upmix(_data)

    # TODO: make this test more tight
    assert (
        np.mean(np.abs((data - _data) ** 2)) < 10
    ), "Data is not equal after up and down conversion."


@backend_equality_check(decimal=4)
def test_hilbert_transform():
    """Test hilbert transform"""

    import keras

    from usbmd import ops_v2 as ops

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
    assert keras.ops.dtype(data_iq) in [
        "complex64",
        "complex128",
    ], f"Data type should be complex, got {keras.ops.dtype(data_iq)} instead."

    data_iq = envelope_detect.to_numpy(data_iq)

    reference_data_iq = hilbert(data, axis=-3)
    np.testing.assert_almost_equal(reference_data_iq, data_iq, decimal=4)

    return data_iq
