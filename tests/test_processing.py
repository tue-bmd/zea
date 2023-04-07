"""Tests for the processing module."""
import numpy as np
import pytest

from usbmd.probes import get_probe
from usbmd.processing import (channels_to_complex, companding,
                              complex_to_channels, demodulate, downsample,
                              normalize, project_to_cartesian_grid,
                              scan_convert, to_8bit, to_image, upmix)
from usbmd.scan import PlaneWaveScan
from usbmd.tensorflow_ultrasound.processing import companding_tf
from usbmd.utils.simulator import UltrasoundSimulator


@pytest.mark.parametrize('comp_type, size, lib', [
    ('a', (2, 1, 128, 32), 'numpy'),
    ('a', (512, 512), 'numpy'),
    ('mu', (2, 1, 128, 32), 'numpy'),
    ('mu', (512, 512), 'numpy'),
    ('a', (2, 1, 128, 32), 'tensorflow'),
    ('a', (512, 512), 'tensorflow'),
    ('mu', (2, 1, 128, 32), 'tensorflow'),
    ('mu', (512, 512), 'tensorflow'),
])
def test_companding(comp_type, size, lib):
    """Test companding function"""
    signal = np.clip((np.random.random(size) - 0.5) *2, -1, 1)
    signal = signal.astype(np.float32)

    if lib == 'tensorflow':
        compand_func = companding_tf
    elif lib == 'numpy':
        compand_func = companding

    signal_out = compand_func(signal, expand=False, comp_type=comp_type)
    signal_out = compand_func(signal_out, expand=True, comp_type=comp_type)

    return np.testing.assert_almost_equal(signal, signal_out, decimal=6)

@pytest.mark.parametrize('size, dynamic_range, input_range', [
    ((2, 1, 128, 32), (-30, -5), None),
    ((512, 512), None, (0, 1)),
    ((1, 128, 32), None, None),
])
def test_converting_to_image(size, dynamic_range, input_range):
    """Test converting to image functions"""
    data = np.random.random(size)
    _data = to_image(data, dynamic_range, input_range)
    _data = to_8bit(data, dynamic_range)
    assert _data.dtype == 'uint8'

@pytest.mark.parametrize('size', [
    (2, 1, 128, 32),
    (512, 512),
    (1, 128, 32),
])
def test_scan_conversion(size):
    """Tests the scan_conversion function with random data"""
    data = np.random.random(size)
    x_axis = np.linspace(-50, 50, 100)
    z_axis = np.linspace(0, 100, 2000)
    scan_convert(data, x_axis, z_axis, n_pixels=500, spline_order=1, fill_value=0)

@pytest.mark.parametrize('size', [
    (128, 32),
    (512, 512),
])
def test_grid_conversion(size):
    """Tests the grid conversion function with random 2d data"""
    data = np.random.random(size)
    x_grid_points = np.linspace(-50, 50, 100)
    z_grid_points = np.linspace(0, 100, 2000)

    x_sample_points = np.deg2rad(x_grid_points) + np.pi / 2
    z_sample_points = np.deg2rad(z_grid_points) + np.pi / 2

    _data = project_to_cartesian_grid(
        data,
        (x_sample_points, z_sample_points),
        (x_grid_points, z_grid_points),
        spline_order=1,
        fill_value=0,
    )

    x_sample_points, x_grid_points = x_grid_points, x_sample_points
    z_sample_points, z_grid_points = z_grid_points, z_sample_points

    _data = project_to_cartesian_grid(
        _data,
        (x_sample_points, z_sample_points),
        (x_grid_points, z_grid_points),
        spline_order=1,
        fill_value=0,
    )
    # probably there is a way to cleverly choose the grid / sample
    # such that it can be inverted
    # np.testing.assert_almost_equal(data, _data)

@pytest.mark.parametrize('size, output_range, input_range', [
    ((2, 1, 128, 32), (-30, -5), (0, 1)),
    ((512, 512), (-2, -1), (-3, 50)),
    ((1, 128, 32), (50, 51), (-2.2, 3.0)),
])
def test_normalize(size, output_range, input_range):
    """Test normalize function"""
    data = np.random.random(size)
    _data = normalize(data, output_range, input_range)
    input_range, output_range = output_range, input_range
    _data = normalize(_data, output_range, input_range)
    # test if default args work too
    _ = normalize(data)
    np.testing.assert_almost_equal(data, _data)

@pytest.mark.parametrize('size, axis', [
    ((2, 1, 128, 32), (2)),
    ((512, 512), (-1)),
    ((1, 128, 32), (0)),
])
def test_complex_to_channels(size, axis):
    """Test complex to channels and back"""
    data = np.random.random(size) + 1j * np.random.random(size)
    _data = complex_to_channels(data, axis=axis)
    _data = channels_to_complex(_data, axis=axis)
    np.testing.assert_almost_equal(data, _data)

@pytest.mark.parametrize('size, axis', [
    ((222, 1, 2, 32), (2)),
    ((512, 512, 2), (-1)),
    ((2, 1, 128, 32), (0)),
])
def test_channels_to_complex(size, axis):
    """Test channels to complex and back"""
    data = np.random.random(size)
    _data = channels_to_complex(data, axis=axis)
    _data = complex_to_channels(_data, axis=axis)
    np.testing.assert_almost_equal(data, _data)

@pytest.mark.parametrize('factor, batch_size', [
    (1, 2), (6, 1), (2, 3),
])
def test_up_and_down_conversion(factor, batch_size):
    """Test rf2iq and iq2rf in sequence"""
    probe = get_probe('verasonics_l11_4v')
    probe_parameters = probe.get_default_scan_parameters()
    fs = probe_parameters['fs']
    fc = probe_parameters['fc']
    scan = PlaneWaveScan(
        N_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        N_ax=2048,
        fs=fs,
        fc=fc,
        angles=np.array([0,]))

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
    _data = upmix(_data, fs=fs, fc=fc, upsampling_rate=factor)
    # TODO: add check if equal. due to filtering / decimation hard to do.
    # np.testing.assert_almost_equal(data, _data)
