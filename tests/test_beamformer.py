"""Tests for the beamformer module"""

import keras
import numpy as np
import pytest

from zea.beamform.beamformer import (
    apply_delays,
    calculate_delays,
    complex_rotate,
    distance_Rx,
    tof_correction,
    transmit_delays,
)
from zea.beamform.delays import compute_t0_delays_planewave
from zea.beamform.lens_correction import compute_lens_corrected_travel_times
from zea.beamform.pixelgrid import cartesian_pixel_grid

from . import backend_equality_check

N_EL = 8  # number of transducer elements
SOUND_SPEED = 1540.0  # m/s
SAMPLING_FREQ = 40e6  # Hz
DEMOD_FREQ = 5e6  # Hz


@pytest.fixture
def probe_geometry():
    """Linear array with *N_EL* elements spanning ±10 mm in x."""
    xs = np.linspace(-10e-3, 10e-3, N_EL)
    return np.stack([xs, np.zeros(N_EL), np.zeros(N_EL)], axis=-1).astype(np.float32)


@pytest.fixture
def flatgrid():
    """Small 2-D Cartesian pixel grid, flattened to (n_pix, 3)."""
    grid = cartesian_pixel_grid(
        xlims=(-5e-3, 5e-3),
        zlims=(5e-3, 20e-3),
        grid_size_x=9,
        grid_size_z=11,
    )
    return grid.reshape(-1, 3).astype(np.float32)


def _make_calculate_delays_inputs(probe_geometry, flatgrid, n_tx=3):
    """Build the full set of inputs required by ``calculate_delays``."""
    n_el = probe_geometry.shape[0]
    polar_angles = np.zeros(n_tx, dtype=np.float32)
    t0_delays = compute_t0_delays_planewave(
        probe_geometry, polar_angles, sound_speed=SOUND_SPEED
    ).astype(np.float32)
    tx_apodizations = np.ones((n_tx, n_el), dtype=np.float32)
    initial_times = np.zeros(n_tx, dtype=np.float32)
    focus_distances = np.zeros(n_tx, dtype=np.float32)
    t_peak = np.zeros(1, dtype=np.float32)
    tx_waveform_indices = np.zeros(n_tx, dtype=np.int32)
    transmit_origins = np.zeros((n_tx, 3), dtype=np.float32)
    return dict(
        grid=flatgrid,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        probe_geometry=probe_geometry,
        initial_times=initial_times,
        sampling_frequency=SAMPLING_FREQ,
        sound_speed=SOUND_SPEED,
        focus_distances=focus_distances,
        polar_angles=polar_angles,
        t_peak=t_peak,
        tx_waveform_indices=tx_waveform_indices,
        transmit_origins=transmit_origins,
    )


def _make_tof_inputs(probe_geometry, flatgrid, n_tx=3, n_ax=64, n_ch=1):
    """Build the full set of inputs required by ``tof_correction``."""
    n_el = probe_geometry.shape[0]
    data = np.random.randn(n_tx, n_ax, n_el, n_ch).astype(np.float32)
    polar_angles = np.zeros(n_tx, dtype=np.float32)
    t0_delays = compute_t0_delays_planewave(
        probe_geometry, polar_angles, sound_speed=SOUND_SPEED
    ).astype(np.float32)
    tx_apodizations = np.ones((n_tx, n_el), dtype=np.float32)
    initial_times = np.zeros(n_tx, dtype=np.float32)
    focus_distances = np.zeros(n_tx, dtype=np.float32)
    t_peak = np.zeros(1, dtype=np.float32)
    tx_waveform_indices = np.zeros(n_tx, dtype=np.int32)
    transmit_origins = np.zeros((n_tx, 3), dtype=np.float32)
    return dict(
        data=data,
        flatgrid=flatgrid,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        sound_speed=SOUND_SPEED,
        probe_geometry=probe_geometry,
        initial_times=initial_times,
        sampling_frequency=SAMPLING_FREQ,
        demodulation_frequency=DEMOD_FREQ,
        f_number=0.0,
        polar_angles=polar_angles,
        focus_distances=focus_distances,
        t_peak=t_peak,
        tx_waveform_indices=tx_waveform_indices,
        transmit_origins=transmit_origins,
    )


def _make_multistatic_inputs(probe_geometry, flatgrid, n_ax=128):
    """Build inputs for a multistatic dataset (n_tx == n_el)."""
    n_el = probe_geometry.shape[0]
    n_tx = n_el  # multistatic requirement
    n_ch = 1
    data = np.random.randn(n_tx, n_ax, n_el, n_ch).astype(np.float32)
    polar_angles = np.zeros(n_tx, dtype=np.float32)
    t0_delays = compute_t0_delays_planewave(
        probe_geometry, polar_angles, sound_speed=SOUND_SPEED
    ).astype(np.float32)
    tx_apodizations = np.ones((n_tx, n_el), dtype=np.float32)
    initial_times = np.zeros(n_tx, dtype=np.float32)
    focus_distances = np.zeros(n_tx, dtype=np.float32)
    t_peak = np.zeros(1, dtype=np.float32)
    tx_waveform_indices = np.zeros(n_tx, dtype=np.int32)
    transmit_origins = np.zeros((n_tx, 3), dtype=np.float32)

    nx_sos, nz_sos = 16, 16
    sos_grid_x = np.linspace(-10e-3, 10e-3, nx_sos).astype(np.float32)
    sos_grid_z = np.linspace(0e-3, 25e-3, nz_sos).astype(np.float32)
    sos_map = np.full((nz_sos, nx_sos), SOUND_SPEED, dtype=np.float32)

    return dict(
        data=data,
        flatgrid=flatgrid,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        sound_speed=SOUND_SPEED,
        probe_geometry=probe_geometry,
        initial_times=initial_times,
        sampling_frequency=SAMPLING_FREQ,
        demodulation_frequency=DEMOD_FREQ,
        f_number=0.0,
        polar_angles=polar_angles,
        focus_distances=focus_distances,
        t_peak=t_peak,
        tx_waveform_indices=tx_waveform_indices,
        transmit_origins=transmit_origins,
        sos_map=sos_map,
        sos_grid_x=sos_grid_x,
        sos_grid_z=sos_grid_z,
    )


# complex_rotate


@backend_equality_check()
def test_complex_rotate_zero_rotation_preserves_data():
    """A rotation by 0 should return the original data."""
    rng = np.random.default_rng(seed=42)
    iq = keras.ops.convert_to_tensor(rng.standard_normal((10, 4, 2)).astype(np.float32))
    theta = keras.ops.zeros((10, 4))
    rotated = complex_rotate(iq, theta)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(rotated),
        keras.ops.convert_to_numpy(iq),
        atol=1e-6,
    )
    return rotated


@backend_equality_check()
def test_complex_rotate_pi_rotation_negates_components():
    """A rotation by π should negate both I and Q (cos π = -1, sin π ≈ 0)."""
    iq = keras.ops.convert_to_tensor([[[1.0, 0.0], [0.0, 1.0]]])
    theta = keras.ops.full((1, 2), np.pi)
    rotated = keras.ops.convert_to_numpy(complex_rotate(iq, theta))
    np.testing.assert_allclose(rotated[0, 0], [-1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(rotated[0, 1], [0.0, -1.0], atol=1e-6)
    return rotated


@backend_equality_check()
def test_complex_rotate_half_pi():
    """Rotating (1, 0) by π/2 should give (0, 1)."""
    iq = keras.ops.convert_to_tensor([[[1.0, 0.0]]])
    theta = keras.ops.full((1, 1), np.pi / 2)
    rotated = keras.ops.convert_to_numpy(complex_rotate(iq, theta))
    np.testing.assert_allclose(rotated[0, 0], [0.0, 1.0], atol=1e-5)
    return rotated


# distance_Rx


@backend_equality_check()
def test_distance_rx_output_shape(flatgrid, probe_geometry):
    """Output should be (n_pix, n_el)."""
    dist = distance_Rx(
        keras.ops.convert_to_tensor(flatgrid),
        keras.ops.convert_to_tensor(probe_geometry),
    )
    assert dist.shape == (flatgrid.shape[0], probe_geometry.shape[0])
    return dist


@backend_equality_check()
def test_distance_rx_positive(flatgrid, probe_geometry):
    """All distances must be non-negative."""
    dist = keras.ops.convert_to_numpy(
        distance_Rx(
            keras.ops.convert_to_tensor(flatgrid),
            keras.ops.convert_to_tensor(probe_geometry),
        )
    )
    assert np.all(dist >= 0)
    return dist


@backend_equality_check()
def test_distance_rx_known_distance():
    """Element at origin, pixel at (0, 0, 1) → distance = 1 m."""
    dist = keras.ops.convert_to_numpy(
        distance_Rx(
            keras.ops.convert_to_tensor([[0.0, 0.0, 1.0]]),
            keras.ops.convert_to_tensor([[0.0, 0.0, 0.0]]),
        )
    )
    np.testing.assert_allclose(dist, [[1.0]], atol=1e-6)
    return dist


# apply_delays


@backend_equality_check()
def test_apply_delays_integer_delays_pick_correct_sample():
    """With integer delays the output should equal the original sample."""
    n_ax, n_el, n_ch = 20, 4, 1
    data = keras.ops.convert_to_tensor(
        np.arange(n_ax * n_el * n_ch, dtype=np.float32).reshape(n_ax, n_el, n_ch)
    )
    delays = keras.ops.full((3, n_el), 5.0)
    result = keras.ops.convert_to_numpy(apply_delays(data, delays, clip_min=0, clip_max=n_ax - 1))
    assert result.shape == (3, n_el, n_ch)
    expected = keras.ops.convert_to_numpy(data[5])
    np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=1e-6)
    return result


@backend_equality_check()
def test_apply_delays_interpolation_midpoint():
    """Delay of 2.5 between samples 2 and 3 should give 50 / 50 interpolation."""
    n_ax, n_el, n_ch = 10, 1, 1
    data_np = np.zeros((n_ax, n_el, n_ch), dtype=np.float32)
    data_np[2, 0, 0] = 0.0
    data_np[3, 0, 0] = 1.0
    data = keras.ops.convert_to_tensor(data_np)
    delays = keras.ops.convert_to_tensor([[2.5]])
    result = keras.ops.convert_to_numpy(apply_delays(data, delays, clip_min=0, clip_max=n_ax - 1))
    np.testing.assert_allclose(result[0, 0, 0], 0.5, atol=1e-6)
    return result


@backend_equality_check()
def test_apply_delays_iq_data_shape():
    """Two-channel (IQ) data should be handled correctly."""
    rng = np.random.default_rng(seed=42)
    n_ax, n_el, n_ch = 10, 2, 2
    data = keras.ops.convert_to_tensor(rng.standard_normal((n_ax, n_el, n_ch)).astype(np.float32))
    delays = keras.ops.full((4, n_el), 3.0)
    result = apply_delays(data, delays, clip_min=0, clip_max=n_ax - 1)
    assert result.shape == (4, n_el, n_ch)
    return result


# transmit_delays


@backend_equality_check()
def test_transmit_delays_planewave_zero_angle(flatgrid, probe_geometry):
    """For a 0° plane wave, transmit delay should equal the min traveltimes."""
    flatgrid_t = keras.ops.convert_to_tensor(flatgrid)
    probe_geometry_t = keras.ops.convert_to_tensor(probe_geometry)
    n_el = probe_geometry.shape[0]
    t0 = keras.ops.zeros((n_el,))
    tx_apod = keras.ops.ones((n_el,))
    rx_delays = distance_Rx(flatgrid_t, probe_geometry_t) / SOUND_SPEED

    txd = transmit_delays(
        flatgrid_t,
        t0,
        tx_apod,
        rx_delays,
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
        transmit_origin=keras.ops.zeros((3,)),
    )
    txd = keras.ops.convert_to_numpy(txd)
    assert txd.shape == (flatgrid.shape[0],)
    assert np.all(np.isfinite(txd))
    return txd


@backend_equality_check()
def test_transmit_delays_focused(flatgrid, probe_geometry):
    """Focused transmit should produce finite delays."""
    flatgrid_t = keras.ops.convert_to_tensor(flatgrid)
    probe_geometry_t = keras.ops.convert_to_tensor(probe_geometry)
    n_el = probe_geometry.shape[0]
    t0 = keras.ops.zeros((n_el,))
    tx_apod = keras.ops.ones((n_el,))
    rx_delays = distance_Rx(flatgrid_t, probe_geometry_t) / SOUND_SPEED

    txd = transmit_delays(
        flatgrid_t,
        t0,
        tx_apod,
        rx_delays,
        np.float32(15e-3),
        np.float32(0.0),
        np.float32(0.0),
        transmit_origin=keras.ops.zeros((3,)),
    )
    txd = keras.ops.convert_to_numpy(txd)
    assert txd.shape == (flatgrid.shape[0],)
    assert np.all(np.isfinite(txd))
    return txd


# calculate_delays


@backend_equality_check()
def test_calculate_delays_output_shapes(probe_geometry, flatgrid):
    """Transmit and receive delays should have the correct shapes."""
    n_tx = 3
    inputs = _make_calculate_delays_inputs(probe_geometry, flatgrid, n_tx=n_tx)
    tx_del, rx_del = calculate_delays(**inputs)
    n_pix = flatgrid.shape[0]
    assert tx_del.shape == (n_pix, n_tx)
    assert rx_del.shape == (n_pix, N_EL)
    return tx_del


@backend_equality_check()
def test_calculate_delays_in_samples(probe_geometry, flatgrid):
    """Returned delays should be in sample units (not seconds)."""
    inputs = _make_calculate_delays_inputs(probe_geometry, flatgrid, n_tx=1)
    tx_del, rx_del = calculate_delays(**inputs)
    rx_del_np = keras.ops.convert_to_numpy(rx_del)
    assert np.all(rx_del_np >= 0)
    assert np.max(rx_del_np) > 1, "Receive delays look too small — possibly still in seconds?"
    return tx_del


# tof_correction


@backend_equality_check()
def test_tof_correction_output_shape_rf(probe_geometry, flatgrid):
    """Output should be (n_tx, n_pix, n_el, n_ch)."""
    n_tx, n_ax, n_ch = 3, 64, 1
    inputs = _make_tof_inputs(probe_geometry, flatgrid, n_tx=n_tx, n_ax=n_ax, n_ch=n_ch)
    result = tof_correction(**inputs)
    n_pix = flatgrid.shape[0]
    assert result.shape == (n_tx, n_pix, N_EL, n_ch)
    return result


@backend_equality_check()
def test_tof_correction_output_shape_iq(probe_geometry, flatgrid):
    """IQ data (n_ch=2) should also work and trigger phase rotation."""
    n_tx, n_ax, n_ch = 2, 64, 2
    inputs = _make_tof_inputs(probe_geometry, flatgrid, n_tx=n_tx, n_ax=n_ax, n_ch=n_ch)
    result = tof_correction(**inputs)
    n_pix = flatgrid.shape[0]
    assert result.shape == (n_tx, n_pix, N_EL, n_ch)
    return result


@backend_equality_check()
def test_tof_correction_with_fnumber(probe_geometry, flatgrid):
    """Using a nonzero f-number should produce masked (zero-valued) regions."""
    n_tx, n_ax = 1, 64
    inputs = _make_tof_inputs(probe_geometry, flatgrid, n_tx=n_tx, n_ax=n_ax)
    inputs["f_number"] = 1.0
    result = keras.ops.convert_to_numpy(tof_correction(**inputs))
    assert np.any(result == 0.0), "Expected some masked-out values with f_number > 0"
    return result


@backend_equality_check()
def test_tof_correction_zero_data(probe_geometry, flatgrid):
    """Zero input data should produce zero output regardless of delays."""
    n_tx, n_ax = 2, 64
    inputs = _make_tof_inputs(probe_geometry, flatgrid, n_tx=n_tx, n_ax=n_ax)
    inputs["data"] = np.zeros_like(inputs["data"])
    result = keras.ops.convert_to_numpy(tof_correction(**inputs))
    np.testing.assert_allclose(result, 0.0, atol=1e-7)
    return result


# tof_correction with sos_grid


@backend_equality_check(backends=["tensorflow", "jax"])
def test_tof_correction_sos_grid_output_shape(probe_geometry, flatgrid):
    """Output shape should be (n_tx, n_pix, n_el, n_ch)."""
    inputs = _make_multistatic_inputs(probe_geometry, flatgrid)
    result = tof_correction(**inputs)
    n_pix = flatgrid.shape[0]
    n_el = probe_geometry.shape[0]
    assert result.shape == (n_el, n_pix, n_el, 1)
    return result


@backend_equality_check(backends=["tensorflow", "jax"])
def test_tof_correction_sos_grid_zero_data(probe_geometry, flatgrid):
    """Zero input data must produce zero output."""
    inputs = _make_multistatic_inputs(probe_geometry, flatgrid)
    inputs["data"] = np.zeros_like(inputs["data"])
    result = keras.ops.convert_to_numpy(tof_correction(**inputs))
    np.testing.assert_allclose(result, 0.0, atol=1e-7)
    return result


@backend_equality_check()
def test_lens_correction_output_shape(probe_geometry, flatgrid):
    """Output should be (n_pix, n_el)."""
    element_pos = keras.ops.convert_to_tensor(probe_geometry)
    pixel_pos = keras.ops.convert_to_tensor(flatgrid)
    tt = compute_lens_corrected_travel_times(
        element_pos,
        pixel_pos,
        lens_thickness=1e-3,
        c_lens=1000.0,
        c_medium=SOUND_SPEED,
    )
    assert tt.shape == (flatgrid.shape[0], probe_geometry.shape[0])
    return tt


@backend_equality_check()
def test_lens_correction_known_vertical_path():
    """Pixel directly above an element gives an analytically known travel time."""
    lens_thickness = 1e-3
    c_lens = 1000.0
    c_medium = SOUND_SPEED
    z_pixel = 20e-3

    element_pos = keras.ops.convert_to_tensor([[0.0, 0.0, 0.0]])
    pixel_pos = keras.ops.convert_to_tensor([[0.0, 0.0, z_pixel]])

    tt = keras.ops.convert_to_numpy(
        compute_lens_corrected_travel_times(
            element_pos,
            pixel_pos,
            lens_thickness=lens_thickness,
            c_lens=c_lens,
            c_medium=c_medium,
            n_iter=5,
        )
    )
    expected = lens_thickness / c_lens + (z_pixel - lens_thickness) / c_medium
    np.testing.assert_allclose(tt[0, 0], expected, rtol=1e-4)
    return tt
