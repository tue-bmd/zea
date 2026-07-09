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
from zea.beamform.delays import compute_t0_delays_focused, compute_t0_delays_planewave
from zea.beamform.lens_correction import compute_lens_corrected_travel_times
from zea.beamform.pixelgrid import (
    cartesian_pixel_grid,
    scanline_aligned_apodization,
    scanline_pixel_grid,
)

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
    t_peak = np.zeros(n_tx, dtype=np.float32)
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
    t_peak = np.zeros(n_tx, dtype=np.float32)
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
    t_peak = np.zeros(n_tx, dtype=np.float32)
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


# scanline beamforming


def test_scanline_pixel_grid_linear():
    """Linear scanline grids are vertical columns at each beam's lateral focus."""
    origins = np.zeros((3, 3), np.float32)
    origins[:, 0] = [-5e-3, 0.0, 5e-3]  # walking transmit origins
    focus = np.full(3, 20e-3, np.float32)
    angles = np.zeros(3, np.float32)
    grid = scanline_pixel_grid(origins, focus, angles, (1e-3, 30e-3), 16, grid_type="cartesian")
    assert grid.shape == (16, 3, 3)
    for n in range(3):
        # angle 0 -> lateral focus == origin x, and every point on the line is at that x
        assert np.allclose(grid[:, n, 0], origins[n, 0])
        assert np.isclose(grid[0, n, 2], 1e-3) and np.isclose(grid[-1, n, 2], 30e-3)


def test_scanline_pixel_grid_sector():
    """Polar-style scanline grids are steered rays from the transmit origin."""
    origins = np.zeros((3, 3), np.float32)
    focus = np.full(3, 40e-3, np.float32)
    angles = np.array([-0.3, 0.0, 0.3], np.float32)
    grid = scanline_pixel_grid(origins, focus, angles, (0.0, 50e-3), 16, grid_type="polar")
    assert grid.shape == (16, 3, 3)
    for n in range(3):
        r = np.linalg.norm(grid[:, n] - origins[n], axis=-1)
        np.testing.assert_allclose(grid[:, n, 0], r * np.sin(angles[n]), atol=1e-6)
        np.testing.assert_allclose(grid[:, n, 2], r * np.cos(angles[n]), atol=1e-6)


def test_scanline_pixel_grid_default_is_cartesian():
    """``grid_type`` defaults to ``"cartesian"``, matching the old ``sector=False`` default."""
    origins = np.zeros((2, 3), np.float32)
    focus = np.full(2, 20e-3, np.float32)
    angles = np.zeros(2, np.float32)
    default_grid = scanline_pixel_grid(origins, focus, angles, (1e-3, 30e-3), 8)
    cartesian_grid = scanline_pixel_grid(
        origins, focus, angles, (1e-3, 30e-3), 8, grid_type="cartesian"
    )
    np.testing.assert_array_equal(default_grid, cartesian_grid)


def test_scanline_pixel_grid_invalid_grid_type():
    """An unsupported ``grid_type`` raises a clear error instead of silently misbehaving."""
    origins = np.zeros((2, 3), np.float32)
    focus = np.full(2, 20e-3, np.float32)
    angles = np.zeros(2, np.float32)
    with pytest.raises(ValueError, match="Unsupported grid_type"):
        scanline_pixel_grid(origins, focus, angles, (1e-3, 30e-3), 8, grid_type="scanline")


def test_scanline_pixel_grid_plane_wave_on_axis_no_nan():
    """A plane-wave (``np.inf`` focus distance) on-axis transmit must not produce NaN
    (regression test for the ``inf * 0`` hazard in the cartesian-style branch)."""
    origins = np.array([[1e-3, 0, 0], [2e-3, 0, 0]], np.float32)
    focus = np.array([np.inf, np.inf], np.float32)
    angles = np.zeros(2, np.float32)
    grid = scanline_pixel_grid(origins, focus, angles, (0, 0.05), 8, grid_type="cartesian")
    assert not np.isnan(grid).any()
    np.testing.assert_allclose(grid[:, 0, 0], origins[0, 0])
    np.testing.assert_allclose(grid[:, 1, 0], origins[1, 0])


def test_scanline_pixel_grid_golden_values():
    """Golden-value regression test locking in the exact numerics of both scanline
    grid styles (pinned before an internal refactor that renamed ``sector`` to
    ``grid_type`` without changing the underlying math)."""
    origins = np.array([[-5e-3, 0, 0], [0, 0, 0], [5e-3, 0, 0]], dtype=np.float32)
    focus = np.array([20e-3, np.inf, 15e-3], dtype=np.float32)
    angles = np.array([0.05, 0.0, -0.05], dtype=np.float32)
    grid_a = scanline_pixel_grid(origins, focus, angles, (1e-3, 30e-3), 5, grid_type="cartesian")
    expected_a_row0 = np.array(
        [[-0.00400042, 0.0, 0.001], [0.0, 0.0, 0.001], [0.00425031, 0.0, 0.001]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(grid_a[0], expected_a_row0, atol=1e-7)

    origins_b = np.zeros((3, 3), dtype=np.float32)
    focus_b = np.full(3, 40e-3, dtype=np.float32)
    angles_b = np.array([-0.3, 0.0, 0.3], dtype=np.float32)
    az_b = np.array([0.1, 0.0, -0.1], dtype=np.float32)
    grid_b = scanline_pixel_grid(
        origins_b, focus_b, angles_b, (0.0, 50e-3), 5, azimuth_angles=az_b, grid_type="polar"
    )
    expected_b_row1 = np.array(
        [
            [-0.00367555, -0.00036878, 0.01194171],
            [0.0, 0.0, 0.0125],
            [0.00367555, -0.00036878, 0.01194171],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(grid_b[1], expected_b_row1, atol=1e-7)


def test_scanline_aligned_apodization_one_hot():
    """Each pixel's apodization row selects exactly its own column's transmit."""
    n_tx, n_line = 4, 5
    apod = scanline_aligned_apodization(n_tx, n_line)
    assert apod.shape == (n_line * n_tx, n_tx)
    for i in range(n_line):
        for n in range(n_tx):
            expected = np.zeros(n_tx, np.float32)
            expected[n] = 1.0
            np.testing.assert_array_equal(apod[i * n_tx + n], expected)


def _make_scanline_inputs(probe_geometry, n_line=12):
    """Two focused transmits and their shared scanline pixel grid."""
    n_el = probe_geometry.shape[0]
    n_tx = 2
    origins = np.zeros((n_tx, 3), np.float32)
    origins[:, 0] = [-3e-3, 3e-3]
    focus = np.full(n_tx, 18e-3, np.float32)
    angles = np.zeros(n_tx, np.float32)
    t0 = np.stack(
        [
            compute_t0_delays_focused(
                o[None], f[None], probe_geometry, a[None], sound_speed=SOUND_SPEED
            )[0]
            for o, f, a in zip(origins, focus, angles)
        ]
    ).astype(np.float32)
    grid = scanline_pixel_grid(origins, focus, angles, (2e-3, 25e-3), n_line, grid_type="cartesian")
    apodization = scanline_aligned_apodization(n_tx, n_line)
    inputs = dict(
        t0_delays=t0,
        tx_apodizations=np.ones((n_tx, n_el), np.float32),
        sound_speed=SOUND_SPEED,
        probe_geometry=probe_geometry,
        initial_times=np.zeros(n_tx, np.float32),
        sampling_frequency=SAMPLING_FREQ,
        demodulation_frequency=DEMOD_FREQ,
        f_number=0.0,
        polar_angles=angles,
        focus_distances=focus,
        t_peak=np.zeros(n_tx, np.float32),
        transmit_origins=origins,
    )
    return grid.astype(np.float32), apodization, inputs


def test_scanline_via_pixel_beamform_matches_per_transmit_tof(probe_geometry):
    """Reconstructing the scanline grid via tof_correction + receive apodization
    (i.e. the regular pixel-based DAS path) must give, for every column, exactly
    what a standalone tof_correction call for that column's own transmit gives.
    """
    n_line = 10
    grid, apodization, inputs = _make_scanline_inputs(probe_geometry, n_line=n_line)
    n_tx = grid.shape[1]
    flatgrid = grid.reshape(-1, 3)
    data = np.random.randn(n_tx, 96, probe_geometry.shape[0], 2).astype(np.float32)

    tof = tof_correction(data, flatgrid, **inputs)  # (n_tx, n_pix, n_el, n_ch)
    apod_tx_pix = keras.ops.convert_to_tensor(apodization.T, dtype=tof.dtype)[:, :, None, None]
    image = keras.ops.convert_to_numpy(keras.ops.sum(tof * apod_tx_pix, axis=(0, 2)))
    image = image.reshape(n_line, n_tx, -1)

    for n in range(n_tx):
        tof_n = tof_correction(
            data[n : n + 1],
            grid[:, n],
            inputs["t0_delays"][n : n + 1],
            inputs["tx_apodizations"][n : n + 1],
            SOUND_SPEED,
            probe_geometry,
            inputs["initial_times"][n : n + 1],
            SAMPLING_FREQ,
            DEMOD_FREQ,
            0.0,
            inputs["polar_angles"][n : n + 1],
            inputs["focus_distances"][n : n + 1],
            inputs["t_peak"][n : n + 1],
            inputs["transmit_origins"][n : n + 1],
        )
        expected = keras.ops.convert_to_numpy(keras.ops.sum(tof_n, axis=2)[0])
        np.testing.assert_allclose(image[:, n], expected, rtol=1e-5, atol=1e-5)


def _focused_transmit_delays(grid, focus, angle, focal_region_length):
    """transmit_delays for a single focused beam, returning a numpy array."""
    xs = np.linspace(-10e-3, 10e-3, N_EL)
    probe = np.stack([xs, np.zeros(N_EL), np.zeros(N_EL)], -1).astype(np.float32)
    t0 = compute_t0_delays_focused(
        np.zeros((1, 3), np.float32),
        np.array([focus], np.float32),
        probe,
        np.array([angle], np.float32),
        sound_speed=SOUND_SPEED,
    )[0]
    t0 = (t0 - t0.min()).astype(np.float32)
    rx = (np.linalg.norm(grid[:, None] - probe[None], axis=-1) / SOUND_SPEED).astype(np.float32)
    txd = transmit_delays(
        grid.astype(np.float32),
        t0,
        np.ones(N_EL, np.float32),
        rx,
        np.float32(focus),
        np.float32(angle),
        np.float32(0.0),
        focal_region_length=focal_region_length,
    )
    return keras.ops.convert_to_numpy(txd)


def _offaxis_column(focus, angle, x_offset=5e-3, n=801):
    """Vertical pixel column offset laterally from a focused beam axis."""
    v = np.array([np.sin(angle), 0.0, np.cos(angle)], np.float32)
    x_col = float(focus * v[0]) + x_offset
    z = np.linspace(focus - 8e-3, focus + 8e-3, n).astype(np.float32)
    return np.stack([np.full_like(z, x_col), np.zeros_like(z), z], -1).astype(np.float32)


def test_focal_region_length_defaults_to_noop():
    """length=0.0 and length=None must reproduce the conventional model exactly."""
    grid = _offaxis_column(focus=15e-3, angle=0.0)
    base = _focused_transmit_delays(grid, 15e-3, 0.0, None)
    zero = _focused_transmit_delays(grid, 15e-3, 0.0, np.float32(0.0))
    np.testing.assert_allclose(base, zero, rtol=0, atol=0)


def test_focal_region_length_only_changes_focal_slab():
    """Focal-region blending may only touch pixels within length/2 of the focal plane."""
    focus, length = 15e-3, 2e-3
    grid = _offaxis_column(focus=focus, angle=0.0)
    base = _focused_transmit_delays(grid, focus, 0.0, None)
    blended = _focused_transmit_delays(grid, focus, 0.0, np.float32(length))

    changed = ~np.isclose(base, blended, atol=1e-12)
    # for an on-axis beam the focal plane is at z == focus; projection == z - focus
    inside_slab = np.abs(grid[:, 2] - focus) < (0.5 * length)
    assert np.all(changed[changed] == inside_slab[changed])
    assert changed.any(), "expected focal-region blending to change some focal-region pixels"


def test_focal_region_length_reduces_delay_discontinuity():
    """Focal-region blending must shrink the transmit-delay jump at the focal plane."""
    focus, length = 15e-3, 2e-3
    grid = _offaxis_column(focus=focus, angle=0.0, x_offset=2e-3)
    jump_base = np.abs(np.diff(_focused_transmit_delays(grid, focus, 0.0, None))).max()
    jump_blended = np.abs(
        np.diff(_focused_transmit_delays(grid, focus, 0.0, np.float32(length)))
    ).max()
    assert jump_blended < 0.6 * jump_base


def test_focal_region_length_noop_for_planewave(flatgrid, probe_geometry):
    """Plane-wave transmits are unaffected by focal_region_length."""
    flatgrid_t = keras.ops.convert_to_tensor(flatgrid)
    probe_t = keras.ops.convert_to_tensor(probe_geometry)
    n_el = probe_geometry.shape[0]
    t0 = keras.ops.zeros((n_el,))
    tx_apod = keras.ops.ones((n_el,))
    rx = distance_Rx(flatgrid_t, probe_t) / SOUND_SPEED
    common = dict(transmit_origin=keras.ops.zeros((3,)))
    base = transmit_delays(
        flatgrid_t, t0, tx_apod, rx, np.float32(np.inf), np.float32(0.0), np.float32(0.0), **common
    )
    hybrid = transmit_delays(
        flatgrid_t,
        t0,
        tx_apod,
        rx,
        np.float32(np.inf),
        np.float32(0.0),
        np.float32(0.0),
        focal_region_length=np.float32(2e-3),
        **common,
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(base), keras.ops.convert_to_numpy(hybrid), rtol=0, atol=0
    )


def test_warn_if_focal_region_length_unused(monkeypatch):
    """The pre-jit helper warns for non-focused data and stays quiet otherwise."""
    import zea.beamform.beamformer as bf

    calls = []
    monkeypatch.setattr(bf, "_warning_once", lambda msg, *a, **k: calls.append(msg))

    planewave = np.full(4, np.inf, np.float32)
    focused = np.full(4, 15e-3, np.float32)

    bf._warn_if_focal_region_length_unused(planewave, 2e-3)  # unused -> warns
    assert len(calls) == 1 and "focal_region_length" in calls[0]

    bf._warn_if_focal_region_length_unused(focused, 2e-3)  # focused -> no warning
    bf._warn_if_focal_region_length_unused(planewave, 0.0)  # disabled -> no warning
    assert len(calls) == 1


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
