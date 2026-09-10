"""Checks of :func:`zea.simulator.pressure_field` against the simulators and analytic physics."""

import keras
import numpy as np
import pytest
from keras import ops

from zea.simulator import pressure_field, simulate_rf, transducer_transfer

SOUND_SPEED = 1540.0
CENTER_FREQUENCY = 3e6
SAMPLING_FREQUENCY = 12e6
N_AX = 512
ELEMENT_WIDTH = 0.27e-3


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))


def _transmit(geometry, n_tx=1, t0_delays=None, **overrides):
    """Unfocused, unapodized transmits shared by the simulator and the pressure field."""
    n_el = len(geometry)
    if t0_delays is None:
        t0_delays = np.zeros((n_tx, n_el), np.float32)
    kwargs = {
        "probe_geometry": np.asarray(geometry, np.float32),
        "sound_speed": SOUND_SPEED,
        "center_frequency": CENTER_FREQUENCY,
        "sampling_frequency": SAMPLING_FREQUENCY,
        "t0_delays": np.asarray(t0_delays, np.float32),
        "initial_times": np.zeros(n_tx, np.float32),
        "element_width": ELEMENT_WIDTH,
        "tx_apodizations": np.ones((n_tx, n_el), np.float32),
        "t_peak": np.zeros(n_tx, np.float32),
        **overrides,
    }
    return {
        k: ops.convert_to_tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
    }


def _scene(transmit, positions, **overrides):
    """Simulator arguments for unit scatterers in the transmit of :func:`_transmit`."""
    positions = np.asarray(positions, np.float32).reshape(-1, 3)
    return {
        **transmit,
        "scatterer_positions": ops.convert_to_tensor(positions),
        "scatterer_magnitudes": ops.ones(len(positions), "float32"),
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "n_ax": N_AX,
        "attenuation_coef": 0.0,
        "scatter_exponent": 0.0,
        **overrides,
    }


def _rel_err(reference, result):
    reference, result = _np(reference), _np(result)
    return np.linalg.norm(reference - result) / np.linalg.norm(reference)


def _spectrum_ratio(numerator, denominator, floor=0.05):
    num, den = np.fft.rfft(_np(numerator)), np.fft.rfft(_np(denominator))
    keep = np.abs(den) > floor * np.abs(den).max()
    return np.abs(num[keep] / den[keep]), keep


def test_single_element_echo_is_the_pressure_delayed_by_the_return_trip():
    # On axis of one element the receive response is the spread and a delay, so the echo is
    # the transmit field shifted by the travel time, spread with unit gain at 1 mm.
    samples = 64
    r = samples * SOUND_SPEED / SAMPLING_FREQUENCY
    transmit = _transmit(np.zeros((1, 3)))
    rf = _np(simulate_rf(**_scene(transmit, [0.0, 0.0, r])))[0, :, 0, 0]
    pressure = _np(pressure_field(np.array([[0.0, 0.0, r]]), **transmit, n_ax=N_AX, output="time"))
    pressure = pressure[0, :, 0]
    expected = np.zeros_like(rf)
    expected[samples:] = pressure[:-samples] * 1e-3 / r
    assert np.abs(pressure).max() > 0
    assert _rel_err(expected, rf) < 1e-3


def test_rms_matches_the_time_waveforms_and_follows_the_grid_shape():
    n_el = 16
    geometry = np.stack([np.linspace(-2e-3, 2e-3, n_el), np.zeros(n_el), np.zeros(n_el)], -1)
    transmit = _transmit(geometry)
    x, z = np.meshgrid(np.linspace(-5e-3, 5e-3, 5), np.linspace(5e-3, 30e-3, 4))
    grid = np.stack([x, np.zeros_like(x), z], -1)
    waveforms = _np(pressure_field(grid, **transmit, output="time"))
    rms = _np(pressure_field(grid, **transmit, n_ax=waveforms.shape[1]))
    assert rms.shape == (1, 4, 5)
    assert waveforms.shape[0] == 1 and waveforms.shape[2:] == (4, 5)
    assert _rel_err(np.sqrt(np.mean(waveforms**2, axis=1)), rms) < 1e-5


def test_chunking_does_not_change_the_result():
    n_el = 8
    geometry = np.stack([np.linspace(-1e-3, 1e-3, n_el), np.zeros(n_el), np.zeros(n_el)], -1)
    transmit = _transmit(geometry, n_tx=3)
    grid = np.stack([np.linspace(-3e-3, 3e-3, 7), np.zeros(7), np.full(7, 15e-3)], -1)
    for output in ("rms", "time"):
        whole = pressure_field(grid, **transmit, n_ax=N_AX, output=output)
        chunked = pressure_field(grid, **transmit, n_ax=N_AX, output=output, max_chunk_gb=2e-4)
        assert _rel_err(whole, chunked) < 1e-5


def test_focused_transmit_adds_coherently_at_the_focus():
    n_el = 16
    geometry = np.stack([np.linspace(-2.5e-3, 2.5e-3, n_el), np.zeros(n_el), np.zeros(n_el)], -1)
    focus = np.array([0.0, 0.0, 30e-3])
    distances = np.linalg.norm(geometry - focus, axis=-1)
    delays = (distances.max() - distances) / SOUND_SPEED
    focused = _transmit(geometry, n_tx=1, t0_delays=delays[None])
    points = np.stack([focus, focus + [3e-3, 0.0, 0.0]])
    rms = _np(pressure_field(points, **focused, n_ax=N_AX))[0]
    assert rms[0] > 3 * rms[1]

    single = _transmit(geometry, n_tx=n_el, t0_delays=np.tile(delays, (n_el, 1)))
    single["tx_apodizations"] = ops.convert_to_tensor(np.eye(n_el, dtype=np.float32))
    per_element = _np(pressure_field(focus[None], **single, n_ax=N_AX))[:, 0]
    assert abs(per_element.sum() - rms[0]) / rms[0] < 1e-2


def test_soft_baffle_scales_by_cos_of_angle():
    angle = np.deg2rad(35.0)
    point = 0.02 * np.array([[np.sin(angle), 0.0, np.cos(angle)]])
    transmit = _transmit(np.zeros((1, 3)))
    rigid = _np(pressure_field(point, **transmit, rigid_baffle=True))
    soft = _np(pressure_field(point, **transmit, rigid_baffle=False))
    # Obliquity once: the field is one way.
    assert _rel_err(np.cos(angle) * rigid, soft) < 1e-4


def test_transducer_bandwidth_enters_one_way():
    transmit = _transmit(np.zeros((1, 3)))
    point = np.array([[0.0, 0.0, 0.02]])
    flat = _np(pressure_field(point, **transmit, n_ax=N_AX, output="time"))[0, :, 0]
    banded = _np(
        pressure_field(
            point,
            **transmit,
            n_ax=N_AX,
            output="time",
            bandwidth_percent=60.0,
            probe_center_frequency=2.5e6,
        )
    )[0, :, 0]
    ratio, keep = _spectrum_ratio(banded, flat)
    freqs = np.fft.rfftfreq(N_AX, 1 / SAMPLING_FREQUENCY)[keep]
    transfer = _np(transducer_transfer(freqs, 2.5e6, 60.0, CENTER_FREQUENCY))
    assert np.allclose(ratio, np.sqrt(transfer), atol=2e-3)


def test_two_dimensional_spreads_cylindrically_in_the_imaging_plane():
    transmit = _transmit(np.zeros((1, 3)), element_height=4e-3)
    points = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.02], [0.0, 3e-3, 0.01]])
    rms = _np(pressure_field(points, **transmit, two_dimensional=True))[0]
    assert abs(rms[0] / rms[1] - np.sqrt(2.0)) < 1e-3
    # Off the plane a point sees the field of its projection onto it.
    assert abs(rms[2] / rms[0] - 1.0) < 1e-6
    spherical = _np(pressure_field(points[:2], **transmit))[0]
    assert abs(spherical[0] / spherical[1] - 2.0) < 1e-3


def test_lens_spreads_the_field_as_a_ray_tube_and_needs_its_sound_speed():
    thickness, c_lens, z = 1e-3, 1000.0, 0.02
    point = np.array([[0.0, 0.0, z]])
    transmit = _transmit(np.zeros((1, 3)))
    plain = _np(pressure_field(point, **transmit, n_ax=N_AX))[0, 0]
    lensed = _np(
        pressure_field(
            point,
            **transmit,
            n_ax=N_AX,
            apply_lens_correction=True,
            lens_thickness=thickness,
            lens_sound_speed=c_lens,
        )
    )[0, 0]
    # On axis the ray tube leaves the face as if from the apparent depth d + (z - d) c / c_lens.
    expected = z / (thickness + (z - thickness) * SOUND_SPEED / c_lens)
    assert abs(lensed / plain / expected - 1.0) < 1e-4
    with pytest.raises(ValueError, match="lens_sound_speed"):
        pressure_field(point, **transmit, n_ax=N_AX, apply_lens_correction=True)


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_under_jit_needs_n_ax_and_n_fft():
    import jax

    transmit = _transmit(np.zeros((1, 3)))
    point = np.array([[0.0, 0.0, 0.02]], np.float32)
    static = ("n_ax", "n_fft", "output", "center_frequency", "sampling_frequency")
    jitted = jax.jit(pressure_field, static_argnames=static)
    with pytest.raises(ValueError, match="n_fft"):
        jitted(point, **transmit)
    reference = pressure_field(point, **transmit, n_ax=N_AX, n_fft=1024)
    assert _rel_err(reference, jitted(point, **transmit, n_ax=N_AX, n_fft=1024)) < 1e-6


def test_element_normals_make_the_field_rotation_invariant():
    n_el = 8
    geometry = np.stack([np.linspace(-1e-3, 1e-3, n_el), np.zeros(n_el), np.zeros(n_el)], -1)
    points = np.array([[4e-3, 0.0, 15e-3], [-2e-3, 1e-3, 20e-3]])
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    normals = np.tile(rotation @ np.array([0.0, 0.0, 1.0]), (n_el, 1))
    reference = pressure_field(points, **_transmit(geometry), n_ax=N_AX, rigid_baffle=False)
    rotated = pressure_field(
        points @ rotation.T,
        **_transmit(geometry @ rotation.T),
        n_ax=N_AX,
        rigid_baffle=False,
        element_normals=normals,
    )
    assert _rel_err(reference, rotated) < 1e-4


def test_chirp_field_correlates_with_the_chirp_and_rejects_bad_output():
    transmit = _transmit(np.zeros((1, 3)))
    point = np.array([[0.0, 0.0, 0.01]])
    tone = _np(pressure_field(point, **transmit, n_ax=N_AX, output="time", n_period=16.0))
    chirp = _np(
        pressure_field(point, **transmit, n_ax=N_AX, output="time", n_period=16.0, chirp_sweep=2e6)
    )
    tone, chirp = tone[0, :, 0], chirp[0, :, 0]
    correlation = np.abs(np.dot(tone, chirp)) / np.linalg.norm(tone) / np.linalg.norm(chirp)
    assert correlation < 0.8
    with pytest.raises(ValueError):
        pressure_field(point, **transmit, output="peak")
