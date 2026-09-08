"""Analytic checks of the simulator physics options in ``simulate_rf``."""

import numpy as np
import pytest
from keras import ops

from zea.simulator import simulate_rf, transducer_transfer

SOUND_SPEED = 1540.0
CENTER_FREQUENCY = 3e6
SAMPLING_FREQUENCY = 12e6
N_AX = 512


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))


def _scene(geometry, positions, magnitudes=None, n_tx=1, **overrides):
    """Unfocused, unapodized transmits, so the response of one scatterer is easy to predict."""
    n_el = len(geometry)
    positions = np.asarray(positions, np.float32).reshape(-1, 3)
    if magnitudes is None:
        magnitudes = np.ones(len(positions), np.float32)
    kwargs = {
        "scatterer_positions": positions,
        "scatterer_magnitudes": np.asarray(magnitudes, np.float32),
        "probe_geometry": np.asarray(geometry, np.float32),
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "sound_speed": SOUND_SPEED,
        "n_ax": N_AX,
        "center_frequency": CENTER_FREQUENCY,
        "sampling_frequency": SAMPLING_FREQUENCY,
        "t0_delays": np.zeros((n_tx, n_el), np.float32),
        "initial_times": np.zeros(n_tx, np.float32),
        "element_width": 0.27e-3,
        "attenuation_coef": 0.0,
        "tx_apodizations": np.ones((n_tx, n_el), np.float32),
        "t_peak": np.zeros(n_tx, np.float32),
        "scatter_exponent": 0.0,
        **overrides,
    }
    return {
        k: ops.convert_to_tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
    }


def _rel_err(reference, result):
    reference, result = _np(reference), _np(result)
    return np.linalg.norm(reference - result) / np.linalg.norm(reference)


def test_soft_baffle_scales_by_cos_of_angle():
    angle = np.deg2rad(35.0)
    scatterer = 0.02 * np.array([np.sin(angle), 0.0, np.cos(angle)])
    scene = _scene(np.zeros((1, 3)), scatterer)
    rigid = simulate_rf(**scene, rigid_baffle=True)
    soft = simulate_rf(**scene, rigid_baffle=False)
    assert _rel_err(rigid, soft) > 0.1
    # Obliquity on transmit and on receive.
    assert _rel_err(np.cos(angle) ** 2 * _np(rigid), soft) < 1e-3


def test_transducer_bandwidth_shapes_the_spectrum():
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 0.02])
    flat = _np(simulate_rf(**scene))[0, :, 0, 0]
    shaped = _np(simulate_rf(**scene, bandwidth_percent=50.0, probe_center_frequency=2.6e6))[
        0, :, 0, 0
    ]
    freqs = np.fft.rfftfreq(N_AX, 1 / SAMPLING_FREQUENCY)
    spectrum_flat, spectrum_shaped = np.fft.rfft(flat), np.fft.rfft(shaped)
    expected = transducer_transfer(freqs, 2.6e6, 50.0, xp=np)
    in_band = np.abs(spectrum_flat) > 0.05 * np.abs(spectrum_flat).max()
    ratio = spectrum_shaped[in_band] / spectrum_flat[in_band]
    np.testing.assert_allclose(ratio, expected[in_band], atol=2e-3)
    # -6 dB at the band edges, per the definition of the fractional bandwidth.
    assert np.isclose(transducer_transfer(2.6e6 * 1.25, 2.6e6, 50.0, xp=np), 0.5)


def _rayleigh_pattern(directions, width, height, wavelength, distance, n=(21, 201)):
    """One-way pattern of a rectangular face by numerical integration, what the sinc approximates.

    Args:
        directions (array-like): Unit vectors from the element centre, of shape (n_dir, 3).
        width (float): Element width [m], along x.
        height (float): Element height [m], along y.
        wavelength (float): Wavelength [m].
        distance (float): Distance to the field point [m].
        n (tuple): Integration points across the width and the height.

    Returns:
        array-like: Field amplitude at each direction, of shape (n_dir,).
    """
    u = (np.arange(n[0]) - (n[0] - 1) / 2) * width / n[0]
    v = (np.arange(n[1]) - (n[1] - 1) / 2) * height / n[1]
    uu, vv = np.meshgrid(u, v, indexing="ij")
    face = np.stack([uu, vv, np.zeros_like(uu)], -1).reshape(-1, 3)
    r = np.linalg.norm(distance * directions[:, None] - face[None], axis=-1)
    return np.abs(np.mean(np.exp(2j * np.pi / wavelength * r) / r, axis=-1))


@pytest.mark.parametrize("lateral_deg", [0.0, 45.0])
def test_whole_element_directivity_uses_direction_cosines(lateral_deg):
    # The sinc of a rectangular element takes the direction cosines lateral / r and elevation / r.
    # Projected angles arctan2(elevation, axial) squeeze the elevation pattern once the scatterer
    # is off to the side: rel L2 0.34 against the face integral at 45 degrees, against 0.02 here.
    width, height, distance = 0.45e-3, 5e-3, 0.15
    elevation = np.deg2rad(np.linspace(-10.0, 10.0, 25))
    lateral = np.deg2rad(lateral_deg)
    cosines = np.stack([np.sin(lateral) * np.cos(elevation), np.sin(elevation)], -1)
    directions = np.concatenate(
        [cosines, np.sqrt(1 - np.sum(cosines**2, -1, keepdims=True))], axis=-1
    )
    # One element per direction, all at the same distance from the one scatterer, and only the
    # element facing it straight on transmits, so each trace holds one factor of the pattern.
    scatterer = np.array([[0.0, 0.0, distance]])
    on_axis = len(directions) // 2
    apodization = np.zeros((1, len(directions)), np.float32)
    apodization[0, on_axis] = 1.0
    scene = _scene(
        scatterer - distance * directions,
        scatterer,
        element_width=width,
        element_height=height,
        n_ax=4096,
        tx_apodizations=apodization,
    )
    rf = _np(simulate_rf(**scene))[0, :, :, 0]
    spectrum = np.abs(np.fft.rfft(rf, axis=0))
    simulated = spectrum[int(round(CENTER_FREQUENCY / SAMPLING_FREQUENCY * rf.shape[0]))]
    reference = _rayleigh_pattern(
        directions, width, height, SOUND_SPEED / CENTER_FREQUENCY, distance
    )
    assert _rel_err(reference / reference[on_axis], simulated / simulated[on_axis]) < 0.05
