"""Analytic checks of the simulator physics options in ``simulate_rf``."""

import numpy as np
import pytest
from keras import ops
from scipy.signal import hilbert
from scipy.special import jv

from zea.probes import create_curved_probe_geometry, curved_probe_normals
from zea.simulator import (
    _element_responses,
    _resolve_sub_elements,
    chirp_spectrum,
    simulate_rf,
    transducer_transfer,
)

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


def _envelope_peak(rf):
    """Peak of the envelope over all samples and elements of an (n_ax, n_el) record."""
    return np.abs(hilbert(rf, axis=0)).max()


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


def test_transducer_transfer_flat_for_none_bandwidth():
    freqs = np.linspace(0.0, 6e6, 16)
    np.testing.assert_array_equal(transducer_transfer(freqs, 2.6e6, None, xp=np), 1.0)


@pytest.mark.parametrize("bandwidth_percent", [0.0, -50.0])
def test_transducer_transfer_rejects_non_positive_bandwidth(bandwidth_percent):
    with pytest.raises(ValueError, match="bandwidth_percent must be positive"):
        transducer_transfer(np.linspace(0.0, 6e6, 16), 2.6e6, bandwidth_percent, xp=np)


def _rotation_about_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@pytest.mark.parametrize("rigid_baffle", [True, False])
def test_element_normals_make_the_scene_rotation_invariant(rigid_baffle):
    n_el = 8
    geometry = np.stack([(np.arange(n_el) - 3.5) * 0.3e-3, np.zeros(n_el), np.zeros(n_el)], -1)
    rng = np.random.default_rng(0)
    positions = np.stack(
        [rng.uniform(-0.01, 0.01, 6), rng.uniform(-1e-3, 1e-3, 6), rng.uniform(0.01, 0.03, 6)], -1
    )
    rotation = _rotation_about_y(np.deg2rad(30.0))
    normals = np.tile(rotation[:, 2], (n_el, 1))

    reference = simulate_rf(**_scene(geometry, positions, rigid_baffle=rigid_baffle))
    rotated = _scene(geometry @ rotation.T, positions @ rotation.T, rigid_baffle=rigid_baffle)
    assert _rel_err(reference, simulate_rf(**rotated)) > 0.05
    assert _rel_err(reference, simulate_rf(**rotated, element_normals=normals)) < 1e-3


def test_curved_probe_normals_point_along_the_arc():
    geometry = create_curved_probe_geometry(16, 0.5e-3, 40e-3)
    normals = curved_probe_normals(geometry)
    angles = (np.arange(16) - 7.5) * 0.5e-3 / 40e-3
    expected = np.stack([np.sin(angles), np.zeros(16), np.cos(angles)], -1)
    np.testing.assert_allclose(normals, expected, atol=1e-6)
    np.testing.assert_allclose(curved_probe_normals(geometry, radius=40e-3), expected, atol=1e-6)


def _correlation(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return a @ b / np.sqrt((a @ a) * (b @ b))


def _half_max_width(rf):
    spectrum = np.abs(np.fft.rfft(rf))
    return np.count_nonzero(spectrum > 0.5 * spectrum.max())


def test_chirp_excitation():
    depth, sweep, n_period = 0.02, 2e6, 16
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, depth], n_period=n_period)
    tone = _np(simulate_rf(**scene))[0, :, 0, 0]
    chirp = _np(simulate_rf(**scene, chirp_sweep=sweep))[0, :, 0, 0]

    # The echo is the chirp waveform delayed by the round trip.
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLING_FREQUENCY)
    delay = np.exp(-2j * np.pi * freqs * 2 * depth / SOUND_SPEED)
    spectrum = chirp_spectrum(n_fft, CENTER_FREQUENCY, SAMPLING_FREQUENCY, n_period, sweep, xp=np)
    expected = np.fft.irfft(spectrum * delay, n_fft)[:N_AX]
    assert _correlation(chirp, expected) > 0.999
    assert _correlation(tone, expected) < 0.8
    assert _half_max_width(chirp) > 2 * _half_max_width(tone)


def test_n_period_sets_the_pulse_length():
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 0.02])
    short = _np(simulate_rf(**scene, n_period=4))[0, :, 0, 0]
    long = _np(simulate_rf(**scene, n_period=12))[0, :, 0, 0]
    support = lambda rf: np.count_nonzero(np.abs(rf) > 1e-2 * np.abs(rf).max())  # noqa: E731
    assert 2.5 * support(short) < support(long) < 3.5 * support(short)


def test_chirp_spectrum_without_sweep_is_the_windowed_tone():
    n_fft = 1024
    spectrum = chirp_spectrum(n_fft, CENTER_FREQUENCY, SAMPLING_FREQUENCY, 4, 0.0)
    waveform = np.fft.irfft(_np(spectrum), n_fft)
    t = np.fft.fftfreq(n_fft, SAMPLING_FREQUENCY / n_fft)
    width = 4 / CENTER_FREQUENCY
    tone = np.where(np.abs(t) < width / 2, np.cos(np.pi * t / width) ** 2, 0) * np.cos(
        2 * np.pi * CENTER_FREQUENCY * t
    )
    np.testing.assert_allclose(waveform, tone, atol=1e-5)
    assert np.isclose(waveform.max(), 1.0)


def test_sub_elements_reproduce_the_sinc_in_the_far_field():
    # The coherent sum of sub-elements tends to the sinc directivity of the whole element: the
    # amplitude spectra agree as 1 / r^2 into the far field of a 2 mm element (w^2 / lambda is
    # 8 mm). The waveforms keep a small Fresnel delay, the mean path being longer than the
    # centre path, so they are not compared directly.
    angle = np.deg2rad(10.0)

    def amplitude_error(r, n_sub):
        scatterer = r * np.array([np.sin(angle), 0.0, np.cos(angle)])
        scene = _scene(np.zeros((1, 3)), scatterer, element_width=2e-3, n_ax=2048)
        whole = np.abs(np.fft.rfft(_np(simulate_rf(**scene))[0, :, 0, 0]))
        divided = np.abs(np.fft.rfft(_np(simulate_rf(**scene, n_sub_elements=n_sub))[0, :, 0, 0]))
        return _rel_err(whole, divided)

    near, far = amplitude_error(0.02, 8), amplitude_error(0.08, 8)
    assert near > 1e-2
    assert far < 2e-3
    assert abs(amplitude_error(0.08, 32) - far) < 2e-4


def test_sub_elements_converge_in_the_near_field():
    # Off the axis of a 5 mm tall element at 8 mm depth, the far-field sinc is wrong and the
    # sub-element sum converges as the count doubles.
    scene = _scene(np.zeros((1, 3)), [0.0, 2e-3, 8e-3], element_height=5e-3)
    counts = (1, 4, 8, 16, 32)
    results = [_np(simulate_rf(**scene, n_sub_elements=(1, n))) for n in counts]
    steps = [_rel_err(results[i + 1], results[i]) for i in range(len(counts) - 1)]
    assert steps[0] > 0.1
    assert steps[1] > steps[2] > steps[3]
    assert steps[3] < 1e-2


def test_auto_sub_elements_follow_the_simus_rule():
    lambda_min = SOUND_SPEED / (CENTER_FREQUENCY * 1.4)
    auto = _resolve_sub_elements("auto", None, 1e-3, 5e-3, SOUND_SPEED, CENTER_FREQUENCY, 80.0)
    assert auto == (int(np.ceil(1e-3 / lambda_min)), int(np.ceil(5e-3 / lambda_min)))
    assert _resolve_sub_elements(None, None, 1e-3, 5e-3, SOUND_SPEED, CENTER_FREQUENCY, 80.0) == (
        1,
        1,
    )
    assert _resolve_sub_elements(3, None, 1e-3, 5e-3, SOUND_SPEED, CENTER_FREQUENCY, None) == (3, 1)
    focused = _resolve_sub_elements(None, 0.02, 1e-3, 5e-3, SOUND_SPEED, CENTER_FREQUENCY, None)
    assert focused == (1, int(np.ceil(5e-3 * CENTER_FREQUENCY / SOUND_SPEED)))
    assert _resolve_sub_elements((2, 3), 0.02, 1e-3, 5e-3, SOUND_SPEED, CENTER_FREQUENCY, None) == (
        2,
        3,
    )


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


def test_elevation_focus_adds_the_elevation_sub_elements_in_phase():
    # At the focus every elevation sub-element arrives together, so the echo of a scatterer
    # there is stronger than without the lens.
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 15e-3], element_height=5e-3)
    unfocused = _np(simulate_rf(**scene, n_sub_elements=(1, 12)))
    focused = _np(simulate_rf(**scene, n_sub_elements=(1, 12), elevation_focus=15e-3))
    assert np.abs(focused).max() > 1.5 * np.abs(unfocused).max()
    with pytest.raises(ValueError):
        simulate_rf(**scene, elevation_slab_2d=True, elevation_focus=15e-3)


def test_lens_layer_delays_the_echo_by_its_travel_time():
    # A uniform lens of 1 mm at 1000 m/s adds 2 d (1 / c_lens - 1 / c) to the round trip of an
    # on-axis scatterer, 8.4 samples here.
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 20e-3], lens_sound_speed=1000.0)
    plain = _np(simulate_rf(**scene))[0, :, 0, 0]
    lensed = _np(simulate_rf(**{**scene, "apply_lens_correction": True}))[0, :, 0, 0]
    xcorr = np.correlate(lensed, plain, "full")
    lag = np.argmax(xcorr) - (len(plain) - 1)
    expected = 2 * 1e-3 * (1 / 1000.0 - 1 / SOUND_SPEED) * SAMPLING_FREQUENCY
    assert abs(lag - expected) <= 1
    # The wave leaves the slow lens as if from 0.65 mm below its face, and spreads through the
    # medium 1.54 times faster: the on-axis spreading distance is d + (z - d) c / c_lens.
    ratio = _envelope_peak(lensed) / _envelope_peak(plain)
    expected = (20e-3 / (1e-3 + 19e-3 * SOUND_SPEED / 1000.0)) ** 2
    assert np.isclose(ratio, expected, rtol=0.01)


def _slab_field(rho, z, d, c_lens, c, frequency, n_u=4000, n_t=200, t_max=1.5):
    """Field of a point source under a flat slab of ``d`` at ``c_lens``, by the Sommerfeld integral.

    Plane-wave expansion of the source, each wave transmitted with its own coefficient (equal
    densities), summed over the propagating and the evanescent branch. Exact for the flat slab.
    """
    k1, k2 = 2 * np.pi * frequency / c_lens, 2 * np.pi * frequency / c
    u = (np.arange(n_u) + 0.5) * (np.pi / 2) / n_u
    t = (np.arange(n_t) + 0.5) * t_max / n_t
    kr = np.concatenate([k1 * np.sin(u), k1 * np.cosh(t)])
    kz1 = np.concatenate([k1 * np.cos(u), 1j * k1 * np.sinh(t)])
    weight = np.concatenate(
        [k1 * np.sin(u) * (np.pi / 2) / n_u, k1 * np.cosh(t) / 1j * t_max / n_t]
    )
    kz2 = np.sqrt(k2**2 - kr**2 + 0j)
    kz2 = np.where(kz2.imag < 0, -kz2, kz2)
    transmission = 2 * kz1 / (kz1 + kz2)
    integrand = weight * transmission * np.exp(1j * kz1 * d)
    bessel = jv(0, np.outer(rho, kr))
    return 1j * np.sum(bessel * integrand * np.exp(1j * kz2 * (z[:, None] - d)), axis=-1)


def test_lens_spreading_matches_the_sommerfeld_slab():
    # One element under a flat 1 mm lens at 1000 m/s, one frequency: the magnitude of the
    # sub-element sum against the exact slab solution, across elevation and along the axis. The
    # phase-path distance 1/(lens_len c / c_lens + medium_len) as the spread is off by 0.10 here.
    height, thickness, c_lens, n_sub = 5e-3, 1e-3, 1000.0, 100
    y = np.concatenate([np.linspace(-6e-3, 6e-3, 13), np.zeros(4)])
    z = np.concatenate([np.full(13, 20e-3), [5e-3, 10e-3, 30e-3, 40e-3]])
    positions = np.stack([np.zeros_like(y), y, z], -1).astype(np.float32)
    _, rx, _ = _element_responses(
        ops.convert_to_tensor(positions),
        ops.convert_to_tensor(np.zeros((1, 3), np.float32)),
        ops.convert_to_tensor(np.array([CENTER_FREQUENCY], np.float32)),
        SOUND_SPEED,
        0.1e-3,
        height,
        0.0,
        thickness,
        c_lens,
        True,
        False,
        True,
        None,
        n_sub_elements=(1, n_sub),
    )
    simulated = np.abs(_np(rx)[:, 0, 0])
    offsets = (np.arange(n_sub) - (n_sub - 1) / 2) * height / n_sub
    reference = np.abs(
        sum(
            _slab_field(np.abs(y - v), z, thickness, c_lens, SOUND_SPEED, CENTER_FREQUENCY)
            for v in offsets
        )
    )
    assert _rel_err(reference / reference.max(), simulated / simulated.max()) < 0.02


def test_lens_thickness_profile_focuses_like_the_ideal_advance():
    # A slow lens thinned towards the elevation edges focuses at elevation_focus: its gain over
    # the uniform lens at the focus matches the gain of the ideal per-sub-element advance over
    # the plain element. The lens lowers both levels alike, as the wave spreads faster past it.
    focus = 20e-3
    scene = _scene(
        np.zeros((1, 3)),
        [0.0, 0.0, focus],
        element_height=5e-3,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        n_sub_elements=(1, 16),
    )
    peak = lambda **kwargs: _envelope_peak(_np(simulate_rf(**kwargs))[0, :, :, 0])  # noqa: E731
    ideal_gain = peak(**scene, elevation_focus=focus) / peak(**scene)
    lens = {**scene, "apply_lens_correction": True}
    physical, uniform = peak(**lens, elevation_focus=focus), peak(**lens)
    assert np.isclose(physical / uniform, ideal_gain, rtol=0.05)
    assert physical > 1.3 * uniform


def test_lens_attenuation_apodizes_and_lowers_the_centre_frequency():
    # 5 dB/cm/MHz over a 1 mm lens twice costs about 3 dB at 3 MHz and tilts the spectrum down.
    scene = _scene(
        np.zeros((1, 3)),
        [0.0, 0.0, 20e-3],
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        apply_lens_correction=True,
    )
    lossless = _np(simulate_rf(**scene))[0, :, 0, 0]
    lossy = _np(simulate_rf(**scene, lens_attenuation_coef=5.0))[0, :, 0, 0]
    energy_db = 10 * np.log10(np.sum(lossy**2) / np.sum(lossless**2))
    assert -4.0 < energy_db < -2.0

    freqs = np.fft.rfftfreq(len(lossless), 1 / SAMPLING_FREQUENCY)

    def centroid(rf):
        power = np.abs(np.fft.rfft(rf)) ** 2
        return np.sum(freqs * power) / np.sum(power)

    assert centroid(lossy) < centroid(lossless) - 2e4


def test_focusing_lens_must_stay_thicker_than_its_sag():
    scene = _scene(
        np.zeros((1, 3)),
        [0.0, 0.0, 20e-3],
        element_height=5e-3,
        lens_thickness=0.2e-3,
        lens_sound_speed=1000.0,
        apply_lens_correction=True,
    )
    with pytest.raises(ValueError, match="too thin"):
        simulate_rf(**scene, elevation_focus=20e-3)
    with pytest.raises(ValueError, match="cannot focus"):
        simulate_rf(**{**scene, "lens_sound_speed": SOUND_SPEED}, elevation_focus=20e-3)


def test_lens_correction_requires_a_lens_sound_speed():
    scene = _scene(
        np.zeros((1, 3)), [0.0, 0.0, 20e-3], apply_lens_correction=True, lens_sound_speed=None
    )
    with pytest.raises(ValueError, match="lens_sound_speed"):
        simulate_rf(**scene)
