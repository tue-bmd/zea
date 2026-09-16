"""Analytic checks of the simulator physics options in ``simulate_rf``."""

import numpy as np
import pytest
from keras import ops
from scipy.signal import hilbert
from scipy.special import jv

from zea.ops import Simulate
from zea.probes import create_curved_probe_geometry, curved_probe_normals
from zea.simulator import (
    _element_model,
    _element_responses,
    _resolve_sub_elements,
    butterworth_transfer,
    gaussian_transfer,
    generalized_normal_transfer,
    measured_pulse,
    obliquity_factor,
    simulate_rf,
    square_burst_pulses,
    square_burst_spectrum,
    transmit_pulse,
    transmit_pulses,
)
from zea.simulator_time_domain import simulate_rf_td

from .simulator_helpers import (
    CENTER_FREQUENCY,
    N_AX,
    SAMPLING_FREQUENCY,
    SOUND_SPEED,
    correlation,
    rel_err,
    stack_padded,
    to_np,
)


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


def _waveform(model, **kwargs):
    """A parametric pulse as the simulators take it: its two-way waveform, sampled at 250 MHz."""
    pulse = transmit_pulse(CENTER_FREQUENCY, pulse_model=model, **kwargs)
    return {"waveforms_two_way": pulse.waveform()}


def _envelope_peak(rf):
    """Peak of the envelope over all samples and elements of an (n_ax, n_el) record."""
    return np.abs(hilbert(rf, axis=0)).max()


@pytest.mark.parametrize("ratio", [float("inf"), 0.57])
def test_baffle_obliquity_scales_by_cos_of_angle(ratio):
    angle = np.deg2rad(35.0)
    scatterer = 0.02 * np.array([np.sin(angle), 0.0, np.cos(angle)])
    scene = _scene(np.zeros((1, 3)), scatterer)
    rigid = simulate_rf(**scene)
    baffled = simulate_rf(**scene, baffle_impedance_ratio=ratio)
    assert rel_err(rigid, baffled) > 0.1
    # Obliquity on transmit and on receive: cos in a soft baffle, cos / (cos + ratio) in general.
    cos = np.cos(angle)
    factor = cos if ratio == float("inf") else cos / (cos + ratio)
    assert rel_err(factor**2 * to_np(rigid), baffled) < 1e-3
    with pytest.raises(ValueError, match="baffle_impedance_ratio"):
        simulate_rf(**scene, baffle_impedance_ratio=-1.0)


def test_obliquity_is_zero_behind_the_element():
    cos_angle = np.array([[0.3, 0.0, -0.5]])
    for ratio in (float("inf"), 0.57):
        factor = np.asarray(obliquity_factor(cos_angle, ratio))
        assert factor[0, 0] > 0.0
        assert factor[0, 1] == 0.0
        assert factor[0, 2] == 0.0
    assert np.all(np.asarray(obliquity_factor(cos_angle, 0.0)) == 1.0)


def test_distances_are_clamped_at_half_a_wavelength():
    # A scatterer on an element: SIMUS replaces distances below lambda / 2 by lambda / 2 for the
    # phase and the 1 / r, so the record is that of a scatterer at lambda / 2, and beyond it the
    # two-way amplitude falls as 1 / r^2 (on axis, so the directivity is 1).
    half_wavelength = SOUND_SPEED / (2 * CENTER_FREQUENCY)
    depths = [0.1 * half_wavelength, half_wavelength, 4 * half_wavelength]
    scene = _scene(
        np.zeros((1, 3)), [[0.0, 0.0, z] for z in depths], t_peak=np.full(1, 1e-6, np.float32)
    )
    for simulate in (simulate_rf, simulate_rf_td):
        records = []
        for k in range(len(depths)):
            one = {**scene, "scatterer_magnitudes": np.eye(len(depths), dtype=np.float32)[k]}
            records.append(to_np(simulate(**one))[0, :, 0, 0])
        assert rel_err(records[1], records[0]) < 1e-5
        ratio = _envelope_peak(records[2]) / _envelope_peak(records[1])
        assert np.isclose(ratio, 1 / 16, rtol=0.03)


def test_transducer_bandwidth_shapes_the_spectrum():
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 0.02])
    flat = to_np(simulate_rf(**scene, **_waveform("hann", bandwidth_percent=None)))[0, :, 0, 0]
    shaped = _waveform("hann", bandwidth_percent=50.0, probe_center_frequency=2.6e6)
    shaped = to_np(simulate_rf(**scene, **shaped))[0, :, 0, 0]
    freqs = np.fft.rfftfreq(N_AX, 1 / SAMPLING_FREQUENCY)
    spectrum_flat, spectrum_shaped = np.fft.rfft(flat), np.fft.rfft(shaped)
    expected = gaussian_transfer(freqs, 2.6e6, 50.0)
    in_band = np.abs(spectrum_flat) > 0.05 * np.abs(spectrum_flat).max()
    ratio = spectrum_shaped[in_band] / spectrum_flat[in_band]
    ratio = ratio / ratio[np.argmax(expected[in_band])]  # both pulses have a unit peak
    np.testing.assert_allclose(ratio, expected[in_band], atol=2e-3)


def _two_way(transfer, f, fc, bandwidth):
    return transfer(f, fc, bandwidth) ** (2 if transfer is butterworth_transfer else 1)


@pytest.mark.parametrize(
    "transfer", [gaussian_transfer, generalized_normal_transfer, butterworth_transfer]
)
def test_transducer_transfer_is_6db_down_at_the_band_edges(transfer):
    edges = 2.6e6 * np.array([0.75, 1.25])
    np.testing.assert_allclose(np.abs(_two_way(transfer, edges, 2.6e6, 50.0)), 0.5, atol=1e-6)
    assert np.abs(_two_way(transfer, 2.6e6 * 0.99, 2.6e6, 50.0)) > 0.95
    with pytest.raises(ValueError, match="bandwidth_percent must be positive"):
        transfer(edges, 2.6e6, 0.0)


def test_butterworth_transfer_is_causal():
    n, fs = 4096, 48e6
    freqs = np.fft.rfftfreq(n, 1 / fs)
    impulse_response = np.fft.irfft(butterworth_transfer(freqs, 3e6, 60.0) ** 2, n)
    energy = impulse_response**2
    assert energy[n // 2 :].sum() < 1e-6 * energy.sum()
    assert energy[: n // 2].argmax() > 0  # rise, then ringdown


def test_hann_transfer_is_flat_for_none_bandwidth():
    np.testing.assert_array_equal(gaussian_transfer(np.linspace(0.0, 6e6, 16), 2.6e6, None), 1.0)


def _rotation_about_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@pytest.mark.parametrize("baffle", [0.0, float("inf")])
def test_element_normals_make_the_scene_rotation_invariant(baffle):
    n_el = 8
    geometry = np.stack([(np.arange(n_el) - 3.5) * 0.3e-3, np.zeros(n_el), np.zeros(n_el)], -1)
    rng = np.random.default_rng(0)
    positions = np.stack(
        [rng.uniform(-0.01, 0.01, 6), rng.uniform(-1e-3, 1e-3, 6), rng.uniform(0.01, 0.03, 6)], -1
    )
    rotation = _rotation_about_y(np.deg2rad(30.0))
    normals = np.tile(rotation[:, 2], (n_el, 1))

    reference = simulate_rf(**_scene(geometry, positions, baffle_impedance_ratio=baffle))
    rotated = _scene(geometry @ rotation.T, positions @ rotation.T, baffle_impedance_ratio=baffle)
    assert rel_err(reference, simulate_rf(**rotated)) > 0.05
    assert rel_err(reference, simulate_rf(**rotated, element_normals=normals)) < 1e-3


def test_curved_probe_normals_point_along_the_arc():
    geometry = create_curved_probe_geometry(16, 0.5e-3, 40e-3)
    normals = curved_probe_normals(geometry)
    angles = (np.arange(16) - 7.5) * 0.5e-3 / 40e-3
    expected = np.stack([np.sin(angles), np.zeros(16), np.cos(angles)], -1)
    np.testing.assert_allclose(normals, expected, atol=1e-6)
    np.testing.assert_allclose(curved_probe_normals(geometry, radius=40e-3), expected, atol=1e-6)


def _half_max_width(rf):
    spectrum = np.abs(np.fft.rfft(rf))
    return np.count_nonzero(spectrum > 0.5 * spectrum.max())


def test_chirp_excitation():
    depth, sweep, n_period = 0.02, 2e6, 16
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, depth])
    kwargs = dict(n_period=n_period, bandwidth_percent=None)
    tone = to_np(simulate_rf(**scene, **_waveform("hann", **kwargs)))[0, :, 0, 0]
    chirp = _waveform("hann", chirp_sweep=sweep, **kwargs)
    chirp = to_np(simulate_rf(**scene, **chirp))[0, :, 0, 0]

    # The echo is the chirp waveform delayed by the round trip.
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLING_FREQUENCY)
    delay = np.exp(-2j * np.pi * freqs * 2 * depth / SOUND_SPEED)
    pulse = transmit_pulse(
        CENTER_FREQUENCY, SAMPLING_FREQUENCY, "hann", n_period, sweep, bandwidth_percent=None
    )
    expected = np.fft.irfft(pulse.spectrum(freqs) * delay, n_fft)[:N_AX]
    assert correlation(chirp, expected) > 0.999
    assert correlation(tone, expected) < 0.8
    assert _half_max_width(chirp) > 2 * _half_max_width(tone)


def _instantaneous_frequency(pulse):
    analytic = hilbert(pulse.waveform())
    frequency = np.gradient(np.unwrap(np.angle(analytic))) * pulse.sampling_frequency / (2 * np.pi)
    inside = np.flatnonzero(np.abs(analytic) > 0.5 * np.abs(analytic).max())
    return frequency[inside[0]], frequency[inside[-1]]


@pytest.mark.parametrize("pulse_model", ["hann", "simus"])
def test_negative_chirp_sweep_runs_down(pulse_model):
    bandwidth = None if pulse_model == "hann" else 100.0
    kwargs = dict(n_period=8, bandwidth_percent=bandwidth)
    up = transmit_pulse(CENTER_FREQUENCY, 48e6, pulse_model, chirp_sweep=1.5e6, **kwargs)
    down = transmit_pulse(CENTER_FREQUENCY, 48e6, pulse_model, chirp_sweep=-1.5e6, **kwargs)
    start, end = _instantaneous_frequency(up)
    assert end - start > 0.5e6
    start, end = _instantaneous_frequency(down)
    assert start - end > 0.5e6


def test_n_period_sets_the_pulse_length():
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 0.02])
    short, long = (
        to_np(simulate_rf(**scene, **_waveform("hann", n_period=n, bandwidth_percent=None)))[
            0, :, 0, 0
        ]
        for n in (4, 12)
    )
    support = lambda rf: np.count_nonzero(np.abs(rf) > 1e-2 * np.abs(rf).max())  # noqa: E731
    assert 2.5 * support(short) < support(long) < 3.5 * support(short)


def _pulse(model, **kwargs):
    kwargs.setdefault("bandwidth_percent", 75.0)
    return transmit_pulse(CENTER_FREQUENCY, SAMPLING_FREQUENCY, model, **kwargs)


def test_pulse_models_have_a_unit_peak_on_the_middle_sample():
    for pulse in (_pulse("realistic"), _pulse("hann"), _pulse("simus", n_period=1.0)):
        waveform = pulse.waveform()
        assert len(waveform) == pulse.n_samples == 2 * max(pulse.n_before, pulse.n_after) + 1
        # Normalised on a 16x oversampled grid; at 4 samples per period the sampled peak of
        # the short realistic pulse falls between samples (~0.94), never above 1.
        assert 0.9 < np.abs(waveform).max() <= 1.0 + 1e-3
        envelope = np.abs(hilbert(waveform))
        assert abs(np.argmax(envelope) - pulse.n_samples // 2) <= 1
        # From the trigger, as a Verasonics waveform: the peak is time_to_peak after sample 0.
        from_trigger = pulse.waveform(from_trigger=True)
        n_before = int(round(pulse.time_to_peak * pulse.sampling_frequency))
        assert len(from_trigger) == n_before + pulse.n_after + 1
        # The same samples from the peak on (each is built on its own FFT length).
        np.testing.assert_allclose(
            from_trigger[n_before:], waveform[pulse.n_samples // 2 :], atol=1e-3
        )
        assert abs(np.argmax(np.abs(hilbert(from_trigger))) - n_before) <= 1


def test_realistic_pulse_rings_down_within_the_band():
    pulse = _pulse("realistic", n_period=0.5)
    assert 0 < pulse.n_before < pulse.n_after  # fast rise, long ringdown
    # Causal: the response starts with the excitation, so the peak is within the support
    # before it (which starts a little earlier, at the -80 dB pre-ringing of the burst edges).
    assert 0 < pulse.time_to_peak <= pulse.n_before / SAMPLING_FREQUENCY
    freqs = np.fft.rfftfreq(8192, 1 / 48e6)
    spectrum = np.abs(transmit_pulse(CENTER_FREQUENCY, 48e6, "realistic", 0.5).spectrum(freqs))
    band = freqs[spectrum > 0.5 * spectrum.max()]
    assert np.isclose((band[-1] - band[0]) / CENTER_FREQUENCY, 0.70, atol=0.05)
    third = (freqs > 2.8 * CENTER_FREQUENCY) & (freqs < 3.2 * CENTER_FREQUENCY)
    assert spectrum[third].max() < 1e-3 * spectrum.max()


def test_equalisation_pulses_follow_the_vantage_pattern():
    # A Verasonics L11-5v transmit, TW.Parameters = [7.8125 MHz, 0.6875, 2 half-cycles], has
    # the tri-level States [-1 6; 0 8; 1 11; 0 5; -1 11; 0 8; 1 6] at 250 MHz: half-width
    # equalisation pulses of opposite sign a quarter period (8 samples) before and after the
    # burst. Vantage rounds the 5.5-sample equalisation width up to 6.
    fs, fc, duty = 250e6, 7.8125e6, 0.6875
    centres, widths, signs = square_burst_pulses(fc, 1.0, duty, equalize=True)
    starts, ends = (centres - widths / 2) * fs, (centres + widths / 2) * fs
    assert np.allclose(ends - starts, [5.5, 11, 11, 5.5])
    assert np.allclose(starts[1:] - ends[:-1], [8, 5, 8])
    assert np.array_equal(signs, [-1, 1, -1, 1])
    plain = square_burst_pulses(fc, 1.0, duty)
    assert np.allclose(plain[0], centres[1:-1]) and np.allclose(plain[1], widths[1:-1])
    # They make the drive zero-mean: a single half-cycle has a DC component, equalised none.
    dc = np.zeros(1)
    assert square_burst_spectrum(dc, fc, 0.5, duty)[0] != 0
    assert np.isclose(square_burst_spectrum(dc, fc, 0.5, duty, equalize=True)[0], 0)


def test_equalisation_pulses_narrow_the_realistic_pulse():
    # With a 2-half-cycle burst the transducer limits the bandwidth (68 % for 77 %, as the
    # L11-5v); the equalisation pulses lengthen the burst and narrow the pulse to 56 %, the
    # width of the Vantage two-way waveform, and delay its peak by about half a period.
    fc, bandwidth = 7.6e6, 76.8
    freqs = np.fft.rfftfreq(1 << 14, 1 / 250e6)
    pulses = {}
    for equalize in (False, True):
        pulse = transmit_pulse(
            fc, 250e6, "realistic", bandwidth_percent=bandwidth, equalize=equalize
        )
        spectrum = np.abs(pulse.spectrum(freqs))
        band = freqs[spectrum > 0.5 * spectrum.max()]
        pulses[equalize] = pulse, (band[-1] - band[0]) / fc
    assert np.isclose(pulses[False][1], 0.68, atol=0.02)
    assert np.isclose(pulses[True][1], 0.56, atol=0.02)
    delay = pulses[True][0].time_to_peak - pulses[False][0].time_to_peak
    assert np.isclose(delay * fc, 0.5, atol=0.1)


def test_measured_pulse_reproduces_its_source():
    # A waveform sampled at 250 MHz, as the waveforms_two_way of a zea file, on a 12 MHz grid.
    fine = transmit_pulse(CENTER_FREQUENCY, 250e6, "realistic", bandwidth_percent=75.0)
    measured = measured_pulse(fine.waveform(), SAMPLING_FREQUENCY, 250e6)
    freqs = np.fft.rfftfreq(2048, 1 / SAMPLING_FREQUENCY)
    source, copy = (np.fft.irfft(p.spectrum(freqs), 2048) for p in (_pulse("realistic"), measured))
    assert rel_err(source, copy) < 5e-3
    # The peak of the supplied waveform is on its middle sample.
    assert np.isclose(measured.time_to_peak, (fine.n_samples // 2) / 250e6, atol=1 / 250e6)
    with pytest.raises(ValueError, match="waveform_two_way"):
        measured_pulse(np.zeros(64), SAMPLING_FREQUENCY)
    with pytest.raises(ValueError, match="pulse_model"):
        _pulse("gaussian")


def test_transmit_pulses_take_one_waveform_per_transmit():
    waveforms = stack_padded(_pulse("realistic").waveform(), _pulse("hann").waveform())
    # Without waveforms every transmit gets the default pulse of transmit_pulse.
    default = transmit_pulses(2, CENTER_FREQUENCY, SAMPLING_FREQUENCY)
    assert default[0] is default[1]
    expected = transmit_pulse(CENTER_FREQUENCY, SAMPLING_FREQUENCY).waveform()
    assert np.array_equal(default[0].waveform(), expected)
    # Identical rows share a pulse, one row serves every transmit.
    kwargs = dict(waveform_sampling_frequency=SAMPLING_FREQUENCY)
    same = transmit_pulses(2, CENTER_FREQUENCY, SAMPLING_FREQUENCY, waveforms[:1], **kwargs)
    shared = transmit_pulses(2, CENTER_FREQUENCY, SAMPLING_FREQUENCY, waveforms[0], **kwargs)
    per_tx = transmit_pulses(2, CENTER_FREQUENCY, SAMPLING_FREQUENCY, waveforms, **kwargs)
    assert same[0] is same[1] and shared[0] is shared[1]
    for pulse in (same[0], shared[1], per_tx[0]):
        assert np.array_equal(pulse.waveform(), same[0].waveform())
    assert not np.array_equal(per_tx[1].waveform(), per_tx[0].waveform())
    with pytest.raises(ValueError, match="waveforms_two_way"):
        transmit_pulses(3, CENTER_FREQUENCY, SAMPLING_FREQUENCY, waveforms)


@pytest.mark.parametrize("pulse_model", ["realistic", "hann", "rf_rate"])
def test_pulse_peak_arrives_at_the_travel_time(pulse_model):
    depth = 0.02
    if pulse_model == "rf_rate":
        # One row per transmit, sampled at the RF rate; the waveform peaks on its middle sample.
        kwargs = dict(waveforms_two_way=np.stack([_pulse("realistic").waveform()] * 2))
        kwargs["waveform_sampling_frequency"] = SAMPLING_FREQUENCY
    else:
        # Two periods: a one-period Hann burst at 4 samples per period reaches Nyquist, where
        # the frequency- and time-domain simulators differ slightly.
        kwargs = _waveform(pulse_model, bandwidth_percent=75.0, n_period=2.0)
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, depth], n_tx=2, **kwargs)
    rf = to_np(simulate_rf(**scene))[0, :, 0, 0]
    peak = np.argmax(np.abs(hilbert(rf)))
    assert abs(peak - 2 * depth / SOUND_SPEED * SAMPLING_FREQUENCY) <= 1
    fast = to_np(simulate_rf_td(**scene))[0, :, 0, 0]
    assert correlation(rf, fast) > 0.99


def test_waveforms_are_placed_by_their_envelope_peak():
    depth = 0.02
    fine = transmit_pulse(CENTER_FREQUENCY, 250e6, "hann", n_period=2.0, bandwidth_percent=60.0)
    waveform = fine.waveform()
    pulse = measured_pulse(waveform, SAMPLING_FREQUENCY)
    # Different waveform per transmit, the second one delayed in the record like a real
    # system's, with t_peak from its envelope peak as Parameters.t_peak derives it: sample 0
    # of each waveform lands at the travel time, so its onset (1 % of the peak) does too.
    delayed = np.concatenate([np.zeros(100), waveform])
    waveforms = np.stack([np.concatenate([waveform, np.zeros(100)]), delayed])
    t_peak = np.array([pulse.time_to_peak, pulse.time_to_peak + 100 / 250e6], np.float32)
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, depth], n_tx=2, t_peak=t_peak)
    onset = np.flatnonzero(np.abs(waveform) > 1e-2 * np.abs(waveform).max())[0] / 250e6
    expected_onset = (2 * depth / SOUND_SPEED + onset) * SAMPLING_FREQUENCY
    for simulate in (simulate_rf, simulate_rf_td):
        rf = to_np(simulate(**scene, waveforms_two_way=waveforms))[:, :, 0, 0]
        onsets = [np.flatnonzero(np.abs(r) > 1e-2 * np.abs(r).max())[0] for r in rf]
        assert abs(onsets[0] - expected_onset) <= 1.5
        assert abs(onsets[1] - onsets[0] - 100 / 250e6 * SAMPLING_FREQUENCY) <= 1
        # The waveform sets the spectrum: hann at 60 % here, not the default pulse.
        spectrum = np.abs(np.fft.rfft(rf[0], 4096))
        freqs = np.fft.rfftfreq(4096, 1 / SAMPLING_FREQUENCY)
        assert correlation(spectrum, np.abs(pulse.spectrum(freqs))) > 0.99
    supplied = simulate_rf(**scene, waveforms_two_way=waveforms)
    assert rel_err(simulate_rf(**scene), supplied) > 0.1


def test_simulate_op_passes_the_waveforms():
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 0.02], n_tx=2)
    waveforms = stack_padded(_pulse("realistic").waveform(), _pulse("hann").waveform())
    kwargs = dict(waveforms_two_way=waveforms, waveform_sampling_frequency=SAMPLING_FREQUENCY)
    direct = simulate_rf(**scene, **kwargs)
    op = Simulate(with_batch_dim=False)
    via_op = op(**scene, **kwargs)[op.output_key]
    # The op sizes the FFT without the scatterer positions, so the lengths (and the float32
    # rounding) differ.
    assert rel_err(direct, via_op) < 1e-4
    # A pulse built with transmit_pulse goes in the same way, at its 250 MHz default rate.
    equalized = _waveform("realistic", equalize=True)
    direct = simulate_rf(**scene, **equalized)
    assert rel_err(direct, op(**scene, **equalized)[op.output_key]) < 1e-4
    assert rel_err(direct, simulate_rf(**scene)) > 0.1
    with pytest.raises(ValueError, match="waveforms_two_way"):
        op(**scene, waveforms_two_way=np.ones((3, 64)))


def test_sub_elements_reproduce_the_sinc_in_the_far_field():
    # The coherent sum of sub-elements tends to the sinc directivity of the whole element: the
    # amplitude spectra agree as 1 / r^2 into the far field of a 2 mm element (w^2 / lambda is
    # 8 mm). The waveforms keep a small Fresnel delay, the mean path being longer than the
    # centre path, so they are not compared directly.
    angle = np.deg2rad(10.0)

    def amplitude_error(r, n_sub):
        scatterer = r * np.array([np.sin(angle), 0.0, np.cos(angle)])
        scene = _scene(np.zeros((1, 3)), scatterer, element_width=2e-3, n_ax=2048)
        whole = np.abs(np.fft.rfft(to_np(simulate_rf(**scene))[0, :, 0, 0]))
        divided = np.abs(np.fft.rfft(to_np(simulate_rf(**scene, n_sub_elements=n_sub))[0, :, 0, 0]))
        return rel_err(whole, divided)

    near, far = amplitude_error(0.02, 8), amplitude_error(0.08, 8)
    assert near > 1e-2
    assert far < 2e-3
    assert abs(amplitude_error(0.08, 32) - far) < 2e-4


def test_sub_elements_converge_in_the_near_field():
    # Off the axis of a 5 mm tall element at 8 mm depth, the far-field sinc is wrong and the
    # sub-element sum converges as the count doubles.
    scene = _scene(np.zeros((1, 3)), [0.0, 2e-3, 8e-3], element_height=5e-3)
    counts = (1, 4, 8, 16, 32)
    results = [to_np(simulate_rf(**scene, n_sub_elements=(1, n))) for n in counts]
    steps = [rel_err(results[i + 1], results[i]) for i in range(len(counts) - 1)]
    assert steps[0] > 0.1
    assert steps[1] > steps[2] > steps[3]
    assert steps[3] < 1e-2


def test_auto_sub_elements_follow_the_simus_rule():
    # lambda_min at the top of the -6 dB band of the pulse: for the default one-cycle burst
    # through a 70 % transducer about 1.28 fc, a little below the transducer's 1.35 fc that
    # SIMUS uses, as the burst narrows the band.
    band_top = transmit_pulse(CENTER_FREQUENCY).band[1]
    assert 1.2 * CENTER_FREQUENCY < band_top < 1.35 * CENTER_FREQUENCY
    lambda_min = SOUND_SPEED / band_top
    auto = _resolve_sub_elements("auto", None, 1e-3, 5e-3, SOUND_SPEED, band_top)
    assert auto == (int(np.ceil(1e-3 / lambda_min)), int(np.ceil(5e-3 / lambda_min)))
    assert _resolve_sub_elements(None, None, 1e-3, 5e-3, SOUND_SPEED, band_top) == (1, 1)
    assert _resolve_sub_elements(3, None, 1e-3, 5e-3, SOUND_SPEED, band_top) == (3, 1)
    focused = _resolve_sub_elements(None, 0.02, 1e-3, 5e-3, SOUND_SPEED, band_top)
    assert focused == (1, int(np.ceil(5e-3 / lambda_min)))
    assert _resolve_sub_elements((2, 3), 0.02, 1e-3, 5e-3, SOUND_SPEED, band_top) == (2, 3)


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
    rf = to_np(simulate_rf(**scene))[0, :, :, 0]
    spectrum = np.abs(np.fft.rfft(rf, axis=0))
    simulated = spectrum[int(round(CENTER_FREQUENCY / SAMPLING_FREQUENCY * rf.shape[0]))]
    reference = _rayleigh_pattern(
        directions, width, height, SOUND_SPEED / CENTER_FREQUENCY, distance
    )
    assert rel_err(reference / reference[on_axis], simulated / simulated[on_axis]) < 0.05


def test_elevation_focus_adds_the_elevation_sub_elements_in_phase():
    # At the focus every elevation sub-element arrives together, so the echo of a scatterer
    # there is stronger than without the lens.
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 15e-3], element_height=5e-3)
    unfocused = to_np(simulate_rf(**scene, n_sub_elements=(1, 12)))
    focused = to_np(simulate_rf(**scene, n_sub_elements=(1, 12), elevation_focus=15e-3))
    assert np.abs(focused).max() > 1.5 * np.abs(unfocused).max()
    with pytest.raises(ValueError):
        simulate_rf(**scene, two_dimensional=True, elevation_focus=15e-3)


def test_two_dimensional_spreads_the_transmit_cylindrically():
    # One element, scatterers at z and 2z: the two-way spread falls by 4 in 3D and by 2 sqrt 2
    # behind the ideal elevation lens, whose transmit falls as 1 / sqrt(r).
    z = 15e-3
    scene = _scene(np.zeros((1, 3)), [[0.0, 0.0, z], [0.0, 0.0, 2 * z]], n_ax=1024)
    for two_dimensional, expected in ((False, 4.0), (True, 2 * np.sqrt(2.0))):
        peaks = []
        for scatterer in range(2):
            single = {
                **scene,
                "scatterer_magnitudes": ops.convert_to_tensor(
                    np.eye(2, dtype=np.float32)[scatterer]
                ),
            }
            rf = to_np(simulate_rf(**single, two_dimensional=two_dimensional))[0, :, 0, 0]
            peaks.append(_envelope_peak(rf[:, None]))
        assert abs(peaks[0] / peaks[1] - expected) < 1e-2 * expected


@pytest.mark.parametrize("simulator", [simulate_rf, simulate_rf_td], ids=["exact", "fast"])
def test_two_dimensional_moves_scatterers_into_the_imaging_plane(simulator):
    # Off the plane a scatterer echoes as its projection; in 3D it is delayed and less directive.
    geometry = np.stack([np.linspace(-2e-3, 2e-3, 8), np.zeros(8), np.zeros(8)], -1)
    in_plane = _scene(geometry, [1e-3, 0.0, 20e-3], element_height=3e-3)
    off_plane = _scene(geometry, [1e-3, 6e-3, 20e-3], element_height=3e-3)
    reference = to_np(simulator(**in_plane, two_dimensional=True))
    assert np.abs(reference).max() > 0
    assert rel_err(reference, simulator(**off_plane, two_dimensional=True)) < 1e-6
    assert rel_err(reference, simulator(**off_plane)) > 0.1
    matrix = np.stack([geometry, geometry + [0.0, 1e-3, 0.0]]).reshape(-1, 3)
    with pytest.raises(ValueError, match="1D probe"):
        simulator(**_scene(matrix, [0.0, 0.0, 20e-3]), two_dimensional=True)


def test_lens_layer_delays_the_echo_by_its_travel_time():
    # A uniform lens of 1 mm at 1000 m/s adds 2 d (1 / c_lens - 1 / c) to the round trip of an
    # on-axis scatterer, 8.4 samples here.
    scene = _scene(np.zeros((1, 3)), [0.0, 0.0, 20e-3], lens_sound_speed=1000.0)
    plain = to_np(simulate_rf(**scene))[0, :, 0, 0]
    lensed = to_np(simulate_rf(**{**scene, "apply_lens_correction": True}))[0, :, 0, 0]
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
    model = _element_model(
        np.zeros((1, 3), np.float32),
        SOUND_SPEED,
        CENTER_FREQUENCY,
        [transmit_pulse(CENTER_FREQUENCY)],
        element_width=0.1e-3,
        element_height=height,
        attenuation_coef=0.0,
        apply_lens_correction=True,
        lens_thickness=thickness,
        lens_sound_speed=c_lens,
        two_dimensional=False,
        baffle_impedance_ratio=1.0,
        element_normals=None,
        n_sub_elements=(1, n_sub),
        elevation_focus=None,
        lens_attenuation_coef=0.0,
    )
    _, rx, _ = _element_responses(
        ops.convert_to_tensor(positions),
        model,
        ops.convert_to_tensor(np.array([CENTER_FREQUENCY], np.float32)),
    )
    simulated = np.abs(to_np(rx)[0, :, 0])
    offsets = (np.arange(n_sub) - (n_sub - 1) / 2) * height / n_sub
    reference = np.abs(
        sum(
            _slab_field(np.abs(y - v), z, thickness, c_lens, SOUND_SPEED, CENTER_FREQUENCY)
            for v in offsets
        )
    )
    assert rel_err(reference / reference.max(), simulated / simulated.max()) < 0.02


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
    peak = lambda **kwargs: _envelope_peak(to_np(simulate_rf(**kwargs))[0, :, :, 0])  # noqa: E731
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
    lossless = to_np(simulate_rf(**scene))[0, :, 0, 0]
    lossy = to_np(simulate_rf(**scene, lens_attenuation_coef=5.0))[0, :, 0, 0]
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
