"""Tests for the frequency-domain simulator (``zea.simulator``)."""

import numpy as np
import pytest
from keras import ops

from zea.simulator import (
    get_measured_pulse_spectrum_fn,
    get_pulse_spectrum_fn,
    simulate_rf,
)

SAMPLING_FREQ = 20e6  # Hz
CENTER_FREQ = 5e6  # Hz
WAVEFORM_SAMPLING_FREQ = 250e6  # Hz
SOUND_SPEED = 1540.0  # m/s


def _hann_sine_pulse(n_cycles=2.0):
    """A Hann-windowed sine like the waveforms stored with zea scans."""
    n = int(n_cycles / CENTER_FREQ * WAVEFORM_SAMPLING_FREQ)
    t = np.arange(n) / WAVEFORM_SAMPLING_FREQ
    return (np.sin(2 * np.pi * CENTER_FREQ * t) * np.hanning(n)).astype(np.float32)


def test_measured_pulse_spectrum_matches_fft():
    """The DTFT-based spectrum matches numpy's FFT at the FFT bin frequencies."""
    waveform = _hann_sine_pulse()
    n = waveform.size
    spectrum_fn = get_measured_pulse_spectrum_fn(waveform, WAVEFORM_SAMPLING_FREQ)

    bin_freqs = np.fft.rfftfreq(n, d=1.0 / WAVEFORM_SAMPLING_FREQ).astype(np.float32)
    spectrum = ops.convert_to_numpy(spectrum_fn(bin_freqs))
    expected = np.fft.rfft(waveform) / WAVEFORM_SAMPLING_FREQ

    np.testing.assert_allclose(spectrum, expected, atol=1e-9)


def test_measured_pulse_spectrum_peaks_at_center_frequency():
    """The measured-pulse spectrum peaks at the pulse's center frequency."""
    waveform = _hann_sine_pulse()
    spectrum_fn = get_measured_pulse_spectrum_fn(waveform, WAVEFORM_SAMPLING_FREQ)

    freqs = np.linspace(0, 20e6, 401).astype(np.float32)
    magnitude = np.abs(ops.convert_to_numpy(spectrum_fn(freqs)))
    peak_freq = freqs[np.argmax(magnitude)]
    assert abs(peak_freq - CENTER_FREQ) < 0.5e6


@pytest.mark.parametrize("use_measured_waveform", [False, True])
def test_simulate_rf_echo_arrival(use_measured_waveform):
    """A single scatterer's echo envelope peaks at the expected travel time.

    With ``pulse_spectrum_fn=None`` the parametric pulse is zero-phase (peak
    at the travel time); a measured waveform starts at time zero instead, so
    its peak arrives ``t_peak`` later — the same convention as
    :class:`zea.inverse.ScattererSimulator`.
    """
    from scipy.signal import hilbert

    n_el = 8
    depth = 10e-3
    probe_geometry = np.stack(
        [np.linspace(-2e-3, 2e-3, n_el), np.zeros(n_el), np.zeros(n_el)], axis=-1
    ).astype(np.float32)

    waveform = _hann_sine_pulse()
    pulse_spectrum_fn = (
        get_measured_pulse_spectrum_fn(waveform, WAVEFORM_SAMPLING_FREQ)
        if use_measured_waveform
        else None
    )

    rf = simulate_rf(
        scatterer_positions=np.array([[0.0, 0.0, depth]], dtype=np.float32),
        scatterer_magnitudes=np.ones(1, dtype=np.float32),
        probe_geometry=probe_geometry,
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        sound_speed=SOUND_SPEED,
        n_ax=512,
        center_frequency=CENTER_FREQ,
        sampling_frequency=SAMPLING_FREQ,
        t0_delays=np.zeros((1, n_el), dtype=np.float32),
        initial_times=np.zeros(1, dtype=np.float32),
        element_width=0.2e-3,
        attenuation_coef=0.0,
        tx_apodizations=np.ones((1, n_el), dtype=np.float32),
        pulse_spectrum_fn=pulse_spectrum_fn,
    )
    rf = ops.convert_to_numpy(rf)
    assert rf.shape == (1, 512, n_el, 1)

    element = n_el // 2
    travel_time = (depth + np.linalg.norm(probe_geometry[element] - [0, 0, depth])) / SOUND_SPEED
    expected_time = travel_time
    if use_measured_waveform:
        expected_time += np.argmax(np.abs(hilbert(waveform))) / WAVEFORM_SAMPLING_FREQ

    envelope = np.abs(hilbert(rf[0, :, element, 0]))
    peak_sample = np.argmax(envelope)
    assert abs(peak_sample - expected_time * SAMPLING_FREQ) <= 2


def test_parametric_and_measured_spectra_agree():
    """A measured Hann-windowed sine reproduces the parametric spectrum shape.

    ``get_pulse_spectrum_fn(center_frequency, n_period)`` is the analytic
    spectrum of a Hann-windowed sine; sampling that same pulse and running it
    through ``get_measured_pulse_spectrum_fn`` must give the same magnitude
    spectrum up to a global scale (the parametric pulse is centered at time
    zero, so phases differ by a linear term).
    """
    n_period = 4.0
    n = int(n_period / CENTER_FREQ * WAVEFORM_SAMPLING_FREQ)
    t = np.arange(n) / WAVEFORM_SAMPLING_FREQ
    waveform = (np.sin(2 * np.pi * CENTER_FREQ * t) * np.hanning(n)).astype(np.float32)

    freqs = np.linspace(0.5e6, 15e6, 200).astype(np.float32)
    parametric = np.abs(
        ops.convert_to_numpy(get_pulse_spectrum_fn(CENTER_FREQ, n_period=n_period)(freqs))
    )
    measured = np.abs(
        ops.convert_to_numpy(
            get_measured_pulse_spectrum_fn(waveform, WAVEFORM_SAMPLING_FREQ)(freqs)
        )
    )

    parametric /= parametric.max()
    measured /= measured.max()
    np.testing.assert_allclose(measured, parametric, atol=0.02)
