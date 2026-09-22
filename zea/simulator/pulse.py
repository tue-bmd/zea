"""The two-way transmit pulse of the simulators: the parametric models of :func:`transmit_pulse`,
a measured waveform through :func:`measured_pulse`, and the spectra behind them."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import hilbert
from scipy.special import fresnel

from zea.internal.core import concrete, round_up_to_power_of_two

PULSE_MODELS = ("realistic", "hann", "simus")


@dataclass(frozen=True)
class Pulse:
    """Two-way transmit pulse of the simulators, with its envelope peak at ``t = 0``.

    Built by :func:`transmit_pulse` or :func:`measured_pulse`. ``spectrum_fn`` is the
    continuous-time spectrum, scaled so that ``irfft`` of its samples on an rfft grid of
    ``sampling_frequency`` recovers the waveform with a unit peak. The support is where the
    envelope is above -80 dB.
    """

    spectrum_fn: Callable
    """Complex spectrum (numpy) at frequencies [Hz]."""
    sampling_frequency: float
    """Sampling frequency [Hz] of :meth:`waveform`."""
    n_before: int
    """Support before the peak, in samples."""
    n_after: int
    """Support after the peak, in samples."""
    time_to_peak: float
    """Time [s] from the transmit trigger (the start of the pulse, or of a measured waveform)
    to the envelope peak: the ``t_peak`` of a real system."""
    band: tuple
    """The -6 dB band [Hz] of the pulse, (low, high)."""

    @property
    def n_samples(self):
        """Length of :meth:`waveform`: odd, with the peak on the middle sample."""
        return 2 * max(self.n_before, self.n_after) + 1

    def spectrum(self, freqs):
        """The spectrum at ``freqs`` [Hz], as complex64."""
        return self.spectrum_fn(np.asarray(freqs, np.float64)).astype(np.complex64)

    def waveform(self, from_trigger=False):
        """The pulse sampled at ``sampling_frequency``, as ``waveforms_two_way`` of the
        simulators.

        By default of length :attr:`n_samples`, odd, with the envelope peak on the middle sample.
        With ``from_trigger`` the waveform starts at the transmit trigger instead, like a
        Verasonics waveform: sample 0 is ``time_to_peak`` before the peak, so the ``t_peak``
        that :class:`zea.Parameters` derives from it is :attr:`time_to_peak`.
        """
        if from_trigger:
            n_before = int(round(self.time_to_peak * self.sampling_frequency))
            n_after = self.n_after
        else:
            n_before = n_after = self.n_samples // 2
        n_fft = round_up_to_power_of_two(2 * (n_before + n_after + 1))
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sampling_frequency)
        waveform = np.fft.irfft(self.spectrum_fn(freqs), n_fft)
        return np.roll(waveform, n_before)[: n_before + n_after + 1].astype(np.float32)


def transmit_pulse(
    center_frequency,
    sampling_frequency=250e6,
    pulse_model="realistic",
    n_period=1.0,
    chirp_sweep=None,
    bandwidth_percent=70.0,
    probe_center_frequency=None,
    excitation_duty_cycle=0.67,
    transducer_order=2,
    equalize=False,
):
    """The parametric two-way (pulse-echo) transmit pulse: excitation times transducer response.

    Its :meth:`Pulse.waveform` is the ``waveforms_two_way`` of :func:`simulate_rf`,
    :func:`simulate_rf_td` and :class:`zea.ops.Simulate`, which use the default pulse of this
    function when none is given::

        pulse = transmit_pulse(5e6, pulse_model="simus", bandwidth_percent=75.0)
        rf = simulate_rf(..., waveforms_two_way=pulse.waveform())

    The radiation factor of a baffled piston (jω) is absorbed into the transducer response, as
    in MUST, so that ``bandwidth_percent`` is the -6 dB pulse-echo bandwidth for every model.
    For a measured pulse see :func:`measured_pulse`.

    The ``"realistic"`` model is the model of the Verasonics Vantage two-way waveform
    (``TW.Wvfm2Wy``): the tri-state burst through the same 2nd-order Butterworth band-pass on
    transmit and on receive, -6 dB two-way at the edges of ``Trans.Bandwidth``, with no
    radiation factor and no normalisation. It reproduces that waveform with
    ``probe_center_frequency`` at ``Trans.frequency``, ``bandwidth_percent`` at
    ``100 * (fHigh - fLow) / Trans.frequency`` (about 77 % for the L11-5v) and ``equalize``
    set, since a Vantage burst has equalisation pulses by default.

    Args:
        center_frequency (float): Centre frequency of the excitation [Hz].
        sampling_frequency (float): Sampling frequency [Hz] of :meth:`Pulse.waveform`; 250 MHz
            like the ``waveforms_two_way`` of a zea file, or the sampling frequency of the RF
            data for the pulse the simulators use internally.
        pulse_model (str): ``"realistic"``: a pulser's tri-state square burst of ``n_period``
            periods (:func:`square_burst_spectrum`) through a causal Butterworth band-pass on
            transmit and on receive (:func:`butterworth_transfer`), so the pulse rises fast and
            rings down like a real system's. ``"hann"``: Hann-windowed cosine or chirp
            (:func:`hann_burst_spectrum`) times a zero-phase Gaussian pulse-echo response
            (:func:`gaussian_transfer`). ``"simus"``: MUST's rectangular-windowed sine or chirp
            (:func:`rect_burst_spectrum`) times its generalized-normal response
            (:func:`generalized_normal_transfer`), for a one-to-one comparison with SIMUS.
        n_period (float): Periods of ``center_frequency`` in the excitation. The default of one
            period (MUST's ``TXnow``) leaves the transducer as the limiting factor, so the pulse
            bandwidth follows ``bandwidth_percent``; a burst of n periods is itself only about
            120 % / n wide and narrows the pulse below the requested bandwidth for n >= 2.
        chirp_sweep (float, optional): Linear frequency sweep [Hz] of the ``"hann"`` and
            ``"simus"`` excitations, from ``center_frequency - chirp_sweep / 2`` to
            ``center_frequency + chirp_sweep / 2``; negative sweeps down (MUST's direction).
        bandwidth_percent (float, optional): Pulse-echo -6 dB fractional bandwidth of the
            transducer in percent of ``probe_center_frequency``. None is a flat response, which
            only ``"hann"`` allows.
        probe_center_frequency (float, optional): Centre of the transducer band [Hz]. Defaults
            to ``center_frequency``.
        excitation_duty_cycle (float): Width of each half-cycle of the ``"realistic"`` burst as
            a fraction of the half period. 0.67 nulls the third harmonic.
        transducer_order (int): Order of the ``"realistic"`` Butterworth band-pass, per way.
        equalize (bool): Add the Verasonics equalisation pulses to the ``"realistic"`` burst
            (:func:`square_burst_pulses`). They lengthen the burst by about a period, so the
            excitation and not the transducer limits the pulse bandwidth: a 2-half-cycle burst
            through a 77 % transducer gives a 56 % pulse instead of 68 %. Off by default, so
            that ``bandwidth_percent`` sets the bandwidth of the pulse.

    Returns:
        Pulse: the pulse, with the envelope peak at t = 0 and a unit peak.
    """
    if pulse_model not in PULSE_MODELS:
        raise ValueError(f"pulse_model ({pulse_model}) must be one of {PULSE_MODELS}.")
    fc = _static_float(center_frequency, "center_frequency")
    fs = _static_float(sampling_frequency, "sampling_frequency")
    fc_probe = fc if probe_center_frequency is None else float(probe_center_frequency)
    sweep = 0.0 if chirp_sweep is None else float(chirp_sweep)
    duration = n_period / fc
    if bandwidth_percent is None and pulse_model != "hann":
        raise ValueError(f"pulse_model='{pulse_model}' needs bandwidth_percent.")
    # The windowed excitations are centred at t = 0; the trigger is where their window starts.
    trigger = -0.5 * n_period / fc
    if pulse_model == "hann":

        def spectrum(f):
            return hann_burst_spectrum(f, fc, n_period, sweep) * gaussian_transfer(
                f, fc_probe, bandwidth_percent
            )

    elif pulse_model == "simus":

        def spectrum(f):
            return rect_burst_spectrum(f, fc, n_period, sweep) * generalized_normal_transfer(
                f, fc_probe, bandwidth_percent
            )

    else:
        duration += (64.0 + equalize) / fc  # ringdown, and the equalisation pulses
        centres, widths, _ = square_burst_pulses(fc, n_period, excitation_duty_cycle, equalize)
        # The trigger is the first edge of the burst, where a Verasonics waveform starts.
        trigger = centres[0] - 0.5 * widths[0]

        def spectrum(f):
            excitation = square_burst_spectrum(f, fc, n_period, excitation_duty_cycle, equalize)
            one_way = butterworth_transfer(f, fc_probe, bandwidth_percent, transducer_order)
            return excitation * one_way**2

    pulse = _calibrate(spectrum, fs, duration, pulse_model == "realistic", trigger)
    if pulse_model != "realistic":
        # A zero-phase transducer responds before the excitation window, the more so the further
        # the excitation is detuned from the band; the trigger has to hold that response.
        return replace(pulse, time_to_peak=max(pulse.time_to_peak, pulse.n_before / fs))
    return pulse


def measured_pulse(waveform_two_way, sampling_frequency, waveform_sampling_frequency=250e6):
    """A sampled two-way (pulse-echo) waveform as the transmit pulse: the ``waveforms_two_way``
    of the simulators, measured, the system's own or built with :func:`transmit_pulse`.

    The waveform already includes the transducer response. It is evaluated on the simulation
    grid by its discrete-time Fourier transform, so any sampling frequency will do, and shifted
    so that its envelope peak is at t = 0. The shift is :attr:`Pulse.time_to_peak`, the time from
    sample 0 to the peak; for a waveform that starts at the transmit trigger, as the
    ``waveforms_two_way`` of a zea file (the Verasonics two-way waveform, sampled at 250 MHz),
    that is the ``t_peak`` of :func:`simulate_rf` which puts the waveform back at the travel
    time (:attr:`zea.Parameters.t_peak` derives the same).

    The Verasonics waveform (``TW.Wvfm2Wy``) is not measured but modelled, by the ``"realistic"``
    model of :func:`transmit_pulse` with its equalisation pulses: the burst through a 2nd-order
    Butterworth band-pass twice, -6 dB two-way at ``Trans.Bandwidth``, without a radiation
    factor or normalisation. Its sample 0 is the trigger and the Vantage simulator places that
    sample at the two-way travel time, so the envelope peak of the waveform is its ``t_peak``.
    The Vantage simulator applies no frequency-dependent scattering to it; see
    ``scatter_exponent`` of :func:`simulate_rf`.

    Args:
        waveform_two_way (array-like): The waveform of shape (n_samples,).
        sampling_frequency (float): Sampling frequency of the RF data [Hz].
        waveform_sampling_frequency (float): Sampling frequency of the waveform [Hz].

    Returns:
        Pulse: the pulse, with the envelope peak at t = 0 and a unit peak.
    """
    samples = np.asarray(waveform_two_way, np.float64)
    if samples.ndim != 1 or len(samples) < 2 or not np.any(samples):
        raise ValueError("waveform_two_way must be a non-zero waveform of shape (n_samples,).")
    fs = _static_float(sampling_frequency, "sampling_frequency")
    times = np.arange(len(samples)) / float(waveform_sampling_frequency)

    def spectrum(f):
        return sampled_spectrum(f, samples, times)

    return _calibrate(spectrum, fs, times[-1], shift_peak=True)


def transmit_pulses(
    n_tx,
    center_frequency,
    sampling_frequency,
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
):
    """The two-way transmit pulse of every transmit, as a list of ``n_tx`` :class:`Pulse`.

    The rows of ``waveforms_two_way`` when given, of shape (n_tx, n_samples), or (n_samples,)
    for the same waveform on every transmit (see :func:`measured_pulse`; identical rows share
    one :class:`Pulse`); otherwise the default pulse of :func:`transmit_pulse` at
    ``center_frequency``, on every transmit.
    """
    if waveforms_two_way is None:
        return [transmit_pulse(center_frequency, sampling_frequency)] * n_tx
    waveforms = np.atleast_2d(np.asarray(waveforms_two_way, np.float64))
    if waveforms.ndim != 2 or waveforms.shape[0] not in (1, n_tx):
        raise ValueError(
            f"waveforms_two_way must have shape (n_tx, n_samples) or (n_samples,), got "
            f"{np.shape(waveforms_two_way)} for {n_tx} transmits."
        )
    pulses = {}
    for waveform in waveforms:
        key = waveform.tobytes()
        if key not in pulses:
            pulses[key] = measured_pulse(waveform, sampling_frequency, waveform_sampling_frequency)
    return [pulses[waveform.tobytes()] for waveform in waveforms] * (n_tx // len(waveforms))


def _calibrate(
    spectrum_fn,
    sampling_frequency,
    duration,
    shift_peak,
    trigger=0.0,
    threshold_db=-80.0,
    oversample=16,
):
    """Shift the envelope peak to t = 0, scale to a unit peak and measure the support, on a grid
    ``oversample`` times finer than ``sampling_frequency`` so that none of it depends on it.
    ``trigger`` is the time [s] of the transmit trigger in the frame of ``spectrum_fn``."""
    fs = oversample * sampling_frequency
    n = round_up_to_power_of_two(max(4 * duration * fs, 256))
    while True:
        freqs = np.fft.rfftfreq(n, 1 / fs)
        waveform = np.fft.irfft(spectrum_fn(freqs), n)
        envelope = np.abs(hilbert(waveform))
        shift = 0.0
        if shift_peak:
            k = int(np.argmax(envelope))
            before, at, after = envelope[k - 1], envelope[k], envelope[(k + 1) % n]
            fraction = 0.5 * (before - after) / (before - 2 * at + after)  # sub-sample peak
            shift = ((k if k < n // 2 else k - n) + fraction) / fs
            waveform = np.fft.irfft(spectrum_fn(freqs) * np.exp(2j * np.pi * freqs * shift), n)
            envelope = np.abs(hilbert(waveform))
        support = np.flatnonzero(envelope > 10 ** (threshold_db / 20) * envelope.max())
        positive, negative = support[support < n // 2], support[support >= n // 2]
        after = int(positive.max()) if len(positive) else 0
        before = int(n - negative.min()) if len(negative) else 0
        if before + after < n // 4 or n >= 2**22:
            break
        n *= 2  # the pulse wraps around the grid
    scale = 1.0 / (oversample * np.abs(waveform).max())  # unit peak on the coarse grid

    def shifted(f):
        return scale * spectrum_fn(f) * np.exp(2j * np.pi * f * shift)

    magnitude = np.abs(spectrum_fn(freqs))
    in_band = np.flatnonzero(magnitude >= 0.5 * magnitude.max())
    band = (float(freqs[in_band[0]]), float(freqs[in_band[-1]]))
    n_before, n_after = (int(np.ceil(k / oversample)) for k in (before, after))
    return Pulse(shifted, sampling_frequency, n_before, n_after, shift - trigger, band)


def _static_float(x, name):
    value = concrete(x)
    if value is None:
        raise ValueError(
            f"{name} must be static (not traced): the transmit pulse is built in numpy."
        )
    return float(value)


def _validate_bandwidth(bandwidth_percent, geometric=False):
    if not np.isfinite(bandwidth_percent) or bandwidth_percent <= 0:
        raise ValueError(f"bandwidth_percent must be positive, got {bandwidth_percent}.")
    if geometric and bandwidth_percent >= 200:
        raise ValueError(f"bandwidth_percent must be below 200, got {bandwidth_percent}.")


def sampled_spectrum(f, samples, times):
    """Continuous-time spectrum of a uniformly sampled waveform at frequencies ``f`` [Hz]: its
    discrete-time Fourier transform times the sampling interval, by Horner's scheme so that no
    (n_freqs, n_samples) table is formed. Zero beyond the Nyquist frequency of the samples."""
    f = np.asarray(f, np.float64)
    dt = times[1] - times[0]
    z = np.exp(-2j * np.pi * f * dt)
    spectrum = np.zeros(f.shape, np.complex128)
    for sample in samples[::-1]:
        spectrum = spectrum * z + sample
    return spectrum * dt * np.exp(-2j * np.pi * f * times[0]) * (np.abs(f) <= 0.5 / dt)


def hann_burst_spectrum(f, fc, n_period, sweep=0.0):
    """Spectrum of a Hann-windowed tone or linear chirp centred at t = 0.

    The window spans ``n_period`` periods of ``fc``, over which the instantaneous frequency
    runs linearly from ``fc - sweep / 2`` to ``fc + sweep / 2``. Evaluated from samples at 64
    per period, so aliasing is far below the 1/f**3 tails of the window.
    """
    width = n_period / fc
    times = np.linspace(-width / 2, width / 2, int(np.ceil(64 * n_period)) + 1)
    window = np.cos(np.pi * times / width) ** 2
    phase = 2 * np.pi * (fc * times + sweep / (2 * width) * times**2)
    return sampled_spectrum(f, window * np.cos(phase), times)


def rect_burst_spectrum(f, fc, n_period, sweep=0.0):
    """Spectrum of MUST's excitation: a rectangular-windowed sine of ``n_period`` periods
    (``TXnow``) centred at t = 0, or its linear chirp over ``|sweep|`` Hz (``TXfreqsweep``),
    as ``getPulseSpectrumFunction`` computes it. MUST inverts the conjugate spectrum and its
    chirp sweeps down, so a positive ``sweep`` is the time-reversed pulse, sweeping up.
    """
    T = n_period / fc
    f = np.asarray(f, np.float64)
    if not sweep:
        spectrum = 1j * (np.sinc(T * (f - fc)) - np.sinc(T * (f + fc)))
    else:
        w, wc, dw = 2 * np.pi * f, 2 * np.pi * fc, 2 * np.pi * abs(sweep)

        def fresnel_integral(x):
            s, c = fresnel(x)
            return c + 1j * s

        def half(w):
            scale = np.sqrt(T / (np.pi * dw))
            return (
                np.sqrt(np.pi * T / dw)
                * np.exp(-1j * (w - wc) ** 2 * T / (2 * dw))
                * (
                    fresnel_integral((dw / 2 + w - wc) * scale)
                    + fresnel_integral((dw / 2 - w + wc) * scale)
                )
            )

        spectrum = (1j * half(w) - 1j * half(-w)) / T
    return np.conj(spectrum) if sweep <= 0 else spectrum


def square_burst_spectrum(f, fc, n_period, duty_cycle=0.67, equalize=False):
    """Spectrum of a pulser's tri-state burst centred at t = 0: ``2 * n_period`` alternating
    rectangular half-cycles, each ``duty_cycle`` of the half period wide. The duty cycle sets the
    harmonics; 0.67 nulls the third. ``equalize`` adds the Verasonics equalisation pulses, see
    :func:`square_burst_pulses`."""
    f = np.asarray(f, np.float64)
    centres, widths, signs = square_burst_pulses(fc, n_period, duty_cycle, equalize)
    pulses = signs * widths * np.sinc(f[:, None] * widths)
    return (pulses * np.exp(-2j * np.pi * f[:, None] * centres)).sum(axis=1)


def square_burst_pulses(fc, n_period, duty_cycle=0.67, equalize=False):
    """The rectangular pulses of :func:`square_burst_spectrum`: their centres [s], widths [s]
    and signs, in order. The half-cycles are centred on the half periods around t = 0, starting
    positive. With ``equalize`` a pulse of half the width and opposite sign precedes the first
    and follows the last half-cycle, a quarter period away from its edge: the equalisation pulses
    a Verasonics system adds by default (``TW.equalize``), which make the drive zero-mean and
    the burst about a period longer."""
    half_period = 0.5 / fc
    n_half = max(1, int(round(2 * n_period)))
    width = duty_cycle * half_period
    centres = (np.arange(n_half) - (n_half - 1) / 2) * half_period
    signs = (-1.0) ** np.arange(n_half)
    widths = np.full(n_half, width)
    if equalize:
        # A quarter period between the edges, so the centres are that plus half of both widths.
        offset = 0.5 * half_period + 0.75 * width
        centres = np.concatenate([[centres[0] - offset], centres, [centres[-1] + offset]])
        signs = np.concatenate([[-signs[0]], signs, [-signs[-1]]])
        widths = np.concatenate([[0.5 * width], widths, [0.5 * width]])
    return centres, widths, signs


def gaussian_transfer(f, fc, bandwidth_percent):
    """Gaussian pulse-echo response of the transducer: unit at ``fc`` and -6 dB at the edges of
    the fractional bandwidth, ``fc * (1 +/- bandwidth_percent / 200)``. None is flat."""
    f = np.asarray(f, np.float64)
    if bandwidth_percent is None:
        return np.ones_like(f)
    _validate_bandwidth(bandwidth_percent)
    half_width = 0.5 * bandwidth_percent / 100 * fc
    return np.exp(-np.log(2) * ((np.abs(f) - fc) / half_width) ** 2)


def generalized_normal_transfer(f, fc, bandwidth_percent):
    """MUST's pulse-echo response: a generalized normal window, unit at ``fc`` and -6 dB at the
    edges of the fractional bandwidth, with exponent ``ln(126) / ln(2 fc / fB)``
    (``getProbeFunction``, squared for the round trip)."""
    _validate_bandwidth(bandwidth_percent, geometric=True)
    f = np.asarray(f, np.float64)
    f_band = bandwidth_percent * fc / 100
    p = np.log(126) / np.log(2 * fc / f_band)
    return np.exp(-((np.abs(np.abs(f) - fc) / (f_band / 2 / np.log(2) ** (1 / p))) ** p))


def butterworth_transfer(f, fc, bandwidth_percent, order=2):
    """One-way Butterworth band-pass response of the transducer: causal and minimum phase,
    unit at the geometric centre of the band and -3 dB at the edges of the fractional bandwidth,
    ``fc * (1 +/- bandwidth_percent / 200)``, so that transmit and receive together are -6 dB
    there. Evaluated as the low-pass prototype at ``j (w**2 - w0**2) / (B w)``."""
    _validate_bandwidth(bandwidth_percent, geometric=True)
    w = 2 * np.pi * np.asarray(f, np.float64)
    low, high = fc * (1 - bandwidth_percent / 200), fc * (1 + bandwidth_percent / 200)
    w0_squared, band = (2 * np.pi) ** 2 * low * high, 2 * np.pi * (high - low)
    w_safe = np.where(w == 0, 1.0, w)
    q = np.asarray(1j * (w**2 - w0_squared) / (band * w_safe))
    poles = np.exp(1j * np.pi * (2 * np.arange(1, order + 1) + order - 1) / (2 * order))
    return np.where(w == 0, 0.0, np.prod(1 / (q[..., None] - poles), axis=-1))
