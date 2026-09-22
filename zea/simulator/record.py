"""The RF record of the frequency-domain simulator: its rfft grid, and the gates that keep
echoes from wrapping around it."""

from dataclasses import dataclass

import numpy as np
from keras import ops

from zea.internal.core import round_up_to_power_of_two


@dataclass(frozen=True)
class Record:
    """The record of ``n_ax`` samples at ``sampling_frequency``, synthesised on an rfft grid of
    ``n_fft`` samples with room for a whole pulse, so that :func:`in_record` never gates the
    end of the record away. Static: built in numpy by :func:`record_grid`.

    Attributes:
        n_ax: Samples in the RF data.
        n_fft: Samples of the synthesis grid, a power of two.
        sampling_frequency: Sampling frequency [Hz] of both.
        freqs: The rfft frequencies [Hz] of the grid, (n_freq,) float32.
        length: Duration [s] of the grid less the pulse tail: the last arrival that fits.
    """

    n_ax: int
    n_fft: int
    sampling_frequency: float
    freqs: np.ndarray
    length: float

    def spectrum(self, pulse):
        """The spectrum of ``pulse`` on the grid, as a complex64 tensor."""
        return ops.convert_to_tensor(pulse.spectrum(self.freqs))


def record_grid(n_ax, sampling_frequency, pulses):
    """The :class:`Record` of ``n_ax`` samples that fits every pulse of ``pulses``, which are
    sampled at ``sampling_frequency``."""
    fs = pulses[0].sampling_frequency
    n_fft = round_up_to_power_of_two(int(n_ax) + max(pulse.n_samples for pulse in pulses))
    freqs = np.fft.rfftfreq(n_fft, 1 / fs).astype(np.float32)
    n_after = max(pulse.n_after for pulse in pulses)
    return Record(int(n_ax), n_fft, fs, freqs, (n_fft - n_after) / fs)


def in_fft(record, one_way_time):
    """Complex gate on the one-way delays that fit the grid, as :func:`delay2` gates the
    transmit delays."""
    return ops.cast(one_way_time < record.n_fft / record.sampling_frequency, "complex64")


def in_record(record, arrival_time):
    """Complex gate on the two-way arrivals that fit the record with their pulse tail."""
    return ops.cast(arrival_time < record.length, "complex64")


def delay2(f, tau, n_fft, sampling_frequency):
    """
    Applies a delay in the frequency domain without phase wrapping.

    Args:
        f (array-like): The input frequencies.
        tau (float): The delay to apply.
        n_fft (int): The number of samples in the FFT.
        sampling_frequency (float): The sampling frequency.

    Returns:
        array-like: The spectrum of the delay.
    """
    arg = ops.array(-1j, dtype="complex64") * ops.cast(2 * np.pi * tau * f, "complex64")
    return ops.where(
        tau < n_fft / sampling_frequency,
        ops.exp(arg),
        ops.array(0.0, dtype="complex64"),
    )
