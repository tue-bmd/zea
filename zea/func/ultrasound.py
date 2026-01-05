import numpy as np
import scipy
from keras import ops

from zea import log
from zea.func.tensor import (
    resample,
)


def demodulate_not_jitable(
    rf_data,
    sampling_frequency=None,
    center_frequency=None,
    bandwidth=None,
    filter_coeff=None,
):
    """Demodulates an RF signal to complex base-band (IQ).

    Demodulates the radiofrequency (RF) bandpass signals and returns the
    Inphase/Quadrature (I/Q) components. IQ is a complex whose real (imaginary)
    part contains the in-phase (quadrature) component.

    This function operates (i.e. demodulates) on the RF signal over the
    (fast-) time axis which is assumed to be the last axis.

    Args:
        rf_data (ndarray): real valued input array of size [..., n_ax, n_el].
            second to last axis is fast-time axis.
        sampling_frequency (float): the sampling frequency of the RF signals (in Hz).
            Only not necessary when filter_coeff is provided.
        center_frequency (float, optional): represents the center frequency (in Hz).
            Defaults to None.
        bandwidth (float, optional): Bandwidth of RF signal in % of center
            frequency. Defaults to None.
            The bandwidth in % is defined by:
            B = Bandwidth_in_% = Bandwidth_in_Hz*(100/center_frequency).
            The cutoff frequency:
            Wn = Bandwidth_in_Hz/sampling_frequency, i.e:
            Wn = B*(center_frequency/100)/sampling_frequency.
        filter_coeff (list, optional): (b, a), numerator and denominator coefficients
            of FIR filter for quadratic band pass filter. All other parameters are ignored
            if filter_coeff are provided. Instead the given filter_coeff is directly used.
            If not provided, a filter is derived from the other params (sampling_frequency,
            center_frequency, bandwidth).
            see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html

    Returns:
        iq_data (ndarray): complex valued base-band signal.

    """
    rf_data = ops.convert_to_numpy(rf_data)
    assert np.isreal(rf_data).all(), f"RF must contain real RF signals, got {rf_data.dtype}"

    input_shape = rf_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, n_el = input_shape
    else:
        n_ax, n_el = input_shape

    if filter_coeff is None:
        assert sampling_frequency is not None, "provide sampling_frequency when no filter is given."
        # Time vector
        t = np.arange(n_ax) / sampling_frequency
        t0 = 0
        t = t + t0

        # Estimate center frequency
        if center_frequency is None:
            # Keep a maximum of 100 randomly selected scanlines
            idx = np.arange(n_el)
            if n_el > 100:
                idx = np.random.permutation(idx)[:100]
            # Power Spectrum
            P = np.sum(
                np.abs(np.fft.fft(np.take(rf_data, idx, axis=-1), axis=-2)) ** 2,
                axis=-1,
            )
            P = P[: n_ax // 2]
            # Carrier frequency
            idx = np.sum(np.arange(n_ax // 2) * P) / np.sum(P)
            center_frequency = idx * sampling_frequency / n_ax

        # Normalized cut-off frequency
        if bandwidth is None:
            Wn = min(2 * center_frequency / sampling_frequency, 0.5)
            bandwidth = center_frequency * Wn
        else:
            assert np.isscalar(bandwidth), "The signal bandwidth (in %) must be a scalar."
            assert (bandwidth > 0) & (bandwidth <= 200), (
                "The signal bandwidth (in %) must be within the interval of ]0,200]."
            )
            # bandwidth in Hz
            bandwidth = center_frequency * bandwidth / 100
            Wn = bandwidth / sampling_frequency
        assert (Wn > 0) & (Wn <= 1), (
            "The normalized cutoff frequency is not within the interval of (0,1). "
            "Check the input parameters!"
        )

        # Down-mixing of the RF signals
        carrier = np.exp(-1j * 2 * np.pi * center_frequency * t)
        # add the singleton dimensions
        carrier = np.reshape(carrier, (*[1] * (n_dim - 2), n_ax, 1))
        iq_data = rf_data * carrier

        # Low-pass filter
        N = 5
        b, a = scipy.signal.butter(N, Wn, "low")

        # factor 2: to preserve the envelope amplitude
        iq_data = scipy.signal.filtfilt(b, a, iq_data, axis=-2) * 2

        # Display a warning message if harmful aliasing is suspected
        # the RF signal is undersampled
        if sampling_frequency < (2 * center_frequency + bandwidth):
            # lower and higher frequencies of the bandpass signal
            fL = center_frequency - bandwidth / 2
            fH = center_frequency + bandwidth / 2
            n = fH // (fH - fL)
            harmless_aliasing = any(
                (2 * fH / np.arange(1, n) <= sampling_frequency)
                & (sampling_frequency <= 2 * fL / np.arange(1, n))
            )
            if not harmless_aliasing:
                log.warning(
                    "rf2iq:harmful_aliasing Harmful aliasing is present: the aliases"
                    " are not mutually exclusive!"
                )
    else:
        b, a = filter_coeff
        iq_data = scipy.signal.lfilter(b, a, rf_data, axis=-2) * 2

    return iq_data


def upmix(iq_data, sampling_frequency, center_frequency, upsampling_rate=6):
    """Upsamples and upmixes complex base-band signals (IQ) to RF.

    Args:
        iq_data (ndarray): complex valued input array of size [..., n_ax, n_el]. second
            to last axis is fast-time axis.
        sampling_frequency (float): the sampling frequency of the input IQ signal (in Hz).
            resulting sampling_frequency of RF data is upsampling_rate times higher.
        center_frequency (float, optional): represents the center frequency (in Hz).

    Returns:
        rf_data (ndarray): output real valued rf data.
    """
    assert iq_data.dtype in [
        "complex64",
        "complex128",
    ], "IQ must contain all complex signals."

    input_shape = iq_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, _ = input_shape
    else:
        n_ax, _ = input_shape

    # Time vector
    n_ax_up = n_ax * upsampling_rate
    sampling_frequency_up = sampling_frequency * upsampling_rate

    t = ops.arange(n_ax_up, dtype="float32") / sampling_frequency_up
    t0 = 0
    t = t + t0

    iq_data_upsampled = resample(
        iq_data,
        n_samples=n_ax_up,
        axis=-2,
        order=1,
    )

    # Up-mixing of the IQ signals
    t = ops.cast(t, dtype="complex64")
    center_frequency = ops.cast(center_frequency, dtype="complex64")
    carrier = ops.exp(1j * 2 * np.pi * center_frequency * t)
    carrier = ops.reshape(carrier, (*[1] * (n_dim - 2), n_ax_up, 1))

    rf_data = iq_data_upsampled * carrier
    rf_data = ops.real(rf_data) * ops.sqrt(2)

    return ops.cast(rf_data, "float32")


def get_band_pass_filter(num_taps, sampling_frequency, f1, f2):
    """Band pass filter

    Args:
        num_taps (int): number of taps in filter.
        sampling_frequency (float): sample frequency in Hz.
        f1 (float): cutoff frequency in Hz of left band edge.
        f2 (float): cutoff frequency in Hz of right band edge.

    Returns:
        ndarray: band pass filter
    """
    bpf = scipy.signal.firwin(num_taps, [f1, f2], pass_zero=False, fs=sampling_frequency)
    return bpf


def get_low_pass_iq_filter(num_taps, sampling_frequency, center_frequency, bandwidth):
    """Design complex low-pass filter.

    The filter is a low-pass FIR filter modulated to the center frequency.

    Args:
        num_taps (int): number of taps in filter.
        sampling_frequency (float): sample frequency.
        center_frequency (float): center frequency.
        bandwidth (float): bandwidth in Hz.

    Raises:
        ValueError: if cutoff frequency (bandwidth / 2) is not within (0, sampling_frequency / 2)

    Returns:
        ndarray: Complex-valued low-pass filter
    """
    cutoff = bandwidth / 2
    if not (0 < cutoff < sampling_frequency / 2):
        raise ValueError(
            f"Cutoff frequency must be within (0, sampling_frequency / 2), "
            f"got {cutoff} Hz, must be within (0, {sampling_frequency / 2}) Hz"
        )
    # Design real-valued low-pass filter
    lpf = scipy.signal.firwin(num_taps, cutoff, pass_zero=True, fs=sampling_frequency)
    # Modulate to center frequency to make it complex
    time_points = np.arange(num_taps) / sampling_frequency
    lpf_complex = lpf * np.exp(1j * 2 * np.pi * center_frequency * time_points)
    return lpf_complex


def complex_to_channels(complex_data, axis=-1):
    """Unroll complex data to separate channels.

    Args:
        complex_data (complex ndarray): complex input data.
        axis (int, optional): on which axis to extend. Defaults to -1.

    Returns:
        ndarray: real array with real and imaginary components
            unrolled over two channels at axis.
    """
    # assert ops.iscomplex(complex_data).any()
    q_data = ops.imag(complex_data)
    i_data = ops.real(complex_data)

    i_data = ops.expand_dims(i_data, axis=axis)
    q_data = ops.expand_dims(q_data, axis=axis)

    iq_data = ops.concatenate((i_data, q_data), axis=axis)
    return iq_data


def channels_to_complex(data):
    """Convert array with real and imaginary components at
    different channels to complex data array.

    Args:
        data (ndarray): input data, with at 0 index of axis
            real component and 1 index of axis the imaginary.

    Returns:
        ndarray: complex array with real and imaginary components.
    """
    assert data.shape[-1] == 2, "Data must have two channels."
    data = ops.cast(data, "complex64")
    return data[..., 0] + 1j * data[..., 1]


def hilbert(x, N: int = None, axis=-1):
    """Manual implementation of the Hilbert transform function. The function
    returns the analytical signal.

    Operated in the Fourier domain.

    Note:
        THIS IS NOT THE MATHEMATICAL THE HILBERT TRANSFORM as you will find it on
        wikipedia, but computes the analytical signal. The implementation reproduces
        the behavior of the `scipy.signal.hilbert` function.

    Args:
        x (ndarray): input data of any shape.
        N (int, optional): number of points in the FFT. Defaults to None.
        axis (int, optional): axis to operate on. Defaults to -1.
    Returns:
        x (ndarray): complex iq data of any shape.k

    """
    input_shape = x.shape
    n_dim = len(input_shape)

    n_ax = input_shape[axis]

    if axis < 0:
        axis = n_dim + axis

    if N is not None:
        if N < n_ax:
            raise ValueError("N must be greater or equal to n_ax.")
        # only pad along the axis, use manual padding
        pad = N - n_ax
        zeros = ops.zeros(
            input_shape[:axis] + (pad,) + input_shape[axis + 1 :],
        )

        x = ops.concatenate((x, zeros), axis=axis)
    else:
        N = n_ax

    # Create filter to zero out negative frequencies
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    idx = list(range(n_dim))
    # make sure axis gets to the end for fft (operates on last axis)
    idx.remove(axis)
    idx.append(axis)
    x = ops.transpose(x, idx)

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[-1] = slice(None)
        h = h[tuple(ind)]

    h = ops.convert_to_tensor(h)
    h = ops.cast(h, "complex64")
    h = h + 1j * ops.zeros_like(h)

    Xf_r, Xf_i = ops.fft((x, ops.zeros_like(x)))

    Xf_r = ops.cast(Xf_r, "complex64")
    Xf_i = ops.cast(Xf_i, "complex64")

    Xf = Xf_r + 1j * Xf_i
    Xf = Xf * h

    # x = np.fft.ifft(Xf)
    # do manual ifft using fft
    Xf_r = ops.real(Xf)
    Xf_i = ops.imag(Xf)
    Xf_r_inv, Xf_i_inv = ops.fft((Xf_r, -Xf_i))

    Xf_i_inv = ops.cast(Xf_i_inv, "complex64")
    Xf_r_inv = ops.cast(Xf_r_inv, "complex64")

    x = Xf_r_inv / N
    x = x + 1j * (-Xf_i_inv / N)

    # switch back to original shape
    idx = list(range(n_dim))
    idx.insert(axis, idx.pop(-1))
    x = ops.transpose(x, idx)
    return x


def demodulate(data, center_frequency, sampling_frequency, axis=-3):
    """Demodulates the input data to baseband. The function computes the analytical
    signal (the signal with negative frequencies removed) and then shifts the spectrum
    of the signal to baseband by multiplying with a complex exponential. Where the
    spectrum was centered around `center_frequency` before, it is now centered around
    0 Hz. The baseband IQ data are complex-valued. The real and imaginary parts
    are stored in two real-valued channels.

    Args:
        data (ops.Tensor): The input data to demodulate of shape `(..., axis, ..., 1)`.
        center_frequency (float): The center frequency of the signal.
        sampling_frequency (float): The sampling frequency of the signal.
        axis (int, optional): The axis along which to demodulate. Defaults to -3.

    Returns:
        ops.Tensor: The demodulated IQ data of shape `(..., axis, ..., 2)`.
    """
    # Compute the analytical signal
    analytical_signal = hilbert(data, axis=axis)

    # Define frequency indices
    frequency_indices = ops.arange(analytical_signal.shape[axis])

    # Expand the frequency indices to match the shape of the RF data
    indexing = [None] * data.ndim
    indexing[axis] = slice(None)
    indexing = tuple(indexing)
    frequency_indices_shaped_like_rf = frequency_indices[indexing]

    # Cast to complex64
    center_frequency = ops.cast(center_frequency, dtype="complex64")
    sampling_frequency = ops.cast(sampling_frequency, dtype="complex64")
    frequency_indices_shaped_like_rf = ops.cast(frequency_indices_shaped_like_rf, dtype="complex64")

    # Shift to baseband
    phasor_exponent = (
        -1j * 2 * np.pi * center_frequency * frequency_indices_shaped_like_rf / sampling_frequency
    )
    iq_data_signal_complex = analytical_signal * ops.exp(phasor_exponent)

    # Split the complex signal into two channels
    iq_data_two_channel = complex_to_channels(ops.squeeze(iq_data_signal_complex, axis=-1))

    return iq_data_two_channel


def compute_time_to_peak_stack(waveforms, center_frequencies, waveform_sampling_frequency=250e6):
    """Compute the time of the peak of each waveform in a stack of waveforms.

    Args:
        waveforms (ndarray): The waveforms of shape (n_waveforms, n_samples).
        center_frequencies (ndarray): The center frequencies of the waveforms in Hz of shape
            (n_waveforms,) or a scalar if all waveforms have the same center frequency.
        waveform_sampling_frequency (float): The sampling frequency of the waveforms in Hz.

    Returns:
        ndarray: The time to peak for each waveform in seconds.
    """
    t_peak = []
    center_frequencies = center_frequencies * ops.ones((waveforms.shape[0],))
    for waveform, center_frequency in zip(waveforms, center_frequencies):
        t_peak.append(compute_time_to_peak(waveform, center_frequency, waveform_sampling_frequency))
    return ops.stack(t_peak)


def compute_time_to_peak(waveform, center_frequency, waveform_sampling_frequency=250e6):
    """Compute the time of the peak of the waveform.

    Args:
        waveform (ndarray): The waveform of shape (n_samples).
        center_frequency (float): The center frequency of the waveform in Hz.
        waveform_sampling_frequency (float): The sampling frequency of the waveform in Hz.

    Returns:
        float: The time to peak for the waveform in seconds.
    """
    n_samples = waveform.shape[0]
    if n_samples == 0:
        raise ValueError("Waveform has zero samples.")

    waveforms_iq_complex_channels = demodulate(
        waveform[..., None], center_frequency, waveform_sampling_frequency, axis=-1
    )
    waveforms_iq_complex = channels_to_complex(waveforms_iq_complex_channels)
    envelope = ops.abs(waveforms_iq_complex)
    peak_idx = ops.argmax(envelope, axis=-1)
    t_peak = ops.cast(peak_idx, dtype="float32") / waveform_sampling_frequency
    return t_peak


def envelope_detect(data, axis=-3):
    """Envelope detection of RF signals.

    If the input data is real, it first applies the Hilbert transform along the specified axis
    and then computes the magnitude of the resulting complex signal.
    If the input data is complex, it computes the magnitude directly.

    Args:
        - data (Tensor): The beamformed data of shape (..., grid_size_z, grid_size_x, n_ch).
        - axis (int): Axis along which to apply the Hilbert transform. Defaults to -3.

    Returns:
        - envelope_data (Tensor): The envelope detected data
            of shape (..., grid_size_z, grid_size_x).
    """
    if data.shape[-1] == 2:
        data = channels_to_complex(data)
    else:
        n_ax = ops.shape(data)[axis]
        n_ax_float = ops.cast(n_ax, "float32")

        # Calculate next power of 2: M = 2^ceil(log2(n_ax))
        # see https://github.com/tue-bmd/zea/discussions/147
        log2_n_ax = ops.log2(n_ax_float)
        M = ops.cast(2 ** ops.ceil(log2_n_ax), "int32")

        data = hilbert(data, N=M, axis=axis)
        indices = ops.arange(n_ax)

        data = ops.take(data, indices, axis=axis)
        data = ops.squeeze(data, axis=-1)

    # data = ops.abs(data)
    real = ops.real(data)
    imag = ops.imag(data)
    data = ops.sqrt(real**2 + imag**2)
    data = ops.cast(data, "float32")
    return data


def log_compress(data, eps=1e-16):
    """Apply logarithmic compression to data."""
    eps = ops.convert_to_tensor(eps, dtype=data.dtype)
    data = ops.where(data == 0, eps, data)  # Avoid log(0)
    return 20 * ops.log10(data)
