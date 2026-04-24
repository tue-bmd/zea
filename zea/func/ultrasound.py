import keras
import numpy as np
import scipy.signal
from keras import ops

from zea import log
from zea.func import split_seed
from zea.func.tensor import (
    resample,
    split_into_windows,
)


def demodulate_not_jitable(
    rf_data,
    sampling_frequency=None,
    demodulation_frequency=None,
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
        demodulation_frequency (float, optional): Modulation frequency (in Hz).
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
        if demodulation_frequency is None:
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
            demodulation_frequency = idx * sampling_frequency / n_ax

        # Normalized cut-off frequency
        if bandwidth is None:
            Wn = min(2 * demodulation_frequency / sampling_frequency, 0.5)
            bandwidth = demodulation_frequency * Wn
        else:
            assert np.isscalar(bandwidth), "The signal bandwidth (in %) must be a scalar."
            assert (bandwidth > 0) & (bandwidth <= 200), (
                "The signal bandwidth (in %) must be within the interval of ]0,200]."
            )
            # bandwidth in Hz
            bandwidth = demodulation_frequency * bandwidth / 100
            Wn = bandwidth / sampling_frequency
        assert (Wn > 0) & (Wn <= 1), (
            "The normalized cutoff frequency is not within the interval of (0,1). "
            "Check the input parameters!"
        )

        # Down-mixing of the RF signals
        carrier = np.exp(-1j * 2 * np.pi * demodulation_frequency * t)
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
        if sampling_frequency < (2 * demodulation_frequency + bandwidth):
            # lower and higher frequencies of the bandpass signal
            fL = demodulation_frequency - bandwidth / 2
            fH = demodulation_frequency + bandwidth / 2
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


def upmix(iq_data, sampling_frequency, demodulation_frequency, upsampling_rate=6):
    """Upsamples and upmixes complex base-band signals (IQ) to RF.

    Args:
        iq_data (ndarray): complex valued input array of size [..., n_ax, n_el]. second
            to last axis is fast-time axis.
        sampling_frequency (float): the sampling frequency of the input IQ signal (in Hz).
            resulting sampling_frequency of RF data is upsampling_rate times higher.
        demodulation_frequency (float, optional): modulation frequency (in Hz).

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
    demodulation_frequency = ops.cast(demodulation_frequency, dtype="complex64")
    carrier = ops.exp(1j * 2 * np.pi * demodulation_frequency * t)
    carrier = ops.reshape(carrier, (*[1] * (n_dim - 2), n_ax_up, 1))

    rf_data = iq_data_upsampled * carrier
    rf_data = ops.real(rf_data) * ops.sqrt(2)

    return ops.cast(rf_data, "float32")


def _sinc(x):
    """Return the normalized sinc function. Equivalent to np.sinc(x)."""
    y = np.pi * ops.where(x == 0, 1.0e-20, x)
    return ops.sin(y) / y


def get_band_pass_filter(num_taps, sampling_frequency, f1, f2, validate=True):
    """Band pass filter

    Compatible with ``jax.jit`` when ``numtaps`` is static. Based on ``scipy.signal.firwin`` with
    hamming window.

    Args:
        num_taps (int): number of taps in filter.
        sampling_frequency (float): sample frequency in Hz.
        f1 (float): cutoff frequency in Hz of left band edge.
        f2 (float): cutoff frequency in Hz of right band edge.
        validate (bool, optional): whether to validate the cutoff frequencies. Defaults to True.

    Returns:
        ndarray: band pass filter
    """
    sampling_frequency = ops.cast(sampling_frequency, "float32")
    f1 = ops.cast(f1, "float32")
    f2 = ops.cast(f2, "float32")

    nyq = 0.5 * sampling_frequency
    f1 = f1 / nyq
    f2 = f2 / nyq

    if validate:
        if f1 <= 0 or f2 >= 1:
            raise ValueError(
                f"Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2. "
                f"Got f1={f1 * nyq} Hz, f2={f2 * nyq} Hz."
            )

        if f1 >= f2:
            raise ValueError(
                f"Invalid cutoff frequencies: the frequencies must be strictly increasing. "
                f"Got f1={f1 * nyq} Hz, f2={f2 * nyq} Hz."
            )

    # Build up the coefficients.
    alpha = 0.5 * (num_taps - 1)
    m = ops.arange(0, num_taps, dtype="float32") - alpha
    h = f2 * _sinc(f2 * m) - f1 * _sinc(f1 * m)

    # Get and apply the window function.
    win = np.hamming(num_taps)
    win = ops.convert_to_tensor(win, dtype=h.dtype)
    h *= win

    # Use center frequency for scaling: 0 for lowpass, 1 (Nyquist) for highpass, or band center
    scale_frequency = ops.where(f1 == 0, 0.0, ops.where(f2 == 1, 1.0, 0.5 * (f1 + f2)))
    c = ops.cos(np.pi * m * scale_frequency)
    s = ops.sum(h * c)
    h /= s

    return h


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
    """Implementation of the Hilbert transform function that computes the analytical signal.

    Operates in the Fourier domain by applying a filter that zeros out negative frequencies
    and doubles positive frequencies.

    .. note::
        This is NOT the mathematical Hilbert transform as defined in the
        `Wikipedia article <https://en.wikipedia.org/wiki/Hilbert_transform>`_,
        but instead computes the analytical signal. The implementation reproduces
        the behavior of the :func:`scipy.signal.hilbert` function.

    Args:
        x (ndarray): Input data of any shape.
        N (int, optional): Number of points to use for the FFT. If specified and greater
            than the length of the data along the specified axis, the data will be
            zero-padded. If None, uses the length of x along the specified axis.
            Defaults to None.
        axis (int, optional): Axis along which to compute the Hilbert transform.
            Defaults to -1 (last axis).

    Returns:
        ndarray: Complex analytical signal with the same shape as the input (or padded
            to length N if specified). The real part is the original signal and the
            imaginary part is the Hilbert transform of the signal.

    Raises:
        ValueError: If N is specified and is less than the length of x along the
            specified axis.

    Example:
        >>> import numpy as np
        >>> from zea.func import hilbert
        >>> x = np.array([1.0, 2.0, 3.0, 4.0])
        >>> analytical_signal = hilbert(x)
        >>> envelope = np.abs(analytical_signal)
    """

    input_shape = x.shape
    n_dim = len(input_shape)

    n_ax = input_shape[axis]

    if axis < 0:
        axis = n_dim + axis

    if N is not None:
        if N < n_ax:
            raise ValueError(f"N must be greater or equal to n_ax, got N={N}, n_ax={n_ax}")

        pad = np.maximum(N - n_ax, 0)

        pad_list = [[0, 0] for _ in range(n_dim)]
        pad_list[axis] = [0, pad]

        x = ops.pad(x, pad_list, mode="constant", constant_values=0.0)
    else:
        N = n_ax

    # Create filter to zero out negative frequencies
    # h[0] = 1, h[1:N//2] = 2, h[N//2] = 1 (if even), rest = 0
    indices = ops.arange(N, dtype="float32")
    h = ops.zeros(N, dtype="float32")

    h = ops.where(indices == 0, 1.0, h)
    h = ops.where((indices > 0) & (indices < N / 2.0), 2.0, h)
    h = ops.where((N % 2 == 0) & (indices == N / 2.0), 1.0, h)

    h = ops.cast(h, "complex64")

    idx = list(range(n_dim))
    # make sure axis gets to the end for fft (operates on last axis)
    idx.remove(axis)
    idx.append(axis)
    x = ops.transpose(x, idx)

    if x.ndim > 1:
        h = ops.reshape(h, [1] * (x.ndim - 1) + [-1])

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
    N = ops.cast(N, "complex64")

    x = Xf_r_inv / N
    x = x + 1j * (-Xf_i_inv / N)

    # switch back to original shape
    idx = list(range(n_dim))
    idx.insert(axis, idx.pop(-1))
    x = ops.transpose(x, idx)
    return x


def demodulate(data, demodulation_frequency, sampling_frequency, axis=-3):
    """Demodulates the input data to baseband. The function computes the analytical
    signal (the signal with negative frequencies removed) and then shifts the spectrum
    of the signal to baseband by multiplying with a complex exponential. Where the
    spectrum was centered around `center_frequency` before, it is now centered around
    0 Hz. The baseband IQ data are complex-valued. The real and imaginary parts
    are stored in two real-valued channels.

    Args:
        data (ops.Tensor): The input data to demodulate of shape `(..., axis, ..., 1)`.
        demodulation_frequency (float): The center frequency of the signal.
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
    demodulation_frequency = ops.cast(demodulation_frequency, dtype="complex64")
    sampling_frequency = ops.cast(sampling_frequency, dtype="complex64")
    frequency_indices_shaped_like_rf = ops.cast(frequency_indices_shaped_like_rf, dtype="complex64")

    # Shift to baseband
    phasor_exponent = (
        -1j
        * 2
        * np.pi
        * demodulation_frequency
        * frequency_indices_shaped_like_rf
        / sampling_frequency
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

        # Calculate next power of 2: M = 2^ceil(log2(n_ax))
        # see https://github.com/tue-bmd/zea/discussions/147
        log2_n_ax = np.log2(n_ax)
        M = int(2 ** np.ceil(log2_n_ax))

        data = hilbert(data, N=M, axis=axis)
        indices = ops.arange(n_ax)

        data = ops.take(data, indices, axis=axis)
        data = ops.squeeze(data, axis=-1)

    data = ops.abs(data)
    return data


def log_compress(data, eps=1e-16):
    """Apply logarithmic compression to data."""
    eps = ops.convert_to_tensor(eps, dtype=data.dtype)
    data = ops.where(data == 0, eps, data)  # Avoid log(0)
    return 20 * ops.log10(data)


def make_tgc_curve(n_ax, attenuation_coef, sampling_frequency, center_frequency, sound_speed=1540):
    """
    Create a Time Gain Compensation (TGC) curve to compensate for depth-dependent attenuation.

    Args:
        n_ax (int): Number of samples in the axial direction
        attenuation_coef (float): Attenuation coefficient in dB/cm/MHz.
            For example, typical value for soft tissue is around 0.5 to 0.75 dB/cm/MHz.
        sampling_frequency (float): Sampling frequency in Hz
        center_frequency (float): Center frequency in Hz
        sound_speed (float): Speed of sound in m/s (default: 1540)

    Returns:
        np.ndarray: TGC gain curve of shape (n_ax,) in linear scale
    """
    # Time vector for each sample
    t = np.arange(n_ax) / sampling_frequency  # seconds

    # Distance traveled (round trip, so divide by 2)
    dist = (t * sound_speed) / 2  # meters

    # Convert distance to cm
    dist_cm = dist * 100

    # Attenuation in dB (two-way: transmit + receive)
    attenuation_db = 2 * attenuation_coef * dist_cm * (center_frequency * 1e-6)

    # Convert dB to linear scale (TGC gain curve)
    tgc_gain_curve = 10 ** (attenuation_db / 20)

    return tgc_gain_curve.astype(np.float32)


def dehaze_nuclear_diffusion(
    hazy_video,
    diffusion_model,
    n_steps: int = 5000,
    initial_step: int = 4500,
    window_size: int = 7,
    window_stride: int | None = None,
    hard_project: bool = True,
    seed=None,
    verbose: bool = True,
    **guidance_kwargs,
):
    r"""Dehaze ultrasound videos using Nuclear Diffusion posterior sampling.

    This function performs video dehazing by combining diffusion posterior sampling
    with low-rank temporal modeling. It processes long video sequences by splitting
    them into overlapping windows, applying `Nuclear Diffusion <https://tue-bmd.github.io/nuclear-diffusion/>`_
    to each window, and averaging predictions across windows for smooth results.

    .. seealso::

        - :doc:`../../notebooks/models/nuclear_dehazing_example`: Detailed tutorial notebook
        - :class:`~zea.models.diffusion.NuclearDiffusion`: The guidance method used for dehazing
        - :func:`~zea.func.split_into_windows`: Window splitting utility

    The method performs posterior sampling to separate the video into:

    - **Tissue component** (:math:`\mathbf{X}`): Dynamic foreground signal with complex structure
    - **Haze component** (:math:`\mathbf{L}`): Low-rank background artifacts

    Nuclear Diffusion replaces the sparsity prior in RPCA with a learned diffusion prior
    while maintaining a nuclear norm penalty on the background component.
    Given video observations :math:`\mathbf{Y} \in \mathbb{R}^{n \times p}`,
    the method jointly samples:

    .. math::

        \mathbf{X}, \mathbf{L} \sim p_\theta(\mathbf{X}, \mathbf{L} \mid \mathbf{Y})

    where :math:`\mathbf{X}` is the dynamic foreground (tissue) and :math:`\mathbf{L}` is the
    low-rank background (haze). The posterior factorizes as:

    .. math::

        p(\mathbf{Y}, \mathbf{L}, \mathbf{X}) = p(\mathbf{Y} \mid \mathbf{L}, \mathbf{X}) \, p(\mathbf{L}) \, p_\theta(\mathbf{X})

    - **Likelihood**: :math:`p(\mathbf{Y} \mid \mathbf{L}, \mathbf{X}) = \mathcal{N}(\mathbf{Y}; \mathbf{L}+\mathbf{X}, \mu^{-1} \mathbf{I})`
    - **Low-rank prior**: :math:`p(\mathbf{L}) \propto \exp(-\gamma \|\mathbf{L}\|_*)` where :math:`\|\mathbf{L}\|_* = \sum_i \sigma_i(\mathbf{L})` is the nuclear norm
    - **Diffusion prior**: :math:`p_\theta(\mathbf{X})` learned from data, capturing complex signal structure

    The method operates by alternating between reverse diffusion and measurement-guided updates,
    minimizing both the data fidelity and the low-rank penalty. This allows it to effectively
    separate structured foreground dynamics from the low-rank haze,
    even when the foreground is not sparse.

    Args:
        hazy_video: Input hazy video as a tensor of shape ``(frames, height, width, channels)``.
        diffusion_model: Pre-trained diffusion model configured with Nuclear Diffusion guidance
            (``guidance="nuclear-dps"``) and haze operator (``operator="linear_interp"``).
        n_steps: Number of diffusion steps for posterior sampling. More steps generally
            produce better quality but take longer. Default is 5000.
        initial_step: Starting step for progressive blending in the diffusion process.
            Must be less than ``n_steps`` and non-negative. Passed to the NuclearDiffusion
            guidance function's ``compute_error`` method. Default is 4500.
        window_size: Number of frames to process together in each window. Larger windows
            capture more temporal context but require more memory. Default is 7.
        window_stride: Stride between consecutive windows. If ``None``, uses non-overlapping
            windows (stride = window_size). Smaller strides create more overlap and smoother
            results but increase computation time.
        hard_project: Whether to preserve bright speckle values from the input by projecting
            positive values from the hazy input. This helps preserve fine tissue texture.
            Default is ``True``.
        seed: Random seed for reproducibility. If ``None``, uses default random state.
        verbose: Whether to display progress information. Default is ``True``.
        **guidance_kwargs: Additional keyword arguments for Nuclear Diffusion guidance:

            - **omega** (float): Weight for measurement error term. Default is 1.0.
            - **gamma** (float): Weight for nuclear norm penalty. Default is 1.0.
            - **rank_weight_factor** (float, optional): Enhanced weighting for larger singular values.

    Returns:
        tuple: A tuple ``(tissue_frames, haze_frames)`` containing:

        - **tissue_frames**: Dehazed tissue component as a numpy array.
        - **haze_frames**: Estimated low-rank haze component as a numpy array.

    Raises:
        ValueError: If the model is not configured with Nuclear Diffusion guidance.

    .. note::
        This function requires a diffusion model with Nuclear Diffusion guidance.
        Initialize your model with ``guidance="nuclear-dps"`` and ``operator="linear_interp"``.

    .. admonition:: Reference

        T. S. W. Stevens, O. Nolan, J.-L. Robert, and R. J. G. van Sloun,
        "Nuclear Diffusion Models for Low-Rank Background Suppression in Videos,"
        *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2026.
        https://arxiv.org/abs/2509.20886

    """  # noqa: E501

    assert initial_step < n_steps, "initial_step must be less than n_steps."
    assert initial_step >= 0, "initial_step must be non-negative."
    assert diffusion_model is not None, (
        "You must pass a diffusion model to `dehaze_nuclear_diffusion`. To see which models are "
        "available on zeahub, visit https://huggingface.co/zeahub/models or"
        "see the available presets: https://github.com/tue-bmd/zea/blob/main/zea/models/presets.py"
    )

    def _nuclear_diffusion_posterior_sample(
        diffusion_model,
        measurements,
        n_steps: int,
        seed=None,
        verbose: bool = True,
        initial_step: int = 100,
        **guidance_kwargs,
    ):
        """Internal method for Nuclear Diffusion posterior sampling.

        This method performs posterior sampling for a single batch/window of frames.
        It alternates between reverse diffusion on the tissue component and gradient updates
        that enforce measurement consistency and low-rank structure on the haze component.

        Args:
            diffusion_model: The diffusion model with Nuclear Diffusion guidance.
            measurements: Measurements of shape ``(batch, frames, H, W, C)``.
            n_steps: Number of diffusion steps.
            seed: Random seed.
            verbose: Whether to show progress.
            initial_step: Starting diffusion step.
            **guidance_kwargs: Guidance parameters (omega, gamma, etc.).

        Returns:
            tuple: ``(tissue_images, haze_images)`` as tensors.
        """

        measurements = ops.convert_to_tensor(measurements)
        image_shape = ops.shape(measurements)

        # Ensure 5D input: (batch, frames, height, width, channels)
        if len(image_shape) != 5:
            raise ValueError(f"Expected 5D input (batch, frames, H, W, C), got shape {image_shape}")

        n_batches, n_frames, image_height, image_width, n_channels = image_shape
        frame_shape = (n_batches, n_frames, image_height, image_width, n_channels)

        # Prepare diffusion: validates params, computes step size, sets up progress tracking
        step_size, progbar = diffusion_model.prepare_diffusion(n_steps, initial_step, verbose)

        # Seed splitting handles None gracefully across all backends
        seed, seed1 = split_seed(seed, 2)
        initial_noise_tissue = keras.random.normal(shape=frame_shape, seed=seed1)
        initial_noise_haze = ops.zeros(frame_shape)

        # Base diffusion times (same pattern as reverse_diffusion / reverse_conditional_diffusion)
        base_diffusion_times = ops.ones((n_batches, n_frames, 1, 1, 1)) * diffusion_model.max_t

        # Initialize noisy samples at the starting diffusion time
        start_diffusion_times = base_diffusion_times - initial_step * step_size
        noise_rates, signal_rates = diffusion_model.diffusion_schedule(start_diffusion_times)
        next_noisy_tissue = signal_rates * measurements + noise_rates * initial_noise_tissue
        next_noisy_haze = initial_noise_haze

        initial_step_t = ops.convert_to_tensor(initial_step, dtype=initial_noise_tissue.dtype)

        # Reverse diffusion loop
        for step in range(initial_step, n_steps):
            noisy_tissue = next_noisy_tissue
            noisy_haze = next_noisy_haze

            # Compute diffusion schedule for current and next step
            diffusion_times = base_diffusion_times - step * step_size
            noise_rates, signal_rates = diffusion_model.diffusion_schedule(diffusion_times)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = diffusion_model.diffusion_schedule(
                next_diffusion_times
            )

            # Compute gradients from guidance function
            (
                (gradients_tissue, gradients_haze),
                (
                    measurement_error,
                    (pred_noises_tissue, pred_tissue, pred_haze, l2_error, nuclear_penalty),
                ),
            ) = diffusion_model.guidance_fn(
                noisy_tissue,
                noisy_haze,
                measurements=measurements,
                noise_rates=noise_rates,
                signal_rates=signal_rates,
                initial_step=initial_step_t,
                step=step,
                total_steps=n_steps,
                **guidance_kwargs,
            )

            # DDIM step for tissue component (deterministic)
            next_noisy_tissue = diffusion_model.reverse_diffusion_step(
                shape=frame_shape,
                pred_images=pred_tissue,
                pred_noises=pred_noises_tissue,
                signal_rates=signal_rates,
                next_signal_rates=next_signal_rates,
                next_noise_rates=next_noise_rates,
            )
            next_noisy_haze = pred_haze

            # Apply guidance updates
            next_noisy_tissue = next_noisy_tissue - gradients_tissue
            next_noisy_haze = next_noisy_haze - gradients_haze

            if progbar is not None:
                progbar.update(
                    step + 1,
                    [
                        ("total_error", measurement_error),
                        ("l2_error", l2_error),
                        ("nuclear_penalty", nuclear_penalty),
                    ],
                )

        return pred_tissue, pred_haze

    # Validate configuration
    if diffusion_model.guidance_fn is None:
        raise ValueError(
            "Model must have guidance function set. Initialize with guidance='nuclear-dps'."
        )

    # Import here to avoid circular dependency
    from zea.models.diffusion import NuclearDiffusion

    if not isinstance(diffusion_model.guidance_fn, NuclearDiffusion):
        raise ValueError(
            f"dehaze_nuclear_diffusion() requires Nuclear Diffusion guidance, "
            f"but model has {type(diffusion_model.guidance_fn).__name__}. "
            "Initialize with guidance='nuclear-dps'."
        )

    # Get sequence length
    seq_len = ops.shape(hazy_video)[0]

    if verbose:
        log.info(f"[Nuclear Diffusion] Processing {seq_len} frames.")

    # Split video into windows
    windows, window_indices = split_into_windows(
        hazy_video, window_size=window_size, stride=window_stride
    )

    if verbose:
        log.info(
            f"[Nuclear Diffusion] Split into {len(windows)} windows with sizes:"
            f" {[len(w) for w in windows]}"
        )

    # Accumulate predictions for each frame
    frame_tissue_preds = [[] for _ in range(int(seq_len))]
    frame_haze_preds = [[] for _ in range(int(seq_len))]

    progbar = keras.utils.Progbar(len(windows), verbose=verbose, unit_name="window")

    # Process each window
    for window_idx, (window, frame_indices) in enumerate(zip(windows, window_indices)):
        window_batch = ops.expand_dims(window, axis=0)  # Add batch dimension
        seed, window_seed = split_seed(seed, 2)

        tissue_images, haze_images = _nuclear_diffusion_posterior_sample(
            diffusion_model,
            measurements=window_batch,
            n_steps=n_steps,
            initial_step=initial_step,
            seed=window_seed,
            verbose=False,  # Disable per-window progress
            **guidance_kwargs,
        )

        # Remove batch dimension
        tissue_frames_window = ops.squeeze(tissue_images, axis=0)
        haze_frames_window = ops.squeeze(haze_images, axis=0)

        # Accumulate predictions for overlapping frames
        for i, frame_idx in enumerate(frame_indices):
            frame_tissue_preds[frame_idx].append(tissue_frames_window[i])
            frame_haze_preds[frame_idx].append(haze_frames_window[i])

        progbar.add(1)

    # Average predictions across overlapping windows
    tissue_frames = []
    haze_frames = []

    for i in range(int(seq_len)):
        # Stack and average tissue predictions
        stacked_tissue = ops.stack(frame_tissue_preds[i], axis=0)
        tissue_frames.append(ops.mean(stacked_tissue, axis=0))

        # Stack and average haze predictions
        stacked_haze = ops.stack(frame_haze_preds[i], axis=0)
        haze_frames.append(ops.mean(stacked_haze, axis=0))

    # Stack frames into sequences
    tissue_frames = ops.stack(tissue_frames, axis=0)
    haze_frames = ops.stack(haze_frames, axis=0)

    # Apply hard projection if requested
    if hard_project:
        tissue_np = ops.convert_to_numpy(tissue_frames)
        hazy_np = ops.convert_to_numpy(hazy_video)

        # Preserve bright speckle values from hazy input
        proj = tissue_np.copy()
        proj[proj > 0] = hazy_np[proj > 0]
        tissue_frames = proj

        # Recompute haze from preserved tissue
        haze_frames = hazy_np - tissue_frames - 1
    else:
        # Convert to numpy
        tissue_frames = ops.convert_to_numpy(tissue_frames)
        haze_frames = ops.convert_to_numpy(haze_frames)
        hazy_np = ops.convert_to_numpy(hazy_video)
        haze_frames = hazy_np - haze_frames - 1

    return tissue_frames, haze_frames
