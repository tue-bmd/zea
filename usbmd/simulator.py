from keras import ops
from usbmd.utils.lens_correction import compute_lens_corrected_travel_times

pi = 3.14159265359


def simulate_rf(
    scatterer_positions,
    scatterer_magnitudes,
    probe_geometry,
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    sound_speed,
    n_ax,
    center_frequency,
    sampling_frequency,
    t0_delays,
    initial_times,
    element_width,
    attenuation_coef,
    tx_apodizations,
):
    """
    Simulates RF data for a given set of scatterers.

    Args:
    scatterer_positions (array-like): The positions of the scatterers [m] of shape (n_scat, 3).
    scatterer_magnitudes (array-like): The magnitudes of the scatterers of shape (n_scat,).
    probe_geometry (array-like): The geometry of the probe [m] of shape (n_el, 3).
    apply_lens_correction (bool): Whether to apply lens correction.
    lens_thickness (float): The thickness of the lens [m].
    lens_sound_speed (float): The speed of sound in the lens [m/s].
    sound_speed (float): The speed of sound in the medium [m/s].
    n_ax (int): The number of samples in the RF data.
    center_frequency (float): The center frequency of the pulse [Hz].
    sampling_frequency (float): The sampling frequency of the RF data [Hz].
    t0_delays (array-like): The delays of the transmitting elements [s] of shape (n_tx, n_el).
    initial_times (array-like): The initial times of the transmitting elements [s] of shape (n_tx,).
    element_width (float): The width of the elements [m].
    attenuation_coef (float): The attenuation coefficient [dB/cm/MHz].
    tx_apodizations (array-like): The apodizations of the transmitting elements of shape (n_tx, n_el).

    Returns:
    rf_data (array-like): The simulated RF data of shape (1, n_ax, n_tx, n_el, 1).
    """

    n_tx = t0_delays.shape[0]

    pulse_spectrum_fn = get_pulse_spectrum_fn(center_frequency, n_period=4)

    if not apply_lens_correction:
        dist = ops.linalg.norm(
            probe_geometry[None] - scatterer_positions[:, None], axis=-1
        )
    else:
        dist = (
            compute_lens_corrected_travel_times(
                probe_geometry,
                scatterer_positions,
                lens_thickness=lens_thickness,
                c_lens=lens_sound_speed,
                c_medium=sound_speed,
                n_iter=3,
            )
            * sound_speed
        )

    freqs = (
        ops.cast(ops.arange(n_ax // 2 + 1) / n_ax, "float32") * sampling_frequency + 1
    )

    waveform_spectrum = pulse_spectrum_fn(freqs)
    parts = []
    for tx in range(n_tx):

        tx_idx = ops.array(tx)

        # [n_scat, n_txel, rxel]
        dist_total = dist[:, None] + dist[:, :, None]

        # [n_scat, n_txel, n_rxel]
        tau_total = dist_total / sound_speed + t0_delays[tx_idx] + initial_times[tx_idx]

        scat_pos_relative_to_probe = scatterer_positions[:, None] - probe_geometry[None]

        # Compute 3D directivity
        theta = ops.arctan2(
            scat_pos_relative_to_probe[:, :, 0], scat_pos_relative_to_probe[:, :, 2]
        )
        phi = ops.arctan2(
            scat_pos_relative_to_probe[:, :, 1], scat_pos_relative_to_probe[:, :, 2]
        )

        directivity_tx = directivity(
            freqs[None, None, None],
            theta[..., None, None],
            element_width,
            sound_speed,
        ) * directivity(
            freqs[None, None, None],
            phi[..., None, None],
            element_width,
            sound_speed,
        )
        directivity_rx = directivity(
            freqs[None, None, None],
            theta[:, None, :, None],
            element_width,
            sound_speed,
        ) * directivity(
            freqs[None, None, None],
            phi[:, None, :, None],
            element_width,
            sound_speed,
        )

        attenuation = attenuate(
            freqs[None, None, None],
            attenuation_coef=attenuation_coef,
            dist=dist_total[..., None],
        )

        spread_atten = spread(dist_total[..., None])

        result = (
            waveform_spectrum[None, None, None]
            * delay2(
                freqs[None, None, None],
                tau_total[..., None],
                n_fft=n_ax,
                fs=sampling_frequency,
            )
            * ops.cast(
                scatterer_magnitudes[:, None, None, None]
                * tx_apodizations[tx, None, :, None, None]
                * directivity_tx
                * directivity_rx
                * attenuation
                * spread_atten,
                "complex64",
            )
        )

        # Sum over all transmitting elements and scatterers
        result = ops.sum(result, axis=[0, 1])

        result = ops.irfft((ops.real(result), ops.imag(result)))

        parts.append(result)

    rf_data = ops.stack(parts, axis=0)
    rf_data = ops.transpose(rf_data, (0, 2, 1))
    return rf_data[None, ..., None]


def travel(distance, sound_speed, frequency):
    return ops.exp(-1j * distance / sound_speed * frequency)


def directivity(f, theta, element_width, sound_speed, rigid_baffle=True):
    """Computes the directivity of a single element.

    Args:
        f (array-like): The input frequencies [Hz].
        theta (array-like): The angles [rad].
        element_width (float): The width of the element [m].
        sound_speed (float): The speed of sound [m/s].
        rigid_baffle (bool): Whether the element is mounted on a rigid baffle,
            impacting the directivity.

    Returns:
        array-like: The directivity of the element.
    """

    wavelength = sound_speed / f

    response = sinc(element_width / wavelength * ops.sin(theta))
    if not rigid_baffle:
        response *= ops.cos(theta)
    return response


def delay2(f, tau, n_fft, fs):
    """
    Applies a delay in the frequency domain without phase wrapping.

    Args:
        f (array-like): The input frequencies.
        tau (float): The delay to apply.
        n_fft (int): The number of samples in the FFT.
        fs (float): The sampling frequency.

    Returns:
        array-like: The spectrum of the delay.
    """
    arg = ops.array(-1j, dtype="complex64") * ops.cast(2 * pi * tau * f, "complex64")
    return ops.where(tau < n_fft / fs, ops.exp(arg), ops.array(0.0, dtype="complex64"))


def attenuate(f, attenuation_coef, dist):
    """
    Applies attenuation to the signal in the frequency domain.

    Args:
        f (array-like): The input frequencies.
        attenuation_coef (float): The attenuation coefficient in dB/cm/MHz.
        dist (float): The distance the signal has traveled.

    Returns:
        array-like: The spectrum of the attenuation.
    """
    return ops.exp(
        -ops.log(10) * attenuation_coef / 20 * dist * 100 * ops.abs(f) * 1e-6
    )


def spread(dist, mindist=1e-4):
    """Function modeling geometric spreading of the wavefront."""
    dist = ops.clip(dist, mindist, float("inf"))
    return mindist / dist


def hann_fd(f, width):
    """The fourier transform of a hann window in the time domain with given width."""
    denom = 1.0 - (f * width) ** 2
    num = 0.5 * sinc(f * width)
    result = num / denom
    result = ops.where(ops.abs(result) > 1.1, 0.25, result)
    return ops.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.25)


def hann_unnormalized(x, width):
    """Hann window function that is 1 at the peak. This means that the integral of the
    window function is not necessarily 1.

    Args:
    x (array-like): The input values.
    width (float): The width of the window. This is the total width from -x to x. The
        window will be nonzero in the range [-width/2, width/2].

    Returns:
    hann_vals (array-like): The values of the Hann window function.
    """
    return ops.where(ops.abs(x) < width / 2, ops.cos(pi * x / width) ** 2, 0)


def get_pulse_spectrum_fn(fc, n_period=3.0):
    """Computes the spectrum of a sine that is windowed with a Hann window.

    Args:
    fc (float): The center frequency of the pulse.
    n_period (float): The number of periods to include in the pulse.

    Returns:
    spectrum_fn (callable): A function that computes the spectrum of the pulse for the
        input frequencies in Hz.
    """
    period = n_period / fc

    def spectrum_fn(f):
        return ops.array(1 / 1j, "complex64") * ops.cast(
            (hann_fd(f - fc, period) - hann_fd(f + fc, period)), "complex64"
        )

    return spectrum_fn


def get_transducer_bandwidth_fn(fc, bandwidth):
    """Computes the spectrum of a probe with a center frequency and bandwidth.

    Args:
    fc (float): The center frequency of the probe.
    bandwidth (float): The bandwidth of the probe.

    Returns
    spectrum_fn (callable): A function that computes the spectrum of the pulse for the
        input frequencies in Hz.
    """

    def bandwidth_fn(f):
        return hann_unnormalized(ops.abs(f) - fc, bandwidth)

    return bandwidth_fn


def sinc(x):
    """The normalized sinc function with a small offset to precent division by zero."""
    x = ops.abs(pi * x) + 1e-9
    return ops.sin(x) / x
