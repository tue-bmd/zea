from usbmd import Scan
from keras import ops
import numpy as np
from usbmd.utils.lens_correction import compute_lens_corrected_travel_times


def simulate_rf(scan: Scan, scat_positions, scat_magnitudes):

    pulse_spectrum_fn = get_pulse_spectrum_fn(scan.fc, n_period=4)

    if not scan.apply_lens_correction:
        dist = ops.linalg.norm(
            scan.probe_geometry[None] - scat_positions[:, None], axis=-1
        )
    else:
        dist = (
            compute_lens_corrected_travel_times(
                scan.probe_geometry,
                scat_positions,
                lens_thickness=scan.lens_thickness,
                c_lens=scan.lens_sound_speed,
                c_medium=scan.sound_speed,
                n_iter=3,
            )
            * scan.sound_speed
        )

    freqs = ops.arange(scan.n_ax // 2 + 1) / scan.n_ax * scan.fs + 1

    waveform_spectrum = pulse_spectrum_fn(freqs)
    parts = []
    for tx in range(scan.n_tx):

        tx_idx = ops.array(tx)

        # [n_scat, n_txel, rxel]
        dist_total = dist[:, None] + dist[:, :, None]

        # [n_scat, n_txel, n_rxel]
        tau_total = (
            dist_total / scan.sound_speed
            + scan.t0_delays[tx_idx]
            + scan.initial_times[tx_idx]
        )

        scat_pos_relative_to_probe = scat_positions[:, None] - scan.probe_geometry[None]

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
            scan.element_width,
            scan.sound_speed,
        ) * directivity(
            freqs[None, None, None],
            phi[..., None, None],
            scan.element_width,
            scan.sound_speed,
        )
        directivity_rx = directivity(
            freqs[None, None, None],
            theta[:, None, :, None],
            scan.element_width,
            scan.sound_speed,
        ) * directivity(
            freqs[None, None, None],
            phi[:, None, :, None],
            scan.element_width,
            scan.sound_speed,
        )

        attenuation = attenuate(
            freqs[None, None, None],
            attenuation_coef=scan.attenuation_coef,
            dist=dist_total[..., None],
        )

        spread_atten = spread(dist_total[..., None])

        result = (
            waveform_spectrum[None, None, None]
            * scat_magnitudes[:, None, None, None]
            * scan.tx_apodizations[tx, None, :, None, None]
            * delay2(
                freqs[None, None, None],
                tau_total[..., None],
                n_fft=scan.n_ax,
                fs=scan.fs,
            )
            * directivity_tx
            * directivity_rx
            * attenuation
            * spread_atten
        )

        # Sum over all transmitting elements and scatterers
        result = ops.sum(result, axis=[0, 1])

        result = ops.irfft((result.real, result.imag))

        parts.append(result)

    rf_data = ops.stack(parts, axis=0)
    rf_data = ops.transpose(rf_data, (0, 2, 1))
    return rf_data


def travel(distance, sound_speed, frequency):
    return ops.exp(-1j * distance / sound_speed * frequency)


# def directivity(theta, frequency, sound_speed):
#     arg = np.pi * sound_speed / frequency * ops.sin(theta)
#     return sinc(arg) * ops.cos(theta)


def directivity(f, theta, element_width, sound_speed, rigid_baffle=True):

    wavelength = sound_speed / f

    response = sinc(element_width / wavelength * ops.sin(theta))
    if not rigid_baffle:
        response *= ops.cos(theta)
    return response


def delay(f, tau):
    """Applies a delay in the frequency domain.

    Parameters
    ----------
    f : array-like
        The input frequencies.
    tau : float
        The delay to apply.

    Returns
    -------
    spect : array-like
        The spectrum of the delay.
    """
    return ops.exp(-1j * 2 * np.pi * tau * f)


def delay2(f, tau, n_fft, fs):
    """Applies a delay in the frequency domain without phase wrapping.

    Parameters
    ----------
    f : array-like
        The input frequencies.
    tau : float
        The delay to apply.
    n_fft : int
        The number of samples in the FFT.
    fs : float
        The sampling frequency.

    Returns
    -------
    spect : array-like
        The spectrum of the delay.
    """
    return ops.where(tau < n_fft / fs, ops.exp(-1j * 2 * np.pi * tau * f), 0)


def attenuate(f, attenuation_coef, dist):
    """Applies attenuation to the signal in the frequency domain.

    Parameters
    ----------
    f : array-like
        The input frequencies.
    attenuation_coef : float
        The attenuation coefficient in dB/cm/MHz.
    dist : float
        The distance the signal has traveled.

    Returns
    -------
    spect : array-like
        The spectrum of the attenuation.
    """
    return ops.exp(
        -ops.log(10) * attenuation_coef / 20 * dist * 100 * ops.abs(f) * 1e-6
    )


def spread(dist, mindist=1e-4):
    dist = ops.clip(dist, mindist, None)
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

    Parameters
    ----------
    x : array-like
        The input values.
    width : float
        The width of the window. This is the total width from -x to x. The window will
        be nonzero in the range [-width/2, width/2].

    Returns
    -------
    hann_vals : array-like
        The values of the Hann window function.
    """
    return ops.where(ops.abs(x) < width / 2, ops.cos(np.pi * x / width) ** 2, 0)


def get_pulse_spectrum_fn(fc, n_period=3.0):
    """Computes the spectrum of a sine that is windowed with a Hann window.

    Parameters
    ----------
    fc : float
        The center frequency of the pulse.
    n_period : float
        The number of periods to include in the pulse.

    Returns
    -------
    spectrum_fn : callable
        A function that computes the spectrum of the pulse for the input frequencies
        in Hz.
    """
    period = n_period / fc

    def spectrum_fn(f):
        return 1 / 1j * (hann_fd(f - fc, period) - hann_fd(f + fc, period))

    return spectrum_fn


def get_transducer_bandwidth_fn(fc, bandwidth):
    """Computes the spectrum of a probe with a center frequency and bandwidth.

    Parameters
    ----------
    fc : float
        The center frequency of the probe.
    bandwidth : float
        The bandwidth of the probe.

    Returns
    -------
    spectrum_fn : callable
        A function that computes the spectrum of the pulse for the input frequencies
        in Hz.
    """

    def bandwidth_fn(f):
        return hann_unnormalized(ops.abs(f) - fc, bandwidth)

    return bandwidth_fn


def sinc(x):
    x = ops.abs(np.pi * x) + 1e-9
    return ops.sin(x) / x


if __name__ == "__main__":
    n_el = 128
    n_scat = 3
    n_tx = 2

    aperture = 30e-3
    from scipy.signal.windows import hann

    tx_apodizations = np.sqrt(hann(n_el)) * np.ones((n_tx, 1))
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )

    # tx_apodizations[:, n_el // 2] = 1.0
    scan = Scan(
        n_tx=n_tx,
        n_ax=512,
        n_el=n_el,
        center_frequency=3.125e6,
        sampling_frequency=12.5e6,
        probe_geometry=probe_geometry,
        t0_delays=np.zeros((n_tx, n_el)),
        tx_apodizations=tx_apodizations,
        element_width=ops.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        apply_lens_correction=True,
        lens_sound_speed=1440,
        lens_thickness=1e-3,
        initial_times=np.zeros((n_tx,)),
        attenuation_coef=0.7,
    )
    scat_x, scat_z = np.meshgrid(
        np.linspace(-10e-3, 10e-3, 5), np.linspace(5e-3, 30e-3, 5), indexing="ij"
    )
    scat_x, scat_z = scat_x.flatten(), scat_z.flatten()
    n_scat = len(scat_x)
    scat_positions = np.stack(
        [
            scat_x,
            np.zeros_like(scat_x),
            scat_z,
        ],
        axis=1,
    )

    rf = simulate_rf(
        scan=scan,
        scat_positions=scat_positions,
        scat_magnitudes=np.ones(n_scat),
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(rf[0], aspect="auto")
    plt.show()
