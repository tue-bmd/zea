from usbmd import Scan
from keras import ops
import numpy as np


def sinc(x):
    x = np.pi * x
    return ops.where(x == 0, 1.0, ops.sin(x) / x)


def simulate_rf(scan: Scan, scat_positions, scat_magnitudes):

    pulse_spectrum_fn = get_pulse_spectrum_fn(scan.fc, n_period=3)

    dist = ops.linalg.norm(scan.probe_geometry[None] - scat_positions[:, None], axis=-1)
    freqs = ops.arange(scan.n_ax // 2 + 1) / scan.n_ax * scan.fs
    waveform_spectrum = pulse_spectrum_fn(freqs)
    parts = []
    for tx in range(scan.n_tx):

        # [n_scat, n_txel]
        tau_tx = dist / scan.sound_speed + scan.t0_delays[tx]
        print(tau_tx.shape)

        # [n_scat, n_txel, n_rxel]
        tau_total = tau_tx[:, :, None] + dist[:, None, :] / scan.sound_speed

        result = waveform_spectrum * delay2(
            freqs[None, None, None], tau_total[..., None], n_fft=scan.n_ax, fs=scan.fs
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


def directivity(theta, frequency, sound_speed):
    arg = np.pi * sound_speed / frequency * ops.sin(theta)
    return sinc(arg) * ops.cos(theta)


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


if __name__ == "__main__":
    n_el = 128
    n_scat = 5
    aperture = 20e-3
    scan = Scan(
        n_tx=1,
        n_ax=512,
        n_el=n_el,
        center_frequency=3.125e6,
        sampling_frequency=12.5e6,
        probe_geometry=ops.stack(
            [
                ops.linspace(-aperture / 2, aperture / 2, n_el),
                ops.zeros(n_el),
                ops.zeros(n_el),
            ],
            axis=1,
        ),
        t0_delays=ops.zeros((1, n_el)),
    )

    rf = simulate_rf(
        scan=scan,
        scat_positions=ops.stack(
            [
                ops.linspace(-10e-3, 10e-3, n_scat),
                ops.zeros(n_scat),
                ops.ones(n_scat) * 10e-3,
            ],
            axis=1,
        ),
        scat_magnitudes=ops.ones(n_scat),
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(rf[0], aspect="auto")
    plt.show()
