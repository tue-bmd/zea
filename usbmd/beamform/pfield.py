"""Automatic pressure field computation used for compounding multiple Tx events"""

import keras
import numpy as np
from keras import ops

from usbmd import log
from usbmd.internal.cache import cache_output


@cache_output(verbose=True)
def compute_pfield(
    scan,
    frequency_step=4,
    db_thresh=-1,
    downsample=10,
    downmix=4,
    alpha=1,
    perc=10,
    norm=True,
    verbose=True,
):
    """Compute the pressure field for ultrasound imaging.

    Args:
        scan (Scan): The ultrasound scan object.
        frequency_step (int, optional): Frequency step. Default is 4.
            Higher is faster but less accurate.
        db_thresh (int, optional): dB threshold. Default is -1.
            Higher is faster but less accurate.
        downsample (int, optional): Downsample the grid for faster computation.
            Default is 10. Higher is faster but less accurate.
        downmix (int, optional): Downmixing the frequency to facilitate a smaller grid.
            Default is 4. Higher requires lower number of grid points but is less accurate.
        alpha (float, optional): Exponent to 'sharpen or smooth' the weighting. Higher is sharper.
            Default is 1.
        perc (int, optional): minimum percentile threshold to keep in the weighting
            Higher is more aggressive) Default is 10.
        norm (bool, optional): per pixel normalization (True) or unnormalized (False)
        verbose (bool, optional): Whether to print progress.

    Returns:
        ops.array: The normalized pressure field (across tx events) of shape (n_tx, Nz, Nx).
    """
    # medium params
    alpha_db = 0  # currently we ignore attenuation in the compounding

    sound_speed = scan.sound_speed

    # probe params
    center_frequency = scan.center_frequency  # central frequency (Hz)
    center_frequency = center_frequency / downmix  # downmixing the frequency

    bandwidth_percent = (
        scan.bandwidth_percent
    )  # pulse-echo 6dB fractional bandwidth (%)

    # pulse params
    num_waveforms = 1  # number of waveforms in the pulse

    # array params
    probe_geometry = ops.convert_to_tensor(scan.probe_geometry, dtype="float32")

    num_elements = scan.n_el  # number of elements
    pitch = probe_geometry[1, 0] - probe_geometry[0, 0]  # element pitch

    kerf = 0.1 * pitch  # for now this is hardcoded
    element_width = pitch - kerf

    num_transmits = len(scan.tx_apodizations)

    # %------------------------------------%
    # % POINT LOCATIONS, DISTANCES & GRIDS %
    # %------------------------------------%

    # subdivide elements into sub elements or not? (to satisfy Fraunhofer approximation)
    lambda_min = sound_speed / (center_frequency * (1 + bandwidth_percent / 200))
    num_sub_elements = ops.ceil(element_width / lambda_min)

    x_orig = ops.convert_to_tensor(scan.grid[:, :, 0], dtype="float32")
    z_orig = ops.convert_to_tensor(scan.grid[:, :, 2], dtype="float32")

    size_orig = ops.shape(x_orig)

    # Nearest-neighbor downsampling the grid
    x = x_orig[::downsample, ::downsample]
    z = z_orig[::downsample, ::downsample]
    size_downsampled = ops.shape(x)

    # Coordinates of the points where pressure is needed
    x = ops.reshape(x, (-1,))
    z = ops.reshape(z, (-1,))

    # Centers of the transducer elements (x- and z-coordinates)
    xe = (ops.arange(0.0, num_elements) - (num_elements - 1) / 2) * pitch
    ze = ops.zeros(num_elements)
    the = ops.zeros(num_elements)

    # Centroids of the sub-elements
    seg_length = element_width / num_sub_elements
    tmp = (
        -element_width / 2
        + seg_length / 2
        + ops.arange(0, num_sub_elements, dtype=seg_length.dtype) * seg_length
    )
    xi = tmp
    zi = ops.zeros((int(num_sub_elements),))

    # Distances between the points and the transducer elements
    x_expanded = x[:, None, None]
    xi_expanded = xi[None, :, None]
    xe_expanded = xe[None, None, :]

    dxi = x_expanded - xi_expanded - xe_expanded

    z_expanded = z[:, None, None]
    zi_expanded = zi[None, :, None]
    ze_expanded = ze[None, None, :]

    d2 = dxi**2 + (z_expanded - zi_expanded - ze_expanded) ** 2
    r = ops.sqrt(d2)
    r_flat = ops.reshape(r, (-1,))

    # Angle between the normal to the transducer and the line joining
    # the point and the transducer
    eps = keras.config.epsilon()
    theta = ops.arcsin((dxi + eps) / (ops.sqrt(d2) + eps)) - the
    sin_theta = ops.sin(theta)

    mysinc = lambda x: ops.sin(ops.abs(x) + eps) / (ops.abs(x) + eps)

    pulse_width = num_waveforms / center_frequency  # temporal pulse width
    wc = 2 * np.pi * center_frequency

    def pulse_spectrum(w):
        imag = mysinc(pulse_width * (w - wc) / 2) - mysinc(pulse_width * (w + wc) / 2)
        return 1j * ops.cast(imag, "complex64")

    # FREQUENCY RESPONSE of the ensemble PZT + probe
    w_bandwidth = bandwidth_percent * wc / 100  # angular frequency bandwidth
    p_shape = ops.log(126) / ops.log(eps + 2 * wc / w_bandwidth)
    probe_spectrum = lambda w: ops.exp(
        -(
            (ops.abs(w - wc) / (w_bandwidth / 2 / ops.log(2) ** (1 / p_shape)))
            ** p_shape
        )
    )

    p_list = []

    if verbose:
        log.info("Computing pressure field for all transmits")
        progbar = keras.utils.Progbar(num_transmits, unit_name="transmits")
    for j in range(0, num_transmits):
        # delays and apodization of transmit event
        delays_tx = ops.convert_to_tensor(scan.t0_delays[j], dtype="float32")
        idx_nan = ops.isnan(delays_tx)
        delays_tx = ops.where(idx_nan, 0, delays_tx)

        tx_apodization = ops.convert_to_tensor(scan.tx_apodizations[j])
        idx_nan = ops.isnan(tx_apodization)
        tx_apodization = ops.where(idx_nan, 0, tx_apodization)
        tx_apodization = ops.squeeze(tx_apodization)

        # The frequency response is a pulse-echo (transmit + receive) response.
        # The spectrum of the pulse (pulse_spectrum) will be then multiplied
        # by the frequency-domain tapering window of the transducer (probe_spectrum)
        # The frequency step df is chosen to avoid interferences due to
        # inadequate discretization.
        # df = frequency step (must be sufficiently small):
        # One has exp[-i(k r + w delay)] = exp[-2i pi(f r/c + f delay)] in the Eq.
        # One wants: the phase increment 2pi(df r/c + df delay) be < 2pi.
        # Therefore: df < 1/(r/c + delay).

        delays_tx_flat = ops.reshape(delays_tx, (-1,))

        df = 1 / (ops.max(r_flat / sound_speed) + ops.max(delays_tx_flat))
        df = frequency_step * df

        # FREQUENCY SAMPLES
        num_freq = 2 * ops.cast(ops.ceil(center_frequency / df), "int32") + 1
        freq = ops.linspace(0, 2 * center_frequency, num_freq)
        df = freq[1]

        # keep the significant components only by using db_thresh
        spectrum = ops.abs(
            pulse_spectrum(2 * np.pi * freq)
            * ops.cast(probe_spectrum(2 * np.pi * freq), "complex64")
        )
        gain_db = 20 * ops.log10(eps + spectrum / (ops.max(spectrum)))
        idx = gain_db > db_thresh

        freq = freq[idx]

        pulse_spect = pulse_spectrum(2 * np.pi * freq)
        probe_spect = probe_spectrum(2 * np.pi * freq)

        # Exponential arrays of size [numel(x) num_elements num_sub_elements]
        kw = 2 * np.pi * freq[0] / sound_speed
        kwa = alpha_db / 8.69 * freq[0] / 1e6 * 1e2

        r_complex = ops.cast(r, dtype="complex64")
        kwa = ops.cast(kwa, dtype="complex64")
        mod_out = ops.cast(ops.mod(kw * r, 2 * np.pi), dtype="complex64")
        exp_arr = ops.exp(-kwa * r_complex + 1j * mod_out)

        # Exponential array for the increment wavenumber dk
        dkw = 2 * np.pi * df / sound_speed
        dkwa = alpha_db / 8.69 * df / 1e6 * 1e2
        dkw = ops.cast(dkw, dtype="complex64")
        dkwa = ops.cast(dkwa, dtype="complex64")

        exp_df = ops.exp((-dkwa + 1j * dkw) * r_complex)

        exp_arr = exp_arr / ops.sqrt(r_complex)
        exp_arr = exp_arr * ops.cast(
            ops.min(ops.sqrt(r)), "complex64"
        )  # normalize the field

        kc = 2 * np.pi * center_frequency / sound_speed
        directivity = mysinc(kc * seg_length / 2 * sin_theta)
        exp_arr = exp_arr * ops.cast(directivity, "complex64")

        # Render pressure field for all relevant frequencies and sum them up
        rp = pfield_freq_loop(
            freq,
            sound_speed,
            delays_tx,
            tx_apodization,
            exp_arr,
            exp_df,
            pulse_spect,
            probe_spect,
            z,
        )

        # RMS acoustic pressure
        p = ops.reshape(ops.sqrt(rp), size_downsampled)

        # resize p to exactly the original grid size
        p = ops.squeeze(
            ops.image.resize(p[..., None], size_orig, interpolation="nearest"), axis=-1
        )

        p_list.append(p)

        if verbose:
            progbar.add(1)

    p_arr = ops.convert_to_tensor(p_list)
    p_arr = ops.where(
        ops.isnan(p_arr), 0, p_arr
    )  # TODO: this is necessary for Jax / TF somehow. not sure why (not for torch)

    if norm:
        p_norm = normalize(p_arr, alpha=alpha, perc=perc)
    else:
        p_norm = p_arr

    return p_norm


def normalize(p_arr, alpha=1, perc=10):
    """Normalize the input array of intensities.

    Args:
        p_arr (array): Sequence of intensity arrays.
        alpha (float, optional): Shape factor to tighten the beams. Higher is sharper.
            Default is 1.
        perc (int, optional): Percentile to keep. Default is 10.

    Returns:
        ops.array: Normalized intensity array.

    """
    # keep only the highest intensities
    # p_arr[p_arr < ops.percentile(p_arr, perc, axis=(1, 2))[:, None, None]] = 0

    # let's do manual percentile calculation
    # Flatten the last two dimensions, sort, and reshape back
    p_flat = ops.reshape(p_arr, (p_arr.shape[0], -1))
    p_sorted = ops.sort(p_flat, axis=1)
    perc_value = p_sorted[:, int(p_arr.shape[1] * p_arr.shape[2] * perc / 100)]
    p_arr = ops.where(p_arr < perc_value[:, None, None], 0, p_arr)

    p_arr = ops.convert_to_tensor(p_arr) ** alpha
    p_norm = p_arr / (keras.config.epsilon() + ops.sum(p_arr, axis=0))

    return p_norm


def pfield_freq_step(
    k,
    freq,
    sound_speed,
    delays_tx,
    tx_apodization,
    exp_arr,
    exp_df,
    pulse_spect,
    probe_spect,
    z,
):
    """
    Calculates the pressure field for a single frequency step.

    Args:
        k (int): Frequency index.
        freq (list): List of frequencies.
        sound_speed (float): Speed of sound.
        delays_tx (list): List of transmit delays.
        tx_apodization (list): List of transmit apodization values (complex64).
        exp_arr (list): List of complex exponentials.
        exp_df (list): List of complex exponential frequency shifts.
        pulse_spect (list): List of pulse spectra.
        probe_spect (list): List of probe spectra (complex64).
        z (list): List of z-coordinates.

    Returns:
        rp_k (Tensor): Pressure field for this frequency.
    """
    kw = 2 * np.pi * freq[k] / sound_speed
    rp_mono = ops.mean(exp_arr * ops.power(exp_df, k), axis=1)

    del_apod = (
        ops.exp(1j * ops.cast(kw * sound_speed * delays_tx, "complex64"))
        * tx_apodization
    )
    rp_k = ops.matmul(rp_mono, del_apod) * pulse_spect[k] * probe_spect[k]
    rp_k = ops.where(z < 0, 0, rp_k)
    return ops.abs(rp_k) ** 2


def pfield_freq_loop(
    freq,
    sound_speed,
    delays_tx,
    tx_apodization,
    exp_arr,
    exp_df,
    pulse_spect,
    probe_spect,
    z,
):
    """Calculates the pressure field using frequency loop method.

    Args:
        freq (list): List of frequencies.
        sound_speed (float): Speed of sound.
        delays_tx (list): List of transmit delays.
        tx_apodization (list): List of transmit apodization values.
        exp_arr (list): List of complex exponentials.
        exp_df (list): List of complex exponential frequency shifts.
        pulse_spect (list): List of pulse spectra.
        probe_spect (list): List of probe spectra.
        z (list): List of z-coordinates.

    Returns:
        (Tensor): Pressure field.
    """

    tx_apodization = ops.cast(tx_apodization, "complex64")
    probe_spect = ops.cast(probe_spect, "complex64")
    stacked = ops.vectorized_map(
        lambda k: pfield_freq_step(
            k,
            freq,
            sound_speed,
            delays_tx,
            tx_apodization,
            exp_arr,
            exp_df,
            pulse_spect,
            probe_spect,
            z,
        ),
        np.arange(0, len(freq), dtype="int32"),
    )
    return ops.sum(stacked, axis=0)
