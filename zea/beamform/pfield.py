"""Pressure field computation for ultrasound imaging.

This module provides routines for automatic computation of the acoustic pressure field
used for compounding multiple transmit (Tx) events in ultrasound imaging.

The pressure field is computed by simulating the acoustic response of the probe and
medium for each transmit event. The computation involves:

- Subdividing each probe element into sub-elements to satisfy the Fraunhofer approximation.
- Calculating the distances and angles between each grid point and each sub-element.
- Computing the frequency response of the probe and the pulse spectrum.
- Summing the contributions from all relevant frequencies, taking into account
  transmit delays, apodization, and directivity.
- Optionally normalizing and thresholding the resulting field for use in
  transmit compounding or adaptive beamforming.

The main entry point is :func:`compute_pfield`, which returns a normalized pressure
field array for all transmit events.

"""

import keras
import numpy as np
from keras import ops

from zea.backend import jit
from zea.func.tensor import sinc, vmap
from zea.internal.cache import cache_output


def _abs_sinc(x):
    return sinc(ops.abs(x))


@cache_output(verbose=True)
def compute_pfield(
    sound_speed,
    center_frequency,
    probe_bandwidth_percent,
    n_el,
    probe_geometry,
    tx_apodizations,
    grid,
    t0_delays,
    frequency_step=4,
    db_thresh=-1.0,
    downsample=10,
    downmix=4,
    alpha=1,
    percentile=10,
    norm=True,
    point_batch_size=2048,
):
    """Compute the pressure field for ultrasound imaging.

    Args:
        sound_speed (float): Speed of sound in the medium.
        center_frequency (float): Center frequency of the transmit pulse in Hz.
        probe_bandwidth_percent (float): Bandwidth of the probe, pulse-echo 6dB
            fractional bandwidth (%)
        n_el (int): Number of elements in the probe.
        probe_geometry (array): Geometry of the probe elements.
        tx_apodizations (array): Transmit apodizations of shape (n_tx, n_el).
        grid (array): Grid points where the pressure field is computed
            of shape (grid_size_z, grid_size_x, 3).
        t0_delays (array): Transmit delays for each transmit event.
        frequency_step (int, optional): Frequency step. Default is 4.
            Higher is faster but less accurate.
        db_thresh (float, optional): dB threshold. Default is -1.0
            Higher is faster but less accurate.
        downsample (int, optional): Downsample the grid for faster computation.
            Default is 10. Higher is faster but less accurate.
        downmix (int, optional): Downmixing the frequency to facilitate a smaller grid.
            Default is 4. Higher requires lower number of grid points but is less accurate.
        alpha (float, optional): Exponent to 'sharpen or smooth' the weighting. Higher is sharper.
            Only works when norm is True. Default is 1.
        percentile (int, optional): minimum percentile threshold to keep in the weighting.
            Only works when norm is True. Higher is more aggressive. Default is 10.
        norm (bool, optional): per pixel normalization (True) or unnormalized (False)
        point_batch_size (int, optional): Batch size for the pressure field computation.
            Higher is slightly faster, but requires more memory. Default is 2048.

    Returns:
        ops.array: The (normalized) pressure field (across tx events)
            of shape (n_tx, grid_size_z, grid_size_x).
    """
    # medium params
    # NOTE: currently we ignore attenuation in the compounding
    attenuation_coef = 0  # dB/cm/MHz, attenuation coefficient of the medium
    attenuation_coef = attenuation_coef / 8.686  # convert to Np/cm/MHz
    attenuation_coef = attenuation_coef / 1e6 / 1e2  # convert to Np/m/Hz

    # cast to float32
    sound_speed = ops.cast(sound_speed, "float32")
    center_frequency = ops.cast(center_frequency, "float32")
    probe_bandwidth_percent = ops.cast(probe_bandwidth_percent, "float32")
    attenuation_coef = ops.cast(attenuation_coef, "float32")
    db_thresh = ops.cast(db_thresh, "float32")

    # to tensor
    probe_geometry = ops.convert_to_tensor(probe_geometry, dtype="float32")
    grid_x = ops.convert_to_tensor(grid[:, :, 0], dtype="float32")
    grid_z = ops.convert_to_tensor(grid[:, :, 2], dtype="float32")
    t0_delays = ops.convert_to_tensor(t0_delays, dtype="float32")
    tx_apodizations = ops.convert_to_tensor(tx_apodizations, dtype="float32")

    # formatting
    t0_delays = ops.where(ops.isnan(t0_delays), 0, t0_delays)
    tx_apodizations = ops.where(ops.isnan(tx_apodizations), 0, tx_apodizations)
    tx_apodizations = ops.cast(tx_apodizations, "complex64")

    # probe params
    fc_original = center_frequency
    center_frequency = center_frequency / downmix  # downmixing the frequency

    # pulse params
    num_waveforms = 1  # number of waveforms in the pulse
    center_wavenumber = 2 * np.pi * center_frequency / sound_speed

    # array params
    pitch = ops.abs(probe_geometry[1, 0] - probe_geometry[0, 0])  # element pitch

    kerf = 0.1 * pitch  # for now this is hardcoded
    element_width = pitch - kerf

    # %------------------------------------%
    # % POINT LOCATIONS, DISTANCES & GRIDS %
    # %------------------------------------%

    # subdivide elements into sub elements or not? (to satisfy Fraunhofer approximation)
    lambda_min = sound_speed / (center_frequency * (1 + probe_bandwidth_percent / 200))
    num_sub_elements = ops.ceil(element_width / lambda_min)

    size_orig = ops.shape(grid_x)

    # Nearest-neighbor downsampling the grid
    grid_x = grid_x[::downsample, ::downsample]
    grid_z = grid_z[::downsample, ::downsample]
    size_downsampled = ops.shape(grid_x)

    # Coordinates of the points where pressure is needed
    grid_x = ops.reshape(grid_x, (-1,))
    grid_z = ops.reshape(grid_z, (-1,))

    # Centers of the transducer elements (x- and z-coordinates)
    element_x = (ops.arange(0.0, n_el) - (n_el - 1) / 2) * pitch
    element_z = ops.zeros(n_el)
    element_theta = ops.zeros(n_el)

    # Centroids of the sub-elements
    seg_length = element_width / num_sub_elements
    sub_element_x = (
        -element_width / 2
        + seg_length / 2
        + ops.arange(0, num_sub_elements, dtype=seg_length.dtype) * seg_length
    )
    sub_element_z = ops.zeros_like(sub_element_x)

    # Distances between the points and the transducer elements
    delta_x = grid_x[:, None, None] - sub_element_x[None, :, None] - element_x[None, None, :]
    delta_z = grid_z[:, None, None] - sub_element_z[None, :, None] - element_z[None, None, :]

    distance = ops.sqrt(delta_x**2 + delta_z**2)

    # Angle between the normal to the transducer and the line joining
    # the point and the transducer
    epsilon = keras.config.epsilon()
    theta = ops.arcsin(ops.clip(delta_x / distance, -1.0, 1.0)) - element_theta
    sin_theta = ops.sin(theta)

    # Clamp distance from below at λ/2; the 1/sqrt(r) Green's function is singular
    # below this scale and the far-field approximation breaks down there.
    min_distance = sound_speed / (2 * fc_original)  # λ/2 at the original (non-downmixed) fc
    distance = ops.maximum(distance, min_distance)

    pulse_width = num_waveforms / center_frequency  # temporal pulse width
    center_angular_freq = 2 * np.pi * center_frequency

    def pulse_spectrum(w):
        imag = _abs_sinc(pulse_width * (w - center_angular_freq) / 2) - _abs_sinc(
            pulse_width * (w + center_angular_freq) / 2
        )
        return 1j * ops.cast(imag, "complex64")

    # FREQUENCY RESPONSE of the ensemble PZT + probe
    w_bandwidth = probe_bandwidth_percent * center_angular_freq / 100  # angular frequency bandwidth
    p_shape = ops.log(126) / ops.log(epsilon + 2 * center_angular_freq / w_bandwidth)

    def probe_spectrum(w):
        # Calculate the normalized frequency difference
        freq_diff = ops.abs(w - center_angular_freq)
        # Calculate the denominator for normalization
        denom = (w_bandwidth / 2) / (ops.log(2) ** (1 / p_shape))
        # Raise the normalized difference to the power of p_shape
        exponent = (freq_diff / denom) ** p_shape
        # Apply the negative sign and exponential
        return ops.cast(ops.exp(-exponent), "complex64")

    # The frequency response is a pulse-echo (transmit + receive) response.
    # The spectrum of the pulse (pulse_spectrum) will be then multiplied
    # by the frequency-domain tapering window of the transducer (probe_spectrum)
    # The frequency step df is chosen to avoid interferences due to
    # inadequate discretization.
    # df = frequency step (must be sufficiently small):
    # One has exp[-i(k r + w delay)] = exp[-2i pi(f r/c + f delay)] in the Eq.
    # One wants: the phase increment 2pi(df r/c + df delay) be < 2pi.
    # Therefore: df < 1/(r/c + delay).

    freq_step = 1 / (ops.max(distance / sound_speed) + ops.max(t0_delays))
    freq_step = frequency_step * freq_step

    # FREQUENCY SAMPLES
    num_freq = 2 * ops.cast(ops.ceil(center_frequency / freq_step), "int32") + 1
    freq = ops.arange(0, num_freq, dtype="float32") * freq_step

    # keep the significant components only by using db_thresh
    spectrum = ops.abs(
        pulse_spectrum(2 * np.pi * freq) * ops.cast(probe_spectrum(2 * np.pi * freq), "complex64")
    )
    gain_db = 20 * ops.log10(keras.config.epsilon() + spectrum / (ops.max(spectrum)))
    idx = gain_db > db_thresh

    freq = freq[idx]

    pulse_spect = pulse_spectrum(2 * np.pi * freq)
    probe_spect = probe_spectrum(2 * np.pi * freq)

    # Exponential arrays of size [numel(x) n_el num_sub_elements]
    wavenumber = 2 * np.pi * freq[0] / sound_speed
    attenuation_wavenumber = attenuation_coef * freq[0]
    attenuation_wavenumber = ops.cast(attenuation_wavenumber, dtype="complex64")

    # Exponential array for the increment wavenumber dk
    wavenumber_step = 2 * np.pi * freq_step / sound_speed
    attenuation_wavenumber_step = attenuation_coef * freq_step
    wavenumber_step = ops.cast(wavenumber_step, dtype="complex64")
    attenuation_wavenumber_step = ops.cast(attenuation_wavenumber_step, dtype="complex64")

    @jit
    def _pfield_freq_loop(distance, sin_theta):
        """Calculates the pressure field using frequency loop method.

        Returns:
            (Tensor): Pressure field of shape (num_points, n_tx).
        """

        distance_complex = ops.cast(distance, dtype="complex64")

        mod_out = ops.cast(ops.mod(wavenumber * distance, 2 * np.pi), dtype="complex64")
        exp_arr = ops.exp(-attenuation_wavenumber * distance_complex + 1j * mod_out)

        exp_freq_step = ops.exp(
            (-attenuation_wavenumber_step + 1j * wavenumber_step) * distance_complex
        )

        exp_arr = exp_arr / ops.sqrt(distance_complex)
        exp_arr = exp_arr * ops.cast(ops.sqrt(min_distance), "complex64")

        directivity = _abs_sinc(center_wavenumber * seg_length / 2 * sin_theta)
        exp_arr = exp_arr * ops.cast(directivity, "complex64")

        monochromatic_pressure = exp_arr / exp_freq_step

        def scan_fn(carry, k):
            monochromatic_pressure, total_pressure_squared = carry
            monochromatic_pressure *= exp_freq_step
            pressure_squared_k = _pfield_freq_step(
                freq[k],
                t0_delays,
                tx_apodizations,
                ops.mean(monochromatic_pressure, axis=1),  # avg over sub-elements
                pulse_spect[k],
                probe_spect[k],
            )
            total_pressure_squared += pressure_squared_k
            return (monochromatic_pressure, total_pressure_squared), None

        num_points, _, _ = ops.shape(monochromatic_pressure)
        n_tx, _ = ops.shape(tx_apodizations)
        (_, total_pressure_squared), _ = ops.scan(
            scan_fn,
            (monochromatic_pressure, ops.zeros((num_points, n_tx), dtype="float32")),
            ops.arange(ops.shape(freq)[0]),
        )

        return total_pressure_squared

    _pfield_freq_loop_mapped = vmap(
        _pfield_freq_loop,
        fn_supports_batch=True,
        batch_size=point_batch_size,
    )

    pressure_squared = _pfield_freq_loop_mapped(distance, sin_theta)  # shape (num_points, n_tx)

    # Zero out pressure behind the transducer (z < 0)
    pressure_squared = ops.where(grid_z[:, None] < 0, 0, pressure_squared)

    # RMS acoustic pressure, reshaped to (n_tx, grid_size_z, grid_size_x)
    pressure = ops.transpose(ops.sqrt(pressure_squared), (1, 0))
    pressure = ops.reshape(pressure, (-1, *size_downsampled))

    # resize pressure to exactly the original grid size
    p_arr = ops.squeeze(
        ops.image.resize(pressure[..., None], size_orig, interpolation="nearest"), axis=-1
    )

    if norm:
        normalized_pfield = normalize_pressure_field(p_arr, alpha=alpha, percentile=percentile)
    else:
        normalized_pfield = p_arr

    return normalized_pfield


def normalize_pressure_field(pfield, alpha: float = 1.0, percentile: float = 10.0):
    """
    Normalize the input array of intensities by zeroing out values below a given percentile.

    Args:
        pfield (array): The unnormalized pressure field array
            of shape (n_tx, grid_size_z, grid_size_x).
        alpha (float, optional): Exponent to 'sharpen or smooth' the weighting.
            Higher values result in sharper weighting. Default is 1.0.
        percentile (int, optional): minimum percentile threshold to keep in the weighting.
            Higher is more aggressive. Default is 10.

    Returns:
        ops.array: Normalized intensity array.
    """
    # Convert percentile to quantile (0–1 range)
    q = percentile / 100.0

    # Compute per-transmitter quantile thresholds
    threshold = ops.quantile(pfield, q, axis=(1, 2), keepdims=True)

    # Zero out values below the threshold
    pfield = ops.where(pfield < threshold, 0, pfield)

    # Sharpen the beam
    pfield = ops.power(pfield, alpha)

    # Normalize over transmit events (axis=0)
    normalized_pfield = pfield / (keras.config.epsilon() + ops.sum(pfield, axis=0, keepdims=True))

    return normalized_pfield


def _pfield_freq_step(
    freq, delays_tx, tx_apodization, monochromatic_pressure, pulse_spect, probe_spect
):
    """
    Calculates the pressure field for a single frequency step.

    Args:
        freq: (float): Frequency of the current step.
        delays_tx (Tensor): Transmit delays of shape (n_tx, n_el).
        tx_apodization (Tensor): Transmit apodization values (complex64) of shape (n_tx, n_el).
        monochromatic_pressure: (Tensor): Per-element, per-field-point complex pressure response
            (including directivity and propagation effects) at the current frequency sample
            of shape (num_points, n_el).
        pulse_spect (complex64): Complex frequency response of the pulse
            at the current frequency sample.
        probe_spect (complex64): Complex frequency response of the pulse and probe
            at the current frequency sample.

    Returns:
        pressure_squared_k (Tensor): Pressure field for this frequency
            of shape (num_points, n_tx).
    """
    angular_frequency = 2 * np.pi * freq
    # Per-transmit complex phasor of shape (n_tx, n_el)
    delay_apodization = (
        ops.exp(1j * ops.cast(angular_frequency * delays_tx, "complex64")) * tx_apodization
    )
    # (num_points, n_el) @ (n_el, n_tx) -> (num_points, n_tx): all transmits batched
    pressure_k = (
        ops.matmul(monochromatic_pressure, ops.transpose(delay_apodization, (1, 0)))
        * pulse_spect
        * probe_spect
    )
    return ops.abs(pressure_k) ** 2
