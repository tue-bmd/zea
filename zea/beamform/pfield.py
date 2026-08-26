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
from zea.func.tensor import vmap
from zea.func.ultrasound import directivity
from zea.internal.cache import cache_output


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
    element_width=None,
    frequency_step=4,
    db_thresh=-1.0,
    downsample=10,
    downmix=4,
    alpha=1,
    percentile=10,
    norm=True,
    point_batch_size=2048,
    interpolation="bilinear",
):
    """Compute the pressure field for ultrasound imaging.

    This implements PFIELD from MUST in its two-dimensional form: element height and elevation
    focus are taken to infinity, so the elevation term of the 3-D theory drops out
    and the Green's function is 1 / sqrt(r) instead of 1 / r (Eqs. 36-37). Equation
    numbers in this module refer to the reference below.

    .. admonition:: Reference

        Garcia, D. (2022). SIMUS: An open-source simulator for medical ultrasound
        imaging. Part I: Theory & examples. *Computer Methods and Programs in
        Biomedicine, 218*, 106726. https://www.biomecardio.com/publis/cmpb22.pdf

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
            Default is 1.
        percentile (int, optional): minimum percentile threshold to keep in the weighting.
            Higher is more aggressive. Default is 10.
        norm (bool, optional): per pixel normalization over the transmit axis (True)
            or unnormalized (False).
        point_batch_size (int, optional): Batch size for the pressure field computation.
            Higher is slightly faster, but requires more memory. Default is 2048.
        interpolation (str, optional): Interpolation used to resize the pressure
            field from the downsampled grid back to the full grid. "nearest"
            is fastest but imprints the downsampled blocks on the
            weighting, which can cause visible steps at the edges of each
            transmit's insonified region; "bilinear" (default) removes those steps at a
            small extra cost and is recommended for display-quality images.

    Returns:
        ops.array: The (normalized) pressure field (across tx events)
            of shape (n_tx, grid_size_z, grid_size_x).
    """
    # medium params
    # NOTE: currently we ignore attenuation in the compounding
    attenuation_coef = 0  # dB/(cm·MHz), attenuation coefficient of the medium
    attenuation_coef = attenuation_coef / 8.686  # convert to Np/(cm·MHz)
    attenuation_coef = attenuation_coef * 1e2 / 1e6  # Np/(m·Hz)

    n_el = int(n_el)

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
    tx_apodizations = ops.convert_to_tensor(tx_apodizations, dtype="complex64")

    # probe params
    fc_original = center_frequency
    center_frequency = center_frequency / downmix  # downmixing the frequency

    # pulse params
    num_waveforms = 1  # number of waveforms in the pulse

    # array params
    if element_width is None:
        from zea.probes import Probe

        pitch = Probe.get_pitch(probe_geometry)
        kerf = 0.1 * pitch
        element_width = pitch - kerf

    # %------------------------------------%
    # % POINT LOCATIONS, DISTANCES & GRIDS %
    # %------------------------------------%

    # subdivide elements into sub elements or not? (to satisfy Fraunhofer approximation, Eq. 21)
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
    element_x = probe_geometry[:, 0]
    element_z = probe_geometry[:, 2]
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

    # Directivity of a sub-element (Eq. 17). It varies over the band, but we approximate it at the
    # center frequency such that it doesn't need to be recomputed for each frequency sample.
    # We use a rigid baffle by default, PFIELD uses a soft baffle (extra cos(theta) factor, Eq. 39).
    sub_element_directivity = ops.cast(
        directivity(center_frequency, theta, seg_length, sound_speed), "complex64"
    )

    # Clamp distance from below at λ/2; the 1/sqrt(r) Green's function is singular
    # below this scale and the far-field approximation breaks down there.
    min_distance = sound_speed / (2 * fc_original)  # λ/2 at the original (non-downmixed) fc
    distance = ops.maximum(distance, min_distance)

    pulse_width = num_waveforms / center_frequency  # temporal pulse width

    # Both spectra are written in ordinary frequency (Hz) rather than angular frequency:
    # the sinc arguments then carry no factors of pi, since keras.ops.sinc is the
    # normalized sin(pi x) / (pi x).
    def pulse_spectrum(f):  # Eq. (25)
        imag = ops.sinc(pulse_width * (f - center_frequency)) - ops.sinc(
            pulse_width * (f + center_frequency)
        )
        return 1j * ops.cast(imag, "complex64")

    # FREQUENCY RESPONSE of the ensemble PZT + probe
    bandwidth = probe_bandwidth_percent * center_frequency / 100  # bandwidth in Hz
    p_shape = ops.log(126) / ops.log(epsilon + 2 * center_frequency / bandwidth)

    def probe_spectrum(f):  # sqrt of Eq. (27), the one-way half of Eq. (28)
        # Calculate the normalized frequency difference
        freq_diff = ops.abs(f - center_frequency)
        # Calculate the denominator for normalization
        denom = (bandwidth / 2) / (ops.log(2) ** (1 / p_shape))
        # Raise the normalized difference to the power of p_shape
        exponent = (freq_diff / denom) ** p_shape
        # The bandwidth is a pulse-echo (two-way) bandwidth; the one-way transmit
        # response is the square root of the two-way response, hence the factor 1/2.
        return ops.cast(ops.exp(-exponent / 2), "complex64")

    # The frequency response is a pulse-echo (transmit + receive) response.
    # The spectrum of the pulse (pulse_spectrum) will be then multiplied
    # by the frequency-domain tapering window of the transducer (probe_spectrum)
    # The frequency step df is chosen to avoid interferences due to
    # inadequate discretization (the criterion below Eq. 42).
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
    spectrum = ops.abs(pulse_spectrum(freq) * ops.cast(probe_spectrum(freq), "complex64"))
    gain_db = 20 * ops.log10(keras.config.epsilon() + spectrum / (ops.max(spectrum)))
    idx = gain_db > db_thresh

    freq = freq[idx]

    pulse_spect = pulse_spectrum(freq)
    probe_spect = probe_spectrum(freq)

    # Exponential arrays of size [numel(x) n_el num_sub_elements]
    wavenumber = 2 * np.pi * freq[0] / sound_speed
    attenuation_wavenumber = attenuation_coef * freq[0]

    # Exponential array for the increment wavenumber dk
    wavenumber_step = 2 * np.pi * freq_step / sound_speed
    attenuation_wavenumber_step = attenuation_coef * freq_step

    @jit(torch=False)
    def _pfield_freq_loop(distance, sub_element_directivity):
        """Calculates the pressure field using frequency loop method.

        Returns:
            (Tensor): Pressure field of shape (num_points, n_tx).
        """

        distance_complex = ops.cast(distance, dtype="complex64")

        mod_out = ops.cast(ops.mod(wavenumber * distance, 2 * np.pi), dtype="complex64")
        exp_arr = ops.exp(
            ops.cast(-attenuation_wavenumber * distance, dtype="complex64") + 1j * mod_out
        )

        exp_freq_step = ops.exp(
            ops.cast(-attenuation_wavenumber_step * distance, dtype="complex64")
            + 1j * ops.cast(wavenumber_step * distance, dtype="complex64")
        )

        exp_arr = exp_arr / ops.sqrt(distance_complex)
        exp_arr = exp_arr * ops.cast(ops.sqrt(min_distance), "complex64")

        exp_arr = exp_arr * sub_element_directivity

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

    pressure_squared = _pfield_freq_loop_mapped(
        distance, sub_element_directivity
    )  # shape (num_points, n_tx)

    # Zero out pressure behind the transducer (z < 0)
    pressure_squared = ops.where(grid_z[:, None] < 0, 0, pressure_squared)

    # Mean over the retained frequency samples, not their sum: the scan accumulates
    # |P(f_k)|^2 with no df factor, so a sum scales with the sample count, i.e. with
    # 1/freq_step -- which is set above from max(t0_delays) and so depends on which
    # transmits are in the stack. The mean keeps the field a true RMS over the
    # retained band, and comparable across transmit subsets (~2e-3 apart, from the
    # differing grids). No effect when norm=True: a global factor cancels below.
    pressure_squared = pressure_squared / ops.cast(ops.shape(freq)[0], "float32")

    # RMS acoustic pressure, reshaped to (n_tx, grid_size_z, grid_size_x)
    pressure = ops.transpose(ops.sqrt(pressure_squared), (1, 0))
    pressure = ops.reshape(pressure, (-1, *size_downsampled))

    # resize pressure to exactly the original grid size (see `interpolation`)
    p_arr = ops.squeeze(
        ops.image.resize(pressure[..., None], size_orig, interpolation=interpolation), axis=-1
    )

    p_arr = shape_pressure_field(p_arr, alpha=alpha, percentile=percentile)

    return normalize_pressure_field(p_arr) if norm else p_arr


def shape_pressure_field(pfield, alpha: float = 1.0, percentile: float = 10.0):
    """Per-transmit shaping of a pressure field: floor the dim tail, sharpen the beam.

    Args:
        pfield (array): Pressure field of shape (n_tx, grid_size_z, grid_size_x).
        alpha (float, optional): Exponent to 'sharpen or smooth' the weighting.
            Higher values result in sharper weighting. Default is 1.0 (no change).
        percentile (int, optional): minimum percentile threshold to keep in the
            weighting, per transmit. Higher is more aggressive. Default is 10.

    Returns:
        ops.array: Shaped pressure field, same shape.
    """
    # Convert percentile to quantile (0–1 range)
    q = percentile / 100.0

    # Compute per-transmitter quantile thresholds
    threshold = ops.quantile(pfield, q, axis=(1, 2), keepdims=True)

    # Zero out values below the threshold
    pfield = ops.where(pfield < threshold, 0, pfield)

    # Sharpen the beam
    return ops.power(pfield, alpha)


def normalize_pressure_field(pfield):
    """Normalize a pressure field per pixel, over the transmit axis.

    Turns the field into compounding weights that sum to 1 at every pixel.

    Args:
        pfield (array): Pressure field of shape (n_tx, grid_size_z, grid_size_x).

    Returns:
        ops.array: Normalized intensity array.
    """
    return pfield / (keras.config.epsilon() + ops.sum(pfield, axis=0, keepdims=True))


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
    # The sum over elements of Eq. (37), as a matmul:
    # (num_points, n_el) @ (n_el, n_tx) -> (num_points, n_tx), all transmits batched
    pressure_k = (
        ops.matmul(monochromatic_pressure, ops.transpose(delay_apodization, (1, 0)))
        * pulse_spect
        * probe_spect
    )
    return ops.abs(pressure_k) ** 2
