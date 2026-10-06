"""Aperture Domain Model Image REconstruction (ADMIRE).

ADMIRE is a model-based beamformer that suppresses off-axis and multipath
(reverberation) clutter. Time-delayed channel data is split into short axial
windows, each window is taken to the frequency domain, and at every frequency the
aperture-domain signal (one complex value per element) is fitted with a linear
model of the wavefronts that point scatterers would produce across the aperture.
Scatterers inside a small region of interest (ROI) around the window center are
kept; everything explained by scatterers *outside* the ROI is clutter and is
discarded. The model is fitted with elastic-net regularized least squares, solved
with cyclic coordinate descent.

The implementation is split in two parts:

- :func:`generate_admire_models` precomputes the models with NumPy. They depend
  only on the imaging geometry (axial sampling, sound speed, center frequency,
  element pitch and sub-aperture size), so they are computed once and reused.
- :func:`apply_admire` fits the models to channel data and reconstructs the
  decluttered channel data with Keras ops, so it runs (and JIT compiles) on every
  backend.

The beamformer is usually used through :class:`zea.ops.ADMIRE`, which handles
the data layout of a zea pipeline.

.. important::
    This is a ``zea`` implementation of ADMIRE. For the original MATLAB, C and
    CUDA code of the Vanderbilt BEAM Lab, see
    `VU-BEAM-Lab/ADMIRE <https://github.com/VU-BEAM-Lab/ADMIRE>`_.

.. note::
    The model generation, its calibration constants and the model application
    are ported from the reference code, Copyright 2020 Christopher Khan,
    Kazuyuki Dei, Siegfried Schlunk and Brett Byram. The coordinate descent
    solver is ported from ``ccd_double_precision.c``, Copyright 2020 Christopher
    Khan. Both are licensed under the
    `Apache License 2.0 <https://github.com/VU-BEAM-Lab/ADMIRE/blob/master/LICENSE>`_;
    see the reference repository's ``NOTICE`` file. The ICA is the FOBI code
    (``ica.m``) from Shlens' tutorial, which the reference includes.

    Changes from the reference: the code is translated to NumPy (model
    generation) and Keras ops (model fitting); all lines, windows and frequencies
    are fitted in one batched solve; IQ data is supported in addition to RF data;
    the phase of the ICA eigenvectors is fixed so the models are reproducible
    across platforms; and the wavenumber calibration table is matched to the
    nearest listed center frequency instead of exactly.

.. citation:: byram2015model, dei2019computationally, khan2021realtime, khan2020genre,
   byram2014ultrasonic, shlens2014tutorial, cardoso1989source

"""

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from keras import ops
from scipy.special import erf

from zea.func.tensor import vmap
from zea.internal.cache import cache_output

#: Wavenumber calibration values of the reference implementation, by transducer
#: center frequency (Hz). Entry ``i`` scales the wavenumber of the ``i``-th selected
#: frequency of every STFT window.
REFERENCE_WAVENUMBER_CALIBRATION = {
    1818200: (1.32, 0.995, 0.835, 0.81),
    2083300: (1.32, 1.0, 0.845, 0.825),
    2500000: (1.315, 0.995, 0.835, 0.81),
    3000000: (1.31, 0.995, 0.835, 0.805),
    3125000: (1.305, 0.985, 0.825, 0.79),
    3500000: (1.29, 0.99, 0.835, 0.815),
    3600000: (1.315, 1.01, 0.855, 0.815),
    4000000: (1.29, 0.99, 0.845, 0.82),
    4500000: (1.325, 1.025, 0.87, 0.825),
    5000000: (1.29, 0.995, 0.85, 0.835),
    5208000: (1.145, 0.995, 0.895, 0.865),
    5500000: (1.28, 0.995, 0.855, 0.845),
    6500000: (1.31, 1.04, 0.89, 0.84),
    7500000: (1.275, 0.975, 0.835, 0.82),
    7800000: (1.295, 1.0, 0.855, 0.825),
    7813000: (1.295, 1.0, 0.855, 0.825),
}

#: Calibration values the reference implementation uses for any other center frequency.
DEFAULT_WAVENUMBER_CALIBRATION = (1.135, 0.995, 0.925, 0.905)


@dataclass(frozen=True)
class ADMIREConfig:
    """Settings of the ADMIRE model space and model fit.

    The defaults are those of the reference implementation. Most users only touch
    the fit settings (``alpha``, ``lambda_scaling_factor``) and the aperture
    settings; the model-space scaling factors and calibration offsets are
    empirical and best left alone.

    Lateral and axial model-space positions are given as multiples of the lateral
    resolution ``res_lat`` (estimated per window and frequency) and of the axial
    resolution ``res_axl = 2 * res_lat``. Ranges are ``(min, spacing, max)``.

    Args:
        bandwidth (float): Fractional bandwidth of the transmitted pulse. Sets the
            STFT window length and the fitted frequency band.
        alpha (float): Elastic-net mixing parameter in ``[0, 1]``: ``1`` is the
            lasso, ``0`` is ridge regression.
        lambda_scaling_factor (float): The regularization strength of each fit is
            ``lambda_scaling_factor * rms(y)``.
        max_iterations (int): Maximum number of coordinate descent sweeps.
        tolerance (float): Coordinate descent stops once the largest change of a
            (standardized) coefficient within a sweep, squared and divided by the
            number of observations, falls below this value.
        ica (bool): Replace each of the ROI and outer models by its independent
            components (FOBI), which shrinks each to (at most) as many predictors
            as there are elements. This is what makes ADMIRE tractable on a GPU;
            ``False`` uses the full models, which can hold hundreds of thousands
            of predictors.
        min_num_elements (int): Smallest sub-aperture that aperture growth may
            select.
        pulse_multiplier (float): Scales the STFT window length (one pulse
            full-width at half-maximum by default).
        frequency_band_scaling (float): The fitted band is
            ``f0 +- 0.5 * frequency_band_scaling * bandwidth * f0``.
        wavenumber_calibration (str or tuple or None): Per-frequency wavenumber
            scaling. ``"reference"`` looks the values up by center frequency like
            the reference implementation; a tuple gives them explicitly; ``None``
            disables calibration.
        cal_shift (float): Calibration depth offset in meters.
        distance_offset_shift (float): Calibration distance offset in meters.
        win_tune (float): Scales the half pulse length in the window amplitude model.
        ellipsoid_constant_1 (float): Pads the ROI acceptance ellipse, in meters.
        ellipsoid_constant_2 (float): Pads the ellipse that the outer model excludes,
            in meters.
        lateral_limit_offset (float): Extends the outer model laterally beyond the
            sub-aperture, in meters.
        roi_x (tuple): ROI lateral positions, in units of ``res_lat``.
        roi_z (tuple): ROI depth positions around the window center, in units of
            ``res_axl``.
        roi_distance_offset_spacing (float): Spacing of the ROI distance offsets, in
            wavelengths.
        roi_distance_offset_limits (tuple): Min and max ROI distance offsets in meters.
        outer_x (tuple): Outer-model lateral positions; min and max are multiples of
            the lateral limit, the spacing is in units of ``res_lat``.
        outer_z (tuple): Outer-model depth positions as
            ``(min_z_center, min_res_axl, spacing_res_axl, max_z_center, max_res_axl,
            max_z_center_extra)``; the minimum is
            ``min_z_center * z_c + min_res_axl * res_axl + outer_z_offset`` and the
            maximum ``(max_z_center + max_z_center_extra) * z_c +
            max_res_axl * res_axl``.
        outer_z_offset (float): Constant added to the minimum outer-model depth, in
            meters.
        outer_distance_offset_spacing (float): Spacing of the outer distance offsets,
            in wavelengths.
        outer_distance_offset_limits (tuple): Min and max outer distance offsets in
            meters.
        predictor_chunk_size (int): Number of predictors generated at once, which
            bounds the memory use of model generation.
        max_cache_bytes (int): Models up to this size are kept in memory between the
            two passes of the ICA; larger ones are generated twice.
    """

    bandwidth: float = 0.6
    alpha: float = 0.9
    lambda_scaling_factor: float = 0.0189
    max_iterations: int = 100000
    tolerance: float = 0.1
    ica: bool = True
    min_num_elements: int = 16
    pulse_multiplier: float = 1.0
    frequency_band_scaling: float = 1.2
    wavenumber_calibration: str | tuple | None = "reference"
    cal_shift: float = 7.75e-6
    distance_offset_shift: float = 3.85e-5
    win_tune: float = 1.0
    ellipsoid_constant_1: float = 0.0
    ellipsoid_constant_2: float = 0.5e-3 + np.finfo(float).eps
    lateral_limit_offset: float = 1e-3
    roi_x: tuple = (-0.5, 0.0179, 0.5)
    roi_z: tuple = (-0.5, 0.1430, 0.5)
    roi_distance_offset_spacing: float = 0.0485
    roi_distance_offset_limits: tuple = (-0.8e-3, 0.4e-3)
    outer_x: tuple = (-1.0, 1.4228, 1.0)
    outer_z: tuple = (0.0, 0.0, 0.7114, 1.0, 0.0, 0.05)
    outer_z_offset: float = 0.0
    outer_distance_offset_spacing: float = 0.1211
    outer_distance_offset_limits: tuple = (-8e-3, 3.2e-3)
    predictor_chunk_size: int = 32768
    max_cache_bytes: int = 2**30

    def replace(self, **changes) -> "ADMIREConfig":
        """Return a copy with some settings changed."""
        return dataclasses.replace(self, **changes)


@dataclass
class ADMIREModels:
    """Precomputed ADMIRE models, as made by :func:`generate_admire_models`.

    There is one model per STFT window and selected frequency. Models are zero
    padded to a common number of rows (elements) and columns (predictors), which
    leaves the fit unchanged: padded columns get zero coefficients and padded rows
    carry no data.

    Attributes:
        models (np.ndarray): Complex models of shape
            ``(n_windows, n_freqs, n_elements, n_predictors)``, with unit-norm columns.
        roi_mask (np.ndarray): Boolean mask of shape
            ``(n_windows, n_freqs, n_predictors)``, true for the ROI predictors used
            to reconstruct the signal.
        aperture_mask (np.ndarray): Boolean mask of shape ``(n_windows, n_elements)``
            with the elements that aperture growth keeps in each window.
        window_starts (np.ndarray): Axial sample index where each window starts.
        window_length (int): Number of axial samples per window.
        fft_length (int): Zero-padded DFT length of a window.
        frequency_bins (np.ndarray): DFT bin of each selected frequency.
        frequencies (np.ndarray): Physical frequency of each selected bin in Hz.
        element_positions (np.ndarray): Lateral element positions of the
            sub-aperture, relative to its center, in meters.
        analytic (bool): Whether the models are made for analytic (IQ) data, where
            only the band around the center frequency is present, or for real RF
            data, where negative frequencies mirror the positive ones.
        config (ADMIREConfig): Settings used to generate the models.
    """

    models: np.ndarray
    roi_mask: np.ndarray
    aperture_mask: np.ndarray
    window_starts: np.ndarray
    window_length: int
    fft_length: int
    frequency_bins: np.ndarray
    frequencies: np.ndarray
    element_positions: np.ndarray
    analytic: bool
    config: ADMIREConfig = field(default_factory=ADMIREConfig)

    @property
    def n_windows(self) -> int:
        """Number of STFT windows."""
        return self.models.shape[0]

    @property
    def n_elements(self) -> int:
        """Number of elements in the sub-aperture."""
        return self.models.shape[2]


def _colon(start, step, stop):
    """MATLAB's ``start:step:stop`` range."""
    if step == 0:
        return np.array([start]) if start == stop else np.array([])
    n = math.floor((stop - start) / step + 1e-10) + 1
    return start + step * np.arange(max(n, 0))


def _resolve_calibration(calibration, center_frequency, n_freqs):
    """Return the wavenumber calibration value of each selected frequency."""
    if calibration is None:
        return np.ones(n_freqs)
    if isinstance(calibration, str):
        if calibration != "reference":
            raise ValueError(
                f"Unknown wavenumber_calibration {calibration!r}, "
                "expected 'reference', a tuple of values, or None."
            )
        # The table keys are rounded (e.g. 7813000 for 7.8125 MHz), so match the nearest
        nearest = min(REFERENCE_WAVENUMBER_CALIBRATION, key=lambda f: abs(f - center_frequency))
        if abs(nearest - center_frequency) <= 1e3:
            calibration = REFERENCE_WAVENUMBER_CALIBRATION[nearest]
        else:
            calibration = DEFAULT_WAVENUMBER_CALIBRATION
    calibration = np.asarray(calibration, dtype=float)
    if n_freqs > calibration.size:
        raise ValueError(
            f"{n_freqs} frequencies are selected per STFT window, but only "
            f"{calibration.size} wavenumber calibration values are available. Pass more "
            "values with `wavenumber_calibration`, or set it to None."
        )
    return calibration[:n_freqs]


def _select_frequencies(
    fft_length, sampling_frequency, center_frequency, config: ADMIREConfig, analytic
):
    """Select the DFT bins in the fitted band and their physical frequencies."""
    half_band = 0.5 * config.bandwidth * center_frequency * config.frequency_band_scaling
    f_min, f_max = center_frequency - half_band, center_frequency + half_band
    bins = np.arange(fft_length)
    bin_frequencies = bins / fft_length * sampling_frequency

    if analytic:
        if 2 * half_band >= sampling_frequency:
            raise ValueError(
                f"The fitted band ({2 * half_band / 1e6:.2f} MHz) does not fit in the axial "
                f"sampling rate ({sampling_frequency / 1e6:.2f} MHz) of the grid. Use a "
                "finer axial grid spacing."
            )
        # IQ data is aliased: map every bin to its alias closest to the center frequency.
        alias = np.round((center_frequency - bin_frequencies) / sampling_frequency)
        frequencies = bin_frequencies + alias * sampling_frequency
    else:
        if f_max >= sampling_frequency / 2:
            raise ValueError(
                f"The fitted band reaches {f_max / 1e6:.2f} MHz, above the Nyquist frequency "
                f"({sampling_frequency / 2e6:.2f} MHz) of the axial grid. Use a finer axial "
                "grid spacing, or IQ data."
            )
        # Only the positive half; the negative frequencies follow by conjugate symmetry.
        frequencies = bin_frequencies[: fft_length // 2]
        bins = bins[: fft_length // 2]

    selected = (frequencies >= f_min) & (frequencies <= f_max)
    order = np.argsort(frequencies[selected])
    return bins[selected][order], frequencies[selected][order]


def _predictor_signals(x, z, offset, element_positions, k, z_center, window_depths, params):
    """Aperture-domain signals of point-scatterer predictors.

    Port of ``generate_modeled_signal_for_predictor.m``. A predictor is a scatterer
    at lateral position ``x`` and depth ``z`` whose transmit path is lengthened by
    ``offset`` (which models multipath). Its signal across the aperture combines the
    element directivity, the part of the pulse that falls within the STFT window, and
    the phase of its perceived depth on each element after receive focusing.

    Args:
        x, z, offset (np.ndarray): Predictor parameters of shape ``(P,)``.
        element_positions (np.ndarray): Element positions of shape ``(n_el,)``.
        k (float): Calibrated wavenumber.
        z_center (float): Center depth of the STFT window.
        window_depths (tuple): First and last depth of the STFT window.
        params (dict): Pitch, sound speed, pulse width and calibration constants.

    Returns:
        np.ndarray: Complex signals of shape ``(n_el, P)``.
    """
    c = params["sound_speed"]
    config: ADMIREConfig = params["config"]
    e = element_positions[:, None]
    offset = offset + config.distance_offset_shift

    # Element directivity (sinc of the element width times the obliquity factor)
    theta = np.arctan2(e - x, z)
    wavelength = 2 * np.pi / k
    argument = np.pi * params["pitch"] * np.sin(theta) / wavelength
    eps = np.finfo(float).eps
    directivity = (np.sin(argument) + eps) / (argument + eps) * np.cos(theta)

    # Round-trip distance, and the depth at which the echo appears on each element
    # after dynamic receive focusing (the distance offset sets the transmit path).
    tau_n0 = (offset + 2 * z_center - (z - config.cal_shift)) / c
    d0 = np.sqrt((e - x) ** 2 + (z - config.cal_shift) ** 2) + c * tau_n0
    perceived_depth = (d0**2 - e**2) / (2 * d0)
    z_distance = d0 - (np.sqrt(e**2 + perceived_depth**2) - perceived_depth)

    # Fraction of a Gaussian pulse centered at z_distance that lies inside the window
    min_travel = 2 * window_depths[0]
    max_travel = 2 * window_depths[1]
    half_pulse = params["half_pulse_width_samples"] / params["sampling_frequency"] * c / 2
    pulse_min = np.maximum(z_distance - half_pulse * config.win_tune, min_travel)
    pulse_max = np.minimum(z_distance + half_pulse * config.win_tune, max_travel)
    st = params["pulse_sigma_distance"]
    gaussian = (
        0.5
        * np.sqrt(np.pi)
        * st
        * (erf((z_distance - pulse_min) / st) - erf((z_distance - pulse_max) / st))
    )
    gaussian[(pulse_min > max_travel) | (pulse_max < min_travel)] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        window_amplitude = np.sqrt(gaussian / gaussian.max(axis=0, keepdims=True))

    # amplitude * exp(1j * phase), without the (slow) complex exponential
    amplitude = window_amplitude * directivity
    phase = z_distance * k
    signals = np.empty(phase.shape, dtype=complex)
    signals.real = amplitude * np.cos(phase)
    signals.imag = amplitude * np.sin(phase)
    return signals


def _predictor_grid(x_range, z_range, offset_range):
    """All predictor parameters of a model space, in the reference ordering."""
    x = _colon(*x_range)
    z = _colon(*z_range)
    offsets = _colon(*offset_range)
    # MATLAB's meshgrid + (:) flattens column-major
    X, Z, offsets = np.meshgrid(x, z, offsets)
    return X.ravel(order="F"), Z.ravel(order="F"), offsets.ravel(order="F")


def _model_chunks(x, z, offset, chunk_size, signal_fn):
    """Yield the valid (nonzero, finite) predictor signals chunk by chunk."""
    for start in range(0, x.size, chunk_size):
        sl = slice(start, start + chunk_size)
        signals = signal_fn(x[sl], z[sl], offset[sl])
        valid = np.all(np.isfinite(signals), axis=0) & np.any(signals != 0, axis=0)
        yield signals[:, valid]


def _ica_basis(chunks_fn, n_el, keep_chunks=True):
    """Basis that replaces a model, from the reference's ICA (FOBI).

    Port of ``ica.m`` (Shlens, "A tutorial on independent component analysis",
    2014) followed by ``pinv(W)``, as in the reference implementation. The model
    columns are the samples, so it is accumulated chunk by chunk without ever
    holding the full model in memory.

    ``ica.m`` was written for real data. On complex models its result depends on the
    phase of the eigenvectors that the linear algebra library returns, which is
    arbitrary, so the reference gives different models on different platforms. Here
    the phase of every eigenvector is fixed by a deterministic rule (see
    :func:`_fix_phase`), so the models are reproducible; the reference would
    produce the same models for that choice of phases. (Textbook complex FOBI
    removes the ambiguity too, but its basis no longer separates on-axis signal
    from clutter, so the reference formulation is kept.)

    Args:
        chunks_fn (callable): Returns an iterator over chunks of model columns.
        n_el (int): Number of rows (elements) of the model.
        keep_chunks (bool): Keep the chunks in memory for the second pass instead
            of generating them again.

    Returns:
        np.ndarray: The ``(n_el, n_el)`` basis that replaces the model.
    """
    # First pass: mean and covariance of the columns
    n = 0
    total = np.zeros(n_el, dtype=complex)
    outer = np.zeros((n_el, n_el), dtype=complex)
    kept = []
    for chunk in chunks_fn():
        if keep_chunks:
            kept.append(chunk)
        n += chunk.shape[1]
        total += chunk.sum(axis=1)
        outer += chunk @ chunk.conj().T
    if n < 2:
        raise ValueError("An ADMIRE model has fewer than two valid predictors.")
    mean = total / n
    covariance = (outer - n * np.outer(mean, mean.conj())) / (n - 1)

    # Whitening, discarding numerically zero directions like MATLAB's pinv
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = _fix_phase(eigenvectors)
    tol = n_el * np.finfo(float).eps * max(np.abs(eigenvalues).max(), np.finfo(float).tiny)
    inv_sqrt = np.where(eigenvalues > tol, 1 / np.sqrt(np.maximum(eigenvalues, tol)), 0.0)
    whitening = inv_sqrt[:, None] * eigenvectors.conj().T

    # Second pass: the fourth-order moment matrix
    moment = np.zeros((n_el, n_el), dtype=complex)
    for chunk in kept if keep_chunks else chunks_fn():
        whitened = whitening @ (chunk - mean[:, None])
        moment += (np.sum(whitened * whitened, axis=0) * whitened) @ whitened.conj().T
    rotation, _, _ = np.linalg.svd(moment)
    rotation = _fix_phase(rotation)

    unmixing = rotation @ whitening
    return np.linalg.pinv(unmixing)


def _fix_phase(vectors):
    """Rotate every column so that a fixed, asymmetric weighting of it is real and positive.

    Eigen- and singular vectors of complex matrices are only defined up to a
    unit-modulus factor, which libraries choose differently.
    """
    weights = np.linspace(1.0, 2.0, vectors.shape[0])
    return vectors * np.exp(-1j * np.angle(weights @ vectors))


def _lateral_resolution(center_signal, n_el, pitch, wavelength, z_center):
    """Lateral resolution estimate from the aperture signal of an on-axis scatterer."""
    n_fft = 2 ** (2 * math.floor(math.log2(n_el)))
    spectrum = np.abs(np.fft.fft(np.abs(center_signal), n_fft))
    spectrum = spectrum / spectrum.max()
    fwhm_width = np.sum(spectrum > 0.5)
    return fwhm_width * (1 / pitch) / n_fft * wavelength * z_center


def _aperture_growth(z_center, n_el, pitch, f_number, min_num_elements):
    """Elements kept by aperture growth (a fixed f-number) at a given depth."""
    if not f_number:
        return np.ones(n_el, dtype=bool)
    n_active = math.ceil(z_center / pitch / f_number)
    n_active = 2 * math.ceil(n_active / 2)
    n_active = min(max(n_active, min_num_elements), n_el)
    offsets = np.arange(1 - n_active / 2, n_active / 2 + 1)
    indices = np.ceil(n_el / 2 + offsets).astype(int) - 1
    mask = np.zeros(n_el, dtype=bool)
    mask[indices] = True
    return mask


def _models_for_window(z_center, window_depths, element_positions, ks, params):
    """Generate the ROI and outer models of every selected frequency of a window."""
    config: ADMIREConfig = params["config"]
    pitch = params["pitch"]
    n_el = element_positions.size
    results = []
    for k in ks:
        wavelength = 2 * np.pi / k

        def signal_fn(x, z, offset, k=k):
            return _predictor_signals(
                x, z, offset, element_positions, k, z_center, window_depths, params
            )

        # Model-space sampling follows from the resolution at the window center
        center = signal_fn(np.zeros(1), np.full(1, z_center), np.zeros(1))[:, 0]
        res_lat = _lateral_resolution(center, n_el, pitch, wavelength, z_center)
        res_axl = 2 * res_lat

        roi_x = tuple(s * res_lat for s in config.roi_x)
        roi_z = (
            z_center + config.roi_z[0] * res_axl,
            config.roi_z[1] * res_axl,
            z_center + config.roi_z[2] * res_axl,
        )
        roi_offsets = (
            config.roi_distance_offset_limits[0],
            config.roi_distance_offset_spacing * wavelength,
            config.roi_distance_offset_limits[1],
        )
        x, z, offset = _predictor_grid(roi_x, roi_z, roi_offsets)
        inside = (x / (res_lat + config.ellipsoid_constant_1)) ** 2 + (
            (z - z_center) / (res_axl + config.ellipsoid_constant_1)
        ) ** 2 <= 1
        roi = (x[inside], z[inside], offset[inside])

        lateral_limit = pitch * n_el / 2 + config.lateral_limit_offset
        sz = config.outer_z
        outer_x = (
            config.outer_x[0] * lateral_limit,
            config.outer_x[1] * res_lat,
            config.outer_x[2] * lateral_limit,
        )
        outer_z = (
            sz[0] * z_center + sz[1] * res_axl + config.outer_z_offset,
            sz[2] * res_axl,
            sz[3] * z_center + sz[4] * res_axl + sz[5] * z_center,
        )
        outer_offsets = (
            config.outer_distance_offset_limits[0],
            config.outer_distance_offset_spacing * wavelength,
            config.outer_distance_offset_limits[1],
        )
        x, z, offset = _predictor_grid(outer_x, outer_z, outer_offsets)
        outside = (x / (res_lat + config.ellipsoid_constant_2)) ** 2 + (
            (z - z_center) / (res_axl + config.ellipsoid_constant_2)
        ) ** 2 > 1
        outer = (x[outside], z[outside], offset[outside])

        parts = []
        for predictors in (roi, outer):

            def chunks_fn(predictors=predictors, signal_fn=signal_fn):
                x, z, offset = predictors
                return _model_chunks(x, z, offset, config.predictor_chunk_size, signal_fn)

            if config.ica:
                n_bytes = predictors[0].size * n_el * np.dtype(complex).itemsize
                keep = n_bytes <= config.max_cache_bytes
                parts.append(_ica_basis(chunks_fn, n_el, keep_chunks=keep))
            else:
                chunks = list(chunks_fn())
                parts.append(np.concatenate(chunks, axis=1))

        model = np.concatenate(parts, axis=1)
        norms = np.linalg.norm(model, axis=0)
        model = np.where(norms > 0, model / np.where(norms > 0, norms, 1), 0)
        is_roi = np.arange(model.shape[1]) < parts[0].shape[1]
        results.append((model, is_roi))
    return results


def _single_threaded_blas():
    """Worker initializer: one BLAS thread per process, so workers don't oversubscribe."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(1)


@cache_output(
    "depths", "sound_speed", "center_frequency", "pitch", "n_elements", "f_number",
    "analytic", "start_depth", "end_depth", "config", verbose=True,
)  # fmt: skip
def generate_admire_models(
    depths,
    sound_speed: float,
    center_frequency: float,
    pitch: float,
    n_elements: int,
    f_number: float = 2.0,
    analytic: bool = True,
    start_depth: float | None = None,
    end_depth: float | None = None,
    config: ADMIREConfig | None = None,
    verbose: bool = False,
    n_workers: int | None = None,
) -> ADMIREModels:
    """Precompute the ADMIRE models for an imaging geometry.

    Port of ``ADMIRE_models_generation_main.m`` and the functions it calls. The
    channel data that the models are applied to must be time-delayed (focused at
    every depth) and sampled at the uniformly spaced ``depths``.

    Generation is done once per geometry, on the CPU, and takes from seconds to
    minutes: deep windows have outer models with hundreds of thousands of
    predictors. The result is cached on disk (see :mod:`zea.internal.cache`).

    Args:
        depths (array-like): Uniformly spaced depths (m) of the axial samples of the
            delayed channel data, i.e. the z-axis of the beamforming grid.
        sound_speed (float): Speed of sound in m/s.
        center_frequency (float): Transmit center frequency in Hz.
        pitch (float): Element pitch in meters.
        n_elements (int): Number of elements in the sub-aperture that is fitted for
            each image line.
        f_number (float): F-number of aperture growth; the sub-aperture is limited
            to ``depth / f_number`` (but at least ``config.min_num_elements``
            elements). ``0`` disables aperture growth.
        analytic (bool): ``True`` for analytic (IQ) data, ``False`` for real RF data.
        start_depth (float, optional): Only windows starting at or below this depth
            are processed. Defaults to the first depth.
        end_depth (float, optional): Only windows ending at or above this depth are
            processed. Defaults to the last depth.
        config (ADMIREConfig, optional): Model and fit settings. Defaults to the
            reference settings.
        verbose (bool): Show a progress bar.
        n_workers (int, optional): Number of processes that generate windows in
            parallel. Defaults to one. Does not affect the result (or its cache key).

    Returns:
        ADMIREModels: The models of every STFT window and selected frequency.
    """
    config = ADMIREConfig() if config is None else config
    depths = np.asarray(depths, dtype=float)
    if depths.ndim != 1 or depths.size < 2:
        raise ValueError("`depths` must be a 1D array with at least two depths.")
    spacing = np.diff(depths)
    dz = spacing.mean()
    if dz <= 0 or not np.allclose(spacing, dz, rtol=1e-3, atol=0):
        raise ValueError("ADMIRE requires uniformly spaced, increasing depths.")
    sampling_frequency = sound_speed / (2 * dz)

    # STFT window: one pulse FWHM of a Gaussian pulse with the given bandwidth
    sff = (config.bandwidth * center_frequency) ** 2 / (8 * np.log(2))
    pulse_fwhm = np.sqrt(8 * np.log(2)) * np.sqrt(1 / (4 * np.pi**2 * sff))
    window_length = math.ceil(config.pulse_multiplier * pulse_fwhm * sampling_frequency)
    fft_length = 2 * window_length

    starts = np.arange(0, depths.size - window_length + 1, window_length)
    start_depth = depths[0] if start_depth is None else start_depth
    end_depth = depths[-1] if end_depth is None else end_depth
    keep = (depths[starts] >= start_depth) & (depths[starts + window_length - 1] <= end_depth)
    starts = starts[keep]
    if starts.size == 0:
        raise ValueError(
            f"No STFT window of {window_length} samples fits in the depth range; the grid "
            "is too short."
        )

    bins, frequencies = _select_frequencies(
        fft_length, sampling_frequency, center_frequency, config, analytic
    )
    if bins.size == 0:
        raise ValueError("No DFT bin falls inside the fitted frequency band.")
    ks = 2 * np.pi * frequencies / sound_speed
    ks = ks * _resolve_calibration(config.wavenumber_calibration, center_frequency, ks.size)

    params = {
        "sound_speed": sound_speed,
        "pitch": pitch,
        "sampling_frequency": sampling_frequency,
        "half_pulse_width_samples": window_length,
        "pulse_sigma_distance": np.sqrt(sound_speed**2 / (4 * np.pi**2 * sff)),
        "config": config,
    }

    element_positions = (np.arange(n_elements) - (n_elements - 1) / 2) * pitch

    windows, aperture_masks = [], []
    for start in starts:
        window_depths = (depths[start], depths[start + window_length - 1])
        z_center = depths[start : start + window_length].mean()
        aperture = _aperture_growth(z_center, n_elements, pitch, f_number, config.min_num_elements)
        aperture_masks.append(aperture)
        windows.append((z_center, window_depths, element_positions[aperture], ks, params))

    if n_workers is not None and n_workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # Spawn: the parent may hold a GPU context or threads that do not survive fork
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            min(n_workers, len(windows)), mp_context=context, initializer=_single_threaded_blas
        ) as pool:
            per_window = pool.map(_models_for_window, *zip(*windows))
            if verbose:
                from tqdm import tqdm

                per_window = tqdm(per_window, total=len(windows), desc="Generating ADMIRE models")
            per_window = list(per_window)
    else:
        iterator = windows
        if verbose:
            from tqdm import tqdm

            iterator = tqdm(windows, desc="Generating ADMIRE models")
        per_window = [_models_for_window(*window) for window in iterator]

    n_predictors = max(model.shape[1] for window in per_window for model, _ in window)
    models = np.zeros((starts.size, ks.size, n_elements, n_predictors), dtype=np.complex64)
    roi_mask = np.zeros((starts.size, ks.size, n_predictors), dtype=bool)
    for w, (window, aperture) in enumerate(zip(per_window, aperture_masks)):
        for f, (model, is_roi) in enumerate(window):
            models[w, f, aperture, : model.shape[1]] = model
            roi_mask[w, f, : model.shape[1]] = is_roi

    return ADMIREModels(
        models=models,
        roi_mask=roi_mask,
        aperture_mask=np.stack(aperture_masks),
        window_starts=starts,
        window_length=window_length,
        fft_length=fft_length,
        frequency_bins=bins,
        frequencies=frequencies,
        element_positions=element_positions,
        analytic=analytic,
        config=config,
    )


def elastic_net_ccd(
    X,
    y,
    row_mask,
    alpha: float,
    lambda_scaling_factor: float,
    tolerance: float,
    max_iterations: int,
):
    r"""Batched elastic-net regression with cyclic coordinate descent.

    Port of ``ccd_double_precision.c``. For every problem it minimizes

    .. math::

        \frac{1}{2N} \lVert \tilde{y} - X \beta \rVert_2^2
        + \tilde{\lambda} \left( \alpha \lVert \beta \rVert_1
        + \frac{1 - \alpha}{2} \lVert \beta \rVert_2^2 \right),

    with :math:`\tilde{y} = y / \sigma_y` and :math:`\tilde{\lambda} = \lambda /
    \sigma_y`, where :math:`\lambda` is ``lambda_scaling_factor`` times the RMS of
    :math:`y`, and returns :math:`\sigma_y \beta`. The columns of ``X`` must have
    unit norm (or be zero). Each problem stops after the first sweep in which the
    largest change of its coefficients falls below ``tolerance``, as in the
    reference; the loop runs until every problem has stopped.

    Args:
        X (Tensor): Design matrices of shape ``(..., N, P)``.
        y (Tensor): Observations of shape ``(..., B, N)``: ``B`` right-hand sides
            share each design matrix.
        row_mask (Tensor): Mask of shape ``(..., N)`` with the rows that hold
            observations. Masked-out rows of ``X`` and ``y`` must be zero.
        alpha (float): Elastic-net mixing parameter.
        lambda_scaling_factor (float): Regularization strength relative to the RMS
            of ``y``.
        tolerance (float): Convergence tolerance.
        max_iterations (int): Maximum number of sweeps over all coefficients.

    Returns:
        Tensor: Coefficients of shape ``(..., B, P)``.
    """
    X = ops.convert_to_tensor(X)
    dtype = X.dtype
    y = ops.convert_to_tensor(y, dtype=dtype)
    row_mask = ops.cast(row_mask, dtype)[..., None, :]  # (..., 1, N)
    n_obs = ops.maximum(ops.sum(row_mask, axis=-1, keepdims=True), 1.0)  # (..., 1, 1)

    # Standardize y by its standard deviation over the observed rows
    mean = ops.sum(y, axis=-1, keepdims=True) / n_obs
    std = ops.sqrt(ops.sum(row_mask * (y - mean) ** 2, axis=-1, keepdims=True) / n_obs)
    lam = lambda_scaling_factor * ops.sqrt(ops.sum(y**2, axis=-1, keepdims=True) / n_obs)
    valid = std > 0
    safe_std = ops.where(valid, std, 1.0)
    residual = ops.where(valid, y / safe_std, 0.0)  # (..., B, N)
    lam = lam / safe_std  # (..., B, 1)

    threshold = lam * alpha
    denominator = 1.0 / n_obs + lam * (1.0 - alpha)
    n_predictors = X.shape[-1]
    X_columns = ops.moveaxis(X, -1, 0)  # (P, ..., N)
    beta = ops.zeros(y.shape[:-1] + (n_predictors,), dtype=dtype)
    beta = ops.moveaxis(beta, -1, 0)  # (P, ..., B)

    def coordinate_step(j, state):
        beta, residual, max_change, active = state
        x_j = ops.take(X_columns, j, axis=0)[..., None, :]  # (..., 1, N)
        beta_j = ops.take(beta, j, axis=0)[..., None]  # (..., B, 1)
        partial = residual + x_j * beta_j
        p_j = ops.sum(x_j * partial, axis=-1, keepdims=True) / n_obs
        new_beta_j = ops.sign(p_j) * ops.maximum(ops.abs(p_j) - threshold, 0.0) / denominator
        # Problems that have converged keep their coefficients (and residual) unchanged
        new_beta_j = ops.where(active, new_beta_j, beta_j)
        residual = ops.where(active, partial - x_j * new_beta_j, residual)
        change = (beta_j - new_beta_j) ** 2 / n_obs  # (..., B, 1)
        beta = ops.slice_update(beta, (j,) + (0,) * (len(beta.shape) - 1), new_beta_j[None, ..., 0])
        return beta, residual, ops.maximum(max_change, change), active

    def sweep(state):
        beta, residual, active, iteration = state
        beta, residual, max_change, _ = ops.fori_loop(
            0, n_predictors, coordinate_step, (beta, residual, ops.zeros_like(lam), active)
        )
        # Like the reference, every problem stops on its own after its first sweep
        # whose largest change falls below the tolerance
        active = ops.logical_and(active, max_change >= tolerance)
        return beta, residual, active, iteration + 1

    def not_converged(beta, residual, active, iteration):
        return ops.logical_and(ops.any(active), iteration < max_iterations)

    state = (beta, residual, valid, ops.zeros((), dtype="int32"))
    state = ops.while_loop(not_converged, lambda *s: sweep(s), state)
    beta = ops.moveaxis(state[0], 0, -1)
    return beta * ops.where(valid, std, 0.0)


def _realify(models):
    """``[[Re, -Im], [Im, Re]]`` real form of complex models ``(..., n_el, P)``."""
    re, im = np.real(models), np.imag(models)
    top = np.concatenate([re, -im], axis=-1)
    bottom = np.concatenate([im, re], axis=-1)
    return np.concatenate([top, bottom], axis=-2).astype(np.float32)


def _dft_matrices(models: ADMIREModels):
    """Forward and inverse DFT matrices restricted to the selected bins.

    Returns:
        tuple: ``cos`` and ``sin`` of shape ``(n_freqs, window_length)`` and the
        inverse weights of shape ``(n_freqs,)``.
    """
    n = np.arange(models.window_length)
    phase = 2 * np.pi * np.outer(models.frequency_bins, n) / models.fft_length
    if models.analytic:
        weights = np.ones(models.frequency_bins.size)
    else:
        # A positive bin also stands in for its conjugate negative twin
        weights = np.where(models.frequency_bins == 0, 1.0, 2.0)
    weights = weights / models.fft_length
    return np.cos(phase).astype(np.float32), np.sin(phase).astype(np.float32), weights


def apply_admire(channel_data, models: ADMIREModels, window_batch_size: int | None = None):
    """Remove clutter from time-delayed channel data with ADMIRE.

    Port of ``apply_ADMIRE_models_CPU.m``: each STFT window is taken to the
    frequency domain, the models are fitted at every selected frequency, the signal
    is rebuilt from the ROI predictors only and taken back to the time domain.
    Frequencies outside the fitted band are removed. Samples that are not covered by
    a window (at the end of the axial range) are passed through unchanged.

    Args:
        channel_data (Tensor): Delayed channel data of shape
            ``(n_z, n_lines, n_elements, n_ch)``: for every image line, the
            sub-aperture centered on that line. ``n_ch`` is ``1`` for RF data and
            ``2`` for IQ data, matching ``models.analytic``.
        models (ADMIREModels): Models from :func:`generate_admire_models`.
        window_batch_size (int, optional): Number of STFT windows fitted at once,
            which bounds memory use. Defaults to all windows at once.

    Returns:
        Tensor: Decluttered channel data with the shape of ``channel_data``.
    """
    n_z, n_lines, n_el, n_ch = channel_data.shape
    if n_el != models.n_elements:
        raise ValueError(
            f"The channel data has {n_el} elements per line, but the ADMIRE models were "
            f"generated for {models.n_elements}."
        )
    if models.analytic != (n_ch == 2):
        kind = "IQ" if models.analytic else "RF"
        raise ValueError(
            f"The ADMIRE models were generated for {kind} data, but the channel data has "
            f"n_ch={n_ch}."
        )
    first = int(models.window_starts[0])
    length = models.window_length
    covered = models.n_windows * length
    if not np.array_equal(models.window_starts, first + length * np.arange(models.n_windows)):
        raise ValueError("ADMIRE expects contiguous, non-overlapping STFT windows.")
    if first + covered > n_z:
        raise ValueError(
            f"The ADMIRE models cover {first + covered} axial samples, but the channel "
            f"data has only {n_z}."
        )
    config = models.config

    data = ops.cast(channel_data, "float32")
    real = data[..., 0]
    imag = data[..., 1] if n_ch == 2 else ops.zeros_like(real)

    cos, sin, inverse_weights = _dft_matrices(models)
    inverse_cos = (cos * inverse_weights[:, None]).astype(np.float32)
    inverse_sin = (sin * inverse_weights[:, None]).astype(np.float32)
    X = _realify(models.models)  # (W, F, 2 n_el, 2 P)
    roi = np.concatenate([models.roi_mask, models.roi_mask], axis=-1).astype(np.float32)
    rows = np.concatenate([models.aperture_mask, models.aperture_mask], axis=-1)
    rows = np.broadcast_to(rows[:, None], X.shape[:-1]).astype(np.float32)

    def to_windows(x):
        x = x[first : first + covered]
        return ops.reshape(x, (models.n_windows, length) + tuple(x.shape[1:]))

    def process(real_w, imag_w, X_w, roi_w, rows_w):
        def forward(matrix, x):
            return ops.einsum("fn,wnle->wfle", matrix, x)

        def inverse(matrix, x):
            return ops.einsum("fn,wfle->wnle", matrix, x)

        # Forward DFT at the selected bins: (W, F, n_lines, n_el)
        spec_re = forward(cos, real_w) + forward(sin, imag_w)
        spec_im = forward(cos, imag_w) - forward(sin, real_w)
        y = ops.concatenate([spec_re, spec_im], axis=-1) * rows_w[:, :, None, :]

        beta = elastic_net_ccd(
            X_w,
            y,
            rows_w,
            alpha=config.alpha,
            lambda_scaling_factor=config.lambda_scaling_factor,
            tolerance=config.tolerance,
            max_iterations=config.max_iterations,
        )
        fitted = ops.einsum("wfnp,wflp->wfln", X_w, beta * roi_w[:, :, None, :])
        fit_re, fit_im = fitted[..., :n_el], fitted[..., n_el:]

        # Inverse DFT from the selected bins back to the window samples
        out_re = inverse(inverse_cos, fit_re) - inverse(inverse_sin, fit_im)
        out_im = inverse(inverse_sin, fit_re) + inverse(inverse_cos, fit_im)
        return out_re, out_im

    fn_args = (to_windows(real), to_windows(imag), X, roi, rows)
    if window_batch_size is None or window_batch_size >= models.n_windows:
        out_re, out_im = process(*fn_args)
    else:
        out_re, out_im = vmap(process, batch_size=window_batch_size, fn_supports_batch=True)(
            *fn_args
        )

    def stitch(original, windows):
        windows = ops.reshape(windows, (covered,) + tuple(original.shape[1:]))
        return ops.concatenate([original[:first], windows, original[first + covered :]], axis=0)

    real = stitch(real, out_re)
    if n_ch == 1:
        return real[..., None]
    return ops.stack([real, stitch(imag, out_im)], axis=-1)
