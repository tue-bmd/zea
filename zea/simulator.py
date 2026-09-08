"""Frequency domain ultrasound simulator.

The simulator works in the frequency domain (RFFT domain) and simulates RF data as a superposition
of scatterer responses. Every scatterer has a location and a magnitude.

To use it in your code, simply call the :func:`simulate_rf` function with the desired
transmit scheme parameters and scatterers. To simulate a sequence of multiple frames,
you can call :func:`simulate_rf` repeatedly with different scatterer positions and magnitudes
and then stack the results.

There is a time-domain variant of the simulator in :mod:`zea.simulator_time_domain`.

Example usage
^^^^^^^^^^^^^

A simple example of simulating RF data with a single scatterer at the center of the probe. For a
more in depth example see the notebook: :doc:`../notebooks/data/zea_simulation_example`.

.. doctest::

    >>> from zea.simulator import simulate_rf
    >>> import numpy as np

    >>> raw_data = simulate_rf(
    ...     scatterer_positions=np.array([[0, 0, 20e-3]]),
    ...     scatterer_magnitudes=np.array([1.0]),
    ...     probe_geometry=np.stack(
    ...         [np.linspace(-20e-3, 20e-3, 64), np.zeros(64), np.zeros(64)], axis=-1
    ...     ),
    ...     apply_lens_correction=True,
    ...     lens_thickness=1e-3,
    ...     lens_sound_speed=1000,
    ...     sound_speed=1540,
    ...     n_ax=1024,
    ...     center_frequency=5e6,
    ...     sampling_frequency=20e6,
    ...     t0_delays=np.zeros((1, 64)),
    ...     initial_times=np.zeros(1),
    ...     element_width=0.2e-3,
    ...     attenuation_coef=0.5,
    ...     tx_apodizations=np.ones((1, 64)),
    ...     t_peak=np.full(1, 1 / 5e6),
    ... )

"""

import functools

import keras
import numpy as np
from keras import ops

from zea import log
from zea.backend import checkpoint, highest_matmul_precision
from zea.beamform.lens_correction import compute_lens_corrected_travel_times
from zea.func.ultrasound import directivity


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
    t_peak,
    elevation_lens=False,
    element_height=None,
    max_chunk_gb=10.0,
    noise_level_db=None,
    tgc_max_db=0.0,
    noise_seed=0,
    noise_reference=None,
    scatter_exponent=2.0,
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
        center_frequency (float): The center frequency of the transmit pulse [Hz].
        sampling_frequency (float): The sampling frequency of the RF data [Hz].
        t0_delays (array-like): The transmit delays [s] of shape (n_tx, n_el).
        initial_times (array-like): The initial times [s] of shape (n_tx,).
        element_width (float): The width of the elements [m].
        attenuation_coef (float): The attenuation coefficient [dB/cm/MHz].
        tx_apodizations (array-like): The transmit apodizations of shape (n_tx, n_el).
        t_peak (array-like): The time of the peak of the transmit pulse [s] of shape (n_tx,).
        elevation_lens (bool): Whether the probe has an elevation lens: drop scatterers outside
            the elevation slab, and focus transmit energy directly downwards (i.e. cylindrical
            instead of spherical spread). For efficient pruning scatterers outside the slab,
            use :class:`zea.ops.Simulate` rather than calling `simulate_rf` directly.
        element_height (float): The elevation height of the elements [m], used for the
            elevation directivity and the elevation slab. If None, defaults to element_width.
        max_chunk_gb (float): Unused here; accepted so :func:`simulate_rf` and
            :func:`zea.simulator_time_domain.simulate_rf_td` share a call signature.
        noise_level_db (float): Electronic noise level in dB relative to the noiseless RF
            maximum. None disables the noise. Must be static under jit.
        tgc_max_db (float): Time gain compensation in dB at the last axial sample, ramped
            linearly in dB from 0 at the first. 0 disables it. Must be static under jit.
        noise_seed (int | SeedGenerator | jax.random.key, optional): Seed for the noise. Vary it
            across transmit batches to keep the realisations independent.
        noise_reference (float): Reference amplitude for the noise level. If None, defaults to the
            noiseless RF maximum. Pass a fixed reference to avoid the noise level changing per
            transmit batch. See :func:`apply_receive_chain`.
        scatter_exponent (float): Weigh the scattered field by
            ``(f / center_frequency)**scatter_exponent``. 2 is Rayleigh scattering (e.g. blood),
            myocardium is approximately 1.5, soft tissue 0.6-0.8. Must be static under jit.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).

    """

    _validate_scatter_exponent(scatter_exponent)

    n_tx = t0_delays.shape[0]

    element_width = _resolve_element_width(probe_geometry, element_width)

    if element_height is None:
        element_height = element_width

    magnitudes = scatterer_magnitudes
    if elevation_lens:
        _warn_if_elevation_extent(probe_geometry)
        scatterer_positions, magnitudes = _apply_elevation_slab(
            scatterer_positions, magnitudes, probe_geometry, element_height
        )

    # tensorflow can't reduce over an empty axis.
    if scatterer_positions.shape[0] == 0:
        shape = (t0_delays.shape[0], int(n_ax), probe_geometry.shape[0], 1)
        return apply_receive_chain(
            ops.zeros(shape, dtype="float32"),
            noise_level_db,
            tgc_max_db,
            noise_seed,
            noise_reference,
        )

    # Phantoms are float64. Cast manually so tensorflow doesn't complain.
    scatterer_positions = ops.cast(scatterer_positions, "float32")
    magnitudes = ops.cast(magnitudes, "float32")

    pulse_spectrum_fn = get_pulse_spectrum_fn(
        center_frequency, n_period=4, sampling_frequency=sampling_frequency
    )

    if not apply_lens_correction:
        dist = ops.linalg.norm(probe_geometry[None] - scatterer_positions[:, None], axis=-1)
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

    # Room for a whole pulse, so record_length below never gates the end of the record away.
    # Traced frequencies give no static pulse length; the record then keeps its old short tail.
    fc_np, fs_np = _concrete(center_frequency), _concrete(sampling_frequency)
    n_pulse = 0 if fc_np is None or fs_np is None else int(np.ceil(4 / fc_np * fs_np))
    n_ax_rounded = float(_round_up_to_power_of_two(int(n_ax) + n_pulse))

    freqs = ops.arange(n_ax_rounded // 2 + 1, dtype="float32") / n_ax_rounded * sampling_frequency

    waveform_spectrum = pulse_spectrum_fn(freqs)

    if scatter_exponent:
        scatter_gain = (freqs / center_frequency) ** scatter_exponent
    else:
        scatter_gain = ops.ones_like(freqs)

    scat_pos_relative_to_probe = scatterer_positions[:, None] - probe_geometry[None]
    theta = ops.arctan2(scat_pos_relative_to_probe[..., 0], scat_pos_relative_to_probe[..., 2])
    phi = ops.arctan2(scat_pos_relative_to_probe[..., 1], scat_pos_relative_to_probe[..., 2])

    # [n_scat, n_el, n_freq]
    directivity_x = directivity(freqs[None, None], theta[..., None], element_width, sound_speed)
    directivity_y = directivity(freqs[None, None], phi[..., None], element_height, sound_speed)
    element_directivity = directivity_x * directivity_y
    attenuation = attenuate(freqs[None, None], attenuation_coef, dist[..., None])
    one_way_phase = delay2(
        freqs[None, None],
        dist[..., None] / sound_speed,
        n_ax_rounded,
        sampling_frequency,
    )
    shared_response = ops.cast(element_directivity * attenuation, "complex64") * one_way_phase

    if elevation_lens:
        tx_response = shared_response * ops.cast(spread(dist[..., None], 0.5), "complex64")
        rx_response = shared_response * ops.cast(spread(dist[..., None], 1.0), "complex64")
    else:
        tx_response = shared_response * ops.cast(spread(dist[..., None], 1.0), "complex64")
        rx_response = tx_response

    # Leave room for the pulse tail
    record_length = n_ax_rounded / sampling_frequency - 2 / center_frequency
    travel_time = dist / sound_speed
    parts = []
    for tx in range(n_tx):
        shifts_not_travel_related = t0_delays[tx][:, None] - initial_times[tx] + t_peak[tx]

        tx_delay = delay2(freqs[None], shifts_not_travel_related, n_ax_rounded, sampling_frequency)
        tx_element_weights = ops.cast(tx_apodizations[tx][:, None], "complex64") * tx_delay

        # delay2 only gates one-way delays. Worst case over the active transmit elements,
        # to never alias in ops.irfft.
        tx_arrival = ops.max(
            ops.where(
                tx_apodizations[tx][None] != 0,
                travel_time + shifts_not_travel_related[None, :, 0],
                -float("inf"),
            ),
            axis=1,
        )
        within_record = ops.cast(tx_arrival[:, None] + travel_time < record_length, "complex64")

        # Explicitly sum over tx dimension before the receive axis exists.
        incident_field = ops.sum(tx_response * tx_element_weights[None], axis=1)
        scattered_field = incident_field * ops.cast(magnitudes[:, None] * scatter_gain, "complex64")
        received_field = scattered_field[:, None] * rx_response * within_record[..., None]
        rf_spectrum = waveform_spectrum * ops.sum(received_field, axis=0)
        parts.append(ops.irfft((ops.real(rf_spectrum), ops.imag(rf_spectrum))))

    rf_data = ops.stack(parts, axis=0)
    rf_data = ops.transpose(rf_data, (0, 2, 1))
    rf_data = rf_data[..., None]
    rf_data = rf_data[:, :n_ax, :, :]
    return apply_receive_chain(rf_data, noise_level_db, tgc_max_db, noise_seed, noise_reference)


def apply_receive_chain(
    rf_data, noise_level_db=None, tgc_max_db=0.0, noise_seed=0, noise_reference=None
):
    """Add electronic noise and time gain compensation to noiseless RF.

    Args:
        rf_data (array-like): Noiseless RF of shape (n_tx, n_ax, n_el, 1), optionally with a
            leading batch axis.
        noise_level_db (float): Noise floor in dB below the peak of ``rf_data``. None disables
            the noise. Must be static when using jit compilation.
        tgc_max_db (float): Gain in dB at the last axial sample. 0 disables it. Must be static when
            using jit compilation.
        noise_seed (int | SeedGenerator | jax.random.key, optional): Seed for the noise. An int
            is stateless, so the same value gives the same realisation; vary it across transmit
            batches. None draws from the global generator and cannot be traced under jit.
        noise_reference (float): Reference amplitude for the noise level. If None, defaults to the
            ``rf_data`` maximum. Pass a fixed reference to avoid the noise level changing per
            transmit batch.

    Returns:
        array-like: RF with same shape as ``rf_data``.
    """
    dtype = keras.backend.standardize_dtype(rf_data.dtype)

    if noise_level_db is not None and noise_level_db > -float("inf"):
        if noise_reference is None:
            # When passing a batch, normalize noise level per item instead of per batch
            noise_reference = ops.max(ops.abs(rf_data), axis=(-4, -3, -2, -1), keepdims=True)
        sigma = noise_reference * 10.0 ** (noise_level_db / 20.0)
        noise = keras.random.normal(ops.shape(rf_data), dtype=dtype, seed=noise_seed)
        rf_data = rf_data + ops.cast(sigma, dtype) * noise

    if tgc_max_db:
        n_ax = int(ops.shape(rf_data)[-3])
        ramp = ops.arange(n_ax, dtype=dtype) / max(n_ax - 1, 1)
        rf_data = rf_data * ops.reshape(10.0 ** (tgc_max_db * ramp / 20.0), (n_ax, 1, 1))

    return rf_data


def _validate_scatter_exponent(scatter_exponent):
    """Reject exponents that make the weighting non-finite: the DC bin is zero, so a
    negative exponent gives infinite gain there, and the NaN spreads over the whole frame."""
    if not np.isfinite(scatter_exponent) or scatter_exponent < 0:
        raise ValueError(
            f"scatter_exponent ({scatter_exponent}) must be finite and non-negative. "
            "2 is Rayleigh scattering (e.g. blood), myocardium is approximately 1.5, "
            "soft tissue 0.6-0.8."
        )


def _resolve_element_width(probe_geometry, element_width):
    """Return the element width, inferring it from the probe pitch when not given."""
    if element_width is not None:
        return element_width
    try:
        geometry = ops.convert_to_numpy(probe_geometry)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(
            "Element width is not provided, and automatic inference is not available for "
            "traced/symbolic probe geometry (for example under JAX JIT or TensorFlow graph "
            "mode). Please provide `element_width` explicitly in the scan/probe parameters."
        ) from exc

    try:
        from zea.probes import Probe

        pitch = Probe.get_pitch(geometry)
    except (ValueError, IndexError, AttributeError) as exc:
        raise ValueError(
            "Element width is not provided and automatic estimation failed from probe "
            "geometry. Please provide `element_width` explicitly or ensure the probe "
            "geometry is a 1-D uniformly spaced linear array. "
            f"Details: {exc}"
        ) from exc
    return pitch * 0.9  # 90% of the pitch


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
    return ops.exp(-ops.log(10) * attenuation_coef / 20 * dist * 100 * ops.abs(f) * 1e-6)


def spread(dist, exponent=1.0, mindist=1e-3):
    """Geometric spreading of the wavefront.

    Args:
        dist (array-like): The distance the wave has traveled.
        exponent (float): 1 for spherical, 0.5 for cylindrical. An elevation lens focuses the
            transmitted energy to a slab, resulting in a cylindrical transmit and a spherical
            receive path.
        mindist (float): Distance that corresponds with unit gain.

    Returns:
        array-like: An amplitude factor in the shape of `dist`.
    """
    dist = ops.clip(dist, mindist, float("inf"))
    return (mindist / dist) ** exponent


def elevation_slab_mask(scatterer_positions, probe_geometry, element_height):
    """Zero out the scatterers an elevation lens never insonifies.

    Returns:
        array-like: 1 inside the slab and 0 outside, of shape (n_scat,).
    """
    if element_height is None:
        raise ValueError("elevation_lens=True requires element_height to be provided.")
    elevation_center = ops.mean(probe_geometry[:, 1])
    offset = ops.abs(scatterer_positions[:, 1] - elevation_center)
    return ops.cast(offset <= element_height / 2, "float32")


def select_elevation_slab(
    scatterer_positions, scatterer_magnitudes, probe_geometry, element_height
):
    """Drop the scatterers an elevation lens never insonifies.

    Not jittable: the output length is data dependent. Under jit use
    :func:`elevation_slab_mask`, which zeroes magnitudes instead and keeps a static shape.

    Returns:
        tuple: the (positions, magnitudes) inside the slab.
    """
    mask = elevation_slab_mask(scatterer_positions, probe_geometry, element_height)
    keep = ops.convert_to_numpy(mask) > 0
    return scatterer_positions[keep], scatterer_magnitudes[keep]


def elevation_slab_bucket(
    scatterer_positions=None,
    scatterer_magnitudes=None,
    probe_geometry=None,
    element_height=None,
    elevation_lens=False,
    bucket_growth=2.0,
    **kwargs,
):
    """
    Prune scatterers outside of the elevation slab. Round up to a power of 2 so jit can cache
    the approximate shape.

    Returns:
        dict: pruned scatterers, or ``{}`` if the input is traced or pruning is disabled.
    """
    del kwargs
    if not elevation_lens or element_height is None:
        return {}
    if scatterer_positions is None or scatterer_magnitudes is None or probe_geometry is None:
        return {}

    try:
        positions = ops.convert_to_numpy(scatterer_positions)
        magnitudes = ops.convert_to_numpy(scatterer_magnitudes)
        geometry = ops.convert_to_numpy(probe_geometry)
    except (RuntimeError, ValueError, TypeError):
        return {}  # traced, fall back to masking

    batched = positions.ndim == 3
    if not batched:
        positions, magnitudes = positions[None], magnitudes[None]

    n_scat = positions.shape[1]
    center = geometry[:, 1].mean()
    inside = np.abs(positions[..., 1] - center) <= element_height / 2

    # ops.map needs a uniform shape when using batched mode
    n_keep = int(inside.sum(axis=1).max())
    if n_keep >= n_scat:
        return {}
    steps = np.ceil(np.log(max(n_keep, 1)) / np.log(bucket_growth))
    bucket = min(n_scat, max(1, int(bucket_growth**steps)))

    index = np.zeros((positions.shape[0], bucket), dtype=np.int64)
    pad_mask = np.ones((positions.shape[0], bucket), dtype=bool)
    for item, row in enumerate(inside):
        kept = np.flatnonzero(row)[:bucket]
        index[item, : len(kept)] = kept
        pad_mask[item, : len(kept)] = False

    positions = np.take_along_axis(positions, index[..., None], axis=1)
    magnitudes = np.where(pad_mask, 0.0, np.take_along_axis(magnitudes, index, axis=1))

    if not batched:
        positions, magnitudes = positions[0], magnitudes[0]
    return {"scatterer_positions": positions, "scatterer_magnitudes": magnitudes}


def _warn_if_elevation_extent(probe_geometry, tol=1e-6):
    """Warn if an elevation lens is used with a seemingly non-1D array probe."""
    try:
        elevation = ops.convert_to_numpy(probe_geometry)[:, 1]
    except (RuntimeError, ValueError, TypeError):
        return  # traced, cannot inspect
    if elevation.max() - elevation.min() > tol:
        log.warning(
            "elevation_lens=True models a 1D probe with a cylindrical lens, but the probe is not "
            f"1D (element elevation min, max: {elevation.min()}, {elevation.max()}) "
            "This is probably a mistake."
        )


def _apply_elevation_slab(
    scatterer_positions, scatterer_magnitudes, probe_geometry, element_height
):
    """Prune to the elevation slab, falling back to masking if positions are traced.

    Under jit `elevation_slab_bucket` has usually pruned already, so the mask only re-zeroes
    padding.
    """
    try:
        return select_elevation_slab(
            scatterer_positions, scatterer_magnitudes, probe_geometry, element_height
        )
    except (RuntimeError, ValueError, TypeError):
        mask = elevation_slab_mask(scatterer_positions, probe_geometry, element_height)
        return scatterer_positions, scatterer_magnitudes * mask


def hann_fd(f, width):
    """The fourier transform of a hann window in the time domain with given width."""
    denom = 1.0 - (f * width) ** 2
    num = 0.5 * ops.sinc(f * width)
    # denom == 0 at f * width == +/-1 is a removable singularity where the Hann
    # window transform equals 0.25. Divide only away from it (using a dummy 1.0
    # at the singular points) and fill the limit in explicitly, so no 0/0 occurs.
    singular = denom == 0
    result = ops.where(singular, 0.25, num / ops.where(singular, 1.0, denom))
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
    return ops.where(ops.abs(x) < width / 2, ops.cos(np.pi * x / width) ** 2, 0)


def get_pulse_spectrum_fn(center_frequency, n_period=3.0, sampling_frequency=None):
    """Computes the spectrum of a sine that is windowed with a Hann window.

    Args:
        center_frequency (float): The center frequency of the transmit pulse.
        n_period (float): The number of periods to include in the pulse.
        sampling_frequency (float): Frequency used for scaling the spectrum such that a waveform
            recovered with ``ops.irfft`` has a unit peak (as ``ops.irfft`` divides the waveform
            by the sampling frequency).

    Returns:
        spectrum_fn (callable): A function that computes the spectrum of the pulse
        for the input frequencies in Hz.
    """
    period = n_period / center_frequency
    scale = 0.5 if sampling_frequency is None else 0.5 * sampling_frequency * period

    def spectrum_fn(f):
        return ops.array(scale, "complex64") * ops.cast(
            (hann_fd(f - center_frequency, period) + hann_fd(f + center_frequency, period)),
            "complex64",
        )

    return spectrum_fn


def get_transducer_bandwidth_fn(probe_center_frequency, bandwidth):
    """Computes the spectrum of a probe with a center frequency and bandwidth.

    Args:
        probe_center_frequency (float): The center frequency of the probe.
        bandwidth (float): The bandwidth of the probe.

    Returns
        spectrum_fn (callable): A function that computes the spectrum of the pulse for
        the input frequencies in Hz.
    """

    def bandwidth_fn(f):
        return hann_unnormalized(ops.abs(f) - probe_center_frequency, bandwidth)

    return bandwidth_fn


def _round_up_to_power_of_two(x):
    """Rounds up to the next power of two."""
    return 2 ** np.ceil(np.log2(x))


# ---------------------------------------------------------------------------------------------
# gem-wave: the same physics as ``simulate_rf`` with the transmit-invariant response shared
# across transmits through two matrix products per frequency block.
# ---------------------------------------------------------------------------------------------


def smooth_size(n):
    """Smallest 2^a 3^b 5^c >= n."""
    best = None
    a = 0
    while 2**a < 2 * n:
        b = 0
        while 2**a * 3**b < 2 * n:
            c = 0
            while 2**a * 3**b * 5**c < n:
                c += 1
            v = 2**a * 3**b * 5**c
            best = v if best is None or v < best else best
            b += 1
        a += 1
    return int(best)


def _hann_fd_np(f, width):
    """:func:`hann_fd` in numpy, for static band selection under an outer jit."""
    denom = 1.0 - (f * width) ** 2
    num = 0.5 * np.sinc(f * width)
    singular = denom == 0
    result = np.where(singular, 0.25, num / np.where(singular, 1.0, denom))
    result = np.where(np.abs(result) > 1.1, 0.25, result)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.25)


def pulse_spectrum_np(freqs, center_frequency, sampling_frequency, n_period):
    """Pulse spectrum of :func:`get_pulse_spectrum_fn` as a numpy array."""
    period = n_period / center_frequency
    f = np.asarray(freqs, np.float32)
    scale = 0.5 * sampling_frequency * period
    return (
        scale
        * (_hann_fd_np(f - center_frequency, period) + _hann_fd_np(f + center_frequency, period))
    ).astype(np.complex64)


def band_bins(n_fft, center_frequency, sampling_frequency, n_period, scatter_exponent, band_db):
    """Contiguous bin range where pulse spectrum times scattering gain exceeds ``band_db``."""
    freqs = np.arange(n_fft // 2 + 1) / n_fft * sampling_frequency
    if band_db is None:
        return 0, len(freqs)
    w = np.abs(pulse_spectrum_np(freqs, center_frequency, sampling_frequency, n_period))
    w = w * (freqs / center_frequency) ** scatter_exponent
    keep = np.flatnonzero(w > w.max() * 10 ** (band_db / 20))
    return int(keep[0]), int(keep[-1]) + 1


def fft_length(
    n_ax,
    sampling_frequency,
    center_frequency,
    sound_speed,
    probe_geometry,
    shift_min,
    shift_max,
    n_period=4.0,
    scatterer_positions=None,
):
    """Smooth FFT length whose echoes never wrap into the first ``n_ax`` samples.

    A kept scatterer has its earliest echo inside the record, so its last one is at most the
    aperture round trip, the spread of the transmit shifts and one pulse later. When the
    positions are given the bound from the farthest scatterer is used if smaller.

    Args:
        n_ax (int): Number of axial samples in the record.
        sampling_frequency (float): Sampling frequency in Hz.
        center_frequency (float): Pulse center frequency in Hz.
        sound_speed (float): Speed of sound in m/s.
        probe_geometry (array-like): Element positions of shape (n_el, 3).
        shift_min (float): Smallest transmit shift (``t0_delays - initial_times + t_peak``).
        shift_max (float): Largest transmit shift.
        n_period (float): Number of periods in the pulse.
        scatterer_positions (array-like, optional): Concrete positions of shape (n_scat, 3).

    Returns:
        int: FFT length, a product of powers of 2, 3 and 5.
    """
    fs, c = float(sampling_frequency), float(sound_speed)
    geometry = np.asarray(probe_geometry, np.float64)
    pulse = 2 * n_period / float(center_frequency)
    aperture = 2 * np.linalg.norm(geometry - geometry.mean(0), axis=1).max()
    n = n_ax + int(np.ceil((2 * aperture / c + float(shift_max - shift_min) + pulse) * fs))
    if scatterer_positions is not None and len(scatterer_positions):
        reach = np.linalg.norm(np.asarray(scatterer_positions, np.float64), axis=1).max()
        reach = reach + np.linalg.norm(geometry, axis=1).max()
        bound = int(np.ceil((2 * reach / c + max(float(shift_max), 0.0) + pulse) * fs))
        n = min(n, max(n_ax, bound))
    return smooth_size(n)


def _concrete(x):
    """numpy view of ``x``, or None when it is traced."""
    if x is None:
        return None
    try:
        return ops.convert_to_numpy(x)
    except (RuntimeError, ValueError, TypeError, NotImplementedError):
        return None


def _to_complex(x):
    return ops.cast(x, "complex64")


def _gem_wave_responses(
    positions,
    geometry,
    freqs,
    sound_speed,
    element_width,
    element_height,
    attenuation_coef,
    lens_thickness,
    lens_sound_speed,
    apply_lens_correction,
    elevation_lens,
):
    """Transmit and receive one-way responses [f, s, e] and the one-way path length [s, e]."""
    relative = positions[:, None] - geometry[None]
    if apply_lens_correction:
        dist = compute_lens_corrected_travel_times(
            geometry,
            positions,
            lens_thickness=lens_thickness,
            c_lens=lens_sound_speed,
            c_medium=sound_speed,
            n_iter=3,
        )
        dist = dist * sound_speed
    else:
        dist = ops.linalg.norm(relative, axis=-1)
    theta = ops.arctan2(relative[..., 0], relative[..., 2])
    phi = ops.arctan2(relative[..., 1], relative[..., 2])
    f3 = freqs[:, None, None]
    amplitude = (
        directivity(f3, theta[None], element_width, sound_speed)
        * directivity(f3, phi[None], element_height, sound_speed)
        * attenuate(f3, attenuation_coef, dist[None])
    )
    phase = ops.exp(
        ops.array(-2j * np.pi, "complex64") * _to_complex(dist[None] * f3 / sound_speed)
    )
    rx = _to_complex(amplitude * spread(dist[None], 1.0)) * phase
    if elevation_lens:
        # An elevation lens focuses the transmit to a slab: cylindrical spread on the way out.
        tx = _to_complex(amplitude * spread(dist[None], 0.5)) * phase
    else:
        tx = rx
    return tx, rx, dist


def _gem_wave_block(
    freqs,
    positions,
    magnitudes,
    geometry,
    shift,
    tx_apodizations,
    center_frequency,
    sound_speed,
    element_width,
    element_height,
    attenuation_coef,
    lens_thickness,
    lens_sound_speed,
    gate_time,
    scatter_exponent,
    apply_lens_correction,
    elevation_lens,
):
    """Band spectrum [f, t, e] of one frequency block over all scatterers.

    Frequency leads every array so the einsums are plain batched matrix products. Scatterers
    whose earliest echo has no support before ``gate_time`` cannot reach the output and are
    dropped, so a long path never wraps into the record.
    """
    tx_response, rx_response, dist = _gem_wave_responses(
        positions,
        geometry,
        freqs,
        sound_speed,
        element_width,
        element_height,
        attenuation_coef,
        lens_thickness,
        lens_sound_speed,
        apply_lens_correction,
        elevation_lens,
    )
    if scatter_exponent:
        gain = (freqs / center_frequency) ** scatter_exponent
    else:
        gain = ops.ones_like(freqs)
    keep = 2 * ops.min(dist, axis=1) / sound_speed + ops.min(shift) < gate_time
    weight = ops.where(keep, magnitudes, 0.0)
    f3 = freqs[:, None, None]
    tx_weights = _to_complex(tx_apodizations[None]) * ops.exp(
        ops.array(-2j * np.pi, "complex64") * _to_complex(shift[None] * f3)
    )
    with highest_matmul_precision():
        incident = ops.einsum("fte,fse->fts", tx_weights, tx_response)
        scattered = incident * _to_complex(weight[None, None, :] * gain[:, None, None])
        return ops.einsum("fts,fse->fte", scattered, rx_response)


def simulate_rf_gem_wave(
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
    t_peak,
    elevation_lens=False,
    element_height=None,
    max_chunk_gb=1.0,
    noise_level_db=None,
    tgc_max_db=0.0,
    noise_seed=0,
    noise_reference=None,
    scatter_exponent=2.0,
    n_period=4.0,
    band_db=-100.0,
    n_fft=None,
):
    """:func:`simulate_rf` with the scatterer response shared across transmits.

    The transmit-independent one-way response is generated once per frequency block and reused
    over all transmits.

    .. code-block:: text

        incident[f, t, s] = sum_e W[f, t, e] R_tx[f, s, e]    W = apod_te exp(-2 pi i f shift_te)
        rf[f, t, e]       = sum_s S[f, t, s] R_rx[f, s, e]    S = incident * mag_s * gain(f)

    :func:`simulate_rf` evaluates the same sums transmit by transmit, re-reading the (large)
    response every time. This version is up to 20x faster for large batches of transmits, and
    only slower for very few transmits (it falls back to simulate_rf with a single transmit,
    to use the more efficient path when there nothing to share).

    Only the bins where the pulse spectrum times the scattering gain exceeds ``band_db`` are
    computed (``band_db=None`` keeps every bin, -100 matches SIMUS defaults), and the record gate is
    applied per scatterer rather than per (transmit, scatterer, element), with the FFT length sized
    so that no kept echo wraps into the record (``n_fft`` pins it). A scatterer is kept when its
    earliest echo still has pulse support inside the record, so the gate is exact up to the pulse
    envelope; echoes that fall past the record are not masked but simply truncated. With
    ``band_db=None`` and the same ``n_fft`` the result matches :func:`simulate_rf` to float32
    precision, except for the last pulse length of the record, where :func:`simulate_rf` retains
    a sub-1e-3 leakage tail from scatterers whose support lies wholly outside it.

    Takes the arguments of :func:`simulate_rf` with the same meaning and defaults, plus:

    Args:
        n_period (float): Periods in the Hann-windowed transmit pulse. :func:`simulate_rf`
            uses 4.
        band_db (float, optional): Bins where the pulse spectrum times the scattering gain is
            below this many dB of its peak are not synthesised. None keeps every bin.
        n_fft (int, optional): FFT length. Derived when None from ``n_ax``, the aperture and
            the transmit shifts (and the scatterer positions when concrete) so that no echo
            wraps into the record. Must be given when the geometry, delays or sound speed are
            traced.
        max_chunk_gb (float): Memory budget for one frequency block. Barely affects GPU speed,
            up to 2x on CPU.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).
    """
    if int(ops.shape(t0_delays)[0]) == 1:
        return simulate_rf(
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
            t_peak,
            elevation_lens=elevation_lens,
            element_height=element_height,
            max_chunk_gb=max_chunk_gb,
            noise_level_db=noise_level_db,
            tgc_max_db=tgc_max_db,
            noise_seed=noise_seed,
            noise_reference=noise_reference,
            scatter_exponent=scatter_exponent,
        )

    _validate_scatter_exponent(scatter_exponent)
    fc, fs = float(center_frequency), float(sampling_frequency)
    n_ax = int(n_ax)
    element_width = _resolve_element_width(probe_geometry, element_width)
    if element_height is None:
        element_height = element_width

    # Concrete views of the raw inputs, before any op puts them into an outer jit.
    raw = [_concrete(x) for x in (t0_delays, initial_times, t_peak, probe_geometry, sound_speed)]
    if n_fft is None:
        if any(x is None for x in raw):
            raise ValueError(
                "n_fft cannot be derived from traced geometry, delays or sound speed; "
                "pass n_fft explicitly."
            )
        t0_np, t_init_np, t_peak_np, geom_np, c_np = raw
        shift_np = t0_np - t_init_np[:, None] + t_peak_np[:, None]
        n_fft = fft_length(
            n_ax,
            fs,
            fc,
            float(c_np),
            geom_np,
            shift_np.min(),
            shift_np.max(),
            n_period,
            _concrete(scatterer_positions),
        )
    n_fft = int(n_fft)

    positions = ops.cast(scatterer_positions, "float32")
    magnitudes = ops.cast(scatterer_magnitudes, "float32")
    geometry = ops.cast(probe_geometry, "float32")
    t0_delays = ops.cast(t0_delays, "float32")
    n_tx, n_el = (int(d) for d in ops.shape(t0_delays))

    if elevation_lens:
        _warn_if_elevation_extent(geometry)
        positions, magnitudes = _apply_elevation_slab(
            positions, magnitudes, geometry, element_height
        )

    def finish(rf):
        return apply_receive_chain(
            rf[..., None], noise_level_db, tgc_max_db, noise_seed, noise_reference
        )

    if int(ops.shape(positions)[0]) == 0:
        return finish(ops.zeros((n_tx, n_ax, n_el), "float32"))

    # Transmit shift per element beyond the travel time, as in simulate_rf.
    shift = (
        t0_delays
        - ops.cast(initial_times, "float32")[:, None]
        + ops.cast(t_peak, "float32")[:, None]
    )
    k0, k1 = band_bins(n_fft, fc, fs, n_period, scatter_exponent, band_db)

    # Forward of one block per bin: the complex responses and the two matrix product outputs.
    n_scat = int(ops.shape(positions)[0])
    per_bin = 8 * ((2 if elevation_lens else 1) * n_scat * n_el + n_tx * n_scat + n_tx * n_el)
    f_block = int(max(1, min(k1 - k0, max_chunk_gb * 2**30 // per_bin)))
    n_blocks = -(-(k1 - k0) // f_block)
    # Spread the band evenly, so the last block is padded by less than a whole block.
    f_block = -(-(k1 - k0) // n_blocks)
    n_band = n_blocks * f_block

    # Band padded to whole blocks; the padded bins get zero pulse weight below.
    freqs_all = np.arange(n_fft // 2 + 1) / n_fft * fs
    freqs = np.full(n_band, freqs_all[k1 - 1], np.float32)
    freqs[: k1 - k0] = freqs_all[k0:k1]
    wave = np.zeros(n_band, np.complex64)
    wave[: k1 - k0] = pulse_spectrum_np(freqs_all, fc, fs, n_period)[k0:k1]
    freqs = ops.convert_to_tensor(freqs)

    def as_f32(x):
        return ops.cast(0.0 if x is None else x, "float32")

    block = checkpoint(
        functools.partial(
            _gem_wave_block,
            positions=positions,
            magnitudes=magnitudes,
            geometry=geometry,
            shift=shift,
            tx_apodizations=ops.cast(tx_apodizations, "float32"),
            center_frequency=fc,
            sound_speed=as_f32(sound_speed),
            element_width=as_f32(element_width),
            element_height=as_f32(element_height),
            attenuation_coef=as_f32(attenuation_coef),
            lens_thickness=as_f32(lens_thickness),
            lens_sound_speed=as_f32(lens_sound_speed),
            gate_time=n_ax / fs + 0.5 * n_period / fc,
            scatter_exponent=float(scatter_exponent),
            apply_lens_correction=bool(apply_lens_correction),
            elevation_lens=bool(elevation_lens),
        )
    )

    def body(i, spectrum):
        start = i * f_block
        block_freqs = ops.slice(freqs, [start], [f_block])
        return ops.slice_update(spectrum, [start, 0, 0], block(block_freqs))

    spectrum = ops.zeros((n_band, n_tx, n_el), "complex64")
    spectrum = ops.fori_loop(0, n_blocks, body, spectrum)
    spectrum = spectrum[: k1 - k0] * ops.convert_to_tensor(wave[: k1 - k0])[:, None, None]

    # limited to prevent large focused line grids from OOM'ing.
    group = min(32, n_tx)
    parts = []
    for start in range(0, n_tx, group):
        band = ops.transpose(spectrum[:, start : start + group], (1, 2, 0))
        pad = ((0, 0), (0, 0), (k0, n_fft // 2 + 1 - k1))
        full = (ops.pad(ops.real(band), pad), ops.pad(ops.imag(band), pad))
        parts.append(ops.irfft(full, fft_length=n_fft)[..., :n_ax])
    rf = ops.transpose(ops.concatenate(parts, axis=0), (0, 2, 1))
    return finish(rf)
