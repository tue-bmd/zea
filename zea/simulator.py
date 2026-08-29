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

import numpy as np
from keras import ops

from zea import log
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

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).

    """

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
        return ops.zeros(shape, dtype="float32")

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

    n_ax_rounded = float(_round_up_to_power_of_two(int(n_ax)))

    freqs = ops.arange(n_ax_rounded // 2 + 1, dtype="float32") / n_ax_rounded * sampling_frequency

    waveform_spectrum = pulse_spectrum_fn(freqs)

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

    record_length = n_ax_rounded / sampling_frequency
    parts = []
    for tx in range(n_tx):
        shifts_not_travel_related = t0_delays[tx][:, None] - initial_times[tx] + t_peak[tx]

        tx_delay = delay2(freqs[None], shifts_not_travel_related, n_ax_rounded, sampling_frequency)
        tx_element_weights = ops.cast(tx_apodizations[tx][:, None], "complex64") * tx_delay

        # delay2 only gates one-way delays. Worst case over elements: drops some in-record
        # pairs, but never aliases in ops.irfft.
        round_trip_time = 2 * ops.max(dist, axis=1) / sound_speed + ops.max(
            shifts_not_travel_related
        )
        within_record = ops.cast(round_trip_time < record_length, "float32")

        # Explicitly sum over tx dimension before the receive axis exists.
        incident_field = ops.sum(tx_response * tx_element_weights[None], axis=1)
        scattered_field = incident_field * ops.cast(
            (magnitudes * within_record)[:, None], "complex64"
        )
        received_field = scattered_field[:, None] * rx_response
        rf_spectrum = waveform_spectrum * ops.sum(received_field, axis=0)
        parts.append(ops.irfft((ops.real(rf_spectrum), ops.imag(rf_spectrum))))

    rf_data = ops.stack(parts, axis=0)
    rf_data = ops.transpose(rf_data, (0, 2, 1))
    rf_data = rf_data[..., None]
    rf_data = rf_data[:, :n_ax, :, :]
    return rf_data


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
