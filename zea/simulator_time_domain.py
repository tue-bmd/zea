"""Time domain ultrasound simulator.

A time-domain alternative to :func:`zea.simulator.simulate_rf`. Instead of synthesizing every
scatterer response frequency by frequency, each echo is splat in an ``(n_ax, n_el)`` map at its
linear two-way delay, which is convolved once per receive channel with the transmit pulse.
Attenuation is evaluated only at the pulse center frequency, so the result is an approximation.

Example usage
^^^^^^^^^^^^^

.. doctest::

    >>> from zea.simulator_time_domain import simulate_rf_td
    >>> import numpy as np

    >>> raw_data = simulate_rf_td(
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

from zea.beamform.lens_correction import compute_lens_corrected_travel_times
from zea.func.ultrasound import directivity
from zea.simulator import (
    _apply_elevation_slab,
    _resolve_element_width,
    _warn_if_elevation_extent,
    attenuate,
    hann_unnormalized,
    spread,
    apply_receive_chain,
)


def simulate_rf_td(
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
):
    """Time-domain (splat-and-convolve) RF simulator.

    A faster alternative to :func:`zea.simulator.simulate_rf` that produces equivalent RF data
    without a per-scatterer, per-frequency Fourier synthesis. Each scatterer
    contribution is splatted, with linear sub-sample interpolation, into an
    ``(n_ax, n_el)`` spike map at its two-way sample delay; the spike map is then
    convolved once per receive channel with a real transmit pulse.

    Directivity, geometric spreading, and attenuation are evaluated at the pulse
    center frequency (a broadband approximation appropriate for the time domain),
    reusing the same helpers as :func:`zea.simulator.simulate_rf`.

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
            use :class:`zea.ops.Simulate` rather than calling `simulate_rf_td` directly.
        element_height (float): The elevation height of the elements [m], used for the
            elevation directivity and the elevation slab. If None, defaults to element_width.
        max_chunk_gb (float): Approximate memory budget [GB] for the (chunk, n_el, n_el)
            tensors held at once while iterating over scatterers. Scatterers are processed
            in chunks sized to this budget, so peak memory no longer scales with the total
            scatterer count. Must be a static (Python) value, not a traced array.
        noise_level_db (float): Electronic noise level in dB relative to the noiseless RF
            maximum. None disables the noise. Must be static under jit.
        tgc_max_db (float): Time gain compensation in dB at the last axial sample, ramped
            linearly in dB from 0 at the first. 0 disables it. Must be static under jit.
        noise_seed (int | SeedGenerator | jax.random.key, optional): Seed for the noise. Vary it
            across transmit batches to keep the realisations independent.
        noise_reference (float): Reference amplitude for the noise level. If None, defaults to the
            noiseless RF maximum. Pass a fixed reference to avoid the noise level changing per
            transmit batch. See :func:`zea.simulator.apply_receive_chain`.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).
    """
    element_width = _resolve_element_width(probe_geometry, element_width)
    if element_height is None:
        element_height = element_width
    n_ax = int(n_ax)
    n_tx = t0_delays.shape[0]
    n_el = probe_geometry.shape[0]
    n_scat = scatterer_positions.shape[0]

    pulse = get_pulse_waveform(center_frequency, sampling_frequency)

    # Chunk so the (n_scat, n_el, n_el) tensors never materialize at once. The factor is
    # approximate memory use after jit fusion, not a count of intermediate tensors.
    bytes_per_scatterer = n_el * n_el * 4 * 6
    chunk_size = max(1, int(max_chunk_gb * 1e9) // max(bytes_per_scatterer, 1))

    spike_maps = [ops.zeros((n_ax, n_el), dtype="float32") for _ in range(n_tx)]
    for start in range(0, n_scat, chunk_size):
        stop = min(start + chunk_size, n_scat)
        base_gain, two_way_time = _precompute_scatterer_response(
            scatterer_positions[start:stop],
            scatterer_magnitudes[start:stop],
            probe_geometry,
            apply_lens_correction,
            lens_thickness,
            lens_sound_speed,
            sound_speed,
            center_frequency,
            element_width,
            element_height,
            attenuation_coef,
            elevation_lens,
        )
        for tx in range(n_tx):
            spike_maps[tx] = spike_maps[tx] + _simulate_transmit(
                base_gain,
                two_way_time,
                t0_delays[tx],
                initial_times[tx],
                tx_apodizations[tx],
                t_peak[tx],
                sampling_frequency,
                n_ax,
                n_el,
            )

    parts = [_convolve_pulse_over_channels(spike_map, pulse) for spike_map in spike_maps]
    rf_data = ops.stack(parts, axis=0)
    rf_data = rf_data[..., None]
    return apply_receive_chain(rf_data, noise_level_db, tgc_max_db, noise_seed, noise_reference)


def _simulate_transmit(
    base_gain,
    two_way_time,
    t0_delays,
    initial_time,
    tx_apodization,
    t_peak,
    sampling_frequency,
    n_ax,
    n_el,
):
    """Build the (n_ax, n_el) spike map for a single transmit event."""
    gain = base_gain * tx_apodization[None, :, None]
    # The pulse is zero-centered, so t_peak shifts it as in simulate_rf.
    tau = two_way_time + t0_delays[None, :, None] - initial_time + t_peak
    sample_positions = tau * sampling_frequency
    return _scatter_spike_map(sample_positions, gain, n_ax, n_el)


def _precompute_scatterer_response(
    scatterer_positions,
    scatterer_magnitudes,
    probe_geometry,
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    sound_speed,
    center_frequency,
    element_width,
    element_height,
    attenuation_coef,
    elevation_lens=False,
):
    """Compute the transmit-independent gain and two-way travel time tensors.

    Returns:
        base_gain (array-like): The (n_scat, n_tx_el, n_rx_el) amplitude of each
            scatterer contribution, excluding the per-transmit apodization.
        two_way_time (array-like): The (n_scat, n_tx_el, n_rx_el) round-trip travel
            time [s], excluding transmit delays and initial times.
    """
    magnitudes = scatterer_magnitudes
    if elevation_lens:
        _warn_if_elevation_extent(probe_geometry)
        scatterer_positions, magnitudes = _apply_elevation_slab(
            scatterer_positions, magnitudes, probe_geometry, element_height
        )

    # See the matching cast in `simulate_rf`.
    scatterer_positions = ops.cast(scatterer_positions, "float32")
    magnitudes = ops.cast(magnitudes, "float32")

    one_way_distance = _one_way_distances(
        probe_geometry,
        scatterer_positions,
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
    )
    travel_time = one_way_distance / sound_speed
    two_way_distance = one_way_distance[:, :, None] + one_way_distance[:, None, :]

    element_directivity = _element_directivity(
        scatterer_positions,
        probe_geometry,
        element_width,
        element_height,
        sound_speed,
        center_frequency,
    )
    directivity_pair = element_directivity[:, :, None] * element_directivity[:, None, :]
    spread_attenuation = (
        spread(one_way_distance[:, :, None], 0.5 if elevation_lens else 1.0)
        * spread(one_way_distance[:, None, :], 1.0)
        * attenuate(center_frequency, attenuation_coef, two_way_distance)
    )

    base_gain = magnitudes[:, None, None] * directivity_pair * spread_attenuation
    two_way_time = travel_time[:, :, None] + travel_time[:, None, :]
    return base_gain, two_way_time


def _one_way_distances(
    probe_geometry,
    scatterer_positions,
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    sound_speed,
):
    """Compute the one-way distance [m] from each scatterer to each element."""
    if not apply_lens_correction:
        return ops.linalg.norm(probe_geometry[None] - scatterer_positions[:, None], axis=-1)
    travel_times = compute_lens_corrected_travel_times(
        probe_geometry,
        scatterer_positions,
        lens_thickness=lens_thickness,
        c_lens=lens_sound_speed,
        c_medium=sound_speed,
        n_iter=3,
    )
    return travel_times * sound_speed


def _element_directivity(
    scatterer_positions, probe_geometry, element_width, element_height, sound_speed, frequency
):
    """3D directivity from each element to each scatterer."""
    relative = scatterer_positions[:, None] - probe_geometry[None]
    theta = ops.arctan2(relative[..., 0], relative[..., 2])
    phi = ops.arctan2(relative[..., 1], relative[..., 2])
    return directivity(frequency, theta, element_width, sound_speed) * directivity(
        frequency, phi, element_height, sound_speed
    )


def _scatter_spike_map(sample_positions, weights, n_ax, n_el):
    """Splat weighted spikes into an (n_ax, n_el) map with linear interpolation.

    Args:
        sample_positions (array-like): The fractional sample index of each
            contribution of shape (n_scat, n_tx_el, n_rx_el).
        weights (array-like): The amplitude of each contribution, same shape.
        n_ax (int): The number of axial samples.
        n_el (int): The number of receive elements.

    Returns:
        array-like: The spike map of shape (n_ax, n_el), summed over scatterers and
        transmit elements.
    """
    lower_index = ops.floor(sample_positions)
    fractional = sample_positions - lower_index
    lower_index = ops.cast(lower_index, "int32")
    rx_index = ops.reshape(ops.arange(n_el, dtype="int32"), (1, 1, n_el))
    lower_map = _accumulate_tap(lower_index, weights * (1 - fractional), rx_index, n_ax, n_el)
    upper_map = _accumulate_tap(lower_index + 1, weights * fractional, rx_index, n_ax, n_el)
    return lower_map + upper_map


def _accumulate_tap(sample_index, weight, rx_index, n_ax, n_el):
    """Scatter-add one interpolation tap into a flattened (n_ax * n_el) buffer."""
    valid = (sample_index >= 0) & (sample_index < n_ax)
    weight = ops.where(valid, weight, ops.zeros_like(weight))
    clamped_index = ops.clip(sample_index, 0, n_ax - 1)
    flat_index = ops.reshape(clamped_index * n_el + rx_index, (-1,))
    flat_weight = ops.reshape(weight, (-1,))
    buffer = ops.segment_sum(flat_weight, flat_index, num_segments=n_ax * n_el)
    return ops.reshape(buffer, (n_ax, n_el))


def _convolve_pulse_over_channels(spike_map, pulse):
    """Convolve every receive channel of the spike map with the pulse ('same' mode).

    Args:
        spike_map (array-like): The spike map of shape (n_ax, n_el).
        pulse (array-like): The real transmit pulse of shape (n_pulse,).

    Returns:
        array-like: The convolved RF data of shape (n_ax, n_el).
    """
    n_ax = spike_map.shape[0]
    n_pulse = pulse.shape[0]
    n_full = n_ax + n_pulse - 1
    signals = ops.pad(ops.transpose(spike_map, (1, 0)), [[0, 0], [0, n_full - n_ax]])
    kernel = ops.pad(ops.reshape(pulse, (1, n_pulse)), [[0, 0], [0, n_full - n_pulse]])
    full = _multiply_spectra(signals, kernel, n_full)
    start = (n_pulse - 1) // 2
    return ops.transpose(full[:, start : start + n_ax], (1, 0))


def _multiply_spectra(signals, kernel, n_full):
    """Convolve along the last axis via the real FFT."""
    signal_real, signal_imag = ops.rfft(signals)
    kernel_real, kernel_imag = ops.rfft(kernel)
    product_real = signal_real * kernel_real - signal_imag * kernel_imag
    product_imag = signal_real * kernel_imag + signal_imag * kernel_real
    return ops.irfft((product_real, product_imag), fft_length=n_full)


def get_pulse_waveform(center_frequency, sampling_frequency, n_period=4, n_samples=129):
    """Generate a real, Hann-windowed sinusoidal transmit pulse in the time domain.

    This is the time-domain counterpart of :func:`zea.simulator.get_pulse_spectrum_fn`: an even,
    zero-centered pulse whose spectrum matches the windowed sine used by
    :func:`zea.simulator.simulate_rf`. The pulse length ``n_samples`` is a fixed (static) sample
    count so the pulse has a compile-time-known shape; the Hann window zeros any
    samples beyond the ``n_period``-period support, so ``n_samples`` only needs to
    be large enough (and odd, to keep the pulse symmetric and delay-aligned) to
    contain that support.

    Args:
        center_frequency (float): The center frequency of the pulse [Hz].
        sampling_frequency (float): The sampling frequency [Hz].
        n_period (float): The number of periods spanned by the Hann window.
        n_samples (int): The (odd) number of samples in the pulse.

    Returns:
        array-like: The pulse waveform of shape (n_samples,).
    """
    width = n_period / center_frequency
    support_samples = width * sampling_frequency
    if support_samples > n_samples:
        raise ValueError(
            f"Hann window support ({support_samples:.1f} samples) exceeds n_samples "
            f"({n_samples}); the pulse would be truncated. Increase n_samples or "
            "reduce sampling_frequency / n_period."
        )
    times = (ops.arange(n_samples, dtype="float32") - n_samples // 2) / sampling_frequency
    window = hann_unnormalized(times, width)
    return window * ops.cos(2 * np.pi * center_frequency * times)
