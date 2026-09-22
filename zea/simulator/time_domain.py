"""Time-domain RF simulator: every echo splat at its two-way delay, convolved once per channel
with the transmit pulse."""

from keras import ops

from zea.beamform.lens_correction import compute_lens_corrected_travel_times
from zea.func.ultrasound import directivity
from zea.internal.core import ndim
from zea.simulator.element import (
    _element_angles,
    _element_frame,
    _scene_positions,
    attenuate,
    element_model,
    spread,
)
from zea.simulator.pulse import transmit_pulses
from zea.simulator.record import _validate_scatter_exponent

__all__ = ["simulate_rf_td"]


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
    *,
    two_dimensional=False,
    element_height=None,
    max_chunk_gb=10.0,
    scatter_exponent=2.0,
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
):
    """Time-domain (splat-and-convolve) RF simulator.

    An approximation of :func:`simulate_rf` without the per-frequency synthesis. Each scatterer
    contribution is splatted, with linear sub-sample interpolation, into an ``(n_ax, n_el)``
    spike map at its two-way sample delay; the spike map is then convolved once per receive
    channel with a real transmit pulse.

    Directivity, geometric spreading, and attenuation are evaluated at the pulse center
    frequency (a broadband approximation appropriate for the time domain), reusing the same
    helpers as :func:`simulate_rf`.

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
        two_dimensional (bool): Simulate in the imaging plane, as a 1D probe behind an ideal
            elevation lens: the scatterers are moved to the probe's elevation center, there is
            no elevation directivity, and the transmit spreads cylindrically rather than
            spherically. Rejects a probe with elevation extent.
        element_height (float): The elevation height of the elements [m], used for the
            elevation directivity. If None, an eighth of the width of a 1D probe (at least
            ``element_width``), or ``element_width`` for a 2D probe.
        max_chunk_gb (float): Approximate memory budget [GB] for the (chunk, n_el, n_el)
            tensors held at once while iterating over scatterers. Scatterers are processed
            in chunks sized to this budget, so peak memory no longer scales with the total
            scatterer count. Must be a static (Python) value, not a traced array.
        scatter_exponent (float): Weigh the scattered waveform spectrum by
            ``(f / center_frequency)**scatter_exponent``. 2 is Rayleigh scattering (e.g. blood),
            myocardium is approximately 1.5, soft tissue 0.6-0.8. Must be static under jit.
            One shared exponent only: the whole medium is splatted into a single spike map and
            convolved with one pulse, so a per-scatterer vector is rejected. Use
            :func:`simulate_rf` for that.
        waveforms_two_way (array-like, optional): Two-way transmit waveforms of shape
            (n_tx, n_samples) or (n_samples,), as in :func:`simulate_rf`; None is the default
            pulse of :func:`transmit_pulse`. Must be static under jit.
        waveform_sampling_frequency (float): Sampling frequency [Hz] of ``waveforms_two_way``.
            Must be static under jit.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1), noiseless:
            the receive chain is :func:`zea.func.apply_receive_chain`.
    """
    if ndim(scatter_exponent) > 0:
        raise ValueError(
            "per-scatterer backscatter coefficients are only supported in the frequency "
            "domain simulator"
        )
    _validate_scatter_exponent(scatter_exponent)
    n_ax = int(n_ax)
    n_tx = int(ops.shape(t0_delays)[0])
    n_el = int(ops.shape(probe_geometry)[0])
    pulses = transmit_pulses(
        n_tx, center_frequency, sampling_frequency, waveforms_two_way, waveform_sampling_frequency
    )
    model = element_model(
        probe_geometry,
        sound_speed,
        center_frequency,
        pulses,
        element_width=element_width,
        element_height=element_height,
        attenuation_coef=attenuation_coef,
        apply_lens_correction=apply_lens_correction,
        lens_thickness=lens_thickness,
        lens_sound_speed=lens_sound_speed,
        two_dimensional=two_dimensional,
    )
    positions = _scene_positions(scatterer_positions, model)
    magnitudes = ops.cast(scatterer_magnitudes, "float32")
    n_scat = int(ops.shape(positions)[0])
    waveforms = {
        pulse: _scattered_waveform(pulse, center_frequency, scatter_exponent) for pulse in pulses
    }

    # Chunk so the (n_scat, n_el, n_el) tensors never materialize at once. The factor is
    # approximate memory use after jit fusion, not a count of intermediate tensors.
    bytes_per_scatterer = n_el * n_el * 4 * 6
    chunk_size = max(1, int(max_chunk_gb * 1e9) // max(bytes_per_scatterer, 1))

    spike_maps = [ops.zeros((n_ax, n_el), dtype="float32") for _ in range(n_tx)]
    for start in range(0, n_scat, chunk_size):
        stop = min(start + chunk_size, n_scat)
        base_gain, two_way_time = _scatterer_response(
            positions[start:stop], magnitudes[start:stop], model, center_frequency
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

    parts = [
        _convolve_pulse_over_channels(spike_map, waveforms[pulse])
        for spike_map, pulse in zip(spike_maps, pulses)
    ]
    return ops.stack(parts, axis=0)[..., None]


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


def _scatterer_response(positions, magnitudes, model, center_frequency):
    """Compute the transmit-independent gain and two-way travel time tensors.

    Returns:
        base_gain (array-like): The (n_scat, n_tx_el, n_rx_el) amplitude of each
            scatterer contribution, excluding the per-transmit apodization.
        two_way_time (array-like): The (n_scat, n_tx_el, n_rx_el) round-trip travel
            time [s], excluding transmit delays and initial times.
    """
    physical_distance = _one_way_distances(positions, model)
    # Half a wavelength at least for the travel time and the spreading, as in simulate_rf.
    # The attenuation keeps the physical path length, as there.
    one_way_distance = ops.maximum(physical_distance, model.min_dist)
    travel_time = one_way_distance / model.sound_speed
    two_way_distance = physical_distance[:, :, None] + physical_distance[:, None, :]

    element_directivity = _element_directivity(positions, model, center_frequency)
    directivity_pair = element_directivity[:, :, None] * element_directivity[:, None, :]
    spread_attenuation = (
        spread(one_way_distance[:, :, None], 0.5 if model.two_dimensional else 1.0, model.min_dist)
        * spread(one_way_distance[:, None, :], 1.0, model.min_dist)
        * attenuate(center_frequency, model.attenuation_coef, two_way_distance)
    )

    base_gain = magnitudes[:, None, None] * directivity_pair * spread_attenuation
    two_way_time = travel_time[:, :, None] + travel_time[:, None, :]
    return base_gain, two_way_time


def _one_way_distances(positions, model):
    """One-way path length [m] from each element center to each position, (n_scat, n_el).

    Through a lens it is the medium distance with the travel time of the refracted path.
    """
    if not model.apply_lens_correction:
        return ops.linalg.norm(positions[:, None] - model.geometry[None], axis=-1)
    travel_times = compute_lens_corrected_travel_times(
        model.geometry,
        positions,
        lens_thickness=model.lens_thickness,
        c_lens=model.lens_sound_speed,
        c_medium=model.sound_speed,
        n_iter=3,
    )
    return travel_times * model.sound_speed


def _element_directivity(positions, model, frequency):
    """Directivity from each element to each scatterer, at the direction cosines of the
    frequency-domain simulator (:func:`_element_angles`); no elevation term in 2D."""
    relative = positions[:, None] - model.geometry[None]
    theta, phi, _ = _element_angles(relative, _element_frame(None, relative.dtype))
    lateral = directivity(frequency, theta, model.element_width, model.sound_speed)
    if model.two_dimensional:
        return lateral
    return lateral * directivity(frequency, phi, model.element_height, model.sound_speed)


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


def _scattered_waveform(pulse, center_frequency, scatter_exponent):
    """The pulse sampled at the RF rate with an odd (static) length and its peak on the middle
    sample, so a spike at the two-way delay convolves to the same record as the frequency-domain
    simulator, weighted by ``(f / center_frequency)**scatter_exponent``."""
    waveform = ops.convert_to_tensor(pulse.waveform())
    if not scatter_exponent:
        return waveform

    n_samples = pulse.n_samples
    freqs = ops.arange(n_samples // 2 + 1, dtype="float32") / n_samples * pulse.sampling_frequency
    scatter_gain = (freqs / center_frequency) ** scatter_exponent
    pulse_real, pulse_imag = ops.rfft(waveform)
    return ops.irfft((pulse_real * scatter_gain, pulse_imag * scatter_gain), fft_length=n_samples)
