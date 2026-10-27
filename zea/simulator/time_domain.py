"""Time-domain RF simulator: every echo splat at its two-way delay, convolved once per channel
with the transmit pulse. It shares the pulses, probe, medium and scene positions with
:func:`~zea.simulator.simulate_rf` (:mod:`~zea.simulator.frequency_domain`), then evaluates
directivity, spreading and attenuation at the center frequency instead of per bin."""

from keras import ops

from zea.func.ultrasound import directivity
from zea.internal.core import ndim
from zea.simulator.response import (
    _element_angles,
    _element_frame,
    _one_way_time,
    _scene_positions,
    attenuate,
    medium_model,
    probe_model,
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

    An approximation of :func:`simulate_rf` without the per-frequency synthesis, faster for 2D
    probes with few transmits. Each scatterer contribution is splatted, with linear sub-sample
    interpolation, into an ``(n_ax, n_el)`` spike map at its two-way sample delay; the spike map
    is then convolved once per receive channel with the transmit pulse. Directivity, geometric
    spreading and attenuation are evaluated at the center frequency (a broadband approximation
    appropriate for the time domain), with the same probe and medium as :func:`simulate_rf`.

    Takes the required arguments of :func:`simulate_rf` with the same meaning, without the
    multi-plane transmits: ``t0_delays`` and ``tx_apodizations`` are (n_tx, n_el) only. Of its
    options it takes ``two_dimensional``, ``element_height``, ``waveforms_two_way`` and
    ``waveform_sampling_frequency`` as there, plus:

    Args:
        max_chunk_gb (float): Approximate memory budget [GB] for the (chunk, n_el, n_el)
            tensors held at once while iterating over scatterers, which bounds the peak memory
            instead of the scatterer count. Not comparable with the budget of
            :func:`simulate_rf`, which bounds a different block of work. Static under jit.
        scatter_exponent (float): Weigh the scattered waveform spectrum by
            ``(f / center_frequency)**scatter_exponent``, as in :func:`simulate_rf`, but one
            shared exponent only: the whole medium is splatted into a single spike map and
            convolved with one pulse, so a per-scatterer vector is rejected. Static under jit.

    The probe model options of :func:`simulate_rf` (``element_normals``,
    ``baffle_impedance_ratio``, ``n_sub_elements``, ``elevation_focus``,
    ``lens_attenuation_coef``, ``simplified_directivity``), its heterogeneous media (the maps)
    and its spectrum and jit options (``band_db``, ``n_fft``, ``scatter_exponent_range``) do
    not apply here.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1), noiseless:
            the receive chain is :func:`zea.func.apply_receive_chain`.
    """
    if ndim(scatter_exponent) > 0:
        raise ValueError(
            "per-scatterer backscatter coefficients are only supported in the frequency "
            "domain simulator"
        )
    if ndim(t0_delays) != 2:
        raise ValueError(
            "simulate_rf_td takes t0_delays of shape (n_tx, n_el); multi-plane transmits are "
            "only supported in the frequency domain simulator"
        )
    _validate_scatter_exponent(scatter_exponent)
    n_ax = int(n_ax)
    n_tx = int(ops.shape(t0_delays)[0])
    n_el = int(ops.shape(probe_geometry)[0])
    pulses = transmit_pulses(
        n_tx, center_frequency, sampling_frequency, waveforms_two_way, waveform_sampling_frequency
    )
    medium = medium_model(sound_speed, center_frequency, attenuation_coef)
    probe = probe_model(
        probe_geometry,
        medium,
        center_frequency,
        pulses,
        element_width=element_width,
        element_height=element_height,
        apply_lens_correction=apply_lens_correction,
        lens_thickness=lens_thickness,
        lens_sound_speed=lens_sound_speed,
        two_dimensional=two_dimensional,
    )
    positions = _scene_positions(scatterer_positions, probe)
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
        tx_gain, rx_gain, travel_time = _scatterer_response(
            positions[start:stop], magnitudes[start:stop], probe, medium, center_frequency
        )
        for tx in range(n_tx):
            spike_maps[tx] = spike_maps[tx] + _simulate_transmit(
                tx_gain,
                rx_gain,
                travel_time,
                t0_delays[tx],
                initial_times[tx],
                tx_apodizations[tx],
                t_peak[tx],
                sampling_frequency,
                n_ax,
            )

    parts = [
        _convolve_pulse_over_channels(spike_map, waveforms[pulse])
        for spike_map, pulse in zip(spike_maps, pulses)
    ]
    return ops.stack(parts, axis=0)[..., None]


def _simulate_transmit(
    tx_gain,
    rx_gain,
    travel_time,
    t0_delays,
    initial_time,
    tx_apodization,
    t_peak,
    sampling_frequency,
    n_ax,
):
    """Build the (n_ax, n_el) spike map for a single transmit event."""
    # The pulse is zero-centered, so t_peak shifts it as in simulate_rf.
    tx_samples = (travel_time + t0_delays[None] - initial_time + t_peak) * sampling_frequency
    rx_samples = travel_time * sampling_frequency
    return _scatter_spike_map(tx_samples, tx_gain * tx_apodization[None], rx_samples, rx_gain, n_ax)


def _scatterer_response(positions, magnitudes, probe, medium, center_frequency):
    """Compute the transmit-independent one-way gains and travel time.

    Every term of a contribution factors into a transmit and a receive element part
    (directivity, spreading and the attenuation of the two-way path), so the
    (n_scat, n_tx_el, n_rx_el) gain is the outer product of the two and the two-way time
    their sum; :func:`_scatter_spike_map` forms both inside its kernel.

    Returns:
        tx_gain (array-like): The (n_scat, n_el) amplitude from each transmit element to
            each scatterer, excluding the per-transmit apodization.
        rx_gain (array-like): The (n_scat, n_el) amplitude from each scatterer to each
            receive element, including the scatterer magnitude.
        travel_time (array-like): The (n_scat, n_el) one-way travel time [s] between each
            scatterer and element, excluding transmit delays and initial times.
    """
    # Through a lens, the medium distance with the travel time of the refracted path.
    physical_distance = medium.sound_speed * _one_way_time(
        positions,
        probe.geometry,
        medium.sound_speed,
        probe.apply_lens_correction,
        probe.lens_thickness,
        probe.lens_sound_speed,
    )
    # Half a wavelength at least for the travel time and the spreading, as in simulate_rf.
    # The attenuation keeps the physical path length, as there.
    one_way_distance = ops.maximum(physical_distance, medium.min_dist)
    travel_time = one_way_distance / medium.sound_speed

    # The attenuation is exponential in the distance, so that of the two-way path is the
    # product of the one-way attenuations.
    one_way_gain = _element_directivity(positions, probe, medium, center_frequency) * attenuate(
        center_frequency, medium.attenuation_coef, physical_distance
    )
    tx_exponent = 0.5 if probe.two_dimensional else 1.0
    tx_gain = one_way_gain * spread(one_way_distance, tx_exponent, medium.min_dist)
    rx_gain = magnitudes[:, None] * one_way_gain * spread(one_way_distance, 1.0, medium.min_dist)
    return tx_gain, rx_gain, travel_time


def _element_directivity(positions, probe, medium, frequency):
    """Directivity from each element to each scatterer, at the direction cosines of the
    frequency-domain simulator (:func:`_element_angles`); no elevation term in 2D."""
    relative = positions[:, None] - probe.geometry[None]
    theta, phi, _ = _element_angles(relative, _element_frame(None, relative.dtype))
    lateral = directivity(frequency, theta, probe.element_width, medium.sound_speed)
    if probe.two_dimensional:
        return lateral
    return lateral * directivity(frequency, phi, probe.element_height, medium.sound_speed)


def _scatter_spike_map(tx_samples, tx_gain, rx_samples, rx_gain, n_ax, n_replicas=16):
    """Splat weighted spikes into an (n_ax, n_el) map with linear interpolation.

    Every (scatterer, transmit element, receive element) contribution lands at the
    fractional sample ``tx_samples + rx_samples`` with amplitude ``tx_gain * rx_gain``. These
    (n_scat, n_tx_el, n_rx_el) tensors are only formed inside this function, so they fuse
    into the scatter (and the gather of the gradient) instead of being held in memory.

    The scatter-add is bound by contention, as neighbouring scatterers land on the same
    samples. Two rewrites reduce it without changing the result (up to the float summation
    order): both taps of a contribution are written as one 2-vector to the same row, the
    upper tap shifted by one sample afterwards, and the scatterers are spread round-robin
    over ``n_replicas`` private copies of the map, summed at the end. The gradient is the
    exact transpose, a gather of the upstream map at both taps (:func:`_gather_spike_map`),
    as the autodiff transpose of the replicated scatter does not fuse and is ~10x slower.

    Args:
        tx_samples (array-like): The (n_scat, n_tx_el) transmit part of the delay [samples].
        tx_gain (array-like): The (n_scat, n_tx_el) transmit part of the amplitude.
        rx_samples (array-like): The (n_scat, n_rx_el) receive part of the delay [samples].
        rx_gain (array-like): The (n_scat, n_rx_el) receive part of the amplitude.
        n_ax (int): The number of axial samples.
        n_replicas (int): The number of private copies of the map, costing
            ``n_replicas * (n_ax + 1) * n_rx_el * 2`` floats. Static under jit.

    Returns:
        array-like: The spike map of shape (n_ax, n_rx_el), summed over scatterers and
        transmit elements.
    """
    n_el = int(ops.shape(rx_samples)[1])

    def contributions(tx_samples, tx_gain, rx_samples, rx_gain):
        sample_positions = tx_samples[:, :, None] + rx_samples[:, None, :]
        weights = tx_gain[:, :, None] * rx_gain[:, None, :]
        return sample_positions, weights

    @ops.custom_gradient
    def splat(tx_samples, tx_gain, rx_samples, rx_gain):
        sample_positions, weights = contributions(tx_samples, tx_gain, rx_samples, rx_gain)
        spike_map = _splat_replicated(sample_positions, weights, n_ax, n_el, n_replicas)

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            sample_positions, weights = contributions(tx_samples, tx_gain, rx_samples, rx_gain)
            d_positions, d_weights = _gather_spike_map(
                sample_positions, weights, upstream, n_ax, n_el
            )
            return (
                ops.sum(d_positions, axis=2),
                ops.sum(d_weights * rx_gain[:, None, :], axis=2),
                ops.sum(d_positions, axis=1),
                ops.sum(d_weights * tx_gain[:, :, None], axis=1),
            )

        return spike_map, grad

    return splat(tx_samples, tx_gain, rx_samples, rx_gain)


def _splat_replicated(sample_positions, weights, n_ax, n_el, n_replicas):
    """Forward of :func:`_scatter_spike_map` on the (n_scat, n_tx_el, n_rx_el) contributions."""
    lower_index = ops.floor(sample_positions)
    fractional = sample_positions - lower_index
    # Row r holds the lower tap of sample r - 1 and the upper tap of sample r.
    row = ops.cast(lower_index, "int32") + 1
    n_rows = n_ax + 1
    n_scat = int(ops.shape(sample_positions)[0])
    replica = ops.reshape(ops.arange(n_scat, dtype="int32") % n_replicas, (-1, 1, 1))
    rx_index = ops.reshape(ops.arange(n_el, dtype="int32"), (1, 1, n_el))

    # Out-of-range rows go to one extra segment, dropped below.
    n_segments = n_replicas * n_rows * n_el
    valid = (row >= 0) & (row < n_rows)
    flat_index = ops.where(valid, (replica * n_rows + row) * n_el + rx_index, n_segments)
    taps = ops.stack([weights * (1 - fractional), weights * fractional], axis=-1)
    buffer = ops.segment_sum(
        ops.reshape(taps, (-1, 2)), ops.reshape(flat_index, (-1,)), num_segments=n_segments + 1
    )
    buffer = ops.sum(ops.reshape(buffer[:-1], (n_replicas, n_rows, n_el, 2)), axis=0)
    return buffer[1:, :, 0] + buffer[:-1, :, 1]


def _gather_spike_map(sample_positions, weights, upstream, n_ax, n_el):
    """Gradient of :func:`_scatter_spike_map`: read the upstream (n_ax, n_el) map at both
    taps of every contribution.

    Returns:
        tuple: The gradients with respect to ``sample_positions`` and ``weights``.
    """
    lower_index = ops.floor(sample_positions)
    fractional = sample_positions - lower_index
    # One zero row on either side: padded row k is sample k - 1, so out-of-range taps
    # clip onto a zero row.
    padded = ops.reshape(ops.pad(upstream, [[1, 1], [0, 0]]), (-1,))
    row = ops.cast(lower_index, "int32") + 1
    rx_index = ops.reshape(ops.arange(n_el, dtype="int32"), (1, 1, n_el))
    lower = ops.take(padded, ops.clip(row, 0, n_ax + 1) * n_el + rx_index)
    upper = ops.take(padded, ops.clip(row + 1, 0, n_ax + 1) * n_el + rx_index)
    return weights * (upper - lower), (1 - fractional) * lower + fractional * upper


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
