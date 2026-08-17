"""Frequency domain ultrasound simulator.

The simulator works in the frequency domain (RFFT domain) and simulates RF data as a superposition
of scatterer responses. Every scatterer has a location and a magnitude.

To use it in your code, simply call the :func:`simulate_rf` function with the desired
transmit scheme parameters and scatterers. To simulate a sequence of multiple frames,
you can call :func:`simulate_rf` repeatedly with different scatterer positions and magnitudes
and then stack the results.

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
    factored=True,
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
        t0_delays (array-like): The delays of the transmitting elements [s] of shape (n_tx, n_el).
        initial_times (array-like): The initial times of the transmitting elements [s] of
            shape (n_tx,).
        element_width (float): The width of the elements [m].
        attenuation_coef (float): The attenuation coefficient [dB/cm/MHz].
        tx_apodizations (array-like): The apodizations of the transmitting elements of
            shape (n_tx, n_el).
        t_peak (array-like): The time of the peak of the transmit pulse [s] of shape (n_tx,).
        factored (bool): Approximate the geometric spreading with a factorizable term. Slight
            amplitude error for scatterers near a large linear probe, but much faster.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).

    """

    n_tx = t0_delays.shape[0]

    if element_width is None:
        if ops.is_tensor(probe_geometry):
            raise ValueError(
                "Element width is not provided, and automatic inference is not available for "
                "traced/symbolic probe geometry (for example under JAX JIT or TensorFlow graph "
                "mode). Please provide `element_width` explicitly in the scan/probe parameters."
            )

        try:
            from zea.probes import Probe

            pitch = Probe.get_pitch(probe_geometry)
        except ValueError as exc:
            raise ValueError(
                "Element width is not provided and automatic estimation failed from probe "
                "geometry. Please provide `element_width` explicitly or ensure the probe "
                "geometry is a 1-D uniformly spaced linear array. "
                f"Details: {exc}"
            ) from exc
        element_width = pitch * 0.9  # 90% of the pitch

    # Phantoms are float64. Cast manually so tensorflow doesn't complain.
    scatterer_positions = ops.cast(scatterer_positions, "float32")
    scatterer_magnitudes = ops.cast(scatterer_magnitudes, "float32")

    pulse_spectrum_fn = get_pulse_spectrum_fn(center_frequency, n_period=4)

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
    parts = []
    for tx in range(n_tx):
        tx_idx = ops.array(tx)

        scat_pos_relative_to_probe = scatterer_positions[:, None] - probe_geometry[None]

        # Compute 3D directivity
        theta = ops.arctan2(
            scat_pos_relative_to_probe[:, :, 0], scat_pos_relative_to_probe[:, :, 2]
        )
        phi = ops.arctan2(scat_pos_relative_to_probe[:, :, 1], scat_pos_relative_to_probe[:, :, 2])

        directivity_tx = directivity(
            freqs[None, None, None],
            theta[..., None, None],
            element_width,
            sound_speed,
        ) * directivity(
            freqs[None, None, None],
            phi[..., None, None],
            element_width,
            sound_speed,
        )
        if factored:
            # [n_scat, n_el, n_freq]
            attenuation = attenuate(
                freqs[None, None],
                attenuation_coef=attenuation_coef,
                dist=dist[..., None],
            )

            # sqrt such that one_way * other_way together yield mindist/(2sqrt(d_tx*d_rx))
            spread_atten = ops.sqrt(spread(2 * dist[..., None]))

            one_way_response = ops.cast(
                directivity_tx[:, :, 0] * attenuation * spread_atten,
                "complex64",
            ) * delay2(
                freqs[None, None],
                dist[..., None] / sound_speed,
                n_fft=n_ax_rounded,
                sampling_frequency=sampling_frequency,
            )

            # When each element fires, relative to the start of the recording. [n_txel, 1]
            tx_delay_offsets = t0_delays[tx][:, None] - initial_times[tx] + t_peak[tx]

            # Apodization and firing phase per transmitting element. [n_txel, n_freq]
            tx_element_weights = ops.cast(tx_apodizations[tx][:, None], "complex64") * delay2(
                freqs[None],
                tx_delay_offsets,
                n_fft=n_ax_rounded,
                sampling_frequency=sampling_frequency,
            )

            # necessary because delay2 never sees the full round-trip distance in this branch
            record_length = n_ax_rounded / sampling_frequency
            round_trip_time = 2 * ops.mean(dist, axis=1) / sound_speed + ops.mean(tx_delay_offsets)
            within_record = ops.cast(round_trip_time < record_length, "float32")
            magnitudes = scatterer_magnitudes * within_record

            # Explicitly sum over tx dimension before the receive axis exists.
            incident_field = ops.sum(one_way_response * tx_element_weights[None], axis=1)
            scattered_field = incident_field * ops.cast(magnitudes[:, None], "complex64")
            received_field = scattered_field[:, None] * one_way_response
            rf_spectrum = waveform_spectrum * ops.sum(received_field, axis=0)
            result = ops.irfft((ops.real(rf_spectrum), ops.imag(rf_spectrum)))
            parts.append(result)
        else:
            directivity_rx = directivity(
                freqs[None, None, None],
                theta[:, None, :, None],
                element_width,
                sound_speed,
            ) * directivity(
                freqs[None, None, None],
                phi[:, None, :, None],
                element_width,
                sound_speed,
            )

            # [n_scat, n_txel, rxel]
            dist_total = dist[:, None] + dist[:, :, None]

            # [n_scat, n_txel, n_rxel]
            tau_total = (
                (dist_total / sound_speed)
                + t0_delays[tx_idx][None, :, None]
                - initial_times[tx_idx]
                + t_peak[tx_idx]
            )

            attenuation = attenuate(
                freqs[None, None, None],
                attenuation_coef=attenuation_coef,
                dist=dist_total[..., None],
            )

            spread_atten = spread(dist_total[..., None])

            result = (
                waveform_spectrum[None, None, None]
                * delay2(
                    freqs[None, None, None],
                    tau_total[..., None],
                    n_fft=n_ax_rounded,
                    sampling_frequency=sampling_frequency,
                )
                * ops.cast(
                    scatterer_magnitudes[:, None, None, None]
                    * tx_apodizations[tx, None, :, None, None]
                    * directivity_tx
                    * directivity_rx
                    * attenuation
                    * spread_atten,
                    "complex64",
                )
            )

            # Sum over all transmitting elements and scatterers
            result = ops.sum(result, axis=[0, 1])

            result = ops.irfft((ops.real(result), ops.imag(result)))

            parts.append(result)

    rf_data = ops.stack(parts, axis=0)
    rf_data = ops.transpose(rf_data, (0, 2, 1))
    rf_data = rf_data[..., None]
    rf_data = rf_data[:, :n_ax, :, :]
    return rf_data


def simulate_rf_fast(
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
):
    """Time-domain (splat-and-convolve) RF simulator.

    A faster alternative to :func:`simulate_rf` that produces equivalent RF data
    without a per-scatterer, per-frequency Fourier synthesis. Each scatterer
    contribution is splatted, with linear sub-sample interpolation, into an
    ``(n_ax, n_el)`` spike map at its two-way sample delay; the spike map is then
    convolved once per receive channel with a real transmit pulse.

    Directivity, geometric spreading, and attenuation are evaluated at the pulse
    center frequency (a broadband approximation appropriate for the time domain),
    reusing the same helpers as :func:`simulate_rf`.

    Args:
        scatterer_positions (array-like): The scatterer positions [m] of shape
            (n_scat, 3).
        scatterer_magnitudes (array-like): The scatterer magnitudes of shape (n_scat,).
        probe_geometry (array-like): The probe geometry [m] of shape (n_el, 3).
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

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).
    """
    element_width = _resolve_element_width(probe_geometry, element_width)
    n_ax = int(n_ax)
    n_tx = t0_delays.shape[0]
    n_el = probe_geometry.shape[0]

    pulse = get_pulse_waveform(center_frequency, sampling_frequency)
    base_gain, two_way_time = _precompute_scatterer_response(
        scatterer_positions,
        scatterer_magnitudes,
        probe_geometry,
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        center_frequency,
        element_width,
        attenuation_coef,
    )

    parts = []
    for tx in range(n_tx):
        spike_map = _simulate_transmit(
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
        parts.append(_convolve_pulse_over_channels(spike_map, pulse))

    rf_data = ops.stack(parts, axis=0)
    return rf_data[..., None]


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
    attenuation_coef,
):
    """Compute the transmit-independent gain and two-way travel time tensors.

    Returns:
        base_gain (array-like): The (n_scat, n_tx_el, n_rx_el) amplitude of each
            scatterer contribution, excluding the per-transmit apodization.
        two_way_time (array-like): The (n_scat, n_tx_el, n_rx_el) round-trip travel
            time [s], excluding transmit delays and initial times.
    """
    # See the matching cast in `simulate_rf`.
    scatterer_positions = ops.cast(scatterer_positions, "float32")
    scatterer_magnitudes = ops.cast(scatterer_magnitudes, "float32")

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
        scatterer_positions, probe_geometry, element_width, sound_speed, center_frequency
    )
    directivity_pair = element_directivity[:, :, None] * element_directivity[:, None, :]
    spread_attenuation = spread(two_way_distance) * attenuate(
        center_frequency, attenuation_coef, two_way_distance
    )

    base_gain = scatterer_magnitudes[:, None, None] * directivity_pair * spread_attenuation
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
    scatterer_positions, probe_geometry, element_width, sound_speed, frequency
):
    """Compute the 3D directivity from each scatterer to each element."""
    relative = scatterer_positions[:, None] - probe_geometry[None]
    theta = ops.arctan2(relative[..., 0], relative[..., 2])
    phi = ops.arctan2(relative[..., 1], relative[..., 2])
    return directivity(frequency, theta, element_width, sound_speed) * directivity(
        frequency, phi, element_width, sound_speed
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


def _resolve_element_width(probe_geometry, element_width):
    """Return the element width, inferring it from the probe pitch when not given."""
    if element_width is not None:
        return element_width
    if ops.is_tensor(probe_geometry):
        raise ValueError(
            "Element width is not provided, and automatic inference is not available for "
            "traced/symbolic probe geometry (for example under JAX JIT or TensorFlow graph "
            "mode). Please provide `element_width` explicitly in the scan/probe parameters."
        )
    try:
        from zea.probes import Probe

        pitch = Probe.get_pitch(probe_geometry)
    except ValueError as exc:
        raise ValueError(
            "Element width is not provided and automatic estimation failed from probe "
            "geometry. Please provide `element_width` explicitly or ensure the probe "
            "geometry is a 1-D uniformly spaced linear array. "
            f"Details: {exc}"
        ) from exc
    return pitch * 0.9  # 90% of the pitch


def get_pulse_waveform(center_frequency, sampling_frequency, n_period=4, n_samples=129):
    """Generate a real, Hann-windowed sinusoidal transmit pulse in the time domain.

    This is the time-domain counterpart of :func:`get_pulse_spectrum_fn`: an even,
    zero-centered pulse whose spectrum matches the windowed sine used by
    :func:`simulate_rf`. The pulse length ``n_samples`` is a fixed (static) sample
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
    times = (ops.arange(n_samples, dtype="float32") - n_samples // 2) / sampling_frequency
    window = hann_unnormalized(times, width)
    return window * ops.cos(2 * np.pi * center_frequency * times)


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


def spread(dist, mindist=1e-4):
    """Function modeling geometric spreading of the wavefront.

    Args:
        dist (array-like): The distance the wave has traveled.
        mindist (float): The minimum distance to prevent division by zero.

    Returns:
        array-like: The geometric spreading factor of same shape as `dist`.
    """
    dist = ops.clip(dist, mindist, float("inf"))
    return mindist / dist


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


def get_pulse_spectrum_fn(center_frequency, n_period=3.0):
    """Computes the spectrum of a sine that is windowed with a Hann window.

    Args:
        center_frequency (float): The center frequency of the transmit pulse.
        n_period (float): The number of periods to include in the pulse.

    Returns:
        spectrum_fn (callable): A function that computes the spectrum of the pulse
        for the input frequencies in Hz.
    """
    period = n_period / center_frequency

    def spectrum_fn(f):
        return ops.array(1 / 2, "complex64") * ops.cast(
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
