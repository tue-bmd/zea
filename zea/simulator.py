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
    ... )

"""

import numpy as np
from keras import ops

from zea.beamform.lens_correction import compute_lens_corrected_travel_times


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
    wavefront_only=False,
    focus_distances=None,
    transmit_origins=None,
    polar_angles=None,
    azimuth_angles=None,
):
    """
    Simulates RF data for a given set of scatterers.

    Two forward models are available, selected by ``wavefront_only``:

    - **Full model** (``wavefront_only=False``, default): every transmitting element
      contributes its own delayed, apodized, attenuated pulse to every scatterer, and
      the responses are summed over all transmitting elements. This is the most
      accurate model but scales with the number of transmitting elements.
    - **Wavefront-only model** (``wavefront_only=True``): only a *single* transmit
      wavefront is simulated per scatterer, collapsing the transmit contribution onto
      one element (per scatterer). This removes the transmit-element axis from the
      inner tensors (``tau_total`` becomes ``[n_scat, 1, n_rxel]``), which is
      substantially faster and lighter on memory, at the cost of a potentially larger
      model error (it does not capture the full transmit wave field). Which element is
      chosen respects the focusing geometry:

      - Non-transmitting elements (zero apodization) are never chosen.
      - For a **focused** transmit, scatterers *before* the focus see the *first*
        arriving element (the converging wavefront), while scatterers *beyond* the
        focus see the *last* arriving element (the wavefront has crossed the focus and
        diverges again). This before/after-focus split requires the transmit geometry
        (``focus_distances``, ``transmit_origins``, ``polar_angles``). If that geometry
        is not provided, the model falls back to first-arrival everywhere, which is
        only valid up to the focal depth.

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
        wavefront_only (bool): If ``True``, use the wavefront-only approximation
            (a single transmit wavefront per scatterer is simulated), collapsing the
            transmit-element axis for a faster, lighter simulation. Defaults to
            ``False`` (full model).
        focus_distances (array-like, optional): The focus distance [m] per transmit of
            shape (n_tx,). Only used by the wavefront-only model to decide, per
            scatterer, whether it lies before or beyond the focus. If ``None`` the
            wavefront-only model uses first-arrival everywhere (valid only up to the
            focal depth).
        transmit_origins (array-like, optional): The transmit beam origins [m] of shape
            (n_tx, 3). Only used by the wavefront-only model (see ``focus_distances``).
            Defaults to the array origin when ``None``.
        polar_angles (array-like, optional): The transmit polar (steering) angles [rad]
            of shape (n_tx,). Only used by the wavefront-only model (see
            ``focus_distances``). Defaults to zeros when ``None``.
        azimuth_angles (array-like, optional): The transmit azimuth angles [rad] of
            shape (n_tx,). Only used by the wavefront-only model. Defaults to zeros.

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

    n_ax_rounded = _round_up_to_power_of_two(int(n_ax)).astype("float32")

    freqs = ops.arange(n_ax_rounded // 2 + 1, dtype="float32") / n_ax_rounded * sampling_frequency

    waveform_spectrum = pulse_spectrum_fn(freqs)
    parts = []
    for tx in range(n_tx):
        tx_idx = ops.array(tx)

        # Transmit distances/apodizations, indexed by transmit element. In the
        # wavefront-only model (eq. 6) we keep only the first wave to reach each
        # scatterer, so we replace the per-element transmit quantities with those of
        # the single earliest-arriving element per scatterer. This collapses the
        # transmit-element axis to size 1 (``tau_total`` becomes [n_scat, 1, n_rxel]),
        # making the simulation much faster and lighter. The full model (eq. 5) keeps
        # all transmit elements.
        dist_tx = dist  # [n_scat, n_txel]
        t0_delays_tx = t0_delays[tx_idx][None, :]  # [1, n_txel]
        tx_apod = tx_apodizations[tx][None, :]  # [1, n_txel]
        if wavefront_only:
            # arrival time at each scatterer from each transmit element, [n_scat, n_txel]
            tau_tx = dist / sound_speed + t0_delays[tx_idx][None, :]

            # Only actively transmitting elements (nonzero apodization) can be the
            # source of the wavefront. For a focused transmit the earliest-*firing*
            # elements are the edges of the aperture (fired first so the beam
            # converges), which lie outside the active subaperture and have zero
            # apodization. Without this mask those edge elements would win the argmin
            # past the focal depth, collapsing the transmit contribution onto a
            # zero-apodization element and zeroing out the scatterer response entirely.
            inactive = tx_apod <= 0  # [1, n_txel]

            # Which single element represents the wavefront at each scatterer depends
            # on the focusing geometry (mirroring
            # ``zea.beamform.beamformer.transmit_delays``): before the focus the
            # converging wavefront arrives *first*, so we take the earliest-arriving
            # element; beyond the focus the wavefront has crossed the focal point and
            # diverges again, so the relevant single arrival is the *latest*. Without
            # the transmit geometry we cannot tell the two apart, so we fall back to
            # first-arrival everywhere (valid only up to the focal depth).
            before_focus = _before_focus_mask(
                scatterer_positions,
                focus_distances[tx_idx] if focus_distances is not None else None,
                transmit_origins[tx_idx] if transmit_origins is not None else None,
                polar_angles[tx_idx] if polar_angles is not None else None,
                azimuth_angles[tx_idx] if azimuth_angles is not None else None,
            )  # [n_scat, 1] boolean, True where the first-arrival should be used

            # First arrival: mask inactive elements to +inf and take the argmin.
            first_idx = ops.argmin(ops.where(inactive, float("inf"), tau_tx), axis=1)
            # Last arrival: mask inactive elements to -inf and take the argmax.
            last_idx = ops.argmax(ops.where(inactive, float("-inf"), tau_tx), axis=1)
            sel_idx = ops.where(before_focus[:, 0], first_idx, last_idx)[:, None]

            dist_tx = ops.take_along_axis(dist, sel_idx, axis=1)  # [n_scat, 1]
            t0_delays_tx = ops.take_along_axis(
                ops.broadcast_to(t0_delays_tx, ops.shape(dist)), sel_idx, axis=1
            )  # [n_scat, 1]
            tx_apod = ops.take_along_axis(tx_apod, sel_idx, axis=1)  # [n_scat, 1]
            first_idx = sel_idx  # reused below for the transmit-side directivity

        # [n_scat, n_txel, rxel] (n_txel is 1 for the wavefront-only model)
        dist_total = dist[:, None] + dist_tx[:, :, None]

        # [n_scat, n_txel, n_rxel]
        tau_total = (dist_total / sound_speed) + t0_delays_tx[..., None] - initial_times[tx_idx]

        scat_pos_relative_to_probe = scatterer_positions[:, None] - probe_geometry[None]

        # Compute 3D directivity
        theta = ops.arctan2(
            scat_pos_relative_to_probe[:, :, 0], scat_pos_relative_to_probe[:, :, 2]
        )
        phi = ops.arctan2(scat_pos_relative_to_probe[:, :, 1], scat_pos_relative_to_probe[:, :, 2])

        # For the wavefront-only model, the transmit-side directivity uses the angle to
        # the first-arriving element only (collapsing the transmit-element axis).
        theta_tx = ops.take_along_axis(theta, first_idx, axis=1) if wavefront_only else theta
        phi_tx = ops.take_along_axis(phi, first_idx, axis=1) if wavefront_only else phi

        directivity_tx = directivity(
            freqs[None, None, None],
            theta_tx[..., None, None],
            element_width,
            sound_speed,
        ) * directivity(
            freqs[None, None, None],
            phi_tx[..., None, None],
            element_width,
            sound_speed,
        )
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
                * tx_apod[..., None, None]
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


def _before_focus_mask(
    scatterer_positions, focus_distance, transmit_origin, polar_angle, azimuth_angle
):
    """Returns, per scatterer, whether it lies before the transmit focus.

    Mirrors the before/after-focus split in
    :func:`zea.beamform.beamformer.transmit_delays`: a scatterer is "before the focus"
    when it sits between the aperture and the focal point along the beam direction, in
    which case the converging wavefront reaches it *first*. Beyond the focus the
    wavefront has crossed the focal point and diverges again, so its *last* arrival is
    the relevant one.

    Args:
        scatterer_positions (array-like): Scatterer positions [m] of shape (n_scat, 3).
        focus_distance (scalar or None): The focus distance [m] for this transmit. A
            positive value focuses in front of the array. If ``None`` (or zero, i.e. a
            plane wave), every scatterer is treated as before the focus (first-arrival).
        transmit_origin (array-like or None): The beam origin [m] of shape (3,). Defaults
            to the array origin when ``None``.
        polar_angle (scalar or None): The polar (steering) angle [rad]. Defaults to 0.
        azimuth_angle (scalar or None): The azimuth angle [rad]. Defaults to 0.

    Returns:
        array-like: Boolean mask of shape (n_scat, 1), ``True`` where the scatterer is
        before the focus (use first arrival).
    """
    n_scat = ops.shape(scatterer_positions)[0]
    if focus_distance is None:
        return ops.ones((n_scat, 1), dtype="bool")

    if polar_angle is None:
        polar_angle = ops.array(0.0, dtype="float32")
    if azimuth_angle is None:
        azimuth_angle = ops.zeros_like(polar_angle)
    if transmit_origin is None:
        transmit_origin = ops.zeros(3, dtype="float32")

    beam_direction = ops.stack(
        [
            ops.sin(polar_angle) * ops.cos(azimuth_angle),
            ops.sin(polar_angle) * ops.sin(azimuth_angle),
            ops.cos(polar_angle),
        ]
    )

    focal_point = transmit_origin + focus_distance * beam_direction  # (3,)
    projection = ops.sum(
        (scatterer_positions - focal_point[None]) * beam_direction[None], axis=-1
    )  # (n_scat,)

    # A positive projection means the scatterer is beyond the focus along the beam. For
    # a diverging transmit (negative focus_distance) the sign flips. A plane wave
    # (focus_distance == 0) has no focus, so everything is "before" (first-arrival).
    before = ops.sign(focus_distance) * projection < 0.0
    before = ops.where(focus_distance == 0.0, ops.ones_like(before), before)
    return before[:, None]


def directivity(f, theta, element_width, sound_speed, rigid_baffle=True):
    """Computes the directivity of a single element.

    Args:
        f (array-like): The input frequencies [Hz].
        theta (array-like): The angles [rad].
        element_width (float): The width of the element [m].
        sound_speed (float): The speed of sound [m/s].
        rigid_baffle (bool): Whether the element is mounted on a rigid baffle,
            impacting the directivity.

    Returns:
        array-like: The directivity of the element.
    """

    if element_width is None:
        response = ops.ones_like(theta)
        return response

    # element_width / wavelength == element_width * f / sound_speed. Using the
    # latter avoids dividing by f, so the DC bin (f == 0) stays finite: the
    # argument is 0 and sinc(0) == 1 (isotropic directivity), the correct limit.
    response = sinc(element_width * f / sound_speed * ops.sin(theta))
    if not rigid_baffle:
        response *= ops.cos(theta)
    return response


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
    num = 0.5 * sinc(f * width)
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


def sinc(x):
    """The normalized sinc function with a small offset to prevent division by zero."""
    x = ops.abs(np.pi * x) + 1e-9
    return ops.sin(x) / x


def _round_up_to_power_of_two(x):
    """Rounds up to the next power of two."""
    return 2 ** np.ceil(np.log2(x))
