"""Frequency domain ultrasound simulator.

The simulator works in the frequency domain (RFFT domain) and simulates RF data as a superposition
of scatterer responses. Every scatterer has a location and a magnitude.

To use it in your code, simply call the :func:`simulate_rf` function with the desired
transmit scheme parameters and scatterers. To simulate a sequence of multiple frames,
you can call :func:`simulate_rf` repeatedly with different scatterer positions and magnitudes
and then stack the results.

:func:`simulate_rf_zea_wave` evaluates the same model with the scatterer response shared across
transmits, which is faster for many transmits, and :func:`pressure_field` evaluates the transmit
field that the simulators scatter on a grid of points.

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
from zea.beamform.lens_correction import (
    compute_lens_corrected_travel_times,
    compute_lens_path_lengths,
)
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
    elevation_slab_2d=False,
    element_height=None,
    max_chunk_gb=10.0,
    noise_level_db=None,
    tgc_max_db=0.0,
    noise_seed=0,
    noise_reference=None,
    scatter_exponent=2.0,
    rigid_baffle=True,
    bandwidth_percent=None,
    probe_center_frequency=None,
    element_normals=None,
    chirp_sweep=None,
    n_period=4.0,
    n_sub_elements=None,
    elevation_focus=None,
    lens_attenuation_coef=0.0,
):
    """
    Simulates RF data for a given set of scatterers.

    Args:
        scatterer_positions (array-like): The positions of the scatterers [m] of shape (n_scat, 3).
        scatterer_magnitudes (array-like): The magnitudes of the scatterers of shape (n_scat,).
        probe_geometry (array-like): The geometry of the probe [m] of shape (n_el, 3).
        apply_lens_correction (bool): Model the acoustic lens as a layer of ``lens_sound_speed``
            in front of the elements. Every sub-element's path refracts through it (Fermat), so
            the lens delay depends on the direction to the scatterer, and the lens attenuates
            with ``lens_attenuation_coef``. With ``elevation_focus`` the layer is a cylindrical
            lens: ``lens_thickness`` at the element centre, thinned (``lens_sound_speed`` below
            ``sound_speed``) or thickened towards the elevation edges so that the normal-incidence
            delay focuses at ``elevation_focus``. The lens face is taken locally flat under each
            sub-element, for the delay and for the spreading of the refracted wave, and the sinc
            directivity uses the geometric angle to the scatterer.
        lens_thickness (float): The thickness of the lens [m] at the element centre.
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
        elevation_slab_2d (bool): Reduce the elevation dimension to a 2D slab: drop the
            scatterers outside it, and spread the transmit cylindrically rather than
            spherically, as an ideal elevation lens focusing to that slab would. This is a
            cheap approximation, not a modelled lens; for the physical lens in 3D use
            ``elevation_focus``, which is exclusive with it. For efficient pruning of the
            scatterers outside the slab, use :class:`zea.ops.Simulate` rather than calling
            `simulate_rf` directly.
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
        rigid_baffle (bool): Element mounted in a rigid baffle (sinc directivity only). False
            models a soft baffle, which adds the obliquity factor cos(angle to the element
            normal), on transmit and on receive. Must be static under jit.
        bandwidth_percent (float, optional): Pulse-echo -6 dB fractional bandwidth of the
            transducer in percent of ``probe_center_frequency``. Applies the Gaussian transfer
            function of :func:`transducer_transfer` to the received spectrum. None is a flat
            transducer response. Must be static under jit.
        probe_center_frequency (float, optional): Centre of the transducer band [Hz]. Defaults
            to ``center_frequency``. Must be static under jit.
        element_normals (array-like, optional): Outward normal of each element of shape
            (n_el, 3), for curved or tilted arrays. The directivity and the obliquity are
            evaluated in each element's own frame: the elevation axis is the projection of
            +y onto the element plane, so a normal must not be parallel to +y. None is every
            element facing +z. See :func:`zea.probes.curved_probe_normals`. The lens correction
            keeps assuming a flat lens.
        chirp_sweep (float, optional): Linear frequency sweep of the transmit pulse [Hz]. The
            instantaneous frequency runs from ``center_frequency - chirp_sweep / 2`` to
            ``center_frequency + chirp_sweep / 2`` over the Hann-windowed pulse (see
            :func:`chirp_spectrum`). None or 0 is the plain windowed tone. Must be static
            under jit.
        n_period (float): Periods of ``center_frequency`` under the Hann window of the transmit
            pulse. Must be static under jit.
        n_sub_elements (optional): Sub-elements per element, summed coherently with their own
            distance and sinc directivity so the response holds in the near field. A pair
            (n_lateral, n_elevation), an int for the lateral count, or ``"auto"`` for the SIMUS
            rule ceil(size / lambda_min) in both directions, with lambda_min at the top of the
            transducer band. None is a single sub-element, except in elevation when
            ``elevation_focus`` is set, which then follows the auto rule. Must be static under
            jit.
        elevation_focus (float, optional): Focal distance [m] of a fixed elevation lens, modelled
            on transmit and on receive through the elevation sub-elements: an ideal focusing
            advance per sub-element, or with ``apply_lens_correction`` the refracted path through
            the lens thickness profile. Exclusive with ``elevation_slab_2d``, the cheap 2D
            approximation of an elevation lens. Must be static under
            jit.
        lens_attenuation_coef (float): Attenuation in the lens [dB/cm/MHz], applied over each
            sub-element's path inside the lens when ``apply_lens_correction`` is set. Apodizes
            the aperture where the lens is thick and lowers the centre frequency.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).

    """

    _validate_scatter_exponent(scatter_exponent)
    _validate_elevation(elevation_slab_2d, elevation_focus)

    n_tx = t0_delays.shape[0]

    element_width = _resolve_element_width(probe_geometry, element_width)

    if element_height is None:
        element_height = element_width
    _validate_lens(
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        elevation_focus,
        element_height,
    )
    n_sub_elements = _resolve_sub_elements(
        n_sub_elements,
        elevation_focus,
        element_width,
        element_height,
        sound_speed,
        center_frequency,
        bandwidth_percent,
    )

    magnitudes = scatterer_magnitudes
    if elevation_slab_2d:
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
        center_frequency, n_period=n_period, sampling_frequency=sampling_frequency
    )

    # Room for a whole pulse, so record_length below never gates the end of the record away.
    # Traced frequencies give no static pulse length; the record then keeps its old short tail.
    fc_np, fs_np = _concrete(center_frequency), _concrete(sampling_frequency)
    n_pulse = 0 if fc_np is None or fs_np is None else int(np.ceil(n_period / fc_np * fs_np))
    n_ax_rounded = float(_round_up_to_power_of_two(int(n_ax) + n_pulse))

    freqs = ops.arange(n_ax_rounded // 2 + 1, dtype="float32") / n_ax_rounded * sampling_frequency

    if chirp_sweep:
        waveform_spectrum = chirp_spectrum(
            n_ax_rounded, center_frequency, sampling_frequency, n_period, chirp_sweep
        )
    else:
        waveform_spectrum = pulse_spectrum_fn(freqs)
    if bandwidth_percent is not None:
        transfer = transducer_transfer(
            freqs, probe_center_frequency, bandwidth_percent, center_frequency
        )
        waveform_spectrum = waveform_spectrum * ops.cast(transfer, "complex64")

    if scatter_exponent:
        scatter_gain = (freqs / center_frequency) ** scatter_exponent
    else:
        scatter_gain = ops.ones_like(freqs)

    # [n_scat, n_el, n_freq]
    tx_response, rx_response, dist = _element_responses(
        scatterer_positions,
        probe_geometry,
        freqs,
        sound_speed,
        element_width,
        element_height,
        attenuation_coef,
        lens_thickness,
        lens_sound_speed,
        apply_lens_correction,
        elevation_slab_2d,
        rigid_baffle,
        element_normals,
        n_sub_elements,
        elevation_focus,
        lens_attenuation_coef,
    )
    # One-way delays past the FFT length are gated, as delay2 does for the transmit shifts.
    in_fft = ops.cast(dist / sound_speed < n_ax_rounded / sampling_frequency, "complex64")
    tx_response = tx_response * in_fft[..., None]
    rx_response = rx_response * in_fft[..., None]

    # Leave room for the pulse tail
    record_length = n_ax_rounded / sampling_frequency - 0.5 * n_period / center_frequency
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


def _validate_elevation(elevation_slab_2d, elevation_focus):
    if elevation_slab_2d and elevation_focus is not None:
        raise ValueError(
            "elevation_slab_2d is the cheap 2D approximation (slab pruning and cylindrical "
            "spread); elevation_focus models the lens in 3D through the elevation "
            "sub-elements. Pick one."
        )


def _lens_sag(v, elevation_focus, sound_speed, lens_sound_speed):
    """Thickness removed from the lens at elevation offset ``v`` to focus at ``elevation_focus``.

    A slower lens is thickest at the centre, a faster one thinnest (negative sag).
    """
    focus = ops.cast(elevation_focus, v.dtype)
    path = ops.sqrt(focus**2 + v**2) - focus
    return path * lens_sound_speed / (sound_speed - lens_sound_speed)


def _lens_spread_distance(lens_len, medium_len, thickness, sound_speed, lens_sound_speed):
    """Distance whose 1/r spreading is the ray-tube divergence of the path refracted at the face.

    The phase path scales the lens leg by c / c_lens, but the wave leaves the face as if from a
    source lens_len * c_lens / c below it (apparent depth). The refracted wavefront is
    astigmatic: that radius holds across the plane of incidence, and within it the radius is
    scaled by cos^2 of the medium angle over cos^2 of the lens angle.
    """
    ratio = lens_sound_speed / sound_speed
    lens_len = ops.maximum(lens_len, 1e-9)
    cos_lens_sq = ops.clip((thickness / lens_len) ** 2, 1e-6, 1.0)
    cos_medium_sq = ops.maximum(1.0 - (1.0 - cos_lens_sq) / ratio**2, 1e-6)
    r_across = lens_len * ratio
    r_within = r_across * cos_medium_sq / cos_lens_sq
    return lens_len * ops.sqrt(
        (r_across + medium_len) * (r_within + medium_len) / (r_across * r_within)
    )


def _validate_lens(
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    sound_speed,
    elevation_focus,
    element_height,
):
    """Static checks of a focusing lens: distinct speeds, and a face above the elements."""
    if not apply_lens_correction:
        return
    if lens_sound_speed is None:
        raise ValueError("apply_lens_correction=True requires lens_sound_speed.")
    if elevation_focus is None:
        return
    values = [_concrete(x) for x in (lens_thickness, lens_sound_speed, sound_speed, element_height)]
    if any(v is None for v in values):
        return
    thickness, c_lens, c, height = (float(v) for v in values)
    if c_lens == c:
        raise ValueError("A lens at the medium's sound speed cannot focus; set lens_sound_speed.")
    sag = (np.sqrt(float(elevation_focus) ** 2 + (height / 2) ** 2) - float(elevation_focus)) * (
        c_lens / (c - c_lens)
    )
    if thickness - sag <= 0:
        raise ValueError(
            f"lens_thickness {thickness:.2e} m is too thin to focus at {elevation_focus} m: the "
            f"lens needs at least {sag:.2e} m at the centre."
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


def _element_frame(element_normals, dtype="float32"):
    """Lateral, elevation and normal unit vectors of the elements, each (n_el, 3) or (1, 3).

    The elevation axis is +y projected onto the element plane, lateral completes the frame.
    """
    if element_normals is None:
        eye = ops.cast(ops.convert_to_tensor(np.eye(3, dtype=np.float32)), dtype)
        return eye[0:1], eye[1:2], eye[2:3]
    normal = ops.cast(element_normals, dtype)
    normal = normal / ops.linalg.norm(normal, axis=-1, keepdims=True)
    y = ops.cast(ops.convert_to_tensor(np.array([0.0, 1.0, 0.0], np.float32)), dtype)
    elevation_axis = y - normal[:, 1:2] * normal
    elevation_axis = elevation_axis / ops.linalg.norm(elevation_axis, axis=-1, keepdims=True)
    lateral_axis = ops.cross(elevation_axis, normal)
    return lateral_axis, elevation_axis, normal


def _element_angles(relative, frame):
    """Lateral and elevation angles and cos of the angle to the element normal.

    The sines of theta and phi are the direction cosines lateral / r and elevation / r, as in
    the Fraunhofer pattern of a rectangular aperture (and SIMUS). Projected angles
    arctan2(lateral, axial) would narrow the elevation pattern for laterally offset scatterers.

    Args:
        relative (array-like): Scatterer positions relative to the elements, (n_scat, n_el, 3).
        frame (tuple): Element axes from :func:`_element_frame`.

    Returns:
        theta, phi, obliquity: arrays of shape (n_scat, n_el).
    """
    lateral_axis, elevation_axis, normal = frame
    lateral = ops.sum(relative * lateral_axis[None], axis=-1)
    elevation = ops.sum(relative * elevation_axis[None], axis=-1)
    axial = ops.sum(relative * normal[None], axis=-1)
    dist = ops.maximum(ops.linalg.norm(relative, axis=-1), 1e-12)
    theta = ops.arcsin(ops.clip(lateral / dist, -1.0, 1.0))
    phi = ops.arcsin(ops.clip(elevation / dist, -1.0, 1.0))
    obliquity = axial / dist
    return theta, phi, obliquity


def _resolve_sub_elements(
    n_sub_elements,
    elevation_focus,
    element_width,
    element_height,
    sound_speed,
    center_frequency,
    bandwidth_percent,
):
    """Sub-elements per element as (n_lateral, n_elevation).

    "auto" is the SIMUS rule ceil(size / lambda_min), lambda_min at the top of the transducer
    band. None and an int keep one elevation sub-element unless there is an elevation focus,
    which needs the elevation subdivision to act at all.
    """
    if isinstance(n_sub_elements, (tuple, list)):
        n_lateral, n_elevation = (int(n) for n in n_sub_elements)
        return max(n_lateral, 1), max(n_elevation, 1)
    focused = elevation_focus is not None
    if n_sub_elements != "auto" and not focused:
        return (1 if n_sub_elements is None else max(int(n_sub_elements), 1)), 1
    values = [_concrete(x) for x in (sound_speed, center_frequency, element_width, element_height)]
    if any(v is None for v in values):
        raise ValueError(
            "The sub-element count cannot be derived from a traced sound speed, frequency or "
            "element size; pass n_sub_elements=(n_lateral, n_elevation) explicitly."
        )
    c, fc, width, height = (float(v) for v in values)
    lambda_min = c / (fc * (1 + (bandwidth_percent or 0.0) / 200))
    n_elevation = max(int(np.ceil(height / lambda_min)), 1)
    if n_sub_elements == "auto":
        return max(int(np.ceil(width / lambda_min)), 1), n_elevation
    return (1 if n_sub_elements is None else max(int(n_sub_elements), 1)), n_elevation


def _sub_element_offsets(n_lateral, n_elevation, element_width, element_height):
    """Centroid offsets (u, v) of the sub-elements in the element frame, each (n_sub,)."""
    u = (ops.arange(n_lateral, dtype="float32") - (n_lateral - 1) / 2) * (
        ops.cast(element_width, "float32") / n_lateral
    )
    v = (ops.arange(n_elevation, dtype="float32") - (n_elevation - 1) / 2) * (
        ops.cast(element_height, "float32") / n_elevation
    )
    return ops.reshape(ops.tile(u[:, None], (1, n_elevation)), (-1,)), ops.tile(v, (n_lateral,))


def _element_responses(
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
    elevation_slab_2d,
    rigid_baffle,
    element_normals,
    n_sub_elements=(1, 1),
    elevation_focus=None,
    lens_attenuation_coef=0.0,
    frequency_first=False,
):
    """Transmit and receive one-way responses and the one-way path length [s, e].

    The responses are [s, e, f], or [f, s, e] when ``frequency_first`` is True. Each element is
    the mean of ``n_sub_elements`` (lateral, elevation) sub-elements with their
    own distance, phase and sinc directivity, so the response holds in the near field too. An
    elevation focus is the ideal focusing advance of each elevation sub-element, or with the lens
    the refracted (Fermat) path through the local lens thickness, which the focus thins towards
    the edges. The lens path is expressed as the medium distance with the same travel time for the
    phase, and spreads as the refracted ray tube (:func:`_lens_spread_distance`); the lens leg is
    attenuated with ``lens_attenuation_coef``. The returned path length is the element centre's.
    """
    n_lateral, n_elevation = n_sub_elements
    n_sub = n_lateral * n_elevation
    relative_center = positions[:, None] - geometry[None]
    dtype = relative_center.dtype
    lateral_axis, elevation_axis, _ = frame = _element_frame(element_normals, dtype)
    dist_center = ops.linalg.norm(relative_center, axis=-1)
    if apply_lens_correction:
        dist = (
            compute_lens_corrected_travel_times(
                geometry,
                positions,
                lens_thickness=lens_thickness,
                c_lens=lens_sound_speed,
                c_medium=sound_speed,
                n_iter=3,
            )
            * sound_speed
        )
    else:
        dist = dist_center
    u, v = _sub_element_offsets(n_lateral, n_elevation, element_width, element_height)
    u, v = ops.cast(u, dtype), ops.cast(v, dtype)
    if elevation_focus is None or apply_lens_correction:
        advance = ops.zeros_like(v)
    else:
        focus = ops.cast(elevation_focus, dtype)
        advance = (ops.sqrt(focus**2 + v**2) - focus) / sound_speed
    if apply_lens_correction and elevation_focus is not None:
        thickness = lens_thickness - _lens_sag(v, elevation_focus, sound_speed, lens_sound_speed)
    else:
        thickness = ops.full_like(v, lens_thickness)
    sub_width = element_width / n_lateral
    sub_height = element_height / n_elevation
    f3 = freqs[:, None, None] if frequency_first else freqs[None, None, :]

    def fx(x):
        """Puts the frequency axis of a [s, e] array where ``frequency_first`` wants it."""
        return x[None] if frequency_first else x[..., None]

    def response(j):
        offset = u[j] * lateral_axis + v[j] * elevation_axis
        relative = relative_center - offset[None]
        theta, phi, obliquity = _element_angles(relative, frame)
        amplitude = directivity(f3, fx(theta), sub_width, sound_speed) * directivity(
            f3, fx(phi), sub_height, sound_speed
        )
        if apply_lens_correction:
            lens_len, medium_len = compute_lens_path_lengths(
                geometry + offset,
                positions,
                lens_thickness=thickness[j],
                c_lens=lens_sound_speed,
                c_medium=sound_speed,
                n_iter=3,
            )
            sub_dist = lens_len * (sound_speed / lens_sound_speed) + medium_len
            spread_dist = _lens_spread_distance(
                lens_len, medium_len, thickness[j], sound_speed, lens_sound_speed
            )
            amplitude = amplitude * attenuate(f3, lens_attenuation_coef, fx(lens_len))
        else:
            medium_len = sub_dist = spread_dist = ops.linalg.norm(relative, axis=-1)
        amplitude = amplitude * attenuate(f3, attenuation_coef, fx(medium_len))
        if not rigid_baffle:
            amplitude = amplitude * fx(obliquity)
        phase = ops.exp(
            ops.array(-2j * np.pi, "complex64")
            * ops.cast((fx(sub_dist) / sound_speed - advance[j]) * f3, "complex64")
        )
        rx = ops.cast(amplitude * spread(fx(spread_dist), 1.0), "complex64") * phase
        if elevation_slab_2d:
            # An elevation lens focuses the transmit to a slab: cylindrical spread on the way out.
            tx = ops.cast(amplitude * spread(fx(spread_dist), 0.5), "complex64") * phase
        else:
            tx = rx
        return tx, rx

    if n_sub == 1:
        tx, rx = response(0)
        return tx, rx, dist

    def body(j, carry):
        tx, rx = response(j)
        return carry[0] + tx, carry[1] + rx

    zeros = ops.zeros(ops.shape(response(0)[0]), "complex64")
    tx, rx = ops.fori_loop(0, n_sub, body, (zeros, zeros))
    scale = ops.array(1.0 / n_sub, "complex64")
    return tx * scale, rx * scale, dist


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
        raise ValueError("elevation_slab_2d=True requires element_height to be provided.")
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
    elevation_slab_2d=False,
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
    if not elevation_slab_2d or element_height is None:
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
            "elevation_slab_2d=True models a 1D probe with a simplified cylindrical elevation lens,"
            " but the probe is not 1D "
            f"(element elevation min, max: {elevation.min()}, {elevation.max()}) "
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


def chirp_spectrum(n_fft, center_frequency, sampling_frequency, n_period, chirp_sweep, xp=ops):
    """Spectrum of a Hann-windowed linear chirp centred at t=0, on the rfft grid of ``n_fft``.

    The window spans ``n_period`` periods of ``center_frequency``, over which the instantaneous
    frequency sweeps linearly from ``center_frequency - chirp_sweep / 2`` to
    ``center_frequency + chirp_sweep / 2``. Scaled like :func:`get_pulse_spectrum_fn`: the
    waveform recovered with ``irfft`` has a unit peak. The waveform is even, so the spectrum is
    real, and with ``chirp_sweep=0`` it is the sampled counterpart of the windowed tone.

    Args:
        n_fft (int): FFT length; the waveform is sampled on its wrapped time grid.
        center_frequency (float): Centre frequency [Hz].
        sampling_frequency (float): Sampling frequency [Hz].
        n_period (float): Periods of ``center_frequency`` under the Hann window.
        chirp_sweep (float): Total frequency sweep [Hz].
        xp: Array module, ``keras.ops`` or ``numpy``.

    Returns:
        array-like: Complex spectrum of shape (n_fft // 2 + 1,).
    """
    n_fft = int(n_fft)
    k = xp.arange(n_fft, dtype="float32")
    t = xp.where(k < n_fft // 2, k, k - n_fft) / sampling_frequency
    width = n_period / center_frequency
    window = xp.where(xp.abs(t) < width / 2, xp.cos(np.pi * t / width) ** 2, 0.0)
    phase = 2 * np.pi * (center_frequency * t + chirp_sweep / (2 * width) * t**2)
    waveform = window * xp.cos(phase)
    if xp is np:
        return np.fft.rfft(waveform).astype(np.complex64)
    real, imag = ops.rfft(waveform)
    return ops.cast(real, "complex64") + ops.array(1j, "complex64") * ops.cast(imag, "complex64")


def transducer_transfer(
    f, probe_center_frequency, bandwidth_percent, center_frequency=None, xp=ops
):
    """Gaussian pulse-echo transfer function of the transducer.

    Unit gain at ``probe_center_frequency`` and -6 dB at the edges of the fractional bandwidth,
    ``probe_center_frequency * (1 +/- bandwidth_percent / 200)``.

    Args:
        f (array-like): Frequencies [Hz].
        probe_center_frequency (float, optional): Centre of the band [Hz]. ``center_frequency``
            when None.
        bandwidth_percent (float): -6 dB fractional bandwidth in percent.
        center_frequency (float, optional): Fallback band centre [Hz].
        xp: Array module, ``keras.ops`` or ``numpy``.

    Returns:
        array-like: The transfer function at ``f``.
    """
    if probe_center_frequency is None:
        probe_center_frequency = center_frequency
    half_width = 0.5 * bandwidth_percent / 100 * probe_center_frequency
    return xp.exp(-np.log(2) * ((xp.abs(f) - probe_center_frequency) / half_width) ** 2)


def _round_up_to_power_of_two(x):
    """Rounds up to the next power of two."""
    return 2 ** np.ceil(np.log2(x))


def _concrete(x):
    """numpy view of ``x``, or None when it is traced."""
    if x is None:
        return None
    try:
        return ops.convert_to_numpy(x)
    except (RuntimeError, ValueError, TypeError, NotImplementedError):
        return None


# ---------------------------------------------------------------------------------------------
# zeaWave: the same physics as ``simulate_rf`` with the transmit-invariant response shared
# across transmits through two matrix products per frequency block.
# ---------------------------------------------------------------------------------------------


def smooth_size(n):
    """Smallest 2^a 3^b 5^c >= n."""
    best = int(_round_up_to_power_of_two(max(n, 1)))
    a = 0
    while 2**a < 2 * n:
        b = 0
        while 2**a * 3**b < 2 * n:
            c = 0
            while 2**a * 3**b * 5**c < n:
                c += 1
            best = min(best, 2**a * 3**b * 5**c)
            b += 1
        a += 1
    return best


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


def _rfft_freqs(n_fft, sampling_frequency):
    """The rfft frequency grid of ``n_fft`` samples, in numpy."""
    return np.arange(n_fft // 2 + 1) / n_fft * sampling_frequency


def _zea_wave_spectrum_np(
    n_fft,
    center_frequency,
    sampling_frequency,
    n_period,
    bandwidth_percent,
    probe_center_frequency,
    chirp_sweep=None,
    one_way=False,
):
    """Transmit pulse times transducer transfer function on the full rfft grid, in numpy.

    ``one_way`` takes the square root of the pulse-echo transfer function, for a transmit field.
    """
    freqs = _rfft_freqs(n_fft, sampling_frequency)
    if chirp_sweep:
        wave = chirp_spectrum(
            n_fft, center_frequency, sampling_frequency, n_period, chirp_sweep, xp=np
        )
    else:
        wave = pulse_spectrum_np(freqs, center_frequency, sampling_frequency, n_period)
    if bandwidth_percent is not None:
        transfer = transducer_transfer(
            freqs, probe_center_frequency, bandwidth_percent, center_frequency, xp=np
        )
        if one_way:
            transfer = np.sqrt(transfer)
        wave = (wave * transfer).astype(np.complex64)
    return wave


def band_bins(
    n_fft,
    center_frequency,
    sampling_frequency,
    n_period,
    scatter_exponent,
    band_db,
    bandwidth_percent=None,
    probe_center_frequency=None,
    chirp_sweep=None,
    one_way=False,
):
    """Contiguous bin range that carries the band.

    The pulse spectrum, the transducer transfer function and the scattering gain together
    exceed ``band_db`` there.
    """
    freqs = _rfft_freqs(n_fft, sampling_frequency)
    if band_db is None:
        return 0, len(freqs)
    w = np.abs(
        _zea_wave_spectrum_np(
            n_fft,
            center_frequency,
            sampling_frequency,
            n_period,
            bandwidth_percent,
            probe_center_frequency,
            chirp_sweep,
            one_way,
        )
    )
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


def _to_complex(x):
    return ops.cast(x, "complex64")


def _zea_wave_block(
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
    elevation_slab_2d,
    rigid_baffle,
    element_normals,
    n_sub_elements,
    elevation_focus,
    lens_attenuation_coef,
):
    """Band spectrum [f, t, e] of one frequency block over all scatterers.

    Frequency leads every array so the einsums are plain batched matrix products. Scatterers
    whose earliest echo has no support before ``gate_time`` cannot reach the output and are
    dropped, so a long path never wraps into the record.
    """
    tx_response, rx_response, dist = _element_responses(
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
        elevation_slab_2d,
        rigid_baffle,
        element_normals,
        n_sub_elements,
        elevation_focus,
        lens_attenuation_coef,
        frequency_first=True,
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


def simulate_rf_zea_wave(
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
    elevation_slab_2d=False,
    element_height=None,
    max_chunk_gb=1.0,
    noise_level_db=None,
    tgc_max_db=0.0,
    noise_seed=0,
    noise_reference=None,
    scatter_exponent=2.0,
    rigid_baffle=True,
    bandwidth_percent=None,
    probe_center_frequency=None,
    element_normals=None,
    chirp_sweep=None,
    n_period=4.0,
    n_sub_elements=None,
    elevation_focus=None,
    band_db=-100.0,
    n_fft=None,
    lens_attenuation_coef=0.0,
):
    """:func:`simulate_rf` with the scatterer response shared across transmits.

    The transmit-independent one-way response is generated once per frequency block and reused
    over all transmits.

    .. code-block:: text

        incident[f, t, s] = sum_e W[f, t, e] R_tx[f, s, e]    W = apod_te exp(-2 pi i f shift_te)
        rf[f, t, e]       = sum_s S[f, t, s] R_rx[f, s, e]    S = incident * mag_s * gain(f)

    :func:`simulate_rf` evaluates the same sums transmit by transmit, re-reading the (large)
    response every time. This version is up to 20x faster for large batches of transmits, and
    only slower for very few transmits (it falls back to :func:`simulate_rf` with a single
    transmit, where there is nothing to share).

    Only the bins where the pulse spectrum, the transducer transfer function and the scattering
    gain together exceed ``band_db`` are computed (``band_db=None`` keeps every bin, -100 matches
    SIMUS defaults), and the record gate is applied per scatterer rather than per (transmit,
    scatterer, element), with the FFT length sized so that no kept echo wraps into the record
    (``n_fft`` pins it). A scatterer is kept when its earliest echo still has pulse support
    inside the record, so the gate is exact up to the pulse envelope; echoes that fall past the
    record are not masked but simply truncated. With ``band_db=None`` and the same ``n_fft`` the
    result matches :func:`simulate_rf` to float32 precision, except for the last pulse length of
    the record, where :func:`simulate_rf` retains a sub-1e-3 leakage tail from scatterers whose
    support lies wholly outside it.

    Takes the arguments of :func:`simulate_rf` with the same meaning, plus:

    Args:
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
            elevation_slab_2d=elevation_slab_2d,
            element_height=element_height,
            max_chunk_gb=max_chunk_gb,
            noise_level_db=noise_level_db,
            tgc_max_db=tgc_max_db,
            noise_seed=noise_seed,
            noise_reference=noise_reference,
            scatter_exponent=scatter_exponent,
            rigid_baffle=rigid_baffle,
            bandwidth_percent=bandwidth_percent,
            probe_center_frequency=probe_center_frequency,
            element_normals=element_normals,
            chirp_sweep=chirp_sweep,
            n_period=n_period,
            n_sub_elements=n_sub_elements,
            elevation_focus=elevation_focus,
            lens_attenuation_coef=lens_attenuation_coef,
        )

    _validate_scatter_exponent(scatter_exponent)
    _validate_elevation(elevation_slab_2d, elevation_focus)
    fc, fs = float(center_frequency), float(sampling_frequency)
    n_ax = int(n_ax)
    element_width = _resolve_element_width(probe_geometry, element_width)
    if element_height is None:
        element_height = element_width
    _validate_lens(
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        elevation_focus,
        element_height,
    )
    n_sub_elements = _resolve_sub_elements(
        n_sub_elements,
        elevation_focus,
        element_width,
        element_height,
        sound_speed,
        fc,
        bandwidth_percent,
    )

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

    if elevation_slab_2d:
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
    if bandwidth_percent is not None:
        bandwidth_percent = float(bandwidth_percent)
    if probe_center_frequency is not None:
        probe_center_frequency = float(probe_center_frequency)
    chirp_sweep = float(chirp_sweep) if chirp_sweep else None
    k0, k1 = band_bins(
        n_fft,
        fc,
        fs,
        n_period,
        scatter_exponent,
        band_db,
        bandwidth_percent,
        probe_center_frequency,
        chirp_sweep,
    )

    # Forward of one block per bin: the complex responses and the two matrix product outputs.
    n_scat = int(ops.shape(positions)[0])
    per_bin = 8 * ((2 if elevation_slab_2d else 1) * n_scat * n_el + n_tx * n_scat + n_tx * n_el)
    f_block = int(max(1, min(k1 - k0, max_chunk_gb * 2**30 // per_bin)))
    n_blocks = -(-(k1 - k0) // f_block)
    # Spread the band evenly, so the last block is padded by less than a whole block.
    f_block = -(-(k1 - k0) // n_blocks)
    n_band = n_blocks * f_block

    # Band padded to whole blocks; the pad repeats the last bin and is dropped after the loop.
    freqs_all = _rfft_freqs(n_fft, fs)
    freqs = np.full(n_band, freqs_all[k1 - 1], np.float32)
    freqs[: k1 - k0] = freqs_all[k0:k1]
    wave = _zea_wave_spectrum_np(
        n_fft, fc, fs, n_period, bandwidth_percent, probe_center_frequency, chirp_sweep
    )[k0:k1]
    freqs = ops.convert_to_tensor(freqs)

    def as_f32(x):
        return ops.cast(0.0 if x is None else x, "float32")

    block = checkpoint(
        functools.partial(
            _zea_wave_block,
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
            elevation_slab_2d=bool(elevation_slab_2d),
            rigid_baffle=bool(rigid_baffle),
            element_normals=None if element_normals is None else as_f32(element_normals),
            n_sub_elements=n_sub_elements,
            elevation_focus=None if elevation_focus is None else float(elevation_focus),
            lens_attenuation_coef=as_f32(lens_attenuation_coef),
        )
    )

    def body(i, spectrum):
        start = i * f_block
        block_freqs = ops.slice(freqs, [start], [f_block])
        return ops.slice_update(spectrum, [start, 0, 0], block(block_freqs))

    spectrum = ops.zeros((n_band, n_tx, n_el), "complex64")
    spectrum = ops.fori_loop(0, n_blocks, body, spectrum)
    spectrum = spectrum[: k1 - k0] * ops.convert_to_tensor(wave)[:, None, None]

    # Transmits in groups, so a long record over many transmits does not allocate at once.
    group = min(32, n_tx)
    parts = []
    for start in range(0, n_tx, group):
        band = ops.transpose(spectrum[:, start : start + group], (1, 2, 0))
        pad = ((0, 0), (0, 0), (k0, n_fft // 2 + 1 - k1))
        full = (ops.pad(ops.real(band), pad), ops.pad(ops.imag(band), pad))
        parts.append(ops.irfft(full, fft_length=n_fft)[..., :n_ax])
    rf = ops.transpose(ops.concatenate(parts, axis=0), (0, 2, 1))
    return finish(rf)


def _pressure_block(
    freqs,
    positions,
    geometry,
    shift,
    tx_apodizations,
    sound_speed,
    element_width,
    element_height,
    attenuation_coef,
    lens_thickness,
    lens_sound_speed,
    apply_lens_correction,
    elevation_slab_2d,
    rigid_baffle,
    element_normals,
    n_sub_elements,
    elevation_focus,
    lens_attenuation_coef,
):
    """Incident field spectrum [f, t, p] of one frequency block, without the pulse."""
    tx_response, _, _ = _element_responses(
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
        elevation_slab_2d,
        rigid_baffle,
        element_normals,
        n_sub_elements,
        elevation_focus,
        lens_attenuation_coef,
        frequency_first=True,
    )
    f3 = freqs[:, None, None]
    tx_weights = _to_complex(tx_apodizations[None]) * ops.exp(
        ops.array(-2j * np.pi, "complex64") * _to_complex(shift[None] * f3)
    )
    with highest_matmul_precision():
        return ops.einsum("fte,fpe->ftp", tx_weights, tx_response)


def pressure_field(
    grid,
    probe_geometry,
    sound_speed,
    center_frequency,
    sampling_frequency,
    t0_delays,
    initial_times,
    element_width,
    tx_apodizations,
    t_peak,
    attenuation_coef=0.0,
    apply_lens_correction=False,
    lens_thickness=0.0,
    lens_sound_speed=None,
    elevation_slab_2d=False,
    element_height=None,
    rigid_baffle=True,
    bandwidth_percent=None,
    probe_center_frequency=None,
    element_normals=None,
    chirp_sweep=None,
    n_period=4.0,
    n_sub_elements=None,
    elevation_focus=None,
    band_db=-100.0,
    n_ax=None,
    n_fft=None,
    output="rms",
    max_chunk_gb=1.0,
    lens_attenuation_coef=0.0,
):
    """Transmit pressure field of :func:`simulate_rf` and :func:`simulate_rf_zea_wave` on a grid.

    The incident field the simulators scatter, evaluated at the grid points instead of at
    scatterers: the same directivity, obliquity, attenuation, spread and transmit weights, times
    the transmit pulse. The transducer transfer function enters once (its square root, as the
    ``bandwidth_percent`` band is pulse-echo), so a unit scatterer at a grid point returns this
    field through the receive response. Behind an elevation lens the field is zero outside the
    elevation slab, as the simulators drop those scatterers. The zea counterpart of SIMUS
    ``pfield``.

    Takes the arguments of :func:`simulate_rf` with the same meaning, except that
    ``attenuation_coef``, ``apply_lens_correction`` and the lens have defaults, plus:

    Args:
        grid (array-like): Points where the field is evaluated [m], of shape (..., 3).
        band_db (float, optional): Bins where the pulse spectrum is below this many dB of its
            peak are not synthesised. None keeps every bin.
        n_ax (int, optional): Samples of the time record at ``sampling_frequency``. Defaults to
            the extent of the field over the grid, so nothing is cut off. ``"rms"`` divides the
            energy of the whole field by it, so a shorter record rescales rather than truncates.
        n_fft (int, optional): FFT length. Derived when None from the grid, the transmit shifts
            and ``n_ax`` so that the field never wraps. Must be given when the grid, the
            geometry, the delays or the sound speed are traced.
        output (str): ``"rms"`` for the root mean square pressure over the ``n_ax`` samples of
            the record, evaluated in the frequency domain, or ``"time"`` for the pressure
            waveforms.
        max_chunk_gb (float): Memory budget for one block of work.

    Returns:
        array-like: The pressure field, of shape (n_tx, ...) for ``"rms"`` and
        (n_tx, n_ax, ...) for ``"time"``, in the units of the transmit pulse (unit peak per
        element at :func:`spread`'s reference distance of 1 mm).
    """
    if output not in ("rms", "time"):
        raise ValueError(f"output must be 'rms' or 'time', got {output!r}.")
    _validate_elevation(elevation_slab_2d, elevation_focus)
    fc, fs = float(center_frequency), float(sampling_frequency)
    n_period = float(n_period)
    grid_shape = tuple(int(d) for d in ops.shape(grid)[:-1])
    positions = ops.reshape(ops.cast(grid, "float32"), (-1, 3))
    geometry = ops.cast(probe_geometry, "float32")
    element_width = _resolve_element_width(geometry, element_width)
    if element_height is None:
        element_height = element_width
    _validate_lens(
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        elevation_focus,
        element_height,
    )
    n_sub_elements = _resolve_sub_elements(
        n_sub_elements,
        elevation_focus,
        element_width,
        element_height,
        sound_speed,
        fc,
        bandwidth_percent,
    )
    t0_delays = ops.cast(t0_delays, "float32")
    n_tx, n_el = (int(d) for d in ops.shape(t0_delays))
    n_points = int(ops.shape(positions)[0])
    shift = (
        t0_delays
        - ops.cast(initial_times, "float32")[:, None]
        + ops.cast(t_peak, "float32")[:, None]
    )

    if n_fft is None or n_ax is None:
        raw = [_concrete(x) for x in (positions, geometry, shift, sound_speed)]
        if any(x is None for x in raw):
            raise ValueError(
                "n_fft and n_ax cannot be derived from a traced grid, geometry, delays or "
                "sound speed; pass them explicitly."
            )
        pos_np, geom_np, shift_np, c_np = raw
        dist_np = np.linalg.norm(
            pos_np[:, None].astype(np.float64) - geom_np[None].astype(np.float64), axis=-1
        )
        arrival = (dist_np / float(c_np))[None] + shift_np[:, None, :]
        extent = int(np.ceil((arrival.max() + n_period / fc) * fs))
        if n_ax is None:
            n_ax = extent
        if n_fft is None:
            n_fft = smooth_size(max(int(n_ax), extent))
    n_ax, n_fft = int(n_ax), int(n_fft)

    if bandwidth_percent is not None:
        bandwidth_percent = float(bandwidth_percent)
    if probe_center_frequency is not None:
        probe_center_frequency = float(probe_center_frequency)
    chirp_sweep = float(chirp_sweep) if chirp_sweep else None
    k0, k1 = band_bins(
        n_fft,
        fc,
        fs,
        n_period,
        0.0,
        band_db,
        bandwidth_percent,
        probe_center_frequency,
        chirp_sweep,
        one_way=True,
    )
    n_kept = k1 - k0
    freqs_all = _rfft_freqs(n_fft, fs)
    wave_all = _zea_wave_spectrum_np(
        n_fft, fc, fs, n_period, bandwidth_percent, probe_center_frequency, chirp_sweep, True
    )

    def as_f32(x):
        return ops.cast(0.0 if x is None else x, "float32")

    block = checkpoint(
        functools.partial(
            _pressure_block,
            geometry=geometry,
            shift=shift,
            tx_apodizations=ops.cast(tx_apodizations, "float32"),
            sound_speed=as_f32(sound_speed),
            element_width=as_f32(element_width),
            element_height=as_f32(element_height),
            attenuation_coef=as_f32(attenuation_coef),
            lens_thickness=as_f32(lens_thickness),
            lens_sound_speed=as_f32(lens_sound_speed),
            apply_lens_correction=bool(apply_lens_correction),
            elevation_slab_2d=bool(elevation_slab_2d),
            rigid_baffle=bool(rigid_baffle),
            element_normals=None if element_normals is None else as_f32(element_normals),
            n_sub_elements=n_sub_elements,
            elevation_focus=None if elevation_focus is None else float(elevation_focus),
            lens_attenuation_coef=as_f32(lens_attenuation_coef),
        )
    )

    def blocked(points, budget):
        """Band spectrum [f, t, p] or its Parseval energy [t, p] over ``points``."""
        n_pts = int(ops.shape(points)[0])
        per_bin = 8 * ((2 if elevation_slab_2d else 1) * n_pts * n_el + n_tx * n_pts)
        f_block = int(max(1, min(n_kept, budget // per_bin)))
        n_blocks = -(-n_kept // f_block)
        f_block = -(-n_kept // n_blocks)
        n_band = n_blocks * f_block
        freqs = np.full(n_band, freqs_all[k1 - 1], np.float32)
        freqs[:n_kept] = freqs_all[k0:k1]
        wave = np.zeros(n_band, np.complex64)
        wave[:n_kept] = wave_all[k0:k1]
        freqs_t = ops.convert_to_tensor(freqs)
        wave_t = ops.convert_to_tensor(wave)

        if output == "time":

            def body(i, spectrum):
                start = i * f_block
                part = block(ops.slice(freqs_t, [start], [f_block]), points)
                return ops.slice_update(spectrum, [start, 0, 0], part)

            spectrum = ops.fori_loop(
                0, n_blocks, body, ops.zeros((n_band, n_tx, n_pts), "complex64")
            )
            return spectrum[:n_kept] * wave_t[:n_kept, None, None]

        # sum_n p[n]^2 = (1/N) sum_k w_k |P_k|^2 with w = 2 except at DC and, for an even
        # transform, at Nyquist.
        parseval = np.full(n_band, 2.0, np.float32)
        if k0 == 0:
            parseval[0] = 1.0
        if k1 == n_fft // 2 + 1 and n_fft % 2 == 0:
            parseval[n_kept - 1] = 1.0
        weight = ops.convert_to_tensor(parseval * np.abs(wave) ** 2)

        def body(i, energy):
            start = i * f_block
            part = block(ops.slice(freqs_t, [start], [f_block]), points)
            w = ops.slice(weight, [start], [f_block])[:, None, None]
            return energy + ops.sum(w * (ops.real(part) ** 2 + ops.imag(part) ** 2), axis=0)

        return ops.fori_loop(0, n_blocks, body, ops.zeros((n_tx, n_pts), "float32"))

    budget = max_chunk_gb * 2**30
    if output == "rms":
        energy = blocked(positions, budget)
        field = ops.sqrt(energy / (n_fft * n_ax))
    else:
        # The band spectrum of a chunk takes half the budget, the blocks the other half.
        chunk = int(max(1, min(n_points, budget // 2 // (8 * n_kept * n_tx))))
        parts = []
        for start in range(0, n_points, chunk):
            spectrum = blocked(positions[start : start + chunk], budget // 2)
            band = ops.transpose(spectrum, (1, 2, 0))
            pad = ((0, 0), (0, 0), (k0, n_fft // 2 + 1 - k1))
            full = (ops.pad(ops.real(band), pad), ops.pad(ops.imag(band), pad))
            parts.append(ops.irfft(full, fft_length=n_fft)[..., :n_ax])
        field = ops.transpose(ops.concatenate(parts, axis=1), (0, 2, 1))

    if elevation_slab_2d:
        _warn_if_elevation_extent(geometry)
        mask = elevation_slab_mask(positions, geometry, element_height)
        field = field * mask
    lead = (n_tx,) if output == "rms" else (n_tx, n_ax)
    return ops.reshape(field, lead + grid_shape)
