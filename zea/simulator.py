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

from collections.abc import Callable
from dataclasses import dataclass, replace

import keras
import numpy as np
from keras import ops
from scipy.signal import hilbert
from scipy.special import fresnel

from zea import log
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
    baffle_impedance_ratio=0.0,
    element_normals=None,
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
    n_sub_elements=None,
    elevation_focus=None,
    lens_attenuation_coef=0.0,
):
    """
    Simulates RF data for a given set of scatterers.

    The two-way (pulse-echo) transmit pulse is ``waveforms_two_way``: the waveform of a zea file
    (the Verasonics ``TW.Wvfm2Wy``), a measured one, or one built with :func:`transmit_pulse`,
    which has the parametric models. Without it the default pulse of :func:`transmit_pulse` is
    used: a one-cycle burst at ``center_frequency`` through a 70 % Butterworth transducer.

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
            The pulse is simulated with its envelope peak at the two-way travel time plus
            ``t_peak``; a real system's is :attr:`Pulse.time_to_peak` of :func:`transmit_pulse`.
        elevation_slab_2d (bool): Reduce the elevation dimension to a 2D slab: drop the
            scatterers outside it, and spread the transmit cylindrically rather than
            spherically, as an ideal elevation lens focusing to that slab would. This is a
            cheap approximation, not a modelled lens; for the physical lens in 3D use
            ``elevation_focus``, which is exclusive with it. For efficient pruning of the
            scatterers outside the slab, use :class:`zea.ops.Simulate` rather than calling
            `simulate_rf` directly.
        element_height (float): The elevation height of the elements [m], used for the
            elevation directivity and the elevation slab. If None, an eighth of the width of a
            1D probe (at least ``element_width``), or ``element_width`` for a 2D probe.
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
            The Verasonics simulator applies no frequency dependence at all: 0 here, with
            ``attenuation_coef=0`` (its attenuation is evaluated at the centre frequency only)
            and ``baffle_impedance_ratio=inf`` (its default element sensitivity is cos times
            sinc), reproduces its spectrum. It also applies no geometric spreading, which zea
            always does.
        baffle_impedance_ratio (float): Impedance of the medium over that of the baffle the
            elements are mounted in, which sets the obliquity factor applied on transmit and on
            receive next to the sinc directivity: 1 for a rigid baffle (0, the default),
            cos(angle to the element normal) for a soft one (``inf``), and in general
            cos / (cos + ratio) (Selfridge et al. 1980, as in SIMUS; 0.57 for epoxy against
            soft tissue). Scatterers behind the element plane get no obliquity factor. Must be
            static under jit.
        element_normals (array-like, optional): Outward normal of each element of shape
            (n_el, 3), for curved or tilted arrays. The directivity and the obliquity are
            evaluated in each element's own frame: the elevation axis is the projection of
            +y onto the element plane, so a normal must not be parallel to +y. None is every
            element facing +z. See :func:`zea.probes.curved_probe_normals`. With
            ``apply_lens_correction`` the lens is conformal: its face is normal to each element.
        waveforms_two_way (array-like, optional): Two-way (pulse-echo) transmit waveforms of
            shape (n_tx, n_samples), or (n_samples,) for one waveform for every transmit,
            sampled at ``waveform_sampling_frequency``. The envelope peak of the waveform is
            placed at the two-way travel time plus ``t_peak``, wherever it is in the waveform;
            see :func:`measured_pulse`. None is the default pulse of :func:`transmit_pulse`.
            Must be static under jit.
        waveform_sampling_frequency (float): Sampling frequency [Hz] of ``waveforms_two_way``,
            250 MHz in zea files and in :meth:`Pulse.waveform`. Must be static under jit.
        n_sub_elements (optional): Sub-elements per element, summed coherently with their own
            distance and sinc directivity so the response holds in the near field. A pair
            (n_lateral, n_elevation), an int for the lateral count, or ``"auto"`` for the SIMUS
            rule ceil(size / lambda_min) in both directions, with lambda_min at the top of the
            -6 dB band of the transmit pulse (SIMUS takes the transducer band, which is a
            little wider than that of a one-cycle burst through it). None is a single
            sub-element, except in elevation when
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
    _validate_baffle(baffle_impedance_ratio)
    _validate_elevation(elevation_slab_2d, elevation_focus)

    n_tx = t0_delays.shape[0]

    element_width = _resolve_element_width(probe_geometry, element_width)
    element_height = _resolve_element_height(probe_geometry, element_width, element_height)
    _validate_lens(
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        elevation_focus,
        element_height,
    )
    pulses = transmit_pulses(
        n_tx, center_frequency, sampling_frequency, waveforms_two_way, waveform_sampling_frequency
    )
    n_sub_elements = _resolve_sub_elements(
        n_sub_elements,
        elevation_focus,
        element_width,
        element_height,
        sound_speed,
        max(pulse.band[1] for pulse in pulses),
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

    # Room for a whole pulse, so record_length below never gates the end of the record away.
    n_pulse = max(pulse.n_samples for pulse in pulses)
    n_ax_rounded = float(_round_up_to_power_of_two(int(n_ax) + n_pulse))
    freqs_np = np.fft.rfftfreq(int(n_ax_rounded), 1 / pulses[0].sampling_frequency)
    freqs = ops.convert_to_tensor(freqs_np.astype(np.float32))
    waveform_spectra = {}
    for pulse in pulses:
        if pulse not in waveform_spectra:
            waveform_spectra[pulse] = ops.convert_to_tensor(pulse.spectrum(freqs_np))

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
        baffle_impedance_ratio,
        element_normals,
        n_sub_elements,
        elevation_focus,
        lens_attenuation_coef,
        min_distance(sound_speed, center_frequency),
    )
    # One-way delays past the FFT length are gated, as delay2 does for the transmit shifts.
    in_fft = ops.cast(dist / sound_speed < n_ax_rounded / sampling_frequency, "complex64")
    tx_response = tx_response * in_fft[..., None]
    rx_response = rx_response * in_fft[..., None]

    # Leave room for the pulse tail
    n_after = max(pulse.n_after for pulse in pulses)
    record_length = (n_ax_rounded - n_after) / pulses[0].sampling_frequency
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
        rf_spectrum = waveform_spectra[pulses[tx]] * ops.sum(received_field, axis=0)
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


def _validate_baffle(baffle_impedance_ratio):
    if not baffle_impedance_ratio >= 0:
        raise ValueError(
            f"baffle_impedance_ratio ({baffle_impedance_ratio}) must be non-negative: 0 for a "
            "rigid baffle, inf for a soft one."
        )


def obliquity_factor(cos_angle, baffle_impedance_ratio):
    """Obliquity factor of an element in a baffle of finite impedance, at the cosine of the
    angle to its normal: 1 in a rigid baffle (ratio 0), the cosine in a soft one (``inf``), and
    cos / (cos + ratio) in between, with the ratio the medium impedance over the baffle's
    (Pesque et al., IEEE Ultrasonics Symposium 1984, as in SIMUS; the soft-baffle cosine is
    Selfridge et al. 1980). With a non-rigid baffle, directions behind the element get 0, not
    a pole.
    """
    if baffle_impedance_ratio == 0:
        return ops.ones_like(cos_angle)
    cos_angle = ops.maximum(cos_angle, 0.0)
    if baffle_impedance_ratio == float("inf"):
        return cos_angle
    return cos_angle / (cos_angle + baffle_impedance_ratio)


def min_distance(sound_speed, center_frequency):
    """Half a wavelength: the distance the simulators clamp the element-scatterer distance to
    for the phase and the spreading, as SIMUS does, so that the 1 / r of a scatterer on an
    element stays finite. The angles keep the true geometry."""
    return sound_speed / (2.0 * center_frequency)


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


def _resolve_element_height(probe_geometry, element_width, element_height, tol=1e-6):
    """Return the element height, inferring it when not given: an eighth of the width of a 1D
    probe (elements without elevation extent; n_el times the pitch), at least the element
    width, and the element width for a 2D probe or a single element. Works on a traced
    geometry, as a traced height; a Python float otherwise."""
    if element_height is not None:
        return element_height
    n_el = int(probe_geometry.shape[0])
    if n_el < 2:
        return element_width
    geometry = ops.cast(probe_geometry, "float32")
    x, y = geometry[:, 0], geometry[:, 1]
    probe_width = (ops.max(x) - ops.min(x)) * (n_el / (n_el - 1))
    one_dimensional = ops.max(y) - ops.min(y) <= tol
    height = ops.where(one_dimensional, ops.maximum(probe_width / 8, element_width), element_width)
    concrete = _concrete(height)
    return height if concrete is None else float(concrete)


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
    max_frequency,
):
    """Sub-elements per element as (n_lateral, n_elevation).

    "auto" is the SIMUS rule ceil(size / lambda_min), lambda_min at ``max_frequency`` [Hz], the
    top of the band. None and an int keep one elevation sub-element unless there is an elevation
    focus, which needs the elevation subdivision to act at all.
    """
    if isinstance(n_sub_elements, (tuple, list)):
        n_lateral, n_elevation = (int(n) for n in n_sub_elements)
        return max(n_lateral, 1), max(n_elevation, 1)
    focused = elevation_focus is not None
    if n_sub_elements != "auto" and not focused:
        return (1 if n_sub_elements is None else max(int(n_sub_elements), 1)), 1
    values = [_concrete(x) for x in (sound_speed, element_width, element_height)]
    if any(v is None for v in values):
        raise ValueError(
            "The sub-element count cannot be derived from a traced sound speed or element "
            "size; pass n_sub_elements=(n_lateral, n_elevation) explicitly."
        )
    c, width, height = (float(v) for v in values)
    lambda_min = c / float(max_frequency)
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
    baffle_impedance_ratio,
    element_normals,
    n_sub_elements=(1, 1),
    elevation_focus=None,
    lens_attenuation_coef=0.0,
    min_dist=0.0,
):
    """Transmit and receive one-way responses [scatterer, element, frequency_bin] and the
    one-way path length [scatterer, element].

    Each element is the mean of ``n_sub_elements`` (lateral, elevation) sub-elements with their
    own distance, phase and sinc directivity, so the response holds in the near field too. The
    sub-element distance is clamped at ``min_dist`` for the phase and the spreading (see
    :func:`min_distance`), not for the angles. An
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
    lateral_axis, elevation_axis, normal = frame = _element_frame(element_normals, dtype)
    # A conformal lens on a curved probe: its face is normal to each element.
    lens_normals = None if element_normals is None else normal
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
                element_normals=lens_normals,
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
    f3 = freqs[None, None, :]

    def response(j):
        offset = u[j] * lateral_axis + v[j] * elevation_axis
        relative = relative_center - offset[None]
        theta, phi, obliquity = _element_angles(relative, frame)
        amplitude = directivity(f3, theta[..., None], sub_width, sound_speed) * directivity(
            f3, phi[..., None], sub_height, sound_speed
        )
        if apply_lens_correction:
            lens_len, medium_len = compute_lens_path_lengths(
                geometry + offset,
                positions,
                lens_thickness=thickness[j],
                c_lens=lens_sound_speed,
                c_medium=sound_speed,
                n_iter=3,
                element_normals=lens_normals,
            )
            sub_dist = lens_len * (sound_speed / lens_sound_speed) + medium_len
            spread_dist = _lens_spread_distance(
                lens_len, medium_len, thickness[j], sound_speed, lens_sound_speed
            )
            amplitude = amplitude * attenuate(f3, lens_attenuation_coef, lens_len[..., None])
        else:
            medium_len = sub_dist = spread_dist = ops.linalg.norm(relative, axis=-1)
        sub_dist = ops.maximum(sub_dist, min_dist)
        spread_dist = ops.maximum(spread_dist, min_dist)
        amplitude = amplitude * attenuate(f3, attenuation_coef, medium_len[..., None])
        amplitude = amplitude * obliquity_factor(obliquity, baffle_impedance_ratio)[..., None]
        phase = ops.exp(
            ops.array(-2j * np.pi, "complex64")
            * ops.cast((sub_dist[..., None] / sound_speed - advance[j]) * f3, "complex64")
        )
        rx = ops.cast(amplitude * spread(spread_dist[..., None], 1.0, min_dist), "complex64")
        rx = rx * phase
        if elevation_slab_2d:
            # An elevation lens focuses the transmit to a slab: cylindrical spread on the way out.
            tx = ops.cast(amplitude * spread(spread_dist[..., None], 0.5, min_dist), "complex64")
            tx = tx * phase
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


def spread(dist, exponent=1.0, mindist=1e-3, reference=1e-3):
    """Geometric spreading of the wavefront.

    Args:
        dist (array-like): The distance the wave has traveled.
        exponent (float): 1 for spherical, 0.5 for cylindrical. An elevation lens focuses the
            transmitted energy to a slab, resulting in a cylindrical transmit and a spherical
            receive path.
        mindist (float): Distances below it are clamped to it. The simulators pass half a
            wavelength, :func:`min_distance`.
        reference (float): Distance of unit gain.

    Returns:
        array-like: An amplitude factor in the shape of `dist`.
    """
    dist = ops.maximum(dist, mindist)
    return (reference / dist) ** exponent


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


PULSE_MODELS = ("realistic", "hann", "simus")


@dataclass(frozen=True)
class Pulse:
    """Two-way transmit pulse of the simulators, with its envelope peak at ``t = 0``.

    Built by :func:`transmit_pulse` or :func:`measured_pulse`. ``spectrum_fn`` is the
    continuous-time spectrum, scaled so
    that ``irfft`` of its samples on an rfft grid of ``sampling_frequency`` recovers the waveform
    with a unit peak. The support is where the envelope is above -80 dB.
    """

    spectrum_fn: Callable
    """Complex spectrum (numpy) at frequencies [Hz]."""
    sampling_frequency: float
    """Sampling frequency [Hz] of :meth:`waveform`."""
    n_before: int
    """Support before the peak, in samples."""
    n_after: int
    """Support after the peak, in samples."""
    time_to_peak: float
    """Time [s] from the transmit trigger (the start of the pulse, or of a measured waveform)
    to the envelope peak: the ``t_peak`` of a real system."""
    band: tuple
    """The -6 dB band [Hz] of the pulse, (low, high)."""

    @property
    def n_samples(self):
        """Length of :meth:`waveform`: odd, with the peak on the middle sample."""
        return 2 * max(self.n_before, self.n_after) + 1

    def spectrum(self, freqs):
        """The spectrum at ``freqs`` [Hz], as complex64."""
        return self.spectrum_fn(np.asarray(freqs, np.float64)).astype(np.complex64)

    def waveform(self, from_trigger=False):
        """The pulse sampled at ``sampling_frequency``, as ``waveforms_two_way`` of the
        simulators.

        By default of length :attr:`n_samples`, odd, with the envelope peak on the middle sample.
        With ``from_trigger`` the waveform starts at the transmit trigger instead, like a
        Verasonics waveform: sample 0 is ``time_to_peak`` before the peak, so the ``t_peak``
        that :class:`zea.Parameters` derives from it is :attr:`time_to_peak`.
        """
        if from_trigger:
            n_before = int(round(self.time_to_peak * self.sampling_frequency))
            n_after = self.n_after
        else:
            n_before = n_after = self.n_samples // 2
        n_fft = int(_round_up_to_power_of_two(2 * (n_before + n_after + 1)))
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sampling_frequency)
        waveform = np.fft.irfft(self.spectrum_fn(freqs), n_fft)
        return np.roll(waveform, n_before)[: n_before + n_after + 1].astype(np.float32)


def transmit_pulse(
    center_frequency,
    sampling_frequency=250e6,
    pulse_model="realistic",
    n_period=1.0,
    chirp_sweep=None,
    bandwidth_percent=70.0,
    probe_center_frequency=None,
    excitation_duty_cycle=0.67,
    transducer_order=2,
    equalize=False,
):
    """The parametric two-way (pulse-echo) transmit pulse: excitation times transducer response.

    Its :meth:`Pulse.waveform` is the ``waveforms_two_way`` of :func:`simulate_rf`,
    :func:`zea.simulator_time_domain.simulate_rf_td` and :class:`zea.ops.Simulate`, which use
    the default pulse of this function when none is given::

        pulse = transmit_pulse(5e6, pulse_model="simus", bandwidth_percent=75.0)
        rf = simulate_rf(..., waveforms_two_way=pulse.waveform())

    The radiation factor of a baffled piston (jω) is absorbed into the transducer response, as
    in MUST, so that ``bandwidth_percent`` is the -6 dB pulse-echo bandwidth for every model.
    For a measured pulse see :func:`measured_pulse`.

    The ``"realistic"`` model is the model of the Verasonics Vantage two-way waveform
    (``TW.Wvfm2Wy``): the tri-state burst through the same 2nd-order Butterworth band-pass on
    transmit and on receive, -6 dB two-way at the edges of ``Trans.Bandwidth``, with no
    radiation factor and no normalisation. It reproduces that waveform with
    ``probe_center_frequency`` at ``Trans.frequency``, ``bandwidth_percent`` at
    ``100 * (fHigh - fLow) / Trans.frequency`` (about 77 % for the L11-5v) and ``equalize``
    set, since a Vantage burst has equalisation pulses by default.

    Args:
        center_frequency (float): Centre frequency of the excitation [Hz].
        sampling_frequency (float): Sampling frequency [Hz] of :meth:`Pulse.waveform`; 250 MHz
            like the ``waveforms_two_way`` of a zea file, or the sampling frequency of the RF
            data for the pulse the simulators use internally.
        pulse_model (str): ``"realistic"``: a pulser's tri-state square burst of ``n_period``
            periods (:func:`square_burst_spectrum`) through a causal Butterworth band-pass on
            transmit and on receive (:func:`butterworth_transfer`), so the pulse rises fast and
            rings down like a real system's. ``"hann"``: Hann-windowed cosine or chirp
            (:func:`hann_burst_spectrum`) times a zero-phase Gaussian pulse-echo response
            (:func:`gaussian_transfer`). ``"simus"``: MUST's rectangular-windowed sine or chirp
            (:func:`rect_burst_spectrum`) times its generalized-normal response
            (:func:`generalized_normal_transfer`), for a one-to-one comparison with SIMUS.
        n_period (float): Periods of ``center_frequency`` in the excitation. The default of one
            period (MUST's ``TXnow``) leaves the transducer as the limiting factor, so the pulse
            bandwidth follows ``bandwidth_percent``; a burst of n periods is itself only about
            120 % / n wide and narrows the pulse below the requested bandwidth for n >= 2.
        chirp_sweep (float, optional): Linear frequency sweep [Hz] of the ``"hann"`` and
            ``"simus"`` excitations, from ``center_frequency - chirp_sweep / 2`` to
            ``center_frequency + chirp_sweep / 2``; negative sweeps down (MUST's direction).
        bandwidth_percent (float, optional): Pulse-echo -6 dB fractional bandwidth of the
            transducer in percent of ``probe_center_frequency``. None is a flat response, which
            only ``"hann"`` allows.
        probe_center_frequency (float, optional): Centre of the transducer band [Hz]. Defaults
            to ``center_frequency``.
        excitation_duty_cycle (float): Width of each half-cycle of the ``"realistic"`` burst as
            a fraction of the half period. 0.67 nulls the third harmonic.
        transducer_order (int): Order of the ``"realistic"`` Butterworth band-pass, per way.
        equalize (bool): Add the Verasonics equalisation pulses to the ``"realistic"`` burst
            (:func:`square_burst_pulses`). They lengthen the burst by about a period, so the
            excitation and not the transducer limits the pulse bandwidth: a 2-half-cycle burst
            through a 77 % transducer gives a 56 % pulse instead of 68 %. Off by default, so
            that ``bandwidth_percent`` sets the bandwidth of the pulse.

    Returns:
        Pulse: the pulse, with the envelope peak at t = 0 and a unit peak.
    """
    if pulse_model not in PULSE_MODELS:
        raise ValueError(f"pulse_model ({pulse_model}) must be one of {PULSE_MODELS}.")
    fc = _static_float(center_frequency, "center_frequency")
    fs = _static_float(sampling_frequency, "sampling_frequency")
    fc_probe = fc if probe_center_frequency is None else float(probe_center_frequency)
    sweep = 0.0 if chirp_sweep is None else float(chirp_sweep)
    duration = n_period / fc
    if bandwidth_percent is None and pulse_model != "hann":
        raise ValueError(f"pulse_model='{pulse_model}' needs bandwidth_percent.")
    # The windowed excitations are centred at t = 0; the trigger is where their window starts.
    trigger = -0.5 * n_period / fc
    if pulse_model == "hann":

        def spectrum(f):
            return hann_burst_spectrum(f, fc, n_period, sweep) * gaussian_transfer(
                f, fc_probe, bandwidth_percent
            )

    elif pulse_model == "simus":

        def spectrum(f):
            return rect_burst_spectrum(f, fc, n_period, sweep) * generalized_normal_transfer(
                f, fc_probe, bandwidth_percent
            )

    else:
        duration += (64.0 + equalize) / fc  # ringdown, and the equalisation pulses
        centres, widths, _ = square_burst_pulses(fc, n_period, excitation_duty_cycle, equalize)
        # The trigger is the first edge of the burst, where a Verasonics waveform starts.
        trigger = centres[0] - 0.5 * widths[0]

        def spectrum(f):
            excitation = square_burst_spectrum(f, fc, n_period, excitation_duty_cycle, equalize)
            one_way = butterworth_transfer(f, fc_probe, bandwidth_percent, transducer_order)
            return excitation * one_way**2

    pulse = _calibrate(spectrum, fs, duration, pulse_model == "realistic", trigger)
    if pulse_model != "realistic":
        # A zero-phase transducer responds before the excitation window, the more so the further
        # the excitation is detuned from the band; the trigger has to hold that response.
        return replace(pulse, time_to_peak=max(pulse.time_to_peak, pulse.n_before / fs))
    return pulse


def measured_pulse(waveform_two_way, sampling_frequency, waveform_sampling_frequency=250e6):
    """A sampled two-way (pulse-echo) waveform as the transmit pulse: the ``waveforms_two_way``
    of the simulators, measured, the system's own or built with :func:`transmit_pulse`.

    The waveform already includes the transducer response. It is evaluated on the simulation
    grid by its discrete-time Fourier transform, so any sampling frequency will do, and shifted
    so that its envelope peak is at t = 0. The shift is :attr:`Pulse.time_to_peak`, the time from
    sample 0 to the peak; for a waveform that starts at the transmit trigger, as the
    ``waveforms_two_way`` of a zea file (the Verasonics two-way waveform, sampled at 250 MHz),
    that is the ``t_peak`` of :func:`simulate_rf` which puts the waveform back at the travel
    time (:attr:`zea.Parameters.t_peak` derives the same).

    The Verasonics waveform (``TW.Wvfm2Wy``) is not measured but modelled, by the ``"realistic"``
    model of :func:`transmit_pulse` with its equalisation pulses: the burst through a 2nd-order
    Butterworth band-pass twice, -6 dB two-way at ``Trans.Bandwidth``, without a radiation
    factor or normalisation. Its sample 0 is the trigger and the Vantage simulator places that
    sample at the two-way travel time, so the envelope peak of the waveform is its ``t_peak``.
    The Vantage simulator applies no frequency-dependent scattering to it; see
    ``scatter_exponent`` of :func:`simulate_rf`.

    Args:
        waveform_two_way (array-like): The waveform of shape (n_samples,).
        sampling_frequency (float): Sampling frequency of the RF data [Hz].
        waveform_sampling_frequency (float): Sampling frequency of the waveform [Hz].

    Returns:
        Pulse: the pulse, with the envelope peak at t = 0 and a unit peak.
    """
    samples = np.asarray(waveform_two_way, np.float64)
    if samples.ndim != 1 or len(samples) < 2 or not np.any(samples):
        raise ValueError("waveform_two_way must be a non-zero waveform of shape (n_samples,).")
    fs = _static_float(sampling_frequency, "sampling_frequency")
    times = np.arange(len(samples)) / float(waveform_sampling_frequency)

    def spectrum(f):
        return sampled_spectrum(f, samples, times)

    return _calibrate(spectrum, fs, times[-1], shift_peak=True)


def transmit_pulses(
    n_tx,
    center_frequency,
    sampling_frequency,
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
):
    """The two-way transmit pulse of every transmit, as a list of ``n_tx`` :class:`Pulse`.

    The rows of ``waveforms_two_way`` when given, of shape (n_tx, n_samples), or (n_samples,)
    for the same waveform on every transmit (see :func:`measured_pulse`; identical rows share
    one :class:`Pulse`); otherwise the default pulse of :func:`transmit_pulse` at
    ``center_frequency``, on every transmit.
    """
    if waveforms_two_way is None:
        return [transmit_pulse(center_frequency, sampling_frequency)] * n_tx
    waveforms = np.atleast_2d(np.asarray(waveforms_two_way, np.float64))
    if waveforms.ndim != 2 or waveforms.shape[0] not in (1, n_tx):
        raise ValueError(
            f"waveforms_two_way must have shape (n_tx, n_samples) or (n_samples,), got "
            f"{np.shape(waveforms_two_way)} for {n_tx} transmits."
        )
    pulses = {}
    for waveform in waveforms:
        key = waveform.tobytes()
        if key not in pulses:
            pulses[key] = measured_pulse(waveform, sampling_frequency, waveform_sampling_frequency)
    return [pulses[waveform.tobytes()] for waveform in waveforms] * (n_tx // len(waveforms))


def _calibrate(
    spectrum_fn,
    sampling_frequency,
    duration,
    shift_peak,
    trigger=0.0,
    threshold_db=-80.0,
    oversample=16,
):
    """Shift the envelope peak to t = 0, scale to a unit peak and measure the support, on a grid
    ``oversample`` times finer than ``sampling_frequency`` so that none of it depends on it.
    ``trigger`` is the time [s] of the transmit trigger in the frame of ``spectrum_fn``."""
    fs = oversample * sampling_frequency
    n = int(_round_up_to_power_of_two(max(4 * duration * fs, 256)))
    while True:
        freqs = np.fft.rfftfreq(n, 1 / fs)
        waveform = np.fft.irfft(spectrum_fn(freqs), n)
        envelope = np.abs(hilbert(waveform))
        shift = 0.0
        if shift_peak:
            k = int(np.argmax(envelope))
            before, at, after = envelope[k - 1], envelope[k], envelope[(k + 1) % n]
            fraction = 0.5 * (before - after) / (before - 2 * at + after)  # sub-sample peak
            shift = ((k if k < n // 2 else k - n) + fraction) / fs
            waveform = np.fft.irfft(spectrum_fn(freqs) * np.exp(2j * np.pi * freqs * shift), n)
            envelope = np.abs(hilbert(waveform))
        support = np.flatnonzero(envelope > 10 ** (threshold_db / 20) * envelope.max())
        positive, negative = support[support < n // 2], support[support >= n // 2]
        after = int(positive.max()) if len(positive) else 0
        before = int(n - negative.min()) if len(negative) else 0
        if before + after < n // 4 or n >= 2**22:
            break
        n *= 2  # the pulse wraps around the grid
    scale = 1.0 / (oversample * np.abs(waveform).max())  # unit peak on the coarse grid

    def shifted(f):
        return scale * spectrum_fn(f) * np.exp(2j * np.pi * f * shift)

    magnitude = np.abs(spectrum_fn(freqs))
    in_band = np.flatnonzero(magnitude >= 0.5 * magnitude.max())
    band = (float(freqs[in_band[0]]), float(freqs[in_band[-1]]))
    n_before, n_after = (int(np.ceil(k / oversample)) for k in (before, after))
    return Pulse(shifted, sampling_frequency, n_before, n_after, shift - trigger, band)


def _static_float(x, name):
    value = _concrete(x)
    if value is None:
        raise ValueError(
            f"{name} must be static (not traced): the transmit pulse is built in numpy."
        )
    return float(value)


def _validate_bandwidth(bandwidth_percent, geometric=False):
    if not np.isfinite(bandwidth_percent) or bandwidth_percent <= 0:
        raise ValueError(f"bandwidth_percent must be positive, got {bandwidth_percent}.")
    if geometric and bandwidth_percent >= 200:
        raise ValueError(f"bandwidth_percent must be below 200, got {bandwidth_percent}.")


def sampled_spectrum(f, samples, times):
    """Continuous-time spectrum of a uniformly sampled waveform at frequencies ``f`` [Hz]: its
    discrete-time Fourier transform times the sampling interval, by Horner's scheme so that no
    (n_freqs, n_samples) table is formed. Zero beyond the Nyquist frequency of the samples."""
    f = np.asarray(f, np.float64)
    dt = times[1] - times[0]
    z = np.exp(-2j * np.pi * f * dt)
    spectrum = np.zeros(f.shape, np.complex128)
    for sample in samples[::-1]:
        spectrum = spectrum * z + sample
    return spectrum * dt * np.exp(-2j * np.pi * f * times[0]) * (np.abs(f) <= 0.5 / dt)


def hann_burst_spectrum(f, fc, n_period, sweep=0.0):
    """Spectrum of a Hann-windowed tone or linear chirp centred at t = 0.

    The window spans ``n_period`` periods of ``fc``, over which the instantaneous frequency
    runs linearly from ``fc - sweep / 2`` to ``fc + sweep / 2``. Evaluated from samples at 64
    per period, so aliasing is far below the 1/f**3 tails of the window.
    """
    width = n_period / fc
    times = np.linspace(-width / 2, width / 2, int(np.ceil(64 * n_period)) + 1)
    window = np.cos(np.pi * times / width) ** 2
    phase = 2 * np.pi * (fc * times + sweep / (2 * width) * times**2)
    return sampled_spectrum(f, window * np.cos(phase), times)


def rect_burst_spectrum(f, fc, n_period, sweep=0.0):
    """Spectrum of MUST's excitation: a rectangular-windowed sine of ``n_period`` periods
    (``TXnow``) centred at t = 0, or its linear chirp over ``|sweep|`` Hz (``TXfreqsweep``),
    as ``getPulseSpectrumFunction`` computes it. MUST inverts the conjugate spectrum and its
    chirp sweeps down, so a positive ``sweep`` is the time-reversed pulse, sweeping up.
    """
    T = n_period / fc
    f = np.asarray(f, np.float64)
    if not sweep:
        spectrum = 1j * (np.sinc(T * (f - fc)) - np.sinc(T * (f + fc)))
    else:
        w, wc, dw = 2 * np.pi * f, 2 * np.pi * fc, 2 * np.pi * abs(sweep)

        def fresnel_integral(x):
            s, c = fresnel(x)
            return c + 1j * s

        def half(w):
            scale = np.sqrt(T / (np.pi * dw))
            return (
                np.sqrt(np.pi * T / dw)
                * np.exp(-1j * (w - wc) ** 2 * T / (2 * dw))
                * (
                    fresnel_integral((dw / 2 + w - wc) * scale)
                    + fresnel_integral((dw / 2 - w + wc) * scale)
                )
            )

        spectrum = (1j * half(w) - 1j * half(-w)) / T
    return np.conj(spectrum) if sweep <= 0 else spectrum


def square_burst_spectrum(f, fc, n_period, duty_cycle=0.67, equalize=False):
    """Spectrum of a pulser's tri-state burst centred at t = 0: ``2 * n_period`` alternating
    rectangular half-cycles, each ``duty_cycle`` of the half period wide. The duty cycle sets the
    harmonics; 0.67 nulls the third. ``equalize`` adds the Verasonics equalisation pulses, see
    :func:`square_burst_pulses`."""
    f = np.asarray(f, np.float64)
    centres, widths, signs = square_burst_pulses(fc, n_period, duty_cycle, equalize)
    pulses = signs * widths * np.sinc(f[:, None] * widths)
    return (pulses * np.exp(-2j * np.pi * f[:, None] * centres)).sum(axis=1)


def square_burst_pulses(fc, n_period, duty_cycle=0.67, equalize=False):
    """The rectangular pulses of :func:`square_burst_spectrum`: their centres [s], widths [s]
    and signs, in order. The half-cycles are centred on the half periods around t = 0, starting
    positive. With ``equalize`` a pulse of half the width and opposite sign precedes the first
    and follows the last half-cycle, a quarter period away from its edge: the equalisation pulses
    a Verasonics system adds by default (``TW.equalize``), which make the drive zero-mean and
    the burst about a period longer."""
    half_period = 0.5 / fc
    n_half = max(1, int(round(2 * n_period)))
    width = duty_cycle * half_period
    centres = (np.arange(n_half) - (n_half - 1) / 2) * half_period
    signs = (-1.0) ** np.arange(n_half)
    widths = np.full(n_half, width)
    if equalize:
        # A quarter period between the edges, so the centres are that plus half of both widths.
        offset = 0.5 * half_period + 0.75 * width
        centres = np.concatenate([[centres[0] - offset], centres, [centres[-1] + offset]])
        signs = np.concatenate([[-signs[0]], signs, [-signs[-1]]])
        widths = np.concatenate([[0.5 * width], widths, [0.5 * width]])
    return centres, widths, signs


def gaussian_transfer(f, fc, bandwidth_percent):
    """Gaussian pulse-echo response of the transducer: unit at ``fc`` and -6 dB at the edges of
    the fractional bandwidth, ``fc * (1 +/- bandwidth_percent / 200)``. None is flat."""
    f = np.asarray(f, np.float64)
    if bandwidth_percent is None:
        return np.ones_like(f)
    _validate_bandwidth(bandwidth_percent)
    half_width = 0.5 * bandwidth_percent / 100 * fc
    return np.exp(-np.log(2) * ((np.abs(f) - fc) / half_width) ** 2)


def generalized_normal_transfer(f, fc, bandwidth_percent):
    """MUST's pulse-echo response: a generalized normal window, unit at ``fc`` and -6 dB at the
    edges of the fractional bandwidth, with exponent ``ln(126) / ln(2 fc / fB)``
    (``getProbeFunction``, squared for the round trip)."""
    _validate_bandwidth(bandwidth_percent, geometric=True)
    f = np.asarray(f, np.float64)
    f_band = bandwidth_percent * fc / 100
    p = np.log(126) / np.log(2 * fc / f_band)
    return np.exp(-((np.abs(np.abs(f) - fc) / (f_band / 2 / np.log(2) ** (1 / p))) ** p))


def butterworth_transfer(f, fc, bandwidth_percent, order=2):
    """One-way Butterworth band-pass response of the transducer: causal and minimum phase,
    unit at the geometric centre of the band and -3 dB at the edges of the fractional bandwidth,
    ``fc * (1 +/- bandwidth_percent / 200)``, so that transmit and receive together are -6 dB
    there. Evaluated as the low-pass prototype at ``j (w**2 - w0**2) / (B w)``."""
    _validate_bandwidth(bandwidth_percent, geometric=True)
    w = 2 * np.pi * np.asarray(f, np.float64)
    low, high = fc * (1 - bandwidth_percent / 200), fc * (1 + bandwidth_percent / 200)
    w0_squared, band = (2 * np.pi) ** 2 * low * high, 2 * np.pi * (high - low)
    w_safe = np.where(w == 0, 1.0, w)
    q = np.asarray(1j * (w**2 - w0_squared) / (band * w_safe))
    poles = np.exp(1j * np.pi * (2 * np.arange(1, order + 1) + order - 1) / (2 * order))
    return np.where(w == 0, 0.0, np.prod(1 / (q[..., None] - poles), axis=-1))


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
