"""Frequency-domain RF simulator: the superposition of the scatterer responses on the rfft grid
of the record, one transmit at a time."""

import keras
import numpy as np
from keras import ops

from zea.simulator.element import (
    _apply_elevation_slab,
    _element_responses,
    _resolve_element_height,
    _resolve_element_width,
    _resolve_sub_elements,
    _validate_baffle,
    _validate_elevation,
    _validate_lens,
    _validate_scatter_exponent,
    _warn_if_elevation_extent,
    min_distance,
)
from zea.simulator.pulse import _round_up_to_power_of_two, transmit_pulses
from zea.simulator.record import delay2


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
    *,
    elevation_slab_2d=False,
    element_height=None,
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
    The RF is noiseless; electronic noise and time gain compensation are
    :func:`apply_receive_chain`, which :class:`zea.ops.Simulate` applies.

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
            cos / (cos + ratio). Scatterers behind the element plane get no obliquity factor. Must
            be static under jit.
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
        return ops.zeros(shape, dtype="float32")

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
    return rf_data[:, :n_ax, :, :]


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
