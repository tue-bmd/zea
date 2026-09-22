"""Frequency-domain RF simulator: the superposition of the scatterer responses on the rfft grid
of the record, one transmit at a time."""

from keras import ops

from zea.simulator.element import (
    _validate_scatter_exponent,
    element_model,
    element_responses,
    prepare_scatterers,
)
from zea.simulator.pulse import transmit_pulses
from zea.simulator.record import delay2, in_fft, in_record, record_grid


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
    :func:`zea.func.apply_receive_chain`, which :class:`zea.ops.Simulate` applies.

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
            evaluated in each element's own frame: the height axis is the projection of
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
            (n_width, n_height), an int for the count along the width, or ``"auto"`` for the
            SIMUS rule ceil(size / lambda_min) in both directions, with lambda_min at the top of
            the -6 dB band of the transmit pulse (SIMUS takes the transducer band, which is a
            little wider than that of a one-cycle burst through it). None is a single
            sub-element, except along the height when ``elevation_focus`` is set, which then
            follows the auto rule. Must be static under jit.
        elevation_focus (float, optional): Focal distance [m] of a fixed elevation lens, modelled
            on transmit and on receive through the sub-elements along the height: an ideal
            focusing advance per sub-element, or with ``apply_lens_correction`` the refracted
            path through the lens thickness profile. Exclusive with ``elevation_slab_2d``, the
            cheap 2D approximation of an elevation lens. Must be static under jit.
        lens_attenuation_coef (float): Attenuation in the lens [dB/cm/MHz], applied over each
            sub-element's path inside the lens when ``apply_lens_correction`` is set. Apodizes
            the aperture where the lens is thick and lowers the centre frequency.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).

    """
    _validate_scatter_exponent(scatter_exponent)
    n_tx = t0_delays.shape[0]
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
        elevation_slab_2d=elevation_slab_2d,
        baffle_impedance_ratio=baffle_impedance_ratio,
        element_normals=element_normals,
        n_sub_elements=n_sub_elements,
        elevation_focus=elevation_focus,
        lens_attenuation_coef=lens_attenuation_coef,
    )
    positions, magnitudes = prepare_scatterers(scatterer_positions, scatterer_magnitudes, model)
    if positions.shape[0] == 0:
        # tensorflow can't reduce over an empty axis.
        return ops.zeros((n_tx, int(n_ax), model.geometry.shape[0], 1), dtype="float32")

    record = record_grid(n_ax, sampling_frequency, pulses)
    freqs = ops.convert_to_tensor(record.freqs)
    spectra = {pulse: record.spectrum(pulse) for pulse in pulses}
    scatter_gain = _scatter_gain(freqs, center_frequency, scatter_exponent)

    # [n_scat, n_el, n_freq]
    tx_response, rx_response, travel_time = element_responses(positions, model, freqs)
    fits = in_fft(record, travel_time)
    tx_response = tx_response * fits[..., None]
    rx_response = rx_response * fits[..., None]

    parts = []
    for tx in range(n_tx):
        shift = _transmit_shift(t0_delays[tx], initial_times[tx], t_peak[tx])
        weights = _transmit_weights(record, freqs, shift, tx_apodizations[tx])
        arrival = _transmit_arrival(travel_time, shift, tx_apodizations[tx])
        fits = in_record(record, arrival[:, None] + travel_time)

        # Explicitly sum over tx dimension before the receive axis exists.
        incident_field = ops.sum(tx_response * weights[None], axis=1)
        scattered_field = incident_field * ops.cast(magnitudes[:, None] * scatter_gain, "complex64")
        received_field = scattered_field[:, None] * rx_response * fits[..., None]
        rf_spectrum = spectra[pulses[tx]] * ops.sum(received_field, axis=0)
        parts.append(ops.irfft((ops.real(rf_spectrum), ops.imag(rf_spectrum))))

    rf_data = ops.transpose(ops.stack(parts, axis=0), (0, 2, 1))[..., None]
    return rf_data[:, : record.n_ax]


def _scatter_gain(freqs, center_frequency, scatter_exponent):
    """The frequency dependence of the scattering, ``(f / fc) ** scatter_exponent``."""
    if scatter_exponent:
        return (freqs / center_frequency) ** scatter_exponent
    return ops.ones_like(freqs)


def _transmit_shift(t0_delays, initial_time, t_peak):
    """The delay [s] of every element's pulse in one transmit, (n_el, 1), apart from its travel."""
    return t0_delays[:, None] - initial_time + t_peak


def _transmit_weights(record, freqs, shift, tx_apodization):
    """Apodization and delay of every element in one transmit, (n_el, n_freq) complex."""
    delay = delay2(freqs[None], shift, record.n_fft, record.sampling_frequency)
    return ops.cast(tx_apodization[:, None], "complex64") * delay


def _transmit_arrival(travel_time, shift, tx_apodization):
    """The latest one-way arrival [s] of one transmit at every scatterer, (n_scat,), over its
    active elements: :func:`delay2` only gates one-way delays, so this gates the round trip."""
    delays = ops.where(tx_apodization[None] != 0, travel_time + shift[None, :, 0], -float("inf"))
    return ops.max(delays, axis=1)
