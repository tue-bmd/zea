"""Frequency-domain RF simulator: the superposition of the scatterer responses on the rfft grid
of the record, with the one-way responses shared by every transmit."""

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
from keras import ops

from zea import log
from zea.backend import checkpoint, highest_matmul_precision
from zea.internal.core import concrete, ndim
from zea.simulator.element import (
    _ray_means,
    _scene_positions,
    _validate_maps,
    element_model,
    element_responses,
)
from zea.simulator.pulse import _pulse_spectra, _pulse_tail, transmit_pulses
from zea.simulator.record import (
    _fft_bound,
    _record_gate_time,
    _record_keep,
    _resolve_scatter_exponent,
    _shift_np,
    _sound_speed_minmax,
    _transmit_shift,
    _validate_scatter_exponent,
    band_bins,
    smooth_size,
)


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
    two_dimensional=False,
    element_height=None,
    max_chunk_gb=4.0,
    scatter_exponent=2.0,
    baffle_impedance_ratio=0.0,
    element_normals=None,
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
    n_sub_elements=None,
    elevation_focus=None,
    lens_attenuation_coef=0.0,
    simplified_directivity=False,
    band_db=-100.0,
    n_fft=None,
    scatter_exponent_range=None,
    sos_map=None,
    map_grid_x=None,
    map_grid_z=None,
    map_grid_y=None,
    n_sos_ray_samples=64,
    attenuation_map=None,
    attenuation_power=1.0,
):
    """Simulates RF data for a given set of scatterers.

    The RF is synthesised in the frequency domain, on the rfft grid of ``n_fft`` samples, as the
    superposition of the scatterer echoes:

    .. code-block:: text

        incident[f, t, s] = sum_e W[f, t, e] R_tx[f, s, e]    W = apod_te exp(-2 pi i f shift_te)
        rf[f, t, e]       = sum_s S[f, t, s] R_rx[f, s, e]    S = incident * mag_s * gain_s(f)

    The one-way responses ``R_tx`` and ``R_rx`` (directivity, obliquity, spreading, attenuation
    and the travel phase) do not depend on the transmit, so they are generated once per
    frequency block and shared across all transmits through the two matrix products. The
    two-way (pulse-echo) transmit pulse ``waveforms_two_way`` multiplies the spectrum of each
    transmit: the waveform of a zea file (the Verasonics ``TW.Wvfm2Wy``), a measured one, or
    one built with :func:`transmit_pulse`, which has the parametric models. Without it the
    default pulse of :func:`transmit_pulse` is used: a one-cycle burst at ``center_frequency``
    through a 70 % Butterworth transducer. Only the bins where the pulse spectrum and the
    scattering gain together exceed ``band_db`` are computed. A scatterer is kept when its
    earliest echo still has pulse support inside the record, and the FFT length is sized so
    that no kept echo wraps into the record; echoes that run past the record are truncated.
    The RF is noiseless: electronic noise and time gain compensation are
    :func:`zea.func.apply_receive_chain`, which :class:`zea.ops.Simulate` applies.

    Args:
        scatterer_positions (array-like): The positions of the scatterers [m] of shape (n_scat, 3).
        scatterer_magnitudes (array-like): The magnitudes of the scatterers of shape (n_scat,).
        probe_geometry (array-like): The geometry of the probe [m] of shape (n_el, 3).
        apply_lens_correction (bool): Model the acoustic lens as a layer of ``lens_sound_speed``
            in front of the elements. Every sub-element's path refracts through it (Fermat), so
            the lens delay depends on the direction to the scatterer, and the lens attenuates
            with ``lens_attenuation_coef``. With ``elevation_focus`` the layer is a cylindrical
            lens: ``lens_thickness`` at the element center, thinned (``lens_sound_speed`` below
            ``sound_speed``) or thickened towards the elevation edges so that the normal-incidence
            delay focuses at ``elevation_focus``. The lens face is taken locally flat under each
            sub-element, for the delay and for the spreading of the refracted wave, and the sinc
            directivity uses the geometric angle to the scatterer.
        lens_thickness (float): The thickness of the lens [m] at the element center.
        lens_sound_speed (float): The speed of sound in the lens [m/s].
        sound_speed (float): The speed of sound in the medium [m/s].
        n_ax (int): The number of samples in the RF data.
        center_frequency (float): The center frequency of the transmit pulse [Hz].
        sampling_frequency (float): The sampling frequency of the RF data [Hz].
        t0_delays (array-like): The transmit delays [s] of shape (n_tx, n_el), or
            (n_tx, n_mpt, n_el) for multi-plane transmits: the delay sets of a transmit fire
            together, so its field is the sum of theirs (SIMUS's MPT).
        initial_times (array-like): The initial times [s] of shape (n_tx,).
        element_width (float): The width of the elements [m].
        attenuation_coef (float): The attenuation coefficient [dB/cm/MHz].
        tx_apodizations (array-like): The transmit apodizations of shape (n_tx, n_el), or
            (n_tx, n_mpt, n_el) to apodize each delay set of a multi-plane transmit.
        t_peak (array-like): The time of the peak of the transmit pulse [s] of shape (n_tx,).
            The pulse is simulated with its envelope peak at the two-way travel time plus
            ``t_peak``; a real system's is :attr:`Pulse.time_to_peak` of :func:`transmit_pulse`.
        two_dimensional (bool): Simulate in the imaging plane, as a 1D probe behind an ideal
            elevation lens: the scatterers are moved to the probe's elevation center, there is
            no elevation directivity, and the transmit spreads cylindrically rather than
            spherically. Exclusive with ``elevation_focus``, the lens modelled in 3D, and
            rejects a probe with elevation extent.
        element_height (float): The elevation height of the elements [m], used for the
            elevation directivity. If None, an eighth of the width of a 1D probe (at least
            ``element_width``), or ``element_width`` for a 2D probe.
        max_chunk_gb (float): Memory budget [GB] for one block of work: a block of frequency
            bins over all scatterers, or, when one bin of every scatterer-element response
            (about ``8 * n_scat * n_el`` bytes) exceeds the budget, one bin over a chunk of
            scatterers, with the chunks summed. Larger blocks run the batched matrix products
            more efficiently: 4 GB is about 25% faster than 1 GB at 100k scatterers on a GPU,
            and up to 2x on CPU. Lower it if memory is tight.
        scatter_exponent (float | array-like): Weigh the scattered field amplitude by
            ``(f / center_frequency)**scatter_exponent``. 2 is Rayleigh scattering (e.g. blood),
            myocardium is approximately 1.5, soft tissue 0.6-0.8. A float sets a global value, an
            array of shape (n_scat,) gives each its own coefficient. If gradients are needed to
            the exponent(s), either pass ``scatter_exponent_range=(min, max)`` or ``band_db=None``.
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
            +y onto the element plane, so a normal must not point along the y axis. None is every
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
            sub-element, except in elevation when ``elevation_focus`` is set, which then
            follows the auto rule. Must be static under jit.
        elevation_focus (float, optional): Focal distance [m] of a fixed elevation lens, modelled
            on transmit and on receive through the elevation sub-elements: an ideal focusing
            advance per sub-element, or with ``apply_lens_correction`` the refracted path through
            the lens thickness profile. Exclusive with ``two_dimensional``, the ideal lens of
            the imaging plane. Must be static under jit.
        lens_attenuation_coef (float): Attenuation in the lens [dB/cm/MHz], applied over each
            sub-element's path inside the lens when ``apply_lens_correction`` is set. Apodizes
            the aperture where the lens is thick and lowers the center frequency.
        simplified_directivity (bool): Evaluate the directivity at the center frequency only.
            Less accurate, but slightly faster. Must be static under jit.
        band_db (float, optional): Bins where the pulse spectrum and the scattering gain
            together are below this many dB of their peak are not synthesised. None disables
            filtering. With per-transmit waveforms the band is the union over the pulses, and
            with per-scatterer exponents the union of the smallest and the largest exponent.
            With traced ``scatter_exponent``, either set ``band_db`` to None or provide an
            explicit ``scatter_exponent_range``.
        n_fft (int, optional): FFT length. Derived when None from ``n_ax``, the aperture, the
            transmit shifts and the pulse (and the scatterer positions when concrete) so that
            no echo wraps into the record, see :func:`fft_length`. Must be given when the
            geometry, the delays or the sound speed are traced, e.g. under ``jax.jit`` without
            closing over them; :class:`zea.ops.Simulate` and :attr:`zea.Parameters.n_fft`
            derive it. ``center_frequency`` and ``sampling_frequency`` must be static.
        scatter_exponent_range (tuple, optional): ``(min, max)`` exponent spanned by
            ``scatter_exponent``, used to pick the band instead of reading the exponents.
            Only needed when ``scatter_exponent`` is traced and ``band_db`` is set; for one
            traced shared exponent it is ``(p, p)``, or the range swept over if the compiled
            kernel is reused. :func:`scatter_exponent_bounds` derives it from a concrete
            exponent, and :class:`zea.ops.Simulate` does so before the jitted call. A range
            that does not cover the exponents in play truncates their band. Must be static
            under jit.
        sos_map (array-like, optional): Sound speed map [m/s] of shape (Nz, Nx) in the x-z
            plane, extruded along y, or (Nz, Nx, Ny) with ``map_grid_y``. Every path from an
            element to a scatterer then runs at the mean slowness along the straight ray between
            them (:func:`zea.func.ultrasound.straight_ray_slowness`), sampled at
            ``n_sos_ray_samples`` points with ``sound_speed`` outside the map; the
            sub-elements of an element share its center ray. The lens, the directivity, the
            spreading and the attenuation keep the geometry of the homogeneous medium at
            ``sound_speed``. None is a homogeneous medium. Differentiable on jax.
        map_grid_x (array-like, optional): Uniform, ascending x coordinates [m] of the maps,
            shape (Nx,).
        map_grid_z (array-like, optional): Uniform, ascending z coordinates [m] of the maps,
            shape (Nz,).
        map_grid_y (array-like, optional): Uniform, ascending y coordinates [m] of 3D maps,
            shape (Ny,). None for 2D maps.
        n_sos_ray_samples (int): Samples of the maps along each ray. Must be static under jit.
        attenuation_map (array-like, optional): Attenuation map [dB/cm/MHz] on the grid of
            ``sos_map`` (the same ``map_grid_*`` arguments, with or without a ``sos_map``).
            :class:`zea.data.spec.AttenuationMap` stores dB/m/Hz: multiply a stored map by 1e4.
            The medium part of every path is then attenuated by the mean coefficient along its
            straight ray (:func:`zea.func.ultrasound.straight_ray_mean`), with
            ``attenuation_coef`` outside the map. The lens part keeps ``lens_attenuation_coef``.
            None attenuates every path with ``attenuation_coef``. Differentiable on jax.
        attenuation_power (float): Exponent ``y`` of the power-law attenuation
            ``attenuation_coef * f**y`` dB/cm, with ``f`` in MHz, of the medium and of
            ``attenuation_map`` (their coefficients are then in dB/cm/MHz^y, as k-Wave's
            ``alpha_coeff`` with ``alpha_power``). Soft tissue is 1 to 1.5. The default 1 is
            linear in frequency. The dispersion that a power law implies is not modelled: the
            sound speed does not depend on frequency. The lens stays linear.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).
    """
    _validate_maps(sos_map, attenuation_map, map_grid_x, map_grid_z, map_grid_y)
    fc, fs = float(center_frequency), float(sampling_frequency)
    n_ax = int(n_ax)
    n_tx, n_el = (int(ops.shape(t0_delays)[i]) for i in (0, -1))
    pulses = transmit_pulses(n_tx, fc, fs, waveforms_two_way, waveform_sampling_frequency)
    model = element_model(
        probe_geometry,
        sound_speed,
        fc,
        pulses,
        element_width=element_width,
        element_height=element_height,
        attenuation_coef=attenuation_coef,
        apply_lens_correction=apply_lens_correction,
        lens_thickness=lens_thickness,
        lens_sound_speed=lens_sound_speed,
        two_dimensional=two_dimensional,
        baffle_impedance_ratio=baffle_impedance_ratio,
        element_normals=element_normals,
        n_sub_elements=n_sub_elements,
        elevation_focus=elevation_focus,
        lens_attenuation_coef=lens_attenuation_coef,
        attenuation_power=attenuation_power,
        simplified_directivity=simplified_directivity,
    )
    positions = _scene_positions(scatterer_positions, model)
    magnitudes = ops.cast(scatterer_magnitudes, "float32")
    n_scat = int(ops.shape(positions)[0])
    _validate_scatter_exponent(scatter_exponent, n_scat)
    scatter_exponent = _resolve_scatter_exponent(scatter_exponent)

    # Concrete views of the raw inputs, before any op puts them into an outer jit.
    raw = [concrete(x) for x in (t0_delays, initial_times, t_peak, probe_geometry, sound_speed)]
    map_np = concrete(sos_map)
    is_concrete = all(x is not None for x in raw) and (sos_map is None or map_np is not None)

    def bound():
        """Samples that hold every echo (see :func:`fft_length`), from the concrete inputs."""
        t0_np, t_init_np, t_peak_np, geom_np, c_np = raw
        shift_np = _shift_np(t0_np, t_init_np, t_peak_np)
        return _fft_bound(
            n_ax,
            fs,
            float(c_np),
            geom_np,
            shift_np.min(),
            shift_np.max(),
            pulses,
            concrete(positions),
            map_np,
        )

    if n_fft is None:
        if not is_concrete:
            raise ValueError(
                "n_fft cannot be derived from traced geometry, delays, sound speed or sound "
                "speed map; pass n_fft explicitly (see fft_length, zea.ops.Simulate or "
                "zea.Parameters.n_fft)."
            )
        n_fft = smooth_size(bound())
    elif map_np is not None and is_concrete and int(n_fft) < bound():
        # A map stretches the echoes of a scatterer; a length sized without it may wrap them.
        log.warning(
            f"n_fft ({int(n_fft)}) is shorter than the {bound()} samples the sound speed map "
            "needs to keep every echo from wrapping into the record; see fft_length."
        )
    n_fft = int(n_fft)

    if n_scat == 0:
        return ops.zeros((n_tx, n_ax, n_el, 1), "float32")

    shift = _transmit_shift(t0_delays, initial_times, t_peak)
    k0, k1 = band_bins(n_fft, fs, pulses, fc, scatter_exponent, band_db, scatter_exponent_range)
    wave = _pulse_spectra(pulses, np.fft.rfftfreq(n_fft, 1 / fs)[k0:k1])

    # Forward of one block per bin: the complex responses and the two matrix product outputs.
    # A single bin can still exceed the budget, so scatterers are chunked as well.
    budget = max_chunk_gb * 2**30
    per_scat = 8 * ((2 if two_dimensional else 1) * n_el + n_tx)
    chunk = int(min(n_scat, max(1, (budget - 8 * n_tx * n_el) // per_scat)))
    n_chunks = -(-n_scat // chunk)
    chunk = -(-n_scat // n_chunks)
    blocks = _band_blocks(n_fft, fs, k0, k1, per_scat * chunk + 8 * n_tx * n_el, budget)
    per_scatterer_exponent = scatter_exponent is not None and ndim(scatter_exponent) == 1

    spectrum = ops.zeros((blocks.n_kept, n_tx, n_el), "complex64")
    for start in range(0, n_scat, chunk):
        part = slice(start, start + chunk)
        # The straight-ray slowness and attenuation of every path, once for all frequency blocks.
        slowness, attenuation = _ray_means(
            positions[part],
            model,
            sos_map,
            attenuation_map,
            map_grid_x,
            map_grid_z,
            map_grid_y,
            n_sos_ray_samples,
        )
        block = _checkpointed_block(
            _rf_block,
            positions=positions[part],
            magnitudes=magnitudes[part],
            model=model,
            slowness=slowness,
            attenuation=attenuation,
            shift=shift,
            tx_apodizations=ops.cast(tx_apodizations, "float32"),
            center_frequency=fc,
            gate_time=_record_gate_time(n_ax, fs, _pulse_tail(pulses)),
            scatter_exponent=scatter_exponent[part] if per_scatterer_exponent else scatter_exponent,
        )
        spectrum = spectrum + _band_spectrum(block, blocks, (n_tx, n_el))
    spectrum = spectrum * ops.convert_to_tensor(wave)[:, :, None]

    # Transmits in groups, so a long record over many transmits does not allocate at once.
    group = min(32, n_tx)
    parts = [
        _band_to_time(spectrum[:, start : start + group], k0, k1, n_fft, n_ax)
        for start in range(0, n_tx, group)
    ]
    rf = ops.transpose(ops.concatenate(parts, axis=0), (0, 2, 1))
    return rf[..., None]


@dataclass(frozen=True)
class _BandBlocks:
    """The kept band [k0, k1) of the FFT grid padded to whole frequency blocks, for a
    :func:`_band_spectrum` loop. ``freqs`` is the padded band [Hz] as a float32 tensor; the pad
    repeats the last bin and is dropped after the loop."""

    freqs: Any
    f_block: int
    n_blocks: int
    n_kept: int

    @property
    def n_band(self):
        return self.n_blocks * self.f_block

    def slice(self, i):
        """Frequencies of block ``i`` (traced), and where it starts in the band."""
        start = i * self.f_block
        return ops.slice(self.freqs, [start], [self.f_block]), start


def _band_blocks(n_fft, sampling_frequency, k0, k1, per_bin, budget):
    """Split the band [k0, k1) into blocks of ``budget`` bytes at ``per_bin`` bytes per bin,
    spread evenly so the last block is padded by less than a whole block."""
    n_kept = k1 - k0
    f_block = int(max(1, min(n_kept, budget // per_bin)))
    n_blocks = -(-n_kept // f_block)
    f_block = -(-n_kept // n_blocks)
    freqs_all = np.fft.rfftfreq(n_fft, 1 / sampling_frequency)
    freqs = np.full(n_blocks * f_block, freqs_all[k1 - 1], np.float32)
    freqs[:n_kept] = freqs_all[k0:k1]
    return _BandBlocks(ops.convert_to_tensor(freqs), f_block, n_blocks, n_kept)


def _checkpointed_block(kernel, **kwargs):
    """``kernel`` with everything but the frequencies bound, under gradient checkpointing.

    The partial keeps the bound arguments out of the checkpoint's: tf.recompute_grad converts
    every argument to a tensor, which None (no map) cannot be.
    """
    return checkpoint(functools.partial(kernel, **kwargs))


def _band_spectrum(block, blocks, shape):
    """Band spectrum [f, *shape] over the kept bins: ``block(freqs)`` of every block of
    ``blocks`` (a :class:`_BandBlocks`) written into place, with the pad dropped."""

    def body(i, spectrum):
        freqs, start = blocks.slice(i)
        return ops.slice_update(spectrum, [start] + [0] * len(shape), block(freqs))

    spectrum = ops.zeros((blocks.n_band, *shape), "complex64")
    return ops.fori_loop(0, blocks.n_blocks, body, spectrum)[: blocks.n_kept]


def _band_to_time(spectrum, k0, k1, n_fft, n_ax):
    """Time records [..., n_ax] of a band spectrum [f, ...] over the bins [k0, k1) of an
    ``n_fft``-point real FFT."""
    band = ops.moveaxis(spectrum, 0, -1)
    pad = [(0, 0)] * (len(ops.shape(spectrum)) - 1) + [(k0, n_fft // 2 + 1 - k1)]
    full = (ops.pad(ops.real(band), pad), ops.pad(ops.imag(band), pad))
    return ops.irfft(full, fft_length=n_fft)[..., :n_ax]


def _rf_block(
    freqs,
    positions,
    magnitudes,
    model,
    slowness,
    attenuation,
    shift,
    tx_apodizations,
    center_frequency,
    gate_time,
    scatter_exponent,
):
    """Band spectrum [f, t, e] of one frequency block over all scatterers.

    Frequency leads every array so the einsums are plain batched matrix products. Scatterers
    whose earliest echo has no support before ``gate_time`` cannot reach the output and are
    dropped, so a long path never wraps into the record. ``slowness`` and ``attenuation`` are
    the mean slowness and attenuation coefficient [s, e] of the straight rays through the maps,
    or None for a homogeneous medium (see :func:`_ray_means`).
    """
    tx_response, rx_response, tau = element_responses(
        positions, model, freqs, slowness=slowness, attenuation=attenuation
    )
    keep = _record_keep(tau, ops.min(shift), gate_time)
    weight = ops.where(keep, magnitudes, 0.0)
    if scatter_exponent is not None:
        # [f, 1] for one shared exponent, [f, s] for one exponent per scatterer.
        weight = weight[None] * (freqs[:, None] / center_frequency) ** scatter_exponent
    else:
        weight = weight[None]
    tx_weights = _transmit_weights(freqs, shift, tx_apodizations)
    with highest_matmul_precision():
        incident = ops.einsum("fte,fse->fts", tx_weights, tx_response)
        scattered = incident * _to_complex(weight)[:, None, :]
        return ops.einsum("fts,fse->fte", scattered, rx_response)


def _transmit_weights(freqs, shift, tx_apodizations):
    """Apodization and delay phasor of every transmit and element, [f, t, e] complex64, summed
    over the delay sets of a multi-plane transmit."""
    n_tx, n_el = ops.shape(shift)[0], ops.shape(shift)[-1]
    shift = ops.reshape(shift, (n_tx, -1, n_el))
    apod = ops.reshape(tx_apodizations, (n_tx, -1, n_el))
    phasor = ops.exp(
        ops.array(-2j * np.pi, "complex64") * _to_complex(shift[None] * freqs[:, None, None, None])
    )
    return ops.sum(_to_complex(apod[None]) * phasor, axis=2)


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
    two_dimensional=False,
    element_height=None,
    baffle_impedance_ratio=0.0,
    element_normals=None,
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
    n_sub_elements=None,
    elevation_focus=None,
    simplified_directivity=False,
    band_db=-100.0,
    n_ax=None,
    n_fft=None,
    output="rms",
    max_chunk_gb=4.0,
    lens_attenuation_coef=0.0,
    sos_map=None,
    map_grid_x=None,
    map_grid_z=None,
    map_grid_y=None,
    n_sos_ray_samples=64,
    attenuation_map=None,
    attenuation_power=1.0,
):
    """Transmit pressure field of :func:`simulate_rf` on a grid.

    The incident field the simulator scatters, evaluated at the grid points instead of at
    scatterers: the same directivity, obliquity, attenuation, spread and transmit weights, times
    the two-way transmit pulse ``waveforms_two_way``. The pulse (which carries the transducer
    response) enters once, here, so a unit scatterer at a grid point returns exactly this field
    through the receive response. In 2D the grid is moved into the imaging plane, as the
    simulator moves its scatterers.

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
    _validate_maps(sos_map, attenuation_map, map_grid_x, map_grid_z, map_grid_y)
    fc, fs = float(center_frequency), float(sampling_frequency)
    n_tx, n_el = (int(ops.shape(t0_delays)[i]) for i in (0, -1))
    pulses = transmit_pulses(n_tx, fc, fs, waveforms_two_way, waveform_sampling_frequency)
    model = element_model(
        probe_geometry,
        sound_speed,
        fc,
        pulses,
        element_width=element_width,
        element_height=element_height,
        attenuation_coef=attenuation_coef,
        apply_lens_correction=apply_lens_correction,
        lens_thickness=lens_thickness,
        lens_sound_speed=lens_sound_speed,
        two_dimensional=two_dimensional,
        baffle_impedance_ratio=baffle_impedance_ratio,
        element_normals=element_normals,
        n_sub_elements=n_sub_elements,
        elevation_focus=elevation_focus,
        lens_attenuation_coef=lens_attenuation_coef,
        attenuation_power=attenuation_power,
        simplified_directivity=simplified_directivity,
    )
    grid_shape = tuple(int(d) for d in ops.shape(grid)[:-1])
    positions = _scene_positions(ops.reshape(grid, (-1, 3)), model)
    n_points = int(ops.shape(positions)[0])
    shift = _transmit_shift(t0_delays, initial_times, t_peak)

    if n_fft is None or n_ax is None:
        raw = [concrete(x) for x in (positions, model.geometry, shift, sound_speed)]
        bounds = _sound_speed_minmax(sound_speed, sos_map)
        if any(x is None for x in raw) or bounds is None:
            raise ValueError(
                "n_fft and n_ax cannot be derived from a traced grid, geometry, delays, "
                "sound speed or sound speed map; pass them explicitly."
            )
        pos_np, geom_np, shift_np, _ = raw
        dist_np = np.linalg.norm(
            pos_np[:, None].astype(np.float64) - geom_np[None].astype(np.float64), axis=-1
        )
        # The slowest speed in play bounds the arrival through the map.
        arrival = (dist_np / bounds[0])[None] + shift_np.reshape(n_tx, -1, n_el).max(1)[:, None]
        extent = int(np.ceil((arrival.max() + _pulse_tail(pulses)) * fs))
        if n_ax is None:
            n_ax = extent
        if n_fft is None:
            n_fft = smooth_size(max(int(n_ax), extent))
    n_ax, n_fft = int(n_ax), int(n_fft)

    k0, k1 = band_bins(n_fft, fs, pulses, fc, 0.0, band_db)
    n_kept = k1 - k0
    wave = _pulse_spectra(pulses, np.fft.rfftfreq(n_fft, 1 / fs)[k0:k1])
    tx_apodizations = ops.cast(tx_apodizations, "float32")
    rays = _ray_means(
        positions,
        model,
        sos_map,
        attenuation_map,
        map_grid_x,
        map_grid_z,
        map_grid_y,
        n_sos_ray_samples,
    )

    def blocked(points, rays, budget):
        """Band spectrum [f, t, p] or its Parseval energy [t, p] over ``points``."""
        block = _checkpointed_block(
            _pressure_block,
            positions=points,
            slowness=rays[0],
            attenuation=rays[1],
            model=model,
            shift=shift,
            tx_apodizations=tx_apodizations,
        )
        n_pts = int(ops.shape(points)[0])
        per_bin = 8 * ((2 if two_dimensional else 1) * n_pts * n_el + n_tx * n_pts)
        blocks = _band_blocks(n_fft, fs, k0, k1, per_bin, budget)

        if output == "time":
            spectrum = _band_spectrum(block, blocks, (n_tx, n_pts))
            return spectrum * ops.convert_to_tensor(wave)[:, :, None]

        # sum_n p[n]^2 = (1/N) sum_k w_k |P_k|^2 with w = 2 except at DC and, for an even
        # transform, at Nyquist. The pad of the band gets no weight.
        parseval = np.full(n_kept, 2.0, np.float32)
        if k0 == 0:
            parseval[0] = 1.0
        if k1 == n_fft // 2 + 1 and n_fft % 2 == 0:
            parseval[-1] = 1.0
        weight = np.zeros((blocks.n_band, n_tx), np.float32)
        weight[:n_kept] = parseval[:, None] * np.abs(wave) ** 2
        weight = ops.convert_to_tensor(weight)

        def body(i, energy):
            freqs, start = blocks.slice(i)
            part = block(freqs)
            w = ops.slice(weight, [start, 0], [blocks.f_block, n_tx])[:, :, None]
            return energy + ops.sum(w * (ops.real(part) ** 2 + ops.imag(part) ** 2), axis=0)

        return ops.fori_loop(0, blocks.n_blocks, body, ops.zeros((n_tx, n_pts), "float32"))

    budget = max_chunk_gb * 2**30
    if output == "rms":
        energy = blocked(positions, rays, budget)
        field = ops.sqrt(energy / (n_fft * n_ax))
    else:
        # The band spectrum of a chunk takes half the budget, the blocks the other half.
        chunk = int(max(1, min(n_points, budget // 2 // (8 * n_kept * n_tx))))
        parts = []
        for start in range(0, n_points, chunk):
            part = tuple(None if r is None else r[start : start + chunk] for r in rays)
            spectrum = blocked(positions[start : start + chunk], part, budget // 2)
            parts.append(_band_to_time(spectrum, k0, k1, n_fft, n_ax))
        field = ops.transpose(ops.concatenate(parts, axis=1), (0, 2, 1))

    lead = (n_tx,) if output == "rms" else (n_tx, n_ax)
    return ops.reshape(field, lead + grid_shape)


def _pressure_block(freqs, positions, slowness, attenuation, model, shift, tx_apodizations):
    """Incident field spectrum [f, t, p] of one frequency block, without the pulse."""
    tx_response, _, _ = element_responses(
        positions, model, freqs, slowness=slowness, attenuation=attenuation
    )
    tx_weights = _transmit_weights(freqs, shift, tx_apodizations)
    with highest_matmul_precision():
        return ops.einsum("fte,fpe->ftp", tx_weights, tx_response)


def _to_complex(x):
    return ops.cast(x, "complex64")
