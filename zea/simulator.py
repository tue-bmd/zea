"""Frequency domain ultrasound simulator.

The simulator works in the frequency domain and simulates RF data as a superposition of scatterer
responses. Every scatterer has a location and a magnitude, and optionally its own backscatter
coefficient: ``scatter_exponent`` is one value shared by the medium, or a vector of one exponent
per scatterer. Warning: the exponent is an amplitude exponent, not an intensity one. That means
Rayleigh scattering is 2, not 4.

Sound speed is set globally.

To use it, you can call :func:`simulate_rf` with the desired transmit scheme parameters and
scatterers directly, but the recommended path is to use :class:`zea.ops.Simulate`, which wraps the
simulator for pipelines and derives the FFT length automatically, and manages the noise level
computation when using multiple batches of scatterers.

:func:`record_reach`, :func:`record_bounds` and :func:`in_record` show which scatterers are
in-record for ``n_ax`` samples; use these to pre-prune your scatterer cloud to avoid wasting compute
on scatterers that are out of view (the simulator doesn't prune them, as moving clouds would
re-trigger jit compilation every frame). On that same note: when using the simulator for dynamic
scenes with varying scatterer numbers, consider padding your scatterer clouds to the next (half)
power of two, so jit only triggers once or twice.

``two_dimensional`` simulates in the imaging plane, as a 1D probe behind an ideal elevation lens.
:mod:`zea.simulator_time_domain` holds a faster, less accurate time-domain variant.

:func:`pressure_field` evaluates the transmit field of the simulator on a grid of points. Should
be a more accurate version of the pfield code used in the beamformer. It will likely be integrated
with the beamformer in the future, but currently only included for visualization purposes.

There is a time-domain variant of the simulator in :mod:`zea.simulator_time_domain`, which is less
accurate, but faster for 2D probes with few transmits.

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
from collections.abc import Callable
from dataclasses import dataclass

import keras
import numpy as np
from keras import ops
from scipy.signal import hilbert
from scipy.special import fresnel

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
    two_dimensional=False,
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
    lens_attenuation_coef=0.0,
    band_db=-100.0,
    n_fft=None,
    scatter_exponent_range=None,
):
    """Simulates RF data for a given set of scatterers.

    The RF is synthesised in the frequency domain, on the rfft grid of ``n_fft`` samples, as the
    superposition of the scatterer echoes:

    .. code-block:: text

        incident[f, t, s] = sum_e W[f, t, e] R_tx[f, s, e]    W = apod_te exp(-2 pi i f shift_te)
        rf[f, t, e]       = sum_s S[f, t, s] R_rx[f, s, e]    S = incident * mag_s * gain_s(f)

    The one-way responses ``R_tx`` and ``R_rx`` (directivity, spreading, attenuation and the
    travel phase) do not depend on the transmit, so they are generated once per frequency block
    and shared across all transmits through the two matrix products. Only the bins where the
    pulse spectrum, the transducer transfer function and the scattering gain together exceed
    ``band_db`` are computed. A scatterer is kept when its earliest echo still has pulse
    support inside the record, and the FFT length is sized so that no kept echo wraps into the
    record; echoes that run past the record are truncated.

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
        two_dimensional (bool): Simulate in the imaging plane, as a 1D probe behind an ideal
            elevation lens: the scatterers are moved to the probe's elevation centre, there is
            no elevation directivity, and the transmit spreads cylindrically rather than
            spherically. Exclusive with ``elevation_focus``, the lens modelled in 3D, and
            rejects a probe with elevation extent.
        element_height (float): The elevation height of the elements [m], used for the
            elevation directivity. If None, an eighth of the width of a 1D probe (at least
            ``element_width``), or ``element_width`` for a 2D probe.
        max_chunk_gb (float): Memory budget [GB] for one frequency block. Barely affects GPU
            speed, up to 2x on CPU.
        noise_level_db (float): Electronic noise level in dB relative to the noiseless RF
            maximum. None disables the noise. Must be static under jit.
        tgc_max_db (float): Time gain compensation in dB at the last axial sample, ramped
            linearly in dB from 0 at the first. 0 disables it. Must be static under jit.
        noise_seed (int | SeedGenerator | jax.random.key, optional): Seed for the noise. Vary it
            across transmit batches to keep the realisations independent.
        noise_reference (float): Reference amplitude for the noise level. If None, defaults to the
            noiseless RF maximum. Pass a fixed reference to avoid the noise level changing per
            transmit batch. See :func:`apply_receive_chain`.
        scatter_exponent (float | array-like): Weigh the scattered field amplitude by
            ``(f / center_frequency)**scatter_exponent``. 2 is Rayleigh scattering (e.g. blood),
            myocardium is approximately 1.5, soft tissue 0.6-0.8. A float sets a global value, an
            array of shape (n_scat,) gives each its own coefficient. If gradients are needed to
            the exponent(s), either pass ``scatter_exponent_range=(min, max)`` or ``band_db=None``.
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
            the lens thickness profile. Exclusive with ``two_dimensional``, the ideal lens of
            the imaging plane. Must be static under jit.
        lens_attenuation_coef (float): Attenuation in the lens [dB/cm/MHz], applied over each
            sub-element's path inside the lens when ``apply_lens_correction`` is set. Apodizes
            the aperture where the lens is thick and lowers the centre frequency.
        band_db (float, optional): Bins where the pulse spectrum, the transducer transfer
            function and the scattering gain together are below this many dB of their peak are
            not synthesised. None disables filtering. With per-scatterer exponents, the band is the
            derived from the union of the smallest and the largest exponent. With traced
            ``scatter_exponent``, either set ``band_db`` to None or provide an explicit
            ``scatter_exponent_range``.
        n_fft (int, optional): FFT length. Derived when None from ``n_ax``, the aperture and
            the transmit shifts (and the scatterer positions when concrete) so that no echo
            wraps into the record, see :func:`fft_length`. Must be given when the geometry, the
            delays or the sound speed are traced, e.g. under ``jax.jit`` without closing over
            them; :class:`zea.ops.Simulate` and :attr:`zea.Parameters.n_fft` derive it.
            ``center_frequency`` and ``sampling_frequency`` must be static.
        scatter_exponent_range (tuple, optional): ``(min, max)`` exponent spanned by
            ``scatter_exponent``, used to pick the band instead of reading the exponents.
            Only needed when ``scatter_exponent`` is traced and ``band_db`` is set; for one
            traced shared exponent it is ``(p, p)``, or the range swept over if the compiled
            kernel is reused. :func:`scatter_exponent_bounds` derives it from a concrete
            exponent, and :class:`zea.ops.Simulate` does so before the jitted call. A range
            that does not cover the exponents in play truncates their band. Must be static
            under jit.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).
    """
    _validate_two_dimensional(two_dimensional, elevation_focus, probe_geometry)
    fc, fs = float(center_frequency), float(sampling_frequency)
    n_ax = int(n_ax)
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
    n_sub_elements = _resolve_sub_elements(
        n_sub_elements,
        elevation_focus,
        element_width,
        element_height,
        sound_speed,
        fc,
        bandwidth_percent,
        two_dimensional,
    )
    positions = ops.cast(scatterer_positions, "float32")
    magnitudes = ops.cast(scatterer_magnitudes, "float32")
    geometry = ops.cast(probe_geometry, "float32")
    _validate_scatter_exponent(scatter_exponent, int(ops.shape(positions)[0]))
    scatter_exponent = _resolve_scatter_exponent(scatter_exponent)
    if two_dimensional:
        positions = _snap_elevation(positions, geometry)

    # Concrete views of the raw inputs, before any op puts them into an outer jit.
    raw = [_concrete(x) for x in (t0_delays, initial_times, t_peak, probe_geometry, sound_speed)]
    if n_fft is None:
        if any(x is None for x in raw):
            raise ValueError(
                "n_fft cannot be derived from traced geometry, delays or sound speed; pass "
                "n_fft explicitly (see fft_length, zea.ops.Simulate or zea.Parameters.n_fft)."
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
            _concrete(positions),
        )
    n_fft = int(n_fft)
    n_tx, n_el = (int(d) for d in ops.shape(t0_delays))

    def finish(rf):
        return apply_receive_chain(
            rf[..., None], noise_level_db, tgc_max_db, noise_seed, noise_reference
        )

    if int(ops.shape(positions)[0]) == 0:
        return finish(ops.zeros((n_tx, n_ax, n_el), "float32"))

    shift = _transmit_shift(t0_delays, initial_times, t_peak)
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
        scatter_exponent_range=scatter_exponent_range,
    )

    # Forward of one block per bin: the complex responses and the two matrix product outputs.
    n_scat = int(ops.shape(positions)[0])
    per_bin = 8 * ((2 if two_dimensional else 1) * n_scat * n_el + n_tx * n_scat + n_tx * n_el)
    f_block = int(max(1, min(k1 - k0, max_chunk_gb * 2**30 // per_bin)))
    n_blocks = -(-(k1 - k0) // f_block)
    # Spread the band evenly, so the last block is padded by less than a whole block.
    f_block = -(-(k1 - k0) // n_blocks)
    n_band = n_blocks * f_block

    # Band padded to whole blocks; the pad repeats the last bin and is dropped after the loop.
    freqs_all = _rfft_freqs(n_fft, fs)
    freqs = np.full(n_band, freqs_all[k1 - 1], np.float32)
    freqs[: k1 - k0] = freqs_all[k0:k1]
    wave = _transmit_spectrum_np(
        n_fft, fc, fs, n_period, bandwidth_percent, probe_center_frequency, chirp_sweep
    )[k0:k1]
    freqs = ops.convert_to_tensor(freqs)

    block = checkpoint(
        functools.partial(
            _rf_block,
            positions=positions,
            magnitudes=magnitudes,
            geometry=geometry,
            shift=shift,
            tx_apodizations=ops.cast(tx_apodizations, "float32"),
            center_frequency=fc,
            sound_speed=_as_f32(sound_speed),
            element_width=_as_f32(element_width),
            element_height=_as_f32(element_height),
            attenuation_coef=_as_f32(attenuation_coef),
            lens_thickness=_as_f32(lens_thickness),
            lens_sound_speed=_as_f32(lens_sound_speed),
            gate_time=_record_gate_time(n_ax, fs, fc, n_period),
            scatter_exponent=scatter_exponent,
            apply_lens_correction=bool(apply_lens_correction),
            two_dimensional=bool(two_dimensional),
            rigid_baffle=bool(rigid_baffle),
            element_normals=None if element_normals is None else _as_f32(element_normals),
            n_sub_elements=n_sub_elements,
            elevation_focus=None if elevation_focus is None else float(elevation_focus),
            lens_attenuation_coef=_as_f32(lens_attenuation_coef),
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


def _rf_block(
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
    two_dimensional,
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
        two_dimensional,
        0.0 if rigid_baffle else float("inf"),
        element_normals,
        n_sub_elements,
        elevation_focus,
        lens_attenuation_coef,
        frequency_first=True,
    )
    keep = _record_keep(dist, sound_speed, ops.min(shift), gate_time)
    weight = ops.where(keep, magnitudes, 0.0)
    if scatter_exponent is not None:
        # [f, 1] for one shared exponent, [f, s] for one exponent per scatterer.
        weight = weight[None] * (freqs[:, None] / center_frequency) ** scatter_exponent
    else:
        weight = weight[None]
    f3 = freqs[:, None, None]
    tx_weights = _to_complex(tx_apodizations[None]) * ops.exp(
        ops.array(-2j * np.pi, "complex64") * _to_complex(shift[None] * f3)
    )
    with highest_matmul_precision():
        incident = ops.einsum("fte,fse->fts", tx_weights, tx_response)
        scattered = incident * _to_complex(weight)[:, None, :]
        return ops.einsum("fts,fse->fte", scattered, rx_response)


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
    """Transmit pressure field of :func:`simulate_rf` on a grid.

    The incident field the simulator scatters, evaluated at the grid points instead of at
    scatterers: the same directivity, obliquity, attenuation, spread and transmit weights, times
    the transmit pulse. The transducer transfer function enters once (its square root, as the
    ``bandwidth_percent`` band is pulse-echo), so a unit scatterer at a grid point returns this
    field through the receive response. In 2D the grid is moved into the imaging plane, as the
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
    _validate_two_dimensional(two_dimensional, elevation_focus, probe_geometry)
    fc, fs = float(center_frequency), float(sampling_frequency)
    n_period = float(n_period)
    grid_shape = tuple(int(d) for d in ops.shape(grid)[:-1])
    positions = ops.reshape(ops.cast(grid, "float32"), (-1, 3))
    geometry = ops.cast(probe_geometry, "float32")
    if two_dimensional:
        positions = _snap_elevation(positions, geometry)
    element_width = _resolve_element_width(geometry, element_width)
    element_height = _resolve_element_height(geometry, element_width, element_height)
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
        two_dimensional,
    )
    n_tx, n_el = (int(d) for d in ops.shape(t0_delays))
    n_points = int(ops.shape(positions)[0])
    shift = _transmit_shift(t0_delays, initial_times, t_peak)

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
    wave_all = _transmit_spectrum_np(
        n_fft, fc, fs, n_period, bandwidth_percent, probe_center_frequency, chirp_sweep, True
    )

    block = checkpoint(
        functools.partial(
            _pressure_block,
            geometry=geometry,
            shift=shift,
            tx_apodizations=ops.cast(tx_apodizations, "float32"),
            sound_speed=_as_f32(sound_speed),
            element_width=_as_f32(element_width),
            element_height=_as_f32(element_height),
            attenuation_coef=_as_f32(attenuation_coef),
            lens_thickness=_as_f32(lens_thickness),
            lens_sound_speed=_as_f32(lens_sound_speed),
            apply_lens_correction=bool(apply_lens_correction),
            two_dimensional=bool(two_dimensional),
            rigid_baffle=bool(rigid_baffle),
            element_normals=None if element_normals is None else _as_f32(element_normals),
            n_sub_elements=n_sub_elements,
            elevation_focus=None if elevation_focus is None else float(elevation_focus),
            lens_attenuation_coef=_as_f32(lens_attenuation_coef),
        )
    )

    def blocked(points, budget):
        """Band spectrum [f, t, p] or its Parseval energy [t, p] over ``points``."""
        n_pts = int(ops.shape(points)[0])
        per_bin = 8 * ((2 if two_dimensional else 1) * n_pts * n_el + n_tx * n_pts)
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

    lead = (n_tx,) if output == "rms" else (n_tx, n_ax)
    return ops.reshape(field, lead + grid_shape)


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
    two_dimensional,
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
        two_dimensional,
        0.0 if rigid_baffle else float("inf"),
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


def _to_complex(x):
    return ops.cast(x, "complex64")


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


# ---------------------------------------------------------------------------------------------
# Record gate
# Which scatterers a record can hold: shared by the synthesis and the public helpers.
# ---------------------------------------------------------------------------------------------


def _record_gate_time(n_ax, sampling_frequency, center_frequency, n_period):
    """Latest arrival [s] of an echo peak with pulse support inside the record."""
    return n_ax / sampling_frequency + 0.5 * n_period / center_frequency


def _record_keep(dist, sound_speed, shift_min, gate_time):
    """The gate of :func:`simulate_rf`: the earliest echo arrives before ``gate_time``."""
    return 2 * ops.min(dist, axis=1) / sound_speed + shift_min < gate_time


def record_reach(
    probe_geometry,
    sound_speed,
    n_ax,
    sampling_frequency,
    center_frequency,
    t0_delays,
    initial_times,
    t_peak,
    n_period=4.0,
    apply_lens_correction=False,
    lens_thickness=0.0,
    lens_sound_speed=None,
    two_dimensional=False,
):
    """Farthest one-way distance [m] from an element at which :func:`simulate_rf` still
    simulates a scatterer.

    A scatterer is simulated while its earliest echo has pulse support inside the record, that
    is while its nearest element is within this distance (see :func:`in_record`). Through a
    lens the distance holds along the element normal, and is high off the normal by a fraction
    of the lens thickness.

    Args:
        probe_geometry (array-like): Element positions [m] of shape (n_el, 3).
        sound_speed (float): Speed of sound [m/s].
        n_ax (int): Number of axial samples in the record.
        sampling_frequency (float): Sampling frequency [Hz].
        center_frequency (float): Pulse center frequency [Hz].
        t0_delays (array-like): Transmit delays [s] of shape (n_tx, n_el).
        initial_times (array-like): Record start times [s] of shape (n_tx,).
        t_peak (array-like): Pulse peak times [s] of shape (n_tx,).
        n_period (float): Number of periods in the pulse.
        apply_lens_correction (bool): Whether the simulation models the lens.
        lens_thickness (float): Lens thickness [m].
        lens_sound_speed (float, optional): Speed of sound in the lens [m/s].
        two_dimensional (bool): Unused; accepted so the record helpers share their arguments.

    Returns:
        float: The reach [m].
    """
    del two_dimensional
    fs, fc, c = float(sampling_frequency), float(center_frequency), float(sound_speed)
    raw = [_concrete(x) for x in (t0_delays, initial_times, t_peak)]
    if any(x is None for x in raw):
        raise ValueError("record_reach needs concrete delays; under jit use in_record.")
    t0_np, t_init_np, t_peak_np = (np.asarray(x, np.float64) for x in raw)
    shift = t0_np - t_init_np[:, None] + t_peak_np[:, None]
    time = (_record_gate_time(int(n_ax), fs, fc, float(n_period)) - float(shift.min())) / 2
    if not apply_lens_correction or lens_sound_speed is None:
        return c * time
    thickness, c_lens = float(lens_thickness), float(lens_sound_speed)
    return thickness + c * (time - thickness / c_lens)


def record_bounds(
    probe_geometry,
    sound_speed,
    n_ax,
    sampling_frequency,
    center_frequency,
    t0_delays,
    initial_times,
    t_peak,
    n_period=4.0,
    apply_lens_correction=False,
    lens_thickness=0.0,
    lens_sound_speed=None,
    two_dimensional=False,
):
    """Box [m] in front of the probe outside which :func:`simulate_rf` simulates no scatterer.

    The bounding box of the elements grown by :func:`record_reach`, starting in z at the
    shallowest element and, in 2D, collapsed onto the imaging plane. A phantom drawn inside it
    wastes no scatterers on the gate; :func:`in_record` gives the exact gate. Takes the
    arguments of :func:`record_reach`.

    Returns:
        ndarray: The box as ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``, shape (2, 3).
    """
    reach = record_reach(
        probe_geometry,
        sound_speed,
        n_ax,
        sampling_frequency,
        center_frequency,
        t0_delays,
        initial_times,
        t_peak,
        n_period,
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
    )
    geometry = _concrete(probe_geometry)
    if geometry is None:
        raise ValueError("record_bounds needs a concrete probe geometry.")
    geometry = np.asarray(geometry, np.float64)
    low, high = geometry.min(0) - reach, geometry.max(0) + reach
    low[2] = geometry[:, 2].min()
    if two_dimensional:
        low[1] = high[1] = geometry[:, 1].mean()
    return np.stack([low, high]).astype(np.float32)


def in_record(
    points,
    probe_geometry,
    sound_speed,
    n_ax,
    sampling_frequency,
    center_frequency,
    t0_delays,
    initial_times,
    t_peak,
    n_period=4.0,
    apply_lens_correction=False,
    lens_thickness=0.0,
    lens_sound_speed=None,
    two_dimensional=False,
):
    """Whether :func:`simulate_rf` simulates a scatterer at each point: its gate.

    A scatterer is dropped when even its earliest echo has no pulse support inside the record.
    Takes the arguments of :func:`record_reach`; ``two_dimensional`` moves the points into the
    imaging plane first, as the simulator does. Jittable.

    Args:
        points (array-like): Positions [m] of shape (n_points, 3).

    Returns:
        array-like: Boolean mask of shape (n_points,).
    """
    positions = ops.cast(points, "float32")
    geometry = ops.cast(probe_geometry, "float32")
    if two_dimensional:
        positions = _snap_elevation(positions, geometry)
    shift = _transmit_shift(t0_delays, initial_times, t_peak)
    dist = _one_way_distance(
        positions,
        geometry,
        bool(apply_lens_correction),
        _as_f32(lens_thickness),
        _as_f32(lens_sound_speed),
        _as_f32(sound_speed),
    )
    gate_time = _record_gate_time(
        int(n_ax), float(sampling_frequency), float(center_frequency), float(n_period)
    )
    return _record_keep(dist, _as_f32(sound_speed), ops.min(shift), gate_time)


# ---------------------------------------------------------------------------------------------
# Element physics
# The one-way response of an element: distance, directivity, obliquity, lens, spreading and
# attenuation, averaged over its sub-elements.
# ---------------------------------------------------------------------------------------------


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
    two_dimensional,
    baffle_impedance_ratio,
    element_normals,
    n_sub_elements=(1, 1),
    elevation_focus=None,
    lens_attenuation_coef=0.0,
    min_dist=0.0,
    frequency_first=False,
):
    """Transmit and receive one-way responses and the one-way path length [s, e].

    The responses are [s, e, f], or [f, s, e] when ``frequency_first`` is True. Each element is
    the mean of ``n_sub_elements`` (lateral, elevation) sub-elements with their
    own distance, phase and sinc directivity, so the response holds in the near field too. The
    sub-element distance is clamped at ``min_dist`` for the phase and the spreading (see
    :func:`min_distance`), not for the angles. An
    elevation focus is the ideal focusing advance of each elevation sub-element, or with the lens
    the refracted (Fermat) path through the local lens thickness, which the focus thins towards
    the edges. The lens path is expressed as the medium distance with the same travel time for the
    phase, and spreads as the refracted ray tube (:func:`_lens_spread_distance`); the lens leg is
    attenuated with ``lens_attenuation_coef``. The returned path length is the element centre's.

    In 2D the positions are expected in the imaging plane (:func:`_snap_elevation`): there is no
    elevation directivity, and the transmit spreads cylindrically, as behind an ideal lens.
    """
    n_lateral, n_elevation = n_sub_elements
    n_sub = n_lateral * n_elevation
    relative_center = positions[:, None] - geometry[None]
    dtype = relative_center.dtype
    lateral_axis, elevation_axis, _ = frame = _element_frame(element_normals, dtype)
    dist = _one_way_distance(
        positions, geometry, apply_lens_correction, lens_thickness, lens_sound_speed, sound_speed
    )
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
        amplitude = directivity(f3, fx(theta), sub_width, sound_speed)
        if not two_dimensional:
            amplitude = amplitude * directivity(f3, fx(phi), sub_height, sound_speed)
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
        sub_dist = ops.maximum(sub_dist, min_dist)
        spread_dist = ops.maximum(spread_dist, min_dist)
        amplitude = amplitude * attenuate(f3, attenuation_coef, fx(medium_len))
        amplitude = amplitude * fx(obliquity_factor(obliquity, baffle_impedance_ratio))
        phase = ops.exp(
            ops.array(-2j * np.pi, "complex64")
            * ops.cast((fx(sub_dist) / sound_speed - advance[j]) * f3, "complex64")
        )
        rx = ops.cast(amplitude * spread(fx(spread_dist), 1.0, min_dist), "complex64") * phase
        if two_dimensional:
            # An ideal elevation lens: cylindrical spread on the way out, spherical back.
            tx = ops.cast(amplitude * spread(fx(spread_dist), 0.5, min_dist), "complex64") * phase
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


def _one_way_distance(
    positions, geometry, apply_lens_correction, lens_thickness, lens_sound_speed, sound_speed
):
    """One-way path length [m] from each position to each element centre, of shape [s, e].

    Through a lens it is the medium distance with the travel time of the refracted path.
    """
    if not apply_lens_correction:
        return ops.linalg.norm(positions[:, None] - geometry[None], axis=-1)
    travel_times = compute_lens_corrected_travel_times(
        geometry,
        positions,
        lens_thickness=lens_thickness,
        c_lens=lens_sound_speed,
        c_medium=sound_speed,
        n_iter=3,
    )
    return travel_times * sound_speed


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
    the Fraunhofer pattern of a rectangular aperture. Projected angles arctan2(lateral, axial) would
    narrow the elevation pattern for laterally offset scatterers.

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


def _sub_element_offsets(n_lateral, n_elevation, element_width, element_height):
    """Centroid offsets (u, v) of the sub-elements in the element frame, each (n_sub,)."""
    u = (ops.arange(n_lateral, dtype="float32") - (n_lateral - 1) / 2) * (
        ops.cast(element_width, "float32") / n_lateral
    )
    v = (ops.arange(n_elevation, dtype="float32") - (n_elevation - 1) / 2) * (
        ops.cast(element_height, "float32") / n_elevation
    )
    return ops.reshape(ops.tile(u[:, None], (1, n_elevation)), (-1,)), ops.tile(v, (n_lateral,))


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


def _snap_elevation(positions, geometry):
    """The positions moved into the imaging plane, the probe's elevation centre."""
    centre = ops.mean(geometry[:, 1])
    return ops.stack(
        [positions[:, 0], ops.zeros_like(positions[:, 0]) + centre, positions[:, 2]], axis=-1
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


# ---------------------------------------------------------------------------------------------
# Spectra
# Pulse, chirp and transducer spectra, the band they occupy and the FFT length that holds the
# record without wrapping.
# ---------------------------------------------------------------------------------------------


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
        bandwidth_percent (float, optional): -6 dB fractional bandwidth in percent. None is a
            flat response.
        center_frequency (float, optional): Fallback band centre [Hz].
        xp: Array module, ``keras.ops`` or ``numpy``.

    Returns:
        array-like: The transfer function at ``f``.
    """
    if bandwidth_percent is None:
        return xp.ones_like(f)
    bandwidth = _concrete(bandwidth_percent)
    if bandwidth is not None and (not np.isfinite(float(bandwidth)) or float(bandwidth) <= 0):
        raise ValueError(f"bandwidth_percent must be positive, got {float(bandwidth)}.")
    if probe_center_frequency is None:
        probe_center_frequency = center_frequency
    if probe_center_frequency is None:
        raise ValueError("transducer_transfer needs probe_center_frequency or center_frequency.")
    half_width = 0.5 * bandwidth_percent / 100 * probe_center_frequency
    return xp.exp(-np.log(2) * ((xp.abs(f) - probe_center_frequency) / half_width) ** 2)


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


def _transmit_spectrum_np(
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
    scatter_exponent_range=None,
):
    """Contiguous fft bin range that is not discarded.

    The pulse spectrum, the transducer transfer function and the scattering gain together
    exceed ``band_db`` there. If ``scatter_exponent`` is a vector of per-scatterer exponents, 
    the band is calculated from the union of the min and max exponents.
    """
    freqs = _rfft_freqs(n_fft, sampling_frequency)
    if band_db is None:
        return 0, len(freqs)
    w = np.abs(
        _transmit_spectrum_np(
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
    lo, hi = _exponent_range(scatter_exponent, scatter_exponent_range)
    k0, k1 = len(freqs), 0
    for exponent in (lo,) if lo == hi else (lo, hi):
        band = w * (freqs / center_frequency) ** exponent
        keep = np.flatnonzero(band > band.max() * 10 ** (band_db / 20))
        k0, k1 = min(k0, int(keep[0])), max(k1, int(keep[-1]) + 1)
    return k0, k1


def _round_up_to_power_of_two(x):
    """Rounds up to the next power of two."""
    return 2 ** np.ceil(np.log2(x))


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


# ---------------------------------------------------------------------------------------------
# Validation and argument resolution
# Static checks and the derivation of the defaults, on concrete values only.
# ---------------------------------------------------------------------------------------------


def _concrete(x):
    """numpy view of ``x``, or None when it is traced."""
    if x is None:
        return None
    try:
        return ops.convert_to_numpy(x)
    except (RuntimeError, ValueError, TypeError, NotImplementedError):
        return None


def _ndim(x):
    """Rank of ``x`` without converting it, so a traced array is not forced to numpy."""
    shape = getattr(x, "shape", None)
    return 0 if shape is None else len(shape)


def _as_f32(x):
    return ops.cast(0.0 if x is None else x, "float32")


def _transmit_shift(t0_delays, initial_times, t_peak):
    """Transmit shift [s] per element beyond the travel time, of shape (n_tx, n_el)."""
    return (
        ops.cast(t0_delays, "float32")
        - ops.cast(initial_times, "float32")[:, None]
        + ops.cast(t_peak, "float32")[:, None]
    )


def _validate_scatter_exponent(scatter_exponent, n_scat=None):
    """Reject invalid exponents (must be scalar or [n_scat], and finite nonnegative)."""
    ndim = _ndim(scatter_exponent)
    if ndim > 1:
        raise ValueError(
            "scatter_exponent must be a scalar or a vector of one exponent per scatterer, "
            f"got {ndim} dimensions."
        )
    if ndim == 1 and n_scat is not None:
        n_given = int(ops.shape(scatter_exponent)[0])
        if n_given != n_scat:
            raise ValueError(
                f"A per-scatterer scatter_exponent needs one value per scatterer: got "
                f"{n_given} exponents for {n_scat} scatterers."
            )
    values = _concrete(scatter_exponent)
    if values is None:
        return
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        shown = values if values.ndim == 0 else np.array2string(values, threshold=8)
        raise ValueError(
            f"scatter_exponent ({shown}) must be finite and non-negative. "
            "2 is Rayleigh scattering (e.g. blood), myocardium is approximately 1.5, "
            "soft tissue 0.6-0.8."
        )


def _resolve_scatter_exponent(scatter_exponent):
    """None when the weighting is the identity, a Python float for a concrete exponent shared
    by every scatterer, else a float32 tensor.

    The rank says whether the exponent is shared or per-scatterer and the gain broadcasts either
    way, so only a concrete scalar is folded into a Python float; a traced one stays a tensor.
    """
    if _ndim(scatter_exponent) == 0:
        value = _concrete(scatter_exponent)
        if value is not None:
            return float(value) or None
    return ops.cast(scatter_exponent, "float32")


def scatter_exponent_bounds(scatter_exponent):
    """Smallest and largest exponent, or None when ``scatter_exponent`` is traced.

    Both edges of the bound move monotonically with the exponent, so the union of the bands of the
    min/max exponents covers every scatterer. ``simulate_rf`` derives them from a concrete exponent;
    input shape is static, so a caller that traces the exponent should pass the pair as
    ``scatter_exponent_range``.
    """
    if scatter_exponent is None:
        return 0.0, 0.0
    values = _concrete(scatter_exponent)
    if values is None:
        return None
    if values.size == 0:
        return 0.0, 0.0
    return float(values.min()), float(values.max())


def _exponent_range(scatter_exponent, given=None):
    """``given`` when it is, else the range spanned by ``scatter_exponent`` itself."""
    if given is not None:
        lo, hi = (float(v) for v in given)
        if not np.isfinite([lo, hi]).all() or lo < 0 or hi < lo:
            raise ValueError(
                f"scatter_exponent_range ({lo}, {hi}) must be a finite, non-negative, "
                "increasing pair."
            )
        return lo, hi
    bounds = scatter_exponent_bounds(scatter_exponent)
    if bounds is None:
        raise ValueError(
            "band_db needs the range of scatter_exponent to pick the band, and a traced "
            "exponent does not give it. Pass scatter_exponent_range=(min, max) (see "
            "scatter_exponent_bounds), or band_db=None to keep every bin."
        )
    return bounds


def _validate_two_dimensional(two_dimensional, elevation_focus, probe_geometry, tol=1e-6):
    """2D is a 1D probe with an ideal elevation lens, so it excludes the 3D lens model."""
    if not two_dimensional:
        return
    if elevation_focus is not None:
        raise ValueError(
            "two_dimensional collapses elevation (cylindrical transmit spread, no elevation "
            "directivity); elevation_focus models the lens in 3D through the elevation "
            "sub-elements. Pick one."
        )
    geometry = _concrete(probe_geometry)
    if geometry is None:
        return
    elevation = np.asarray(geometry)[:, 1]
    if elevation.max() - elevation.min() > tol:
        raise ValueError(
            "two_dimensional=True needs a 1D probe, but the elements span "
            f"{elevation.min():.2e} to {elevation.max():.2e} m in elevation."
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


def _resolve_sub_elements(
    n_sub_elements,
    elevation_focus,
    element_width,
    element_height,
    sound_speed,
    center_frequency,
    bandwidth_percent,
    two_dimensional=False,
):
    """Sub-elements per element as (n_lateral, n_elevation).

    "auto" splits an element into patches of ceil(size / lambda_min), i.e. at most one wavelength,
    to make sure that the far-field assumption is valid.
    """
    if isinstance(n_sub_elements, (tuple, list)):
        n_lateral, n_elevation = (int(n) for n in n_sub_elements)
        return max(n_lateral, 1), 1 if two_dimensional else max(n_elevation, 1)
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
    n_elevation = 1 if two_dimensional else max(int(np.ceil(height / lambda_min)), 1)
    if n_sub_elements == "auto":
        return max(int(np.ceil(width / lambda_min)), 1), n_elevation
    return (1 if n_sub_elements is None else max(int(n_sub_elements), 1)), n_elevation


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
    (Selfridge et al. 1980, as in SIMUS). With a non-rigid baffle, directions behind the
    element get 0, not a pole.
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
    """Time [s] from the transmit trigger (the start of the excitation, or of a measured
    waveform) to the envelope peak: the ``t_peak`` of a real system."""
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
        that :class:`zea.Parameters` derives from it is :attr:`time_to_peak`. A model whose
        transducer response is not causal (``"hann"``, ``"simus"``) then loses the part of its
        response before the trigger, which is small.
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

    return _calibrate(spectrum, fs, duration, pulse_model == "realistic", trigger)


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
    (n_freqs, n_samples) table is formed."""
    f = np.asarray(f, np.float64)
    dt = times[1] - times[0]
    z = np.exp(-2j * np.pi * f * dt)
    spectrum = np.zeros(f.shape, np.complex128)
    for sample in samples[::-1]:
        spectrum = spectrum * z + sample
    return spectrum * dt * np.exp(-2j * np.pi * f * times[0])


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
