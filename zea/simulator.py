"""Frequency domain ultrasound simulator.

The simulator works in the frequency domain and simulates RF data as a superposition of scatterer
responses. Every scatterer has a location and a magnitude, and optionally its own backscatter
coefficient: ``scatter_exponent`` is one value shared by the medium, or a vector of one exponent
per scatterer. Warning: the exponent is an amplitude exponent, not an intensity one. That means
Rayleigh scattering is 2, not 4.

Sound speed is one value for the medium, or a map: ``sos_map`` with its grid ``sos_grid_x``,
``sos_grid_z`` (and ``sos_grid_y`` for a 3D map) makes every element-scatterer path run at the mean
slowness along the straight ray between them (:func:`zea.func.ultrasound.straight_ray_slowness`),
with ``sound_speed`` outside the map. Straight rays keep the geometry, so the directivity, the
spreading and the attenuation are those of the homogeneous medium; only the travel times change.

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

import keras
import numpy as np
from keras import ops

from zea.backend import checkpoint, highest_matmul_precision
from zea.beamform.lens_correction import compute_lens_path_lengths
from zea import log
from zea.func.ultrasound import directivity, straight_ray_slowness


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
    sos_map=None,
    sos_grid_x=None,
    sos_grid_z=None,
    sos_grid_y=None,
    n_sos_ray_samples=64,
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
        t0_delays (array-like): The transmit delays [s] of shape (n_tx, n_el).
        initial_times (array-like): The initial times [s] of shape (n_tx,).
        element_width (float): The width of the elements [m].
        attenuation_coef (float): The attenuation coefficient [dB/cm/MHz].
        tx_apodizations (array-like): The transmit apodizations of shape (n_tx, n_el).
        t_peak (array-like): The time of the peak of the transmit pulse [s] of shape (n_tx,).
        two_dimensional (bool): Simulate in the imaging plane, as a 1D probe behind an ideal
            elevation lens: the scatterers are moved to the probe's elevation center, there is
            no elevation directivity, and the transmit spreads cylindrically rather than
            spherically. Exclusive with ``elevation_focus``, the lens modelled in 3D, and
            rejects a probe with elevation extent.
        element_height (float): The elevation height of the elements [m], used for the
            elevation directivity. If None, defaults to element_width.
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
        probe_center_frequency (float, optional): center of the transducer band [Hz]. Defaults
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
            the aperture where the lens is thick and lowers the center frequency.
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
        sos_map (array-like, optional): Sound speed map [m/s] of shape (Nz, Nx) in the x-z
            plane, extruded along y, or (Nz, Nx, Ny) with ``sos_grid_y``. Every path from an
            element to a scatterer then runs at the mean slowness along the straight ray between
            them (:func:`zea.func.ultrasound.straight_ray_slowness`), sampled at
            ``n_sos_ray_samples`` points with ``sound_speed`` outside the map; the
            sub-elements of an element share its center ray. The lens, the directivity, the
            spreading and the attenuation keep the geometry of the homogeneous medium at
            ``sound_speed``. None is a homogeneous medium. Differentiable on jax.
        sos_grid_x (array-like, optional): Uniform, ascending x coordinates [m] of the map,
            shape (Nx,).
        sos_grid_z (array-like, optional): Uniform, ascending z coordinates [m] of the map,
            shape (Nz,).
        sos_grid_y (array-like, optional): Uniform, ascending y coordinates [m] of a 3D map,
            shape (Ny,). None for a 2D map.
        n_sos_ray_samples (int): Samples of the map along each ray. Must be static under jit.

    Returns:
        rf_data (array-like): The simulated RF data of shape (n_tx, n_ax, n_el, 1).
    """
    _validate_two_dimensional(two_dimensional, elevation_focus, probe_geometry)
    _validate_sos_map(sos_map, sos_grid_x, sos_grid_z, sos_grid_y)
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
    map_np = _concrete(sos_map)
    concrete = all(x is not None for x in raw) and (sos_map is None or map_np is not None)

    def bound():
        """Samples that hold every echo (see :func:`fft_length`), from the concrete inputs."""
        t0_np, t_init_np, t_peak_np, geom_np, c_np = raw
        shift_np = t0_np - t_init_np[:, None] + t_peak_np[:, None]
        return _fft_bound(
            n_ax,
            fs,
            fc,
            float(c_np),
            geom_np,
            shift_np.min(),
            shift_np.max(),
            n_period,
            _concrete(positions),
            map_np,
        )

    if n_fft is None:
        if not concrete:
            raise ValueError(
                "n_fft cannot be derived from traced geometry, delays, sound speed or sound "
                "speed map; pass n_fft explicitly (see fft_length, zea.ops.Simulate or "
                "zea.Parameters.n_fft)."
            )
        n_fft = smooth_size(bound())
    elif map_np is not None and concrete and int(n_fft) < bound():
        # A map stretches the echoes of a scatterer; a length sized without it may wrap them.
        log.warning(
            f"n_fft ({int(n_fft)}) is shorter than the {bound()} samples the sound speed map "
            "needs to keep every echo from wrapping into the record; see fft_length."
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

    # The straight-ray slowness of every path, once for all frequency blocks.
    slowness = _ray_slowness(
        positions,
        geometry,
        _as_f32(sound_speed),
        sos_map,
        sos_grid_x,
        sos_grid_z,
        sos_grid_y,
        n_sos_ray_samples,
        bool(apply_lens_correction),
        _as_f32(lens_thickness),
        element_normals,
    )

    block = checkpoint(
        functools.partial(
            _rf_block,
            positions=positions,
            magnitudes=magnitudes,
            geometry=geometry,
            slowness=slowness,
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
    slowness,
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
    dropped, so a long path never wraps into the record. ``slowness`` is the mean slowness
    [s, e] of the straight rays through a sound speed map, or None for a homogeneous medium.
    """
    tx_response, rx_response, tau = _element_responses(
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
        rigid_baffle,
        element_normals,
        n_sub_elements,
        elevation_focus,
        lens_attenuation_coef,
        frequency_first=True,
        slowness=slowness,
    )
    keep = _record_keep(tau, ops.min(shift), gate_time)
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
    sos_map=None,
    sos_grid_x=None,
    sos_grid_z=None,
    sos_grid_y=None,
    n_sos_ray_samples=64,
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
    _validate_sos_map(sos_map, sos_grid_x, sos_grid_z, sos_grid_y)
    fc, fs = float(center_frequency), float(sampling_frequency)
    n_period = float(n_period)
    grid_shape = tuple(int(d) for d in ops.shape(grid)[:-1])
    positions = ops.reshape(ops.cast(grid, "float32"), (-1, 3))
    geometry = ops.cast(probe_geometry, "float32")
    if two_dimensional:
        positions = _snap_elevation(positions, geometry)
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
        two_dimensional,
    )
    n_tx, n_el = (int(d) for d in ops.shape(t0_delays))
    n_points = int(ops.shape(positions)[0])
    shift = _transmit_shift(t0_delays, initial_times, t_peak)

    if n_fft is None or n_ax is None:
        raw = [_concrete(x) for x in (positions, geometry, shift, sound_speed)]
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
        arrival = (dist_np / bounds[0])[None] + shift_np[:, None, :]
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

    # workaround for tensorflow. tf.recompute_grad converts every argument to a tensor, crashing
    # when no sound speed map is available (i.e. None).
    block_kwargs = dict(
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

    slowness = _ray_slowness(
        positions,
        geometry,
        _as_f32(sound_speed),
        sos_map,
        sos_grid_x,
        sos_grid_z,
        sos_grid_y,
        n_sos_ray_samples,
        bool(apply_lens_correction),
        _as_f32(lens_thickness),
        element_normals,
    )

    def blocked(points, slow, budget):
        """Band spectrum [f, t, p] or its Parseval energy [t, p] over ``points``."""
        block = checkpoint(
            functools.partial(_pressure_block, positions=points, slowness=slow, **block_kwargs)
        )
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
                part = block(ops.slice(freqs_t, [start], [f_block]))
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
            part = block(ops.slice(freqs_t, [start], [f_block]))
            w = ops.slice(weight, [start], [f_block])[:, None, None]
            return energy + ops.sum(w * (ops.real(part) ** 2 + ops.imag(part) ** 2), axis=0)

        return ops.fori_loop(0, n_blocks, body, ops.zeros((n_tx, n_pts), "float32"))

    budget = max_chunk_gb * 2**30
    if output == "rms":
        energy = blocked(positions, slowness, budget)
        field = ops.sqrt(energy / (n_fft * n_ax))
    else:
        # The band spectrum of a chunk takes half the budget, the blocks the other half.
        chunk = int(max(1, min(n_points, budget // 2 // (8 * n_kept * n_tx))))
        parts = []
        for start in range(0, n_points, chunk):
            slow = None if slowness is None else slowness[start : start + chunk]
            spectrum = blocked(positions[start : start + chunk], slow, budget // 2)
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
    slowness,
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
        rigid_baffle,
        element_normals,
        n_sub_elements,
        elevation_focus,
        lens_attenuation_coef,
        frequency_first=True,
        slowness=slowness,
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


def _record_keep(tau, shift_min, gate_time):
    """The gate of :func:`simulate_rf`: the earliest echo arrives before ``gate_time``.

    ``tau`` is the one-way travel time [s, e] of :func:`_one_way_time`.
    """
    return 2 * ops.min(tau, axis=1) + shift_min < gate_time


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
    sos_map=None,
    sos_grid_x=None,
    sos_grid_z=None,
    sos_grid_y=None,
    n_sos_ray_samples=64,
):
    """Farthest one-way distance [m] from an element at which :func:`simulate_rf` still
    simulates a scatterer.

    A scatterer is simulated while its earliest echo has pulse support inside the record, that
    is while its nearest element is within this distance (see :func:`in_record`). Through a
    lens the distance holds along the element normal, and is high off the normal by a fraction
    of the lens thickness. If a sound speed map is provided, uses the fastest speed in the map.

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
        sos_map (array-like, optional): Sound speed map of :func:`simulate_rf`. Its grids and
            ``n_sos_ray_samples`` are unused, but accepted so the record helpers share arguments.

    Returns:
        float: The reach [m].
    """
    # unused, but accepted so all record helpers share arguments
    del probe_geometry, two_dimensional, sos_grid_x, sos_grid_z, sos_grid_y, n_sos_ray_samples
    fs, fc = float(sampling_frequency), float(center_frequency)
    raw = [_concrete(x) for x in (t0_delays, initial_times, t_peak)]
    minmax = _sound_speed_minmax(sound_speed, sos_map)
    if any(x is None for x in raw) or minmax is None:
        raise ValueError(
            "record_reach needs concrete delays, sound speed and map; under jit use in_record."
        )
    c_max = minmax[1]
    t0_np, t_init_np, t_peak_np = (np.asarray(x, np.float64) for x in raw)
    shift = t0_np - t_init_np[:, None] + t_peak_np[:, None]
    time = (_record_gate_time(int(n_ax), fs, fc, float(n_period)) - float(shift.min())) / 2
    if not apply_lens_correction or lens_sound_speed is None:
        return c_max * time
    thickness, c_lens = float(lens_thickness), float(lens_sound_speed)
    return thickness + c_max * (time - thickness / c_lens)


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
    sos_map=None,
    sos_grid_x=None,
    sos_grid_z=None,
    sos_grid_y=None,
    n_sos_ray_samples=64,
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
        sos_map=sos_map,
    )
    del sos_grid_x, sos_grid_z, sos_grid_y, n_sos_ray_samples
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
    sos_map=None,
    sos_grid_x=None,
    sos_grid_z=None,
    sos_grid_y=None,
    n_sos_ray_samples=64,
    element_normals=None,
):
    """Whether :func:`simulate_rf` simulates a scatterer at each point: its gate.

    A scatterer is dropped when even its earliest echo has no pulse support inside the record.
    Takes the arguments of :func:`record_reach`; ``two_dimensional`` moves the points into the
    imaging plane first, as the simulator does, and a sound speed map times the paths along
    their straight rays (``element_normals`` places the lens face for those rays). Jittable.

    Args:
        points (array-like): Positions [m] of shape (n_points, 3).
        element_normals (array-like, optional): Element normals of :func:`simulate_rf`, used
            only with a lens and a sound speed map to start the rays at the lens face.

    Returns:
        array-like: Boolean mask of shape (n_points,).
    """
    _validate_sos_map(sos_map, sos_grid_x, sos_grid_z, sos_grid_y)
    positions = ops.cast(points, "float32")
    geometry = ops.cast(probe_geometry, "float32")
    if two_dimensional:
        positions = _snap_elevation(positions, geometry)
    shift = _transmit_shift(t0_delays, initial_times, t_peak)
    slowness = _ray_slowness(
        positions,
        geometry,
        _as_f32(sound_speed),
        sos_map,
        sos_grid_x,
        sos_grid_z,
        sos_grid_y,
        n_sos_ray_samples,
        bool(apply_lens_correction),
        _as_f32(lens_thickness),
        element_normals,
    )
    tau = _one_way_time(
        positions,
        geometry,
        bool(apply_lens_correction),
        _as_f32(lens_thickness),
        _as_f32(lens_sound_speed),
        _as_f32(sound_speed),
        slowness,
    )
    gate_time = _record_gate_time(
        int(n_ax), float(sampling_frequency), float(center_frequency), float(n_period)
    )
    return _record_keep(tau, ops.min(shift), gate_time)


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
    rigid_baffle,
    element_normals,
    n_sub_elements=(1, 1),
    elevation_focus=None,
    lens_attenuation_coef=0.0,
    frequency_first=False,
    slowness=None,
):
    """Transmit and receive one-way responses and the one-way travel time [s, e].

    The responses are [s, e, f], or [f, s, e] when ``frequency_first`` is True. Each element is
    the mean of ``n_sub_elements`` (lateral, elevation) sub-elements with their
    own distance, phase and sinc directivity, so the response holds in the near field too. An
    elevation focus is the ideal focusing advance of each elevation sub-element, or with the lens
    the refracted (Fermat) path through the local lens thickness, which the focus thins towards
    the edges. The lens path is expressed as the medium distance with the same travel time for the
    phase, and spreads as the refracted ray tube (:func:`_lens_spread_distance`); the lens part is
    attenuated with ``lens_attenuation_coef``. The returned travel time is the element center's.

    ``slowness`` is the mean slowness [s, e] of the straight rays through a sound speed map
    (:func:`_ray_slowness`), or None for ``1 / sound_speed``. It times the medium leg of every
    sub-element's path (the sub-elements are within an element width of the center ray, so they
    share its slowness to first order); the lens leg, the directivity, the spreading and the
    attenuation keep the homogeneous geometry.

    In 2D the positions are expected in the imaging plane (:func:`_snap_elevation`): there is no
    elevation directivity, and the transmit spreads cylindrically, as behind an ideal lens.
    """
    n_lateral, n_elevation = n_sub_elements
    n_sub = n_lateral * n_elevation
    relative_center = positions[:, None] - geometry[None]
    dtype = relative_center.dtype
    lateral_axis, elevation_axis, _ = frame = _element_frame(element_normals, dtype)
    tau = _one_way_time(
        positions,
        geometry,
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        slowness,
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

    def medium_time(length):
        """Travel time [s, e] over a medium leg: at the ray's slowness, or at ``1 / c``."""
        return length / sound_speed if slowness is None else length * slowness

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
            sub_time = lens_len / lens_sound_speed + medium_time(medium_len)
            spread_dist = _lens_spread_distance(
                lens_len, medium_len, thickness[j], sound_speed, lens_sound_speed
            )
            amplitude = amplitude * attenuate(f3, lens_attenuation_coef, fx(lens_len))
        else:
            medium_len = spread_dist = ops.linalg.norm(relative, axis=-1)
            sub_time = medium_time(medium_len)
        amplitude = amplitude * attenuate(f3, attenuation_coef, fx(medium_len))
        if not rigid_baffle:
            amplitude = amplitude * fx(obliquity)
        phase = ops.exp(
            ops.array(-2j * np.pi, "complex64")
            * ops.cast((fx(sub_time) - advance[j]) * f3, "complex64")
        )
        rx = ops.cast(amplitude * spread(fx(spread_dist), 1.0), "complex64") * phase
        if two_dimensional:
            # An ideal elevation lens: cylindrical spread on the way out, spherical back.
            tx = ops.cast(amplitude * spread(fx(spread_dist), 0.5), "complex64") * phase
        else:
            tx = rx
        return tx, rx

    if n_sub == 1:
        tx, rx = response(0)
        return tx, rx, tau

    def body(j, carry):
        tx, rx = response(j)
        return carry[0] + tx, carry[1] + rx

    zeros = ops.zeros(ops.shape(response(0)[0]), "complex64")
    tx, rx = ops.fori_loop(0, n_sub, body, (zeros, zeros))
    scale = ops.array(1.0 / n_sub, "complex64")
    return tx * scale, rx * scale, tau


def _one_way_time(
    positions,
    geometry,
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    sound_speed,
    slowness=None,
):
    """One-way travel time [s] from each position to each element center.

    ``slowness`` is the mean slowness of each straight ray (:func:`_ray_slowness`), or None for
    ``1 / sound_speed``.
    """
    if not apply_lens_correction:
        length = ops.linalg.norm(positions[:, None] - geometry[None], axis=-1)
        return length / sound_speed if slowness is None else length * slowness
    lens_len, medium_len = compute_lens_path_lengths(
        geometry,
        positions,
        lens_thickness=lens_thickness,
        c_lens=lens_sound_speed,
        c_medium=sound_speed,
        n_iter=3,
    )
    medium_time = medium_len / sound_speed if slowness is None else medium_len * slowness
    return lens_len / lens_sound_speed + medium_time


def _ray_slowness(
    positions,
    geometry,
    sound_speed,
    sos_map,
    sos_grid_x,
    sos_grid_z,
    sos_grid_y,
    n_samples,
    apply_lens_correction=False,
    lens_thickness=0.0,
    element_normals=None,
):
    """Mean slowness [s, e] of the straight rays from the elements to the positions, or None
    without a map.

    Through a lens the rays start at the lens face, ``lens_thickness`` along the element normal,
    as the lens part is timed separately (:func:`_one_way_time`).
    """
    if sos_map is None:
        return None
    start = geometry
    if apply_lens_correction:
        normal = _element_frame(element_normals, geometry.dtype)[2]
        start = geometry + ops.cast(lens_thickness, geometry.dtype) * normal
    return straight_ray_slowness(
        positions,
        start,
        sos_map,
        sos_grid_x,
        sos_grid_z,
        sound_speed,
        sos_grid_y=sos_grid_y,
        n_samples=int(n_samples),
    )


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

    A slower lens is thickest at the center, a faster one thinnest (negative sag).
    """
    focus = ops.cast(elevation_focus, v.dtype)
    path = ops.sqrt(focus**2 + v**2) - focus
    return path * lens_sound_speed / (sound_speed - lens_sound_speed)


def _lens_spread_distance(lens_len, medium_len, thickness, sound_speed, lens_sound_speed):
    """Distance whose 1/r spreading is the ray-tube divergence of the path refracted at the face.

    The phase path scales the lens part by c / c_lens, but the wave leaves the face as if from a
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
    """The positions moved into the imaging plane, the probe's elevation center."""
    center = ops.mean(geometry[:, 1])
    return ops.stack(
        [positions[:, 0], ops.zeros_like(positions[:, 0]) + center, positions[:, 2]], axis=-1
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
    """Spectrum of a Hann-windowed linear chirp centerd at t=0, on the rfft grid of ``n_fft``.

    The window spans ``n_period`` periods of ``center_frequency``, over which the instantaneous
    frequency sweeps linearly from ``center_frequency - chirp_sweep / 2`` to
    ``center_frequency + chirp_sweep / 2``. Scaled like :func:`get_pulse_spectrum_fn`: the
    waveform recovered with ``irfft`` has a unit peak. The waveform is even, so the spectrum is
    real, and with ``chirp_sweep=0`` it is the sampled counterpart of the windowed tone.

    Args:
        n_fft (int): FFT length; the waveform is sampled on its wrapped time grid.
        center_frequency (float): center frequency [Hz].
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
        probe_center_frequency (float, optional): center of the band [Hz]. ``center_frequency``
            when None.
        bandwidth_percent (float, optional): -6 dB fractional bandwidth in percent. None is a
            flat response.
        center_frequency (float, optional): Fallback band center [Hz].
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
        raise ValueError("bandwidth_percent needs probe_center_frequency or center_frequency.")
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
    sos_map=None,
):
    """Smooth FFT length whose echoes never wrap into the first ``n_ax`` samples.

    A kept scatterer has its earliest echo inside the record, so its last one is at most the
    aperture round trip, the spread of the transmit shifts and one pulse later. When the
    positions are given the bound from the farthest scatterer is used if smaller. When using a sound
    speed map, uses the worst case based on the min/max speeds in the map.

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
        sos_map (array-like, optional): Concrete sound speed map [m/s] of :func:`simulate_rf`.

    Returns:
        int: FFT length, a product of powers of 2, 3 and 5.
    """
    return smooth_size(
        _fft_bound(
            n_ax,
            sampling_frequency,
            center_frequency,
            sound_speed,
            probe_geometry,
            shift_min,
            shift_max,
            n_period,
            scatterer_positions,
            sos_map,
        )
    )


def _fft_bound(
    n_ax,
    sampling_frequency,
    center_frequency,
    sound_speed,
    probe_geometry,
    shift_min,
    shift_max,
    n_period=4.0,
    scatterer_positions=None,
    sos_map=None,
):
    """Samples that hold every kept echo, before rounding: see :func:`fft_length`."""
    n_ax, fs, fc = int(n_ax), float(sampling_frequency), float(center_frequency)
    n_period, shift_min, shift_max = float(n_period), float(shift_min), float(shift_max)
    c_min, c_max = _sound_speed_minmax(sound_speed, sos_map)
    geometry = np.asarray(probe_geometry, np.float64)
    pulse = 2 * n_period / fc
    aperture = 2 * np.linalg.norm(geometry - geometry.mean(0), axis=1).max()
    # A kept scatterer is within c_max * (gate - shift_min) / 2 of its nearest element, and
    # that path may run at c_max while its farthest runs at c_min.
    gate = _record_gate_time(n_ax, fs, fc, n_period)
    spread = (c_max / c_min - 1) * max(gate - shift_min, 0.0)
    n = n_ax + int(np.ceil((2 * aperture / c_min + spread + shift_max - shift_min + pulse) * fs))
    if scatterer_positions is not None and len(scatterer_positions):
        reach = np.linalg.norm(np.asarray(scatterer_positions, np.float64), axis=1).max()
        reach = reach + np.linalg.norm(geometry, axis=1).max()
        bound = int(np.ceil((2 * reach / c_min + max(shift_max, 0.0) + pulse) * fs))
        n = min(n, max(n_ax, bound))
    return n


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


def _validate_sos_map(sos_map, sos_grid_x, sos_grid_z, sos_grid_y=None):
    """Static checks of a sound speed map: a map with its x and z grid, y for a 3D map, of
    matching shapes and, when concrete, uniform ascending grids and positive speeds."""
    grids = (("sos_grid_x", sos_grid_x), ("sos_grid_z", sos_grid_z), ("sos_grid_y", sos_grid_y))
    if sos_map is None:
        given = [name for name, grid in grids if grid is not None]
        if given:
            raise ValueError(f"{', '.join(given)} given without sos_map.")
        return
    if sos_grid_x is None or sos_grid_z is None:
        raise ValueError("sos_map needs its coordinates sos_grid_x and sos_grid_z.")
    grids = [("sos_grid_z", sos_grid_z), ("sos_grid_x", sos_grid_x)]
    if sos_grid_y is not None:
        grids.append(("sos_grid_y", sos_grid_y))
    shape = tuple(int(d) for d in ops.shape(sos_map))
    expected = tuple(int(ops.shape(grid)[0]) for _, grid in grids)
    if shape != expected:
        raise ValueError(
            f"sos_map of shape {shape} does not match its grids: expected (Nz, Nx) for a 2D "
            f"map or (Nz, Nx, Ny) with sos_grid_y for a 3D map, here {expected}."
        )
    for name, grid in grids:
        values = _concrete(grid)
        if values is None:
            continue
        steps = np.diff(np.asarray(values, np.float64))
        if len(values) < 2 or steps.min() <= 0:
            raise ValueError(f"{name} must be ascending with at least two points.")
        if not np.allclose(steps, steps[0], rtol=1e-3):
            raise ValueError(f"{name} must be uniformly spaced.")
    values = _concrete(sos_map)
    if values is not None and (not np.all(np.isfinite(values)) or np.any(values <= 0)):
        raise ValueError("sos_map must hold finite, positive sound speeds.")


def _sound_speed_minmax(sound_speed, sos_map=None):
    """Slowest and fastest speed of the medium as floats, or None when either is traced."""
    c = _concrete(sound_speed)
    if c is None:
        return None
    c = float(c)
    if sos_map is None:
        return c, c
    values = _concrete(sos_map)
    if values is None:
        return None
    return min(c, float(values.min())), max(c, float(values.max()))


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
            f"lens needs at least {sag:.2e} m at the center."
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
