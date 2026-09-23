"""The RF record of the simulators: the transmit shifts that place the echoes in it, the gate on
which scatterers it can hold, the FFT length and band it is synthesised on, and the scatter
exponent that shapes that band."""

import numpy as np
from keras import ops

from zea.internal.core import concrete, ndim, round_up_to_power_of_two
from zea.simulator.element import (
    _as_f32,
    _element_frame,
    _one_way_time,
    _ray_slowness,
    _ray_starts,
    _snap_elevation,
    _validate_maps,
)
from zea.simulator.pulse import _pulse_span, _pulse_tail, _unique_pulses, transmit_pulses


def _transmit_shift(t0_delays, initial_times, t_peak):
    """Transmit shift [s] per element beyond the travel time, shaped like ``t0_delays``."""
    t0 = ops.cast(t0_delays, "float32")
    per_tx = ops.cast(t_peak, "float32") - ops.cast(initial_times, "float32")
    return t0 + ops.reshape(per_tx, (-1,) + (1,) * (len(ops.shape(t0)) - 1))


def _shift_np(t0_delays, initial_times, t_peak):
    """:func:`_transmit_shift` on concrete inputs, in float64."""
    t0 = np.asarray(t0_delays, np.float64)
    per_tx = np.asarray(t_peak, np.float64) - np.asarray(initial_times, np.float64)
    return t0 + per_tx.reshape((-1,) + (1,) * (t0.ndim - 1))


# ---------------------------------------------------------------------------------------------
# Record gate
# Which scatterers a record can hold: shared by the synthesis and the public helpers.
# ---------------------------------------------------------------------------------------------


def _record_gate_time(n_ax, sampling_frequency, pulse_tail):
    """Latest arrival [s] of an echo peak with pulse support inside the record, for a pulse
    with ``pulse_tail`` [s] of support after its peak (:func:`_pulse_tail`)."""
    return n_ax / sampling_frequency + pulse_tail


def _record_keep(tau, shift_min, gate_time):
    """The gate of :func:`simulate_rf`: the earliest echo arrives before ``gate_time``.

    ``tau`` is the one-way travel time [s, e] of :func:`_one_way_time`.
    """
    return 2 * ops.min(tau, axis=1) + shift_min < gate_time


def record_reach(
    sound_speed,
    n_ax,
    sampling_frequency,
    center_frequency,
    t0_delays,
    initial_times,
    t_peak,
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
    apply_lens_correction=False,
    lens_thickness=0.0,
    lens_sound_speed=None,
    sos_map=None,
):
    """Farthest one-way distance [m] from an element at which :func:`simulate_rf` still
    simulates a scatterer.

    A scatterer is simulated while its earliest echo has pulse support inside the record, that
    is while its nearest element is within this distance (see :func:`in_record`). Through a
    lens the distance holds along the element normal, and is high off the normal by a fraction
    of the lens thickness. If a sound speed map is provided, uses the fastest speed in the map.

    Args:
        sound_speed (float): Speed of sound [m/s].
        n_ax (int): Number of axial samples in the record.
        sampling_frequency (float): Sampling frequency [Hz].
        center_frequency (float): Pulse center frequency [Hz].
        t0_delays (array-like): Transmit delays [s] of shape (n_tx, n_el) or (n_tx, n_mpt, n_el).
        initial_times (array-like): Record start times [s] of shape (n_tx,).
        t_peak (array-like): Pulse peak times [s] of shape (n_tx,).
        waveforms_two_way (array-like, optional): The transmit waveforms of
            :func:`simulate_rf`; None is its default pulse. The pulse support after the peak
            sets how far past the record an echo peak may arrive and still be simulated.
        waveform_sampling_frequency (float): Sampling frequency [Hz] of ``waveforms_two_way``.
        apply_lens_correction (bool): Whether the simulation models the lens.
        lens_thickness (float): Lens thickness [m].
        lens_sound_speed (float, optional): Speed of sound in the lens [m/s].
        sos_map (array-like, optional): Sound speed map of :func:`simulate_rf`; only its
            fastest speed matters here.

    Returns:
        float: The reach [m].
    """
    fs, fc = float(sampling_frequency), float(center_frequency)
    raw = [concrete(x) for x in (t0_delays, initial_times, t_peak)]
    minmax = _sound_speed_minmax(sound_speed, sos_map)
    if any(x is None for x in raw) or minmax is None:
        raise ValueError(
            "record_reach needs concrete delays, sound speed and map; under jit use in_record."
        )
    c_max = minmax[1]
    shift = _shift_np(*raw)
    pulses = transmit_pulses(None, fc, fs, waveforms_two_way, waveform_sampling_frequency)
    gate_time = _record_gate_time(int(n_ax), fs, _pulse_tail(pulses))
    time = (gate_time - float(shift.min())) / 2
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
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
    apply_lens_correction=False,
    lens_thickness=0.0,
    lens_sound_speed=None,
    two_dimensional=False,
    sos_map=None,
):
    """Box [m] in front of the probe outside which :func:`simulate_rf` simulates no scatterer.

    The bounding box of the elements grown by :func:`record_reach`, starting in z at the
    shallowest element and, in 2D, collapsed onto the imaging plane. A phantom drawn inside it
    wastes no scatterers on the gate; :func:`in_record` gives the exact gate. Takes the
    arguments of :func:`record_reach`, plus:

    Args:
        probe_geometry (array-like): Element positions [m] of shape (n_el, 3).
        two_dimensional (bool): Collapse the box onto the imaging plane.

    Returns:
        ndarray: The box as ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``, shape (2, 3).
    """
    reach = record_reach(
        sound_speed,
        n_ax,
        sampling_frequency,
        center_frequency,
        t0_delays,
        initial_times,
        t_peak,
        waveforms_two_way=waveforms_two_way,
        waveform_sampling_frequency=waveform_sampling_frequency,
        apply_lens_correction=apply_lens_correction,
        lens_thickness=lens_thickness,
        lens_sound_speed=lens_sound_speed,
        sos_map=sos_map,
    )
    geometry = concrete(probe_geometry)
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
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
    apply_lens_correction=False,
    lens_thickness=0.0,
    lens_sound_speed=None,
    two_dimensional=False,
    sos_map=None,
    map_grid_x=None,
    map_grid_z=None,
    map_grid_y=None,
    n_sos_ray_samples=64,
    element_normals=None,
):
    """Whether :func:`simulate_rf` simulates a scatterer at each point: its gate.

    A scatterer is dropped when even its earliest echo has no pulse support inside the record.
    Takes the arguments of :func:`record_reach`; ``two_dimensional`` moves the points into the
    imaging plane first, as the simulator does, and a sound speed map with its grids times the
    paths along their straight rays. Jittable.

    Args:
        points (array-like): Positions [m] of shape (n_points, 3).
        probe_geometry (array-like): Element positions [m] of shape (n_el, 3).
        two_dimensional (bool): Gate the points projected onto the imaging plane.
        sos_map, map_grid_x, map_grid_z, map_grid_y, n_sos_ray_samples: The sound speed map
            of :func:`simulate_rf` with its grids and ray sampling.
        element_normals (array-like, optional): Element normals of :func:`simulate_rf`. With
            a lens they orient its face: the lens path follows them, and the rays through a map
            start at the face.

    Returns:
        array-like: Boolean mask of shape (n_points,).
    """
    _validate_maps(sos_map, None, map_grid_x, map_grid_z, map_grid_y)
    positions = ops.cast(points, "float32")
    geometry = ops.cast(probe_geometry, "float32")
    if two_dimensional:
        positions = _snap_elevation(positions, geometry)
    apply_lens_correction = bool(apply_lens_correction)
    shift = _transmit_shift(t0_delays, initial_times, t_peak)
    slowness = _ray_slowness(
        positions,
        _ray_starts(geometry, apply_lens_correction, lens_thickness, element_normals),
        _as_f32(sound_speed),
        sos_map,
        map_grid_x,
        map_grid_z,
        map_grid_y,
        n_sos_ray_samples,
    )
    lens_normals = None if element_normals is None else _element_frame(element_normals).normal
    tau = _one_way_time(
        positions,
        geometry,
        apply_lens_correction,
        _as_f32(lens_thickness),
        _as_f32(lens_sound_speed),
        _as_f32(sound_speed),
        slowness,
        lens_normals,
    )
    pulses = transmit_pulses(
        None,
        float(center_frequency),
        float(sampling_frequency),
        waveforms_two_way,
        waveform_sampling_frequency,
    )
    gate_time = _record_gate_time(int(n_ax), float(sampling_frequency), _pulse_tail(pulses))
    return _record_keep(tau, ops.min(shift), gate_time)


# ---------------------------------------------------------------------------------------------
# FFT length and band
# ---------------------------------------------------------------------------------------------


def band_bins(
    n_fft,
    sampling_frequency,
    pulses,
    center_frequency,
    scatter_exponent,
    band_db,
    scatter_exponent_range=None,
):
    """Contiguous fft bin range that is not discarded.

    The pulse spectrum (the largest over the transmits) and the scattering gain together exceed
    ``band_db`` there. If ``scatter_exponent`` is a vector of per-scatterer exponents, the band
    is calculated from the union of the min and max exponents.
    """
    freqs = np.fft.rfftfreq(n_fft, 1 / sampling_frequency)
    if band_db is None:
        return 0, len(freqs)
    w = np.max([np.abs(pulse.spectrum(freqs)) for pulse in _unique_pulses(pulses)], axis=0)
    lo, hi = _exponent_range(scatter_exponent, scatter_exponent_range)
    k0, k1 = len(freqs), 0
    for exponent in (lo,) if lo == hi else (lo, hi):
        band = w * (freqs / center_frequency) ** exponent
        keep = np.flatnonzero(band > band.max() * 10 ** (band_db / 20))
        k0, k1 = min(k0, int(keep[0])), max(k1, int(keep[-1]) + 1)
    return k0, k1


def smooth_size(n):
    """Smallest 2^a 3^b 5^c >= n."""
    best = round_up_to_power_of_two(max(n, 1))
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
    waveforms_two_way=None,
    waveform_sampling_frequency=250e6,
    scatterer_positions=None,
    sos_map=None,
):
    """Smooth FFT length whose echoes never wrap into the first ``n_ax`` samples.

    A kept scatterer has its earliest echo inside the record, so its last one is at most the
    aperture round trip, the spread of the transmit shifts and one pulse later. When the
    positions are given the bound from the farthest scatterer is used if smaller. When using a
    sound speed map, uses the worst case based on the min/max speeds in the map.

    Args:
        n_ax (int): Number of axial samples in the record.
        sampling_frequency (float): Sampling frequency in Hz.
        center_frequency (float): Pulse center frequency in Hz.
        sound_speed (float): Speed of sound in m/s.
        probe_geometry (array-like): Element positions of shape (n_el, 3).
        shift_min (float): Smallest transmit shift (``t0_delays - initial_times + t_peak``).
        shift_max (float): Largest transmit shift.
        waveforms_two_way (array-like, optional): The transmit waveforms of
            :func:`simulate_rf`; None is its default pulse.
        waveform_sampling_frequency (float): Sampling frequency [Hz] of ``waveforms_two_way``.
        scatterer_positions (array-like, optional): Concrete positions of shape (n_scat, 3).
        sos_map (array-like, optional): Concrete sound speed map [m/s] of :func:`simulate_rf`.

    Returns:
        int: FFT length, a product of powers of 2, 3 and 5.
    """
    pulses = transmit_pulses(
        None,
        float(center_frequency),
        float(sampling_frequency),
        waveforms_two_way,
        waveform_sampling_frequency,
    )
    return smooth_size(
        _fft_bound(
            n_ax,
            sampling_frequency,
            sound_speed,
            probe_geometry,
            shift_min,
            shift_max,
            pulses,
            scatterer_positions,
            sos_map,
        )
    )


def _fft_bound(
    n_ax,
    sampling_frequency,
    sound_speed,
    probe_geometry,
    shift_min,
    shift_max,
    pulses,
    scatterer_positions=None,
    sos_map=None,
):
    """Samples that hold every kept echo, before rounding: see :func:`fft_length`."""
    n_ax, fs = int(n_ax), float(sampling_frequency)
    shift_min, shift_max = float(shift_min), float(shift_max)
    c_min, c_max = _sound_speed_minmax(sound_speed, sos_map)
    geometry = np.asarray(probe_geometry, np.float64)
    pulse = 2 * _pulse_span(pulses)
    aperture = 2 * np.linalg.norm(geometry - geometry.mean(0), axis=1).max()
    # A kept scatterer is within c_max * (gate - shift_min) / 2 of its nearest element, and
    # that path may run at c_max while its farthest runs at c_min.
    gate = _record_gate_time(n_ax, fs, _pulse_tail(pulses))
    spread = (c_max / c_min - 1) * max(gate - shift_min, 0.0)
    n = n_ax + int(np.ceil((2 * aperture / c_min + spread + shift_max - shift_min + pulse) * fs))
    if scatterer_positions is not None and len(scatterer_positions):
        reach = np.linalg.norm(np.asarray(scatterer_positions, np.float64), axis=1).max()
        reach = reach + np.linalg.norm(geometry, axis=1).max()
        bound = int(np.ceil((2 * reach / c_min + max(shift_max, 0.0) + pulse) * fs))
        n = min(n, max(n_ax, bound))
    return n


def _sound_speed_minmax(sound_speed, sos_map=None):
    """Slowest and fastest speed of the medium as floats, or None when either is traced."""
    c = concrete(sound_speed)
    if c is None:
        return None
    c = float(c)
    if sos_map is None:
        return c, c
    values = concrete(sos_map)
    if values is None:
        return None
    return min(c, float(values.min())), max(c, float(values.max()))


# ---------------------------------------------------------------------------------------------
# Scatter exponent
# The frequency weighting of the scatterers, ``(f / fc) ** exponent``, shared or per scatterer.
# ---------------------------------------------------------------------------------------------


def _validate_scatter_exponent(scatter_exponent, n_scat=None):
    """Reject invalid exponents (must be scalar or [n_scat], and finite nonnegative)."""
    rank = ndim(scatter_exponent)
    if rank > 1:
        raise ValueError(
            "scatter_exponent must be a scalar or a vector of one exponent per scatterer, "
            f"got {rank} dimensions."
        )
    if rank == 1 and n_scat is not None:
        n_given = int(np.shape(scatter_exponent)[0])
        if n_given != n_scat:
            raise ValueError(
                f"A per-scatterer scatter_exponent needs one value per scatterer: got "
                f"{n_given} exponents for {n_scat} scatterers."
            )
    values = concrete(scatter_exponent)
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
    if ndim(scatter_exponent) == 0:
        value = concrete(scatter_exponent)
        if value is not None:
            return float(value) or None
    return ops.cast(scatter_exponent, "float32")


def scatter_exponent_bounds(scatter_exponent):
    """Smallest and largest exponent, or None when ``scatter_exponent`` is traced.

    Both edges of the band move monotonically with the exponent, so the union of the bands of
    the min and max exponents covers every scatterer. :func:`simulate_rf` derives them from a
    concrete exponent; a caller that traces the exponent should pass the pair as
    ``scatter_exponent_range``.
    """
    if scatter_exponent is None:
        return 0.0, 0.0
    values = concrete(scatter_exponent)
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
