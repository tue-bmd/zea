"""Pixel grid calculation for ultrasound beamforming."""

import numpy as np
from keras import ops

from zea import log
from zea.beamform.geometry import compute_element_normals, is_plane_wave

eps = 1e-10


def _steering_direction(polar: float, azimuth: float) -> np.ndarray:
    r"""Unit steering vector :math:`(\sin\alpha\cos\beta, \sin\alpha\sin\beta, \cos\alpha)`."""
    polar, azimuth = float(polar), float(azimuth)
    return np.array(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)]
    )


def _focused_ray_direction_bounds(focal_point, element_positions) -> np.ndarray:
    """Bounding box of the ray directions from the elements to a focal point.

    Each element sends a ray towards the focus, or away from the virtual source of
    a diverging wave (the direction is flipped behind the probe). Each ray direction
    is a unit vector. For each axis, this returns the lowest and highest value that
    coordinate takes over all rays.

    Args:
        focal_point: Focal point or virtual source, shape ``(3,)``.
        element_positions: Positions of the active elements, shape ``(n_active, 3)``.

    Returns:
        np.ndarray: Shape ``(3, 2)``, one ``(lo, hi)`` row per axis x, y, z.
    """
    rays = focal_point - element_positions
    rays *= np.where(rays[:, 2:] < 0, -1.0, 1.0) / np.maximum(
        np.linalg.norm(rays, axis=1, keepdims=True), eps
    )
    lo, hi = rays.min(axis=0), rays.max(axis=0)
    # The rays fill the aperture, so the ray nearest the axis is on axis wherever
    # the x and y bounds straddle zero.
    nearest = np.where(lo * hi <= 0, 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    hi[2] = np.sqrt(max(1.0 - nearest[0] ** 2 - nearest[1] ** 2, 0.0))
    return np.stack([lo, hi], axis=-1)


def transmit_direction_bounds(
    focus_distances,
    probe_geometry,
    tx_apodizations,
    polar_angles,
    transmit_origins,
    azimuth_angles=None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Bounding box of the ray directions of each transmit.

    For focused and diverging transmits, each active element (non-zero
    ``tx_apodizations``) sends a ray through the focal point or virtual source,
    see :func:`_focused_ray_direction_bounds`. A plane wave travels in one
    direction, its steering direction, so its box is a single point.

    Args:
        focus_distances: Focus distance per transmit, shape ``(n_tx,)``. Negative for
            diverging waves; see :func:`zea.beamform.geometry.is_plane_wave` for plane waves.
        probe_geometry: Element positions, shape ``(n_el, 3)``.
        tx_apodizations: Transmit apodization per transmit, shape ``(n_tx, n_el)``.
        polar_angles: Steering angle per transmit in radians, shape ``(n_tx,)``.
            ``None`` means unsteered.
        transmit_origins: Origin of each transmit, shape ``(n_tx, 3)``.
        azimuth_angles: Steering azimuth per transmit in radians, shape ``(n_tx,)``.
            ``None`` means steering in the x-z plane.

    Returns:
        tuple: ``(bounds, plane)``. ``bounds`` has shape ``(n_tx, 3, 2)``: per
        transmit, one ``(lo, hi)`` row per axis x, y, z of the unit ray directions. ``plane``
        marks the plane waves. Transmits without active elements or with a ``NaN``
        focus are skipped. ``(None, None)`` if no usable transmit remains.
    """
    focus_distances = np.asarray(focus_distances, dtype=float)
    probe_geometry = np.asarray(probe_geometry, dtype=float)
    tx_apodizations = np.asarray(tx_apodizations, dtype=float)

    if focus_distances.size == 0 or tx_apodizations.ndim != 2:
        return None, None

    if polar_angles is None:
        polar_angles = np.zeros_like(focus_distances)
    polar_angles = np.asarray(polar_angles, dtype=float)
    transmit_origins = np.asarray(transmit_origins, dtype=float)
    if azimuth_angles is None:
        azimuth_angles = np.zeros_like(polar_angles)
    azimuth_angles = np.asarray(azimuth_angles, dtype=float)

    plane = is_plane_wave(focus_distances)
    bounds, kept_plane = [], []
    for tx, focus in enumerate(focus_distances):
        active = np.nonzero(np.abs(tx_apodizations[tx]) > eps)[0]
        if active.size == 0 or np.isnan(focus):
            continue
        steering = _steering_direction(polar_angles[tx], azimuth_angles[tx])
        if plane[tx]:
            bounds.append(np.stack([steering, steering], axis=-1))
        else:
            focal_point = transmit_origins[tx] + float(focus) * steering
            bounds.append(_focused_ray_direction_bounds(focal_point, probe_geometry[active]))
        kept_plane.append(bool(plane[tx]))

    if not bounds:
        return None, None
    return np.array(bounds), np.array(kept_plane)


# A rectangular element of width ``w`` has the far-field directivity
# ``sinc(w sin(theta) / lambda)``, which falls to -6 dB at ``w sin(theta) / lambda``
# equal to this constant (``np.sinc`` convention, ``sinc(x) = sin(pi x) / (pi x)``).
# The soft-baffle model of Selfridge et al. (Appl. Phys. Lett. 37, 35, 1980) adds a
# ``cos(theta)`` factor, which only narrows the pattern, so leaving it out is conservative.
DIRECTIVITY_6DB = 0.603


def _max_transverse_spatial_frequency(t, sin_theta_r, f_min, f_max, cap_t, cap_r, sound_speed):
    """Highest spatial frequency ``f * (t + r) / c`` along x or y, over the band.

    ``r`` ranges up to ``sin_theta_r``. Each path's term ``f * |sin(theta)|`` is
    capped (in Hz) by element directivity. The result is piecewise linear in ``f``
    with kinks only where a term reaches its cap, so evaluating the band edges and
    those kinks is exact. Negate ``t`` and the result to get the lowest instead.

    Args:
        t: x (or y) of the transmit direction per transmit, shape ``(n_tx,)``.
        sin_theta_r: Largest receive angle sine.
        f_min, f_max: The band in Hz.
        cap_t: Directivity cap of the transmit term per transmit, shape ``(n_tx,)``.
        cap_r: Directivity cap of the receive term.
        sound_speed: Speed of sound in m/s.

    Returns:
        float: The highest spatial frequency in cycles per metre.
    """
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kink_t = np.clip(np.nan_to_num(cap_t / np.abs(t), nan=f_max, posinf=f_max), f_min, f_max)
        kink_r = np.clip(cap_r / sin_theta_r, f_min, f_max) if sin_theta_r > 0 else f_min
    f = np.stack(np.broadcast_arrays(f_min, f_max, kink_t, kink_r))
    k = np.sign(t) * np.minimum(f * np.abs(t), cap_t) + np.minimum(f * sin_theta_r, cap_r)
    return float(k.max()) / sound_speed


def _frequency_band(center_frequency, demodulation_frequency, bandwidth) -> tuple[float, float]:
    """The RF band ``(f_min, f_max)`` in Hz of the data reaching the grid.

    Centred on ``demodulation_frequency``, or on ``center_frequency`` when that is
    unset or 0 (either can be 0, depending on where the data was demodulated).
    Per-transmit carriers widen the band. An unknown ``bandwidth`` collapses the
    band to its centre.
    """
    if demodulation_frequency is None:
        demodulation_frequency = 0.0
    band_center = np.ravel(np.asarray(demodulation_frequency, dtype=float))
    if np.all(band_center <= 0):
        band_center = np.ravel(np.asarray(center_frequency, dtype=float))
    half_band = float(bandwidth) / 2 if bandwidth is not None else 0.0
    f_min = max(float(band_center.min()) - half_band, 0.0)
    f_max = float(band_center.max()) + half_band
    return f_min, f_max


def _max_element_tilt(probe_geometry) -> float:
    """Largest angle in radians between an element normal and ``+z``; ``0`` for a flat array.

    Uses the same normals as the receive f-number mask, see
    :func:`zea.beamform.geometry.compute_element_normals`.
    """
    probe_geometry = np.asarray(probe_geometry, dtype=float)
    if probe_geometry.ndim != 2 or probe_geometry.shape[0] < 2 or probe_geometry.shape[1] != 3:
        return 0.0
    normals = ops.convert_to_numpy(compute_element_normals(ops.convert_to_tensor(probe_geometry)))
    return float(np.arccos(np.clip(np.min(normals[:, 2]), -1.0, 1.0)))


def _receive_angle_sine(f_number, tilt: float = 0.0) -> float:
    """Largest receive angle sine relative to ``+z``.

    The f-number cone has half-angle ``arctan(1 / (2 F))`` around each element's
    normal (``90`` degrees for ``F = 0``). Normals tilted by up to ``tilt`` widen
    it to ``sin(arctan(1 / (2 F)) + tilt)``, which is ``1 / sqrt(1 + 4 F^2)`` for
    a flat array.
    """
    f_number = float(f_number)
    theta_r = np.arctan(1.0 / (2.0 * f_number)) if f_number > 0 else np.pi / 2
    return float(np.sin(min(theta_r + tilt, np.pi / 2)))


def _symmetric_transmit_direction_bounds(sin_theta_r) -> tuple[np.ndarray, np.ndarray]:
    """Fallback transmit direction bounds matching the receive cone.

    Used when no transmit is usable. Gives the classic ``lambda * F / 2`` lateral limit.

    Same layout as :func:`transmit_direction_bounds`, with a single non-plane transmit.
    """
    cos_r_min = np.sqrt(1.0 - sin_theta_r**2)
    bounds = np.array(
        [[[-sin_theta_r, sin_theta_r], [-sin_theta_r, sin_theta_r], [cos_r_min, 1.0]]]
    )
    return bounds, np.array([False])


def _directivity_cap(width, sound_speed) -> float:
    """Largest ``f * sin(theta)`` in Hz a path through an element of ``width`` can carry."""
    return DIRECTIVITY_6DB * sound_speed / float(width) if width else np.inf


def _transverse_spatial_frequency_range(
    bounds, plane, sin_theta_r, f_min, f_max, cap, sound_speed
) -> tuple[float, float]:
    """Lowest and highest spatial frequency along x or y in the image, in cycles per metre.

    The spatial frequency along the axis is ``k = f * (t + r) / c``. Each end is
    taken at the temporal frequency ``f`` that pushes it furthest out, within the
    directivity caps. Plane waves are not capped.

    Args:
        bounds: ``(lo, hi)`` of the x (or y) of the transmit directions per
            transmit, shape ``(n_tx, 2)``. See :func:`transmit_direction_bounds`.
        plane: Plane-wave mask per transmit, shape ``(n_tx,)``.
        sin_theta_r: Largest receive angle sine.
        f_min, f_max: The band in Hz.
        cap: Directivity cap in Hz, see :func:`_directivity_cap`.
        sound_speed: Speed of sound in m/s.
    """
    cap_t = np.where(plane, np.inf, cap)
    band = (sin_theta_r, f_min, f_max, cap_t, cap, sound_speed)
    k_hi = _max_transverse_spatial_frequency(bounds[:, 1], *band)
    k_lo = -_max_transverse_spatial_frequency(-bounds[:, 0], *band)
    return k_lo, k_hi


def _axial_spatial_frequency_range(
    z_bounds, sin_theta_r, f_min, f_max, sound_speed
) -> tuple[float, float]:
    """Lowest and highest spatial frequency along z in the image, in cycles per metre.

    The spatial frequency along z is ``k_z = f * (t_z + r_z) / c``. The highest
    comes from the most on-axis transmit with an on-axis receive, the lowest from
    the most oblique transmit with the most oblique receive.

    Args:
        z_bounds: ``(lo, hi)`` of the z of the transmit directions per
            transmit, shape ``(n_tx, 2)``. See :func:`transmit_direction_bounds`.
    """
    cos_r_min = np.sqrt(1.0 - sin_theta_r**2)
    kz_hi = f_max * (z_bounds[:, 1].max() + 1.0) / sound_speed
    kz_lo = f_min * (z_bounds[:, 0].min() + cos_r_min) / sound_speed
    return kz_lo, kz_hi


def _pitch_limit(spatial_frequency_range) -> float:
    """Largest pixel pitch that samples a spatial frequency range without aliasing.

    The grid repeats the spectrum every ``1 / pitch`` cycles per metre, so the
    width of the range must fit in that period.
    """
    k_lo, k_hi = spatial_frequency_range
    return 1.0 / (k_hi - k_lo) if k_hi > k_lo else np.inf


def _axial_pitch_limit(kz_range, sin_theta_r, f_min, f_max, iq: bool) -> tuple[float, str]:
    """Largest axial pitch and a readable string on what sets it.

    ``kz_range`` is the axial spatial frequency range, see
    :func:`_axial_spatial_frequency_range`. An IQ image only needs the width of
    that range to fit in one grid period. An RF image is real, so the negative
    frequencies are present too and the grid must reach the highest frequency.
    """
    kz_hi = kz_range[1]
    if iq:
        basis = (
            f"IQ, height of the band in k_z for {f_min / 1e6:.2f} - {f_max / 1e6:.2f} MHz "
            f"and sin(theta_r) = {sin_theta_r:.3f}"
        )
        return _pitch_limit(kz_range), basis
    basis = f"RF, top of the band in k_z for f_max = {f_max / 1e6:.2f} MHz"
    return (1.0 / (2.0 * kz_hi) if kz_hi > 0 else np.inf), basis


def aliasing_limits(
    sound_speed: float,
    center_frequency: float | np.ndarray,
    f_number: float,
    focus_distances,
    probe_geometry,
    tx_apodizations,
    polar_angles,
    transmit_origins,
    demodulation_frequency: float | np.ndarray | None = None,
    demodulated: bool = False,
    bandwidth: float | None = None,
    azimuth_angles=None,
    element_width: float | None = None,
    element_height: float | None = None,
) -> dict:
    """Largest pixel pitch the beamforming grid can use without spatial aliasing.

    A transmit in direction ``t`` and a receive in direction ``r`` at frequency
    ``f`` give spatial frequency ``k = f * (t + r) / c`` in the image. Sweeping
    ``f`` over the band, ``r`` over the receive cone (set by ``f_number``) and
    ``t`` over the transmit rays (see :func:`transmit_direction_bounds`) gives the
    image's k-space support, as in the k-space description of pulse-echo imaging by
    Walker and Trahey, IEEE Trans. Ultrason. Ferroelectr. Freq. Control 45 (3),
    pp. 541-558, 1998, https://doi.org/10.1109/58.677599. The limits follow from
    its bounding box:

    - Lateral: ``dx <= 1 / (k_x_max - k_x_min)``, about ``lambda_min * F / 2``
      for equal transmit and receive apertures. Elevational ``dy`` likewise.
    - Axial, RF: ``dz <= 1 / (2 * k_z_max)``, about ``lambda_min / 4``.
    - Axial, IQ: ``dz <= 1 / (k_z_max - k_z_min)``, about ``c / (2 B)`` on axis.

    Given an element size, element directivity (-6 dB, see :data:`DIRECTIVITY_6DB`)
    limits how oblique a path can be and still carry energy. Plane waves are not
    limited this way, as their direction is fixed by the steering.

    For a curved array the receive cone is measured from each element's tilted
    normal (as in :func:`zea.beamform.beamformer.fnumber_mask`), so it is widened
    by the largest tilt. Directivity is then not applied, since it too is relative
    to each element's normal.

    Args:
        sound_speed: Speed of sound in m/s.
        center_frequency: Transmit center frequency in Hz, scalar or per transmit.
        f_number: Receive f-number; ``0`` means the full aperture.
        focus_distances, probe_geometry, tx_apodizations, polar_angles,
            transmit_origins, azimuth_angles: Transmit geometry, see
            :func:`transmit_direction_bounds`.
        demodulation_frequency: Demodulation frequency in Hz. ``None`` or ``0``
            falls back to ``center_frequency``.
        demodulated: Whether the data reaching the grid is baseband IQ.
        bandwidth: Bandwidth of the data in Hz. ``None`` means unknown, in which
            case the RF axial limit is used.
        element_width: Element width in m. ``None`` disables the lateral
            directivity limit.
        element_height: Element height in m. ``None`` falls back to ``element_width``.

    Returns:
        dict: ``dx_max``, ``dy_max`` and ``dz_max`` in metres, plus the quantities
        behind them for reporting: ``lambda_min``, ``transmit_direction``
        (``(lo, hi)`` of the x, y and z of the transmit directions), ``sin_theta_r``,
        ``sin_theta_directivity`` (along x and y), ``axial_basis`` (a readable
        explanation) and
        ``kx_range`` / ``ky_range`` / ``kz_range`` (the ``(lowest, highest)``
        spatial frequency of the image along each axis, in cycles per metre).
    """
    sound_speed = float(sound_speed)
    f_min, f_max = _frequency_band(center_frequency, demodulation_frequency, bandwidth)
    lambda_min = sound_speed / f_max if f_max > 0 else np.inf
    tilt = _max_element_tilt(probe_geometry)
    sin_theta_r = _receive_angle_sine(f_number, tilt)

    bounds, plane = transmit_direction_bounds(
        focus_distances,
        probe_geometry,
        tx_apodizations,
        polar_angles,
        transmit_origins,
        azimuth_angles,
    )
    if bounds is None or plane is None:
        bounds, plane = _symmetric_transmit_direction_bounds(sin_theta_r)

    if element_height is None:
        element_height = element_width
    cap_x = _directivity_cap(element_width, sound_speed)
    cap_y = _directivity_cap(element_height, sound_speed)
    if tilt > 0:
        # Directivity bounds the angle to each element's own normal, not to +z.
        cap_x = cap_y = np.inf

    band = (sin_theta_r, f_min, f_max)
    # Spatial frequency ranges of the image along x, y and z.
    kx_range = _transverse_spatial_frequency_range(bounds[:, 0], plane, *band, cap_x, sound_speed)
    ky_range = _transverse_spatial_frequency_range(bounds[:, 1], plane, *band, cap_y, sound_speed)
    kz_range = _axial_spatial_frequency_range(bounds[:, 2], *band, sound_speed)
    dz_max, axial_basis = _axial_pitch_limit(
        kz_range, *band, iq=demodulated and bandwidth is not None
    )

    return {
        "dx_max": _pitch_limit(kx_range),
        "dy_max": _pitch_limit(ky_range),
        "dz_max": dz_max,
        "lambda_min": lambda_min,
        "transmit_direction": np.stack(
            [bounds[..., 0].min(axis=0), bounds[..., 1].max(axis=0)], -1
        ),
        "sin_theta_r": sin_theta_r,
        "sin_theta_directivity": tuple(
            min(cap / f_max, 1.0) if f_max > 0 else 1.0 for cap in (cap_x, cap_y)
        ),
        "axial_basis": axial_basis,
        "kx_range": kx_range,
        "ky_range": ky_range,
        "kz_range": kz_range,
    }


def _grid_pitch(lims, size) -> tuple[float, float]:
    """``(extent, pitch)`` of a grid axis with pixels on both limits; pitch 0 for one pixel."""
    extent = abs(float(lims[1] - lims[0]))
    return extent, extent / (size - 1) if size > 1 else 0.0


def _transverse_aliasing_message(name, axis, axis_index, extent, d, limits) -> str:
    """Warning for a lateral or elevational pitch ``d`` exceeding its limit."""
    d_max = limits[f"d{axis}_max"]
    lo, hi = limits["transmit_direction"][axis_index]
    return (
        f"{name} grid pitch {d * 1e6:.1f} um exceeds {d_max * 1e6:.1f} um "
        f"(lambda_min = {limits['lambda_min'] * 1e6:.1f} um, "
        f"{axis} of the transmit directions in [{lo:.3f}, {hi:.3f}], "
        f"sin(theta_r) = +-{limits['sin_theta_r']:.3f}, "
        f"element directivity sin(theta) <= {limits['sin_theta_directivity'][axis_index]:.3f}). "
        f"Consider increasing grid_size_{axis} to {int(np.ceil(extent / d_max)) + 1} "
        "or more (for an automatically sized grid, raise pixels_per_wavelength), "
        "or narrowing the apertures "
        f"(f_number / transmit steering) so the data stays inside the grid's {name.lower()} "
        "band."
    )


def _axial_aliasing_message(depth, dz, limits) -> str:
    """Warning for an axial pitch ``dz`` exceeding its limit."""
    return (
        f"Axial grid pitch {dz * 1e6:.1f} um exceeds {limits['dz_max'] * 1e6:.1f} um "
        f"({limits['axial_basis']}). "
        f"Consider increasing grid_size_z to {int(np.ceil(depth / limits['dz_max'])) + 1} "
        "or more (for an automatically sized grid, raise pixels_per_wavelength), "
        "or band-limiting the data further."
    )


def check_for_aliasing(
    xlims,
    zlims,
    grid_size_x: int,
    grid_size_z: int,
    sound_speed: float,
    center_frequency: float | np.ndarray,
    f_number: float,
    focus_distances,
    probe_geometry,
    tx_apodizations,
    polar_angles,
    transmit_origins,
    demodulation_frequency: float | np.ndarray | None = None,
    demodulated: bool = False,
    bandwidth: float | None = None,
    azimuth_angles=None,
    ylims=(0.0, 0.0),
    grid_size_y: int = 1,
    element_width: float | None = None,
    element_height: float | None = None,
) -> list:
    """Warn if the grid's pixel pitch exceeds :func:`aliasing_limits`.

    Only valid for regular cartesian grids (2D or 3D).

    Args:
        xlims, zlims: Lateral and axial extent ``(min, max)`` of the grid in metres.
        grid_size_x, grid_size_z: Number of pixels along x and z, with pixels on
            both limits as in :func:`cartesian_pixel_grid`. An axis with a single
            pixel is not checked.
        sound_speed, center_frequency, f_number, focus_distances, probe_geometry,
            tx_apodizations, polar_angles, transmit_origins, demodulation_frequency,
            demodulated, bandwidth, azimuth_angles, element_width, element_height:
            Forwarded to :func:`aliasing_limits`.
        ylims, grid_size_y: Elevational extent and number of pixels. The default
            is a 2D grid.

    Returns:
        list of str: The messages emitted; empty when the grid is adequate.
    """
    limits = aliasing_limits(
        sound_speed,
        center_frequency,
        f_number,
        focus_distances,
        probe_geometry,
        tx_apodizations,
        polar_angles,
        transmit_origins,
        demodulation_frequency=demodulation_frequency,
        demodulated=demodulated,
        bandwidth=bandwidth,
        azimuth_angles=azimuth_angles,
        element_width=element_width,
        element_height=element_height,
    )

    messages = []
    transverse = [
        ("Lateral", "x", 0, xlims, grid_size_x),
        ("Elevational", "y", 1, ylims, grid_size_y),
    ]
    for name, axis, axis_index, lims, size in transverse:
        extent, d = _grid_pitch(lims, size)
        if d > limits[f"d{axis}_max"]:
            messages.append(_transverse_aliasing_message(name, axis, axis_index, extent, d, limits))
    depth, dz = _grid_pitch(zlims, grid_size_z)
    if dz > limits["dz_max"]:
        messages.append(_axial_aliasing_message(depth, dz, limits))

    for message in messages:
        log.warning_once(message, key=message)
    return messages


def cartesian_pixel_grid(
    xlims,
    zlims,
    ylims=(0.0, 0.0),
    grid_size_x=None,
    grid_size_y=None,
    grid_size_z=None,
    dx=None,
    dy=None,
    dz=None,
):
    """Generate a Cartesian pixel grid.

    Behaviour:
      - If ylims has zero extent (abs(ymax - ymin) < eps) the function returns a 2D grid
        with shape (nz, nx, 3) that contains (x, y=0, z) per-pixel (y omitted as a dimension).
      - If ylims has non-zero extent the function returns a 3D grid with shape
        (nz, nx, ny, 3) containing (x, y, z) per-voxel.

    Args:
        xlims (tuple): [xmin, xmax]
        ylims (tuple): [ymin, ymax] — if ymax == ymin (within tol) treated as "no y extent"
        zlims (tuple): [zmin, zmax]
        grid_size_x, grid_size_y, grid_size_z (int): number of samples along each axis.
            For 2D (no y extent) only grid_size_x and grid_size_z are required if using sizes.
        dx, dy, dz (float): spacings along axes.
            For 2D, only dx and dz are required if using spacings.

    Returns:
        np.ndarray:
            - 2D: shape (nz, nx, 3) with per-pixel [x, y, z] (y will be zeros)
            - 3D: shape (nz, nx, ny, 3) with per-voxel [x, y, z]
    """
    is_3d = abs(ylims[1] - ylims[0]) > eps

    # Validate: must provide either all sizes OR all spacings (exclusive)
    if is_3d:
        sizes_provided = (
            (grid_size_x is not None) and (grid_size_y is not None) and (grid_size_z is not None)
        )
        spacings_provided = (dx is not None) and (dy is not None) and (dz is not None)
    else:
        sizes_provided = (grid_size_x is not None) and (grid_size_z is not None)
        spacings_provided = (dx is not None) and (dz is not None)
        grid_size_y = 1  # Make grid 'flat' in the y direction for 2D case

    if sizes_provided == spacings_provided:
        if is_3d:
            raise ValueError(
                "For 3D (non-zero y extent) either provide grid_size_x/grid_size_y/grid_size_z "
                "OR provide dx/dy/dz (but not both)."
            )
        else:
            raise ValueError(
                "For 2D (no y extent) either provide grid_size_x & grid_size_z "
                "OR provide dx & dz (but not both)."
            )

    # Build coordinate vectors
    if sizes_provided:
        assert grid_size_x is not None and grid_size_y is not None and grid_size_z is not None
        x = np.linspace(xlims[0], xlims[1], grid_size_x)
        y = np.linspace(ylims[0], ylims[1], grid_size_y)
        z = np.linspace(zlims[0], zlims[1], grid_size_z)
    else:
        assert dx is not None and dz is not None
        sign_x = np.sign(xlims[1] - xlims[0]) if xlims[1] != xlims[0] else 1.0
        sign_z = np.sign(zlims[1] - zlims[0]) if zlims[1] != zlims[0] else 1.0
        x = np.arange(xlims[0], xlims[1] + sign_x * eps, sign_x * dx)
        z = np.arange(zlims[0], zlims[1] + sign_z * eps, sign_z * dz)
        if is_3d:
            assert dy is not None
            sign_y = np.sign(ylims[1] - ylims[0]) if ylims[1] != ylims[0] else 1.0
            y = np.arange(ylims[0], ylims[1] + sign_y * eps, sign_y * dy)
        else:
            y = np.array([0.0])

    # Build grid: always (nz, nx, ny, 3)
    z_grid, x_grid, y_grid = np.meshgrid(z, x, y, indexing="ij")
    grid = np.stack((x_grid, y_grid, z_grid), axis=-1)

    # Squeeze y dimension for 2D case: (nz, nx, 1, 3) -> (nz, nx, 3)
    if not is_3d:
        grid = grid.squeeze(axis=2)

    return grid


def radial_pixel_grid(rlims, dr, oris, dirs, num_radial_pixels=None):
    """Generate a focused pixel grid based on input parameters.

    To accommodate the multitude of ways of defining a focused transmit grid, we define
    pixel "rays" or "lines" according to their origins (oris) and directions (dirs).
    The position along the ray is defined by its limits (rlims) and spacing (dr).

    Args:
        rlims (tuple): Radial limits of pixel grid ([rmin, rmax]) with respect to each ray
            origin. Both ends are included when ``dr`` divides the range evenly, matching
            :func:`cartesian_pixel_grid`'s treatment of ``zlims``.
        dr (float): Pixel spacing in radius
        oris (np.ndarray): Origin of each ray in Cartesian coordinates (x, y, z)
            with shape (nrays, 3)
        dirs (np.ndarray): Steering direction of each ray in azimuth, in units of
            radians (nrays, 2)
        num_radial_pixels (int, optional): Number of samples along each ray. Derived from
            ``rlims`` and ``dr`` when not given.

    Returns:
        grid (np.ndarray): Pixel grid of size (nr, nrays, 3) in
            Cartesian coordinates (x, y, z), with nr being the number of radial pixels.
    """
    # Get focusing positions in rho-theta coordinates. Counting the samples (rather than
    # np.arange over a float range) keeps rmax in the grid instead of dropping it to a
    # rounding error, so rlims are the radii of the first and last sample exactly.
    if num_radial_pixels is None:
        num_radial_pixels = int(np.floor((rlims[1] - rlims[0]) / dr + 1e-9)) + 1
    r = rlims[0] + np.arange(num_radial_pixels) * dr  # Depth rho
    t = dirs[:, 0]  # Use azimuthal angle theta (ignore elevation angle)
    tt, rr = np.meshgrid(t, r, indexing="ij")

    # Convert the focusing grid to Cartesian coordinates
    xx = rr * np.sin(tt) + oris[:, [0]]
    zz = rr * np.cos(tt) + oris[:, [2]]
    yy = 0 * xx
    grid = np.stack((xx, yy, zz), axis=-1)
    return grid


def polar_pixel_grid(
    polar_limits,
    zlims,
    num_radial_pixels: int,
    num_polar_pixels: int,
    distance_to_apex: float = 0.0,
):
    """Generate a polar grid.

    Uses radial_pixel_grid but based on parameters that are present in the scan class.
    Currently only 2D grids (no elevation steering) are supported.

    Args:
        polar_limits (tuple): Polar limits of pixel grid ([polar_min, polar_max])
        zlims (tuple): Depth limits of pixel grid ([zmin, zmax]), measured from the
            transducer surface (``z = 0``) just like for a Cartesian grid.
        num_radial_pixels (int, optional): Number of depth pixels.
        num_polar_pixels (int, optional): Number of polar pixels.
        distance_to_apex (float, optional): Distance from the transducer surface to the
            apex of the pixel grid, the point the rays fan out from. For a curved probe
            that point is the centre of curvature, so this equals the probe's radius of
            curvature (see :func:`~zea.probes.fit_curved_probe_radius`).

    Returns:
        grid (np.ndarray): Pixel grid of size (num_radial_pixels, num_polar_pixels, 3)
        in Cartesian coordinates (x, y, z)
    """
    assert len(polar_limits) == 2, "polar_limits must be a tuple of length 2."
    assert len(zlims) == 2, "zlims must be a tuple of length 2."

    # Radii are measured from the apex, depths from the transducer surface.
    rlims = (zlims[0] + distance_to_apex, zlims[1] + distance_to_apex)
    dr = (rlims[1] - rlims[0]) / max(num_radial_pixels - 1, 1)

    oris = np.array([0, 0, -distance_to_apex])
    oris = np.tile(oris, (num_polar_pixels, 1))
    dirs_az = np.linspace(*polar_limits, num_polar_pixels)  # ty: ignore[no-matching-overload]

    dirs_el = np.zeros(num_polar_pixels)
    dirs = np.vstack((dirs_az, dirs_el)).T

    return radial_pixel_grid(rlims, dr, oris, dirs, num_radial_pixels).transpose(1, 0, 2)


def scanline_pixel_grid(
    transmit_origins,
    focus_distances,
    polar_angles,
    zlims,
    num_depth_pixels,
    azimuth_angles=None,
    grid_type="cartesian",
):
    """Pixel grid for scanline beamforming: one column of pixels per transmit.

    Scanline (line-by-line) imaging is a special case of pixel-based DAS
    beamforming where each transmit is beamformed to a single column of
    pixels. This builds that grid in the same
    ``(grid_size_z, grid_size_x, 3)`` layout as :func:`cartesian_pixel_grid`
    (one column ``n`` per transmit ``n``, ``num_depth_pixels`` rows), so it can
    be beamformed by the regular pixel-based :class:`~zea.ops.Beamform`
    pipeline. Pair with :func:`scanline_aligned_apodization` (fed to
    :class:`~zea.ops.AlignedApodization`) to zero out every transmit except
    the one whose column each pixel belongs to.

    Args:
        transmit_origins (np.ndarray): Beam origins ``(n_tx, 3)`` in meters.
        focus_distances (np.ndarray): Focus distances ``(n_tx,)`` in meters.
        polar_angles (np.ndarray): Steering angles ``(n_tx,)`` in radians.
        zlims (tuple): Depth range ``(z_min, z_max)`` in meters.
        num_depth_pixels (int): Number of samples along each line.
        azimuth_angles (np.ndarray, optional): Azimuth angles ``(n_tx,)``.
            Defaults to zeros.
        grid_type (str): ``"cartesian"`` for a vertical column at each beam's
            lateral focus position (linear-scan geometry), matching
            :func:`cartesian_pixel_grid`. ``"polar"`` for a steered ray from
            each transmit's own origin (sector / phased-array geometry),
            matching :func:`polar_pixel_grid`. Defaults to ``"cartesian"``.

    Returns:
        np.ndarray: Pixel positions of shape ``(num_depth_pixels, n_tx, 3)`` in
        Cartesian ``(x, y, z)`` coordinates (meters); column ``n`` is the beam
        line of transmit ``n``.
    """
    origins = np.asarray(transmit_origins, dtype=np.float32)
    focus_distances = np.asarray(focus_distances, dtype=np.float32)
    polar_angles = np.asarray(polar_angles, dtype=np.float32)
    azimuth_angles = (
        np.zeros_like(polar_angles)
        if azimuth_angles is None
        else np.asarray(azimuth_angles, dtype=np.float32)
    )

    line_depths = np.linspace(zlims[0], zlims[1], num_depth_pixels).astype(np.float32)
    beam_directions = np.stack(
        [
            np.sin(polar_angles) * np.cos(azimuth_angles),
            np.sin(polar_angles) * np.sin(azimuth_angles),
            np.cos(polar_angles),
        ],
        axis=-1,
    )

    # Every scanline column is a ray sampled along ``line_depths``; the two grid
    # types differ only in each column's origin and direction.
    if grid_type == "polar":
        # Steered rays from each transmit's own origin.
        column_origins = origins
        column_directions = beam_directions
    elif grid_type == "cartesian":
        # Vertical columns at each beam's lateral focus position. Non-finite
        # (e.g. plane-wave np.inf) focus distances must contribute no lateral
        # offset, otherwise inf * 0 gives NaN for on-axis transmits.
        finite_focus_distances = np.where(np.isfinite(focus_distances), focus_distances, 0.0)
        column_origins = np.zeros_like(origins)
        column_origins[:, 0] = origins[:, 0] + finite_focus_distances * beam_directions[:, 0]
        column_directions = np.zeros_like(beam_directions)
        column_directions[:, 2] = 1.0
    else:
        raise ValueError(
            f"Unsupported grid_type: {grid_type!r}. Supported types are 'cartesian' and 'polar'."
        )

    grid = column_origins[None, :, :] + line_depths[:, None, None] * column_directions[None, :, :]
    return grid.astype(np.float32)


def scanline_aligned_apodization(n_tx, num_depth_pixels):
    """Compounding apodization mask that isolates each pixel's owning transmit.

    For a grid built by :func:`scanline_pixel_grid` (shape
    ``(num_depth_pixels, n_tx, 3)``, flattened row-major to ``(n_pix, 3)``),
    pixel ``(i, n)`` at flat index ``i * n_tx + n`` belongs to transmit ``n``.
    This returns the corresponding one-hot weight (1 for the owning transmit,
    0 for every other transmit) to feed to
    :class:`zea.ops.AlignedApodization`, turning the regular pixel-based DAS
    pipeline into scanline (line-by-line) beamforming.

    Args:
        n_tx (int): Number of transmits (grid columns).
        num_depth_pixels (int): Number of depth samples per line (grid rows).

    Returns:
        np.ndarray: Apodization weights of shape
        ``(num_depth_pixels * n_tx, n_tx)``.
    """
    return np.tile(np.eye(n_tx, dtype=np.float32), (num_depth_pixels, 1))
