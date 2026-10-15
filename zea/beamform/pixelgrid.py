"""Pixel grid calculation for ultrasound beamforming."""

import numpy as np

from zea import log
from zea.beamform.geometry import is_plane_wave

eps = 1e-10


def transmit_direction_range(
    focus_distances,
    probe_geometry,
    tx_apodizations,
    polar_angles,
    transmit_origins,
    azimuth_angles=None,
) -> np.ndarray | None:
    r"""Range of the ray directions of each transmit.

    This is the transmit half of the image's k-space support (see
    :func:`aliasing_limits`). Two cases, both taken from the transmit geometry:

    * **focused / diverging** -- the focal point lies at
      :math:`\mathbf{o} + z_f \hat{\mathbf{d}}` for origin :math:`\mathbf{o}`,
      focus distance :math:`z_f` and steering direction
      :math:`\hat{\mathbf{d}} = (\sin\alpha\cos\beta, \sin\alpha\sin\beta, \cos\alpha)`
      with polar angle :math:`\alpha` and azimuth :math:`\beta`. The range spans
      the rays from the active elements (non-zero ``tx_apodizations``) through
      it. For an unsteered focus centred on an aperture of width :math:`W` the
      x component spans :math:`\pm s` with
      :math:`s = \frac{W/2}{\sqrt{z_f^2 + (W/2)^2}}`; steering shifts and skews it.
    * **plane** -- a single direction :math:`\hat{\mathbf{d}}`, so each range
      collapses to a point. The sign is kept: a single steered plane wave shifts
      the support but does not widen it.

    The ``z`` range runs from the most oblique ray to the ray nearest the axis.
    The rays fill the aperture, so the latter is taken on axis wherever the ``x``
    and ``y`` ranges straddle zero.

    Args:
        focus_distances: Focus distance per transmit along the steering
            direction, shape ``(n_tx,)``. Negative for diverging waves. See
            :func:`zea.beamform.geometry.is_plane_wave` for how plane waves are marked.
        probe_geometry: Element positions, shape ``(n_el, 3)``.
        tx_apodizations: Transmit apodization per transmit, shape ``(n_tx, n_el)``.
        polar_angles: Steering angle per transmit in radians, shape ``(n_tx,)``.
        transmit_origins: Origin of each transmit, shape ``(n_tx, 3)``.
        azimuth_angles: Azimuth of the steering per transmit in radians, shape
            ``(n_tx,)``. ``None`` means zeros, i.e. steering in the x-z plane.

    Returns:
        np.ndarray or None: ``(lo, hi)`` of the ``x``, ``y`` and ``z`` components
        of the unit ray direction per transmit, shape ``(n_tx, 3, 2)``, with
        transmits that have no active elements left out. ``None`` when the
        transmit geometry is empty or malformed, in which case callers should fall
        back to the symmetric assumption.
    """
    focus_distances = np.asarray(focus_distances, dtype=float)
    probe_geometry = np.asarray(probe_geometry, dtype=float)
    tx_apodizations = np.asarray(tx_apodizations, dtype=float)

    if focus_distances.size == 0 or tx_apodizations.ndim != 2:
        return None

    polar_angles = np.asarray(polar_angles, dtype=float)
    transmit_origins = np.asarray(transmit_origins, dtype=float)
    if azimuth_angles is None:
        azimuth_angles = np.zeros_like(polar_angles)
    azimuth_angles = np.asarray(azimuth_angles, dtype=float)

    ranges = []
    for tx, focus in enumerate(focus_distances):
        polar, azimuth = float(polar_angles[tx]), float(azimuth_angles[tx])
        steering = np.array(
            [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)]
        )
        if is_plane_wave(focus):
            ranges.append(np.stack([steering, steering], axis=-1))
            continue
        active = np.nonzero(np.abs(tx_apodizations[tx]) > eps)[0]
        if active.size == 0:
            continue
        focal_point = transmit_origins[tx] + float(focus) * steering
        # Rays run from each element towards a focus, or away from the virtual
        # source of a diverging wave, so flip the direction behind the probe.
        rays = focal_point - probe_geometry[active]
        rays *= np.sign(rays[:, 2:]) / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), eps)
        lo, hi = rays.min(axis=0), rays.max(axis=0)
        nearest = np.where(lo * hi <= 0, 0.0, np.minimum(np.abs(lo), np.abs(hi)))
        lo[2] = rays[:, 2].min()
        hi[2] = np.sqrt(max(1.0 - nearest[0] ** 2 - nearest[1] ** 2, 0.0))
        ranges.append(np.stack([lo, hi], axis=-1))

    return np.array(ranges) if ranges else None


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
) -> dict:
    """Largest pixel pitch the beamforming grid can carry without spatial aliasing.

    A transmit in unit direction ``t`` and a receive in unit direction ``r`` at
    temporal frequency ``f`` put a plane wave in the image with spatial frequency
    ``k = f * (t + r) / c``. In the x-z plane, with angles ``theta`` to the z axis::

        k_x = f * (sin(theta_t) + sin(theta_r)) / c
        k_z = f * (cos(theta_t) + cos(theta_r)) / c

    Sweeping ``f`` over the band, ``r`` over the receive cone of half-angle
    ``sin(theta_r) = 1 / sqrt(1 + 4 F^2)`` and ``t`` over the transmit rays (see
    :func:`transmit_direction_range`) gives the image's k-space support, a single patch
    at positive ``k_z``. The limits below use the bounding box of that patch.

    **Axial.** An RF image is real, so the mirror patch at ``-k`` is present too
    and the grid must reach the top of the patch: ``dz <= 1 / (2 * k_z_max)``,
    i.e. ``lambda_min / 4`` for an unsteered transmit. An IQ image (carrier phase
    restored by the beamformer) only needs the patch height to fit in one grid
    period: ``dz <= 1 / (k_z_max - k_z_min)``, on axis ``c / (2 B)``. Oblique
    paths make the patch taller, so wide apertures need a finer IQ grid.

    **Lateral.** RF and IQ share one limit, since the mirror patch lies at
    negative ``k_z``: ``dx <= 1 / (k_x_max - k_x_min)``, paraxially
    ``lambda_min * F / 2`` for equal apertures. A steered plane wave shifts the
    patch sideways, while compounding over ``+-alpha`` widens it. When no
    transmit has active elements, ``sin(theta_t)`` is assumed to span
    ``+-sin(theta_r)``.

    **Elevation.** For 3D grids, ``dy`` follows from ``k_y`` exactly as ``dx``
    does from ``k_x``, using the y components of ``t`` and ``r``.

    Args:
        sound_speed: Speed of sound in m/s.
        center_frequency: Transmit center frequency in Hz, scalar or per transmit.
        f_number: Receive f-number; ``0`` means the full half-space aperture.
        focus_distances: Focus depth per transmit, see :func:`transmit_direction_range`.
        probe_geometry: Element positions, see :func:`transmit_direction_range`.
        tx_apodizations: Transmit apodizations, see :func:`transmit_direction_range`.
        polar_angles: Steering angle per transmit, see :func:`transmit_direction_range`.
        transmit_origins: Origin of each transmit, see :func:`transmit_direction_range`.
        demodulation_frequency: Demodulation frequency in Hz. ``None`` or ``0``
            falls back to ``center_frequency`` for the band's RF position.
        demodulated: Whether the data reaching the grid is baseband IQ.
        bandwidth: The RF band retained by the pipeline's band-pass, in Hz.
            ``None`` means unknown: the band is then just its centre, and IQ
            falls back to the RF limit.
        azimuth_angles: Azimuth of the steering per transmit, see
            :func:`transmit_direction_range`.

    Returns:
        dict: The limits plus the quantities behind them, so callers can report
        *why* a limit is what it is:

        - ``dx_max`` / ``dy_max`` / ``dz_max``: Maximum lateral / elevational /
          axial pixel spacing in metres.
        - ``lambda_min``: Shortest RF wavelength used.
        - ``transmit_direction``: ``(lo, hi)`` of the x, y and z components of the
          transmit directions over all transmits, shape ``(3, 2)``.
        - ``sin_theta_r``: Largest receive angle sine, set by ``f_number``.
        - ``axial_basis``: Readable string on what sets the axial limit.
        - ``kx_range`` / ``ky_range`` / ``kz_range``: Bounding box of the support
          in cycles per metre.
    """
    sound_speed = float(sound_speed)

    # Both the RF axial limit and the lateral limit need the shortest *RF*
    # wavelength, so they need the band's position on the RF axis. Either
    # frequency can legitimately read 0 -- `demodulation_frequency` when the data
    # was stored at baseband, `center_frequency` once `Demodulate` has zeroed it --
    # so take whichever is still carrying the carrier. Per-transmit carriers widen
    # the band to cover all of them.
    if demodulation_frequency is None:
        demodulation_frequency = 0.0
    band_center = np.ravel(np.asarray(demodulation_frequency, dtype=float))
    if np.all(band_center <= 0):
        band_center = np.ravel(np.asarray(center_frequency, dtype=float))
    half_band = float(bandwidth) / 2 if bandwidth is not None else 0.0
    f_min = max(float(band_center.min()) - half_band, 0.0)
    f_max = float(band_center.max()) + half_band
    lambda_min = sound_speed / f_max if f_max > 0 else np.inf

    f_number = float(f_number)
    sin_theta_r = 1.0 / np.sqrt(1.0 + 4.0 * f_number**2) if f_number > 0 else 1.0
    cos_r_min = np.sqrt(1.0 - sin_theta_r**2)
    ranges = transmit_direction_range(
        focus_distances,
        probe_geometry,
        tx_apodizations,
        polar_angles,
        transmit_origins,
        azimuth_angles,
    )
    if ranges is None:
        # Symmetric, the transmit cone matching the receive cone: lambda * F / 2.
        ranges = np.array(
            [[[-sin_theta_r, sin_theta_r], [-sin_theta_r, sin_theta_r], [cos_r_min, 1.0]]]
        )

    # Lateral and elevational: k = f * u / c with u = t + r along the axis. Each
    # edge of u reaches furthest from zero at the frequency that scales it outwards.
    def transverse_band(axis):
        u_lo = ranges[:, axis, 0].min() - sin_theta_r
        u_hi = ranges[:, axis, 1].max() + sin_theta_r
        k_lo = (f_max if u_lo < 0 else f_min) * u_lo / sound_speed
        k_hi = (f_max if u_hi > 0 else f_min) * u_hi / sound_speed
        return (k_lo, k_hi), 1.0 / (k_hi - k_lo) if k_hi > k_lo else np.inf

    kx_range, dx_max = transverse_band(0)
    ky_range, dy_max = transverse_band(1)

    # Axial: k_z = f * (t_z + r_z) / c. The top of the patch is the most on-axis
    # transmit with an on-axis receive, the bottom the most oblique transmit with
    # the most oblique receive.
    kz_hi = f_max * (ranges[:, 2, 1].max() + 1.0) / sound_speed
    kz_lo = f_min * (ranges[:, 2, 0].min() + cos_r_min) / sound_speed

    if demodulated and bandwidth is not None:
        dz_max = 1.0 / (kz_hi - kz_lo)
        axial_basis = (
            f"IQ, height of the band in k_z for {f_min / 1e6:.2f} - {f_max / 1e6:.2f} MHz "
            f"and sin(theta_r) = {sin_theta_r:.3f}"
        )
    else:
        dz_max = 1.0 / (2.0 * kz_hi) if kz_hi > 0 else np.inf
        axial_basis = f"RF, top of the band in k_z for f_max = {f_max / 1e6:.2f} MHz"

    return {
        "dx_max": dx_max,
        "dy_max": dy_max,
        "dz_max": dz_max,
        "lambda_min": lambda_min,
        "transmit_direction": np.stack(
            [ranges[..., 0].min(axis=0), ranges[..., 1].max(axis=0)], -1
        ),
        "sin_theta_r": sin_theta_r,
        "axial_basis": axial_basis,
        "kx_range": kx_range,
        "ky_range": ky_range,
        "kz_range": (kz_lo, kz_hi),
    }


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
) -> list:
    """Warn when the beamforming grid under-samples the data it is about to carry.

    Compares the grid's pixel pitch against :func:`aliasing_limits`. Content
    beyond these limits cannot be represented by *any* image on this grid, so it
    folds back as aliasing. Only valid for regular cartesian grids, 2D or 3D.

    Args:
        xlims: Lateral extent ``(xmin, xmax)`` of the grid in metres.
        zlims: Axial extent ``(zmin, zmax)`` of the grid in metres.
        grid_size_x: Number of pixels along x. Pixels sit on both limits, as in
            :func:`cartesian_pixel_grid`, so the pitch is ``width / (grid_size_x - 1)``.
            A single pixel has no pitch and is not checked.
        grid_size_z: Number of pixels along z, likewise.
        sound_speed, center_frequency, f_number, focus_distances, probe_geometry,
            tx_apodizations, polar_angles, transmit_origins, demodulation_frequency,
            demodulated, bandwidth, azimuth_angles: Forwarded to
            :func:`aliasing_limits`.
        ylims: Elevational extent ``(ymin, ymax)`` of the grid in metres. The
            default, no extent, is a 2D grid.
        grid_size_y: Number of pixels along y, likewise.

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
    )

    def pitch(lims, size):
        extent = float(lims[1] - lims[0])
        return extent, extent / (size - 1) if size > 1 else 0.0

    messages = []
    transverse = [
        ("Lateral", "x", 0, xlims, grid_size_x),
        ("Elevational", "y", 1, ylims, grid_size_y),
    ]
    for name, axis, component, lims, size in transverse:
        extent, d = pitch(lims, size)
        d_max = limits[f"d{axis}_max"]
        if d <= d_max:
            continue
        lo, hi = limits["transmit_direction"][component]
        messages.append(
            f"{name} grid pitch {d * 1e6:.1f} um exceeds {d_max * 1e6:.1f} um "
            f"(lambda_min = {limits['lambda_min'] * 1e6:.1f} um, "
            f"transmit direction {axis} component in [{lo:.3f}, {hi:.3f}], "
            f"sin(theta_r) = +-{limits['sin_theta_r']:.3f}). "
            f"Consider increasing grid_size_{axis} to {int(np.ceil(extent / d_max)) + 1} "
            "or more, "
            "unsetting it to size the grid automatically, or narrowing the apertures "
            f"(f_number / transmit steering) so the data stays inside the grid's {name.lower()} "
            "band."
        )
    depth, dz = pitch(zlims, grid_size_z)
    if dz > limits["dz_max"]:
        messages.append(
            f"Axial grid pitch {dz * 1e6:.1f} um exceeds {limits['dz_max'] * 1e6:.1f} um "
            f"({limits['axial_basis']}). "
            f"Consider increasing grid_size_z to {int(np.ceil(depth / limits['dz_max'])) + 1} "
            "or more, "
            "unsetting it to size the grid automatically, or band-limiting the data further."
        )

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
