"""Pixel grid calculation for ultrasound beamforming."""

import numpy as np

from zea import log

eps = 1e-10


def check_for_aliasing(parameters):
    """Checks if the :class:`~zea.Parameters` will cause spatial aliasing due to a too low pixel
    density. If so, a warning is printed with a suggestion to increase the pixel density by either
    increasing the number of pixels, or decreasing the pixel spacing, depending on which parameter
    was set by the user."""
    width = parameters.xlims[1] - parameters.xlims[0]
    depth = parameters.zlims[1] - parameters.zlims[0]
    wvln = parameters.wavelength

    if width / parameters.grid_size_x > wvln / 2:
        log.warning(
            f"width/grid_size_x = {width / parameters.grid_size_x:.7f} > "
            f"wavelength/2 = {wvln / 2:.7f}. "
            f"Consider increasing grid_size_x to {int(np.ceil(width / (wvln / 2)))} "
            "or more, or unsetting it to size the grid automatically."
        )
    if depth / parameters.grid_size_z > wvln / 2:
        log.warning(
            f"depth/grid_size_z = {depth / parameters.grid_size_z:.7f} > "
            f"wavelength/2 = {wvln / 2:.7f}. "
            f"Consider increasing grid_size_z to {int(np.ceil(depth / (wvln / 2)))} "
            "or more, or unsetting it to size the grid automatically."
        )


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
        x = np.linspace(xlims[0], xlims[1] + eps, grid_size_x)
        y = np.linspace(ylims[0], ylims[1] + eps, grid_size_y)
        z = np.linspace(zlims[0], zlims[1] + eps, grid_size_z)
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


def radial_pixel_grid(rlims, dr, oris, dirs):
    """Generate a focused pixel grid based on input parameters.

    To accommodate the multitude of ways of defining a focused transmit grid, we define
    pixel "rays" or "lines" according to their origins (oris) and directions (dirs).
    The position along the ray is defined by its limits (rlims) and spacing (dr).

    Args:
        rlims (tuple): Radial limits of pixel grid ([rmin, rmax]) with respect to each ray origin
        dr (float): Pixel spacing in radius
        oris (np.ndarray): Origin of each ray in Cartesian coordinates (x, y, z)
            with shape (nrays, 3)
        dirs (np.ndarray): Steering direction of each ray in azimuth, in units of
            radians (nrays, 2)

    Returns:
        grid (np.ndarray): Pixel grid of size (nr, nrays, 3) in
            Cartesian coordinates (x, y, z), with nr being the number of radial pixels.
    """
    # Get focusing positions in rho-theta coordinates
    r = np.arange(rlims[0], rlims[1], dr)  # Depth rho
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
        zlims (tuple): Depth limits of pixel grid ([zmin, zmax])
        num_radial_pixels (int, optional): Number of depth pixels.
        num_polar_pixels (int, optional): Number of polar pixels.
        distance_to_apex (float, optional): Distance from transducer to apex of pixel grid.

    Returns:
        grid (np.ndarray): Pixel grid of size (num_radial_pixels, num_polar_pixels, 3)
        in Cartesian coordinates (x, y, z)
    """
    assert len(polar_limits) == 2, "polar_limits must be a tuple of length 2."
    assert len(zlims) == 2, "zlims must be a tuple of length 2."

    rlims = (zlims[0], zlims[1] + distance_to_apex)
    dr = (rlims[1] - rlims[0]) / num_radial_pixels

    oris = np.array([0, 0, -distance_to_apex])
    oris = np.tile(oris, (num_polar_pixels, 1))
    dirs_az = np.linspace(*polar_limits, num_polar_pixels)  # ty: ignore[no-matching-overload]

    dirs_el = np.zeros(num_polar_pixels)
    dirs = np.vstack((dirs_az, dirs_el)).T

    grid = radial_pixel_grid(rlims, dr, oris, dirs).transpose(1, 0, 2)

    # In case of rounding errors, trim the grid to the correct number of radial pixels
    return grid[:num_radial_pixels, :, :]


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
