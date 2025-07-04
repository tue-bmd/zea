"""Pixel grid calculation for ultrasound beamforming."""

import numpy as np

from zea import log

eps = 1e-10


def check_for_aliasing(scan):
    """Checks if the scan class parameters will cause spatial aliasing due to a too low pixel
    density. If so, a warning is printed with a suggestion to increase the pixel density by either
    increasing the number of pixels, or decreasing the pixel spacing, depending on which parameter
    was set by the user."""
    width = scan.xlims[1] - scan.xlims[0]
    depth = scan.zlims[1] - scan.zlims[0]
    wvln = scan.wavelength

    if width / scan.Nx > wvln / 2:
        log.warning(
            f"width/Nx = {width / scan.Nx:.7f} < wavelength/2 = {wvln / 2}. "
            f"Consider either increasing scan.Nx to {int(np.ceil(width / (wvln / 2)))} or more, or "
            "increasing scan.pixels_per_wavelength to 2 or more."
        )
    if depth / scan.Nz > wvln / 2:
        log.warning(
            f"depth/Nz = {depth / scan.Nz:.7f} < wavelength/2 = {wvln / 2:.7f}. "
            f"Consider either increasing scan.Nz to {int(np.ceil(depth / (wvln / 2)))} or more, or "
            "increasing scan.pixels_per_wavelength to 2 or more."
        )


def cartesian_pixel_grid(xlims, zlims, Nx=None, Nz=None, dx=None, dz=None):
    """Generate a Cartesian pixel grid based on input parameters.

    Args:
        xlims (tuple): Azimuthal limits of pixel grid ([xmin, xmax])
        zlims (tuple): Depth limits of pixel grid ([zmin, zmax])
        Nx (int): Number of azimuthal pixels, overrides dx and dz parameters
        Nz (int): Number of depth pixels, overrides dx and dz parameters
        dx (float): Pixel spacing in azimuth
        dz (float): Pixel spacing in depth

    Raises:
        ValueError: Either Nx and Nz or dx and dz must be defined.

    Returns:
        grid (np.ndarray): Pixel grid of size (nz, nx, 3) in
            Cartesian coordinates (x, y, z)
    """
    assert (bool(Nx) and bool(Nz)) ^ (bool(dx) and bool(dz)), (
        "Either Nx and Nz or dx and dz must be defined."
    )

    # Determine the grid spacing
    if Nx is not None and Nz is not None:
        x = np.linspace(xlims[0], xlims[1] + eps, Nx)
        z = np.linspace(zlims[0], zlims[1] + eps, Nz)
    elif dx is not None and dz is not None:
        sign = np.sign(xlims[1] - xlims[0])
        x = np.arange(xlims[0], xlims[1] + eps, sign * dx)
        z = np.arange(zlims[0], zlims[1] + eps, sign * dz)
    else:
        raise ValueError("Either Nx and Nz or dx and dz must be defined.")

    # Create the pixel grid
    z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
    y_grid = 0 * x_grid  # Assume y = 0
    grid = np.stack((x_grid, y_grid, z_grid), axis=-1)
    return grid


def radial_pixel_grid(rlims, dr, oris, dirs):
    """Generate a focused pixel grid based on input parameters.

    To accommodate the multitude of ways of defining a focused transmit grid, we define
    pixel "rays" or "lines" according to their origins (oris) and directions (dirs).
    The position along the ray is defined by its limits (rlims) and spacing (dr).

    Args:
        rlims (tuple): Radial limits of pixel grid ([rmin, rmax])
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


def polar_pixel_grid(polar_limits, zlims, Nz: int, Nr: int):
    """Generate a polar grid.

    Uses radial_pixel_grid but based on parameters that are present in the scan class.

    Args:
        polar_limits (tuple): Azimuthal limits of pixel grid ([azimuth_min, azimuth_max])
        zlims (tuple): Depth limits of pixel grid ([zmin, zmax])
        Nz (int, optional): Number of depth pixels.
        Nr (int, optional): Number of azimuthal pixels.

    Returns:
        grid (np.ndarray): Pixel grid of size (Nz, Nr, 3) in Cartesian coordinates (x, y, z)
    """
    assert len(polar_limits) == 2, "polar_limits must be a tuple of length 2."
    assert len(zlims) == 2, "zlims must be a tuple of length 2."

    dr = (zlims[1] - zlims[0]) / Nz

    oris = np.array([0, 0, 0])
    oris = np.tile(oris, (Nr, 1))
    dirs_az = np.linspace(*polar_limits, Nr)

    dirs_el = np.zeros(Nr)
    dirs = np.vstack((dirs_az, dirs_el)).T
    return radial_pixel_grid(zlims, dr, oris, dirs).transpose(1, 0, 2)
