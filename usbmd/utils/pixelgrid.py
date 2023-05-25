"""Pixel grid calculation for beamforming

- **Author(s)**     : Dongwoon Hyun, Ben Luijten
- **Date**          : 2020-04-03
"""
import numpy as np

eps = 1e-10


def get_grid(scan):
    """Creates a pixelgrid based on scan class parameters."""
    xlims = scan.xlims
    zlims = scan.zlims
    Nx = scan.Nx
    Nz = scan.Nz

    if Nx and Nz:
        grid = cartesian_pixel_grid(xlims, zlims, Nx=Nx, Nz=Nz)
    else:
        wvln = scan.c / scan.fc
        dx = wvln / scan.pixels_per_wavelength
        dz = dx
        grid = cartesian_pixel_grid(xlims, zlims, dx=dx, dz=dz)
        print(
            f'Pixelgrid was set automatically to Nx: {grid.shape[1]}, Nz: {grid.shape[0]}, '
            f'using {scan.pixels_per_wavelength} pixels per wavelength.'
        )
    return grid

def check_for_aliasing(scan):
    """Checks if the scan class parameters will cause aliasing."""
    wvln = scan.c / scan.fc
    dx = wvln / scan.pixels_per_wavelength
    dz = dx

    width = scan.xlims[1] - scan.xlims[0]
    depth = scan.zlims[1] - scan.zlims[0]

    if scan.Nx and scan.Nz:
        if width/scan.Nx > wvln/2:
            print(
                f'WARNING: width/Nx = {width/scan.Nx} < wvln/2 = {wvln/2}. '
                f'Consider increasing Nx'
            )
        if depth/scan.Nz > wvln/2:
            print(
                f'WARNING: depth/Nz = {depth/scan.Nz} < wvln/2 = {wvln/2}. '
                f'Consider increasing Nz'
            )
    else:
        if dx > wvln/2:
            print(
                f'WARNING: dx = {dx} > wvln/2 = {wvln/2}. '
                f'Consider increasing pixels_per_wavelength to 2 or more'
            )
        if dz > wvln/2:
            print(
                f'WARNING: dz = {dz} > wvln/2 = {wvln/2}. '
                f'Consider increasing pixels_per_wavelength to 2 or more'
            )


def cartesian_pixel_grid(xlims, zlims, **kwargs):
    """
    Generate a Cartesian pixel grid based on input parameters.
    The output has shape (nx, nz, 3).
    Either Nx and Nz or dx and dz must be defined.

    INPUTS
    xlims   Azimuthal limits of pixel grid ([xmin, xmax])
    zlims   Depth limits of pixel grid ([zmin, zmax])
    dx      Pixel spacing in azimuth
    dz      Pixel spacing in depth
    Nx      Number of azimuthal pixels, overrides dx and dz parameters
    Nz      Number of depth pixels, overrides dx and dz parameters

    OUTPUTS
    grid    Pixel grid of size (nx, nz, 3) in Cartesian coordinates (x, y, z)
    """
    # Check if Nx and Nz are defined in kwargs and if they are not None
    Nx = kwargs.get("Nx", None)
    Nz = kwargs.get("Nz", None)
    dx = kwargs.get("dx", None)
    dz = kwargs.get("dz", None)

    # Determine the grid spacing
    if Nx is not None and Nz is not None:
        x = np.linspace(xlims[0], xlims[1] + eps, Nx)
        z = np.linspace(zlims[0], zlims[1] + eps, Nz)
    elif dx is not None and dz is not None:
        x = np.arange(xlims[0], xlims[1] + eps, dx)
        z = np.arange(zlims[0], zlims[1] + eps, dz)
    else:
        raise ValueError("Either Nx and Nz or dx and dz must be defined.")

    # Create the pixel grid
    z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
    y_grid = 0 * x_grid # Assume y = 0
    grid = np.stack((x_grid, y_grid, z_grid), axis=-1)
    return grid

def radial_pixel_grid(rlims, dr, oris, dirs):
    """
    Generate a focused pixel grid based on input parameters.
    To accommodate the multitude of ways of defining a focused transmit grid, we define
    pixel "rays" or "lines" according to their origins (oris) and directions (dirs).
    The position along the ray is defined by its limits (rlims) and spacing (dr).

    INPUTS
    rlims   Radial limits of pixel grid ([rmin, rmax])
    dr      Pixel spacing in radius
    oris    Origin of each ray in Cartesian coordinates (x, y, z) with shape (nrays, 3)
    dirs    Steering direction of each ray in azimuth, in units of radians (nrays, 2)

    OUTPUTS
    grid    Pixel grid of size (nr, nrays, 3) in Cartesian coordinates (x, y, z)
    """
    # Get focusing positions in rho-theta coordinates
    r = np.arange(rlims[0], rlims[1] + eps, dr)  # Depth rho
    t = dirs[:, 0]  # Use azimuthal angle theta (ignore elevation angle)
    tt, rr = np.meshgrid(t, r, indexing="ij")

    # Convert the focusing grid to Cartesian coordinates
    xx = rr * np.sin(tt) + oris[:, [0]]
    zz = rr * np.cos(tt) + oris[:, [2]]
    yy = 0 * xx
    grid = np.stack((xx, yy, zz), axis=-1)
    return grid
