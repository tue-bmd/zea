"""Display functionality for USBMD.

All functionality related to displaying ultrasound images.

- **Author(s)**     : Tristan Stevens
- **Date**          : 02/11/2023
"""

from typing import Tuple, Union

import numpy as np
import scipy
from keras import ops
from PIL import Image
from skimage.transform import resize

from usbmd.utils import find_first_nonzero_index, translate


def to_8bit(image, dynamic_range: Union[None, tuple] = None, pillow: bool = True):
    """Convert image to 8 bit image [0, 255]. Clip between dynamic range.

    Args:
        image (ndarray): Input image(s). Should be in between dynamic range.
        dynamic_range (tuple, optional): Dynamic range of input image(s).
        pillow (bool, optional): Whether to return PIL image. Defaults to True.

    Returns:
        image (ndarray): Output 8 bit image(s) [0, 255].

    """
    if dynamic_range is None:
        dynamic_range = (-60, 0)

    image = ops.clip(image, *dynamic_range)
    image = translate(image, dynamic_range, (0, 255))
    image = ops.convert_to_numpy(image)
    image = image.astype(np.uint8)
    if pillow:
        image = Image.fromarray(image)
    return image


def scan_convert_2d(
    image,
    rho_range: Tuple,
    theta_range: Tuple,
    resolution: Union[float, None] = None,
    method: str = "linear",
):
    """
    Perform scan conversion on a 2D ultrasound image from polar coordinates
    (rho, theta) to Cartesian coordinates (x, z).

    Args:
        image (ndarray): The input 2D ultrasound image in polar coordinates.
            Has dimensions (n_rho, n_theta).
        rho_range (tuple): A tuple specifying the range of rho values
            (min_rho, max_rho). Defined in meters.
        theta_range (tuple): A tuple specifying the range of theta values
            (min_theta, max_theta). Defined in radians.
        resolution (float, optional): The resolution for the Cartesian grid.
            If None, it is calculated based on the input image.
        method (str, optional): The interpolation method to use. Defaults to
            'linear'. See `scipy.interpolate.interpn` for available methods.

    Returns:
        ndarray: The scan-converted 2D ultrasound image in Cartesian coordinates.
            Has dimensions (n_z, n_x). Coordinates outside the input image
            ranges are filled with NaNs.

    Note:
        Polar grid is inferred from the input image shape and the supplied
        rho and theta ranges. Cartesian grid is computed based on polar grid
        with resolutions specified by resolution parameter.

    TODO: Change `scipy.interpolate.interpn` to keras equivalent (requires
        custom implementation currently on keras 3.6)

    """

    rho = ops.linspace(rho_range[0], rho_range[1], image.shape[0], dtype=image.dtype)
    theta = ops.linspace(
        theta_range[0], theta_range[1], image.shape[1], dtype=image.dtype
    )

    rho_grid, theta_grid = ops.meshgrid(rho, theta, indexing="ij")

    x_grid, z_grid = frustum_convert_rt2xz(rho_grid, theta_grid)

    x_lim = [ops.min(x_grid), ops.max(x_grid)]
    z_lim = [ops.min(z_grid), ops.max(z_grid)]

    if resolution is None:
        d_rho = rho[1] - rho[0]
        d_theta = theta[1] - theta[0]
        # arc length along constant phi at 1/4 depth
        sRT = 0.25 * (rho[0] + rho[-1]) * d_theta
        # average of arc lengths and radial step
        resolution = ops.mean([sRT, d_rho])

    x_lim = [ops.min(x_grid), ops.max(x_grid)]
    z_lim = [ops.min(z_grid), ops.max(z_grid)]

    x_vec = ops.arange(x_lim[0], x_lim[1], resolution)
    z_vec = ops.arange(z_lim[0], z_lim[1], resolution)

    z_grid, x_grid = ops.meshgrid(z_vec, x_vec)

    rho_grid_interp, theta_grid_interp = frustum_convert_xz2rt(
        x_grid, z_grid, theta_limits=[theta[0], theta[-1]]
    )

    xi = ops.stack([rho_grid_interp, theta_grid_interp], axis=-1)

    image = ops.convert_to_numpy(image)
    rho = ops.convert_to_numpy(rho)
    theta = ops.convert_to_numpy(theta)
    xi = ops.convert_to_numpy(xi)
    image_sc = scipy.interpolate.interpn(
        (rho, theta), image, xi, method=method, bounds_error=False
    )
    image_sc = ops.convert_to_tensor(image_sc)
    image_sc = ops.transpose(image_sc)

    return image_sc


def scan_convert_3d(
    image,
    rho_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    phi_range: Tuple[float, float],
    resolution: Union[float, None] = None,
    method: str = "linear",
):
    """
    Perform scan conversion on a 3D ultrasound image from polar coordinates
    (rho, theta, phi) to Cartesian coordinates (z, x, y).

    Args:
        image (ndarray): The input 3D ultrasound image in polar coordinates.
            Has dimensions (n_rho, n_theta, n_phi).
        rho_range (tuple): A tuple specifying the range of rho values
            (min_rho, max_rho). Defined in meters.
        theta_range (tuple): A tuple specifying the range of theta values
            (min_theta, max_theta). Defined in radians.
        phi_range (tuple): A tuple specifying the range of phi values
            (min_phi, max_phi). Defined in radians.
        resolution (float, optional): The resolution for the Cartesian grid.
            If None, it is calculated based on the input image.
        method (str, optional): The interpolation method to use. Defaults to
            'linear'. See `scipy.interpolate.interpn` for available methods.

    Returns:
        ndarray: The scan-converted 3D ultrasound image in Cartesian coordinates.
            Has dimensions (n_z, n_x, n_y). Coordinates outside the input image
            ranges are filled with NaNs.

    Note:
        Polar grid is inferred from the input image shape and the supplied
        rho, theta and phi ranges. Cartesian grid is computed based on polar grid
        with resolutions specified by resolution parameter.

    TODO: Change `scipy.interpolate.interpn` to keras equivalent (requires
        custom implementation currently on keras 3.6)
    """

    rho = ops.linspace(rho_range[0], rho_range[1], image.shape[0], dtype=image.dtype)
    theta = ops.linspace(
        theta_range[0], theta_range[1], image.shape[1], dtype=image.dtype
    )
    phi = ops.linspace(phi_range[0], phi_range[1], image.shape[2], dtype=image.dtype)

    rho_grid, theta_grid, phi_grid = ops.meshgrid(rho, theta, phi, indexing="ij")

    x_grid, y_grid, z_grid = frustum_convert_rtp2xyz(rho_grid, theta_grid, phi_grid)

    x_lim = [ops.min(x_grid), ops.max(x_grid)]
    y_lim = [ops.min(y_grid), ops.max(y_grid)]
    z_lim = [ops.min(z_grid), ops.max(z_grid)]

    lims = ops.array([x_lim, y_lim, z_lim])

    if resolution is None:
        d_rho = rho[1] - rho[0]
        d_theta = theta[1] - theta[0]
        d_phi = phi[1] - phi[0]

        # arc length along constant phi at 1/4 depth
        sRT = 0.25 * (rho[0] + rho[-1]) * d_theta
        # arc length along constant theta at 1/4 depth
        sRP = 0.25 * (rho[0] + rho[-1]) * d_phi
        # average of arc lengths and radial step
        resolution = ops.mean([sRT, sRP, d_rho])

    dims = ops.round(ops.abs(ops.diff(lims, axis=1)) / resolution / 16) * 16
    dims = ops.maximum(dims, 1)

    dim_centers = 0.5 * ops.array(dims)
    lim_centers = ops.mean(lims, axis=1)

    # create vectors x, y, z centered at the center of the volume at the resolution of the volume
    x_vec = (ops.arange(dims[0][0]) - dim_centers[0]) * resolution + lim_centers[0]
    y_vec = (ops.arange(dims[1][0]) - dim_centers[1]) * resolution + lim_centers[1]
    z_vec = (ops.arange(dims[2][0]) - dim_centers[2]) * resolution + lim_centers[2]

    z_grid, x_grid, y_grid = ops.meshgrid(z_vec, x_vec[::-1], y_vec[::-1])

    rho_grid_interp, theta_grid_interp, phi_grid_interp = frustum_convert_xyz2rtp(
        x_grid,
        y_grid,
        z_grid,
        theta_limits=[theta[0], theta[-1]],
        phi_limits=[phi[0], phi[-1]],
    )

    xi = ops.stack([rho_grid_interp, theta_grid_interp, phi_grid_interp], axis=-1)

    image = ops.convert_to_numpy(image)
    rho = ops.convert_to_numpy(rho)
    theta = ops.convert_to_numpy(theta)
    phi = ops.convert_to_numpy(phi)
    xi = ops.convert_to_numpy(xi)
    volume = scipy.interpolate.interpn(
        (rho, theta, phi), image, xi, method=method, bounds_error=False
    )
    volume = ops.convert_to_tensor(volume)
    volume = ops.transpose(volume, (1, 0, 2))

    return volume


def cart2pol(x, y):
    """Convert x, y cartesian coordinates to polar coordinates theta, rho."""
    theta = ops.mod(ops.arctan2(x, -y), np.pi * 2)
    rho = ops.sqrt(x**2 + y**2)
    return (theta, rho)


def transform_sc_image_to_polar(image_sc, output_size=None, fit_outline=True):
    """
    Transform a scan converted input image (cone) into square
        using radial stretching and downsampling. Note that it assumes the background to be zero!
        Please verify if your results make sense, especially if the image contains black parts
        at the edges. This function is not perfect by any means, but it works for most cases.

    Args:
        image (numpy.ndarray): Input image as a 2D numpy array (height, width).
        output_size (tuple, optional): Output size of the image as a tuple.
            Defaults to image_sc.shape.
        fit_outline (bool, optional): Whether to fit a polynomial the outline of the image.
            Defaults to True. If this is set to False, and the ultrasound image contains
            some black parts at the edges, weird artifacts can occur, because the jagged outline
            is stretched to the desired width.

    Returns:
        numpy.ndarray: Squared image as a 2D numpy array (height, width).
    """
    assert len(image_sc.shape) == 2, "function only allows for 2D data"

    # Default output size is the input size
    if output_size is None:
        output_size = image_sc.shape

    # Initialize an empty target array for polar_image
    polar_image = np.zeros_like(image_sc)

    # Flip along the x axis (such that curve of image_sc is pointing up)
    flipped_image = np.flip(image_sc, axis=0)

    # Find index of first non zero element along y axis (for every vertical line)
    non_zeros_flipped = find_first_nonzero_index(flipped_image, 0)

    # Remove any black vertical lines (columns) that do not contain image data
    remove_vertical_lines = np.where(non_zeros_flipped == -1)[0]
    polar_image = np.delete(polar_image, remove_vertical_lines, axis=1)
    non_zeros_flipped = np.delete(non_zeros_flipped, remove_vertical_lines)

    if fit_outline:
        model_fitted_bottom = np.poly1d(
            np.polyfit(range(len(non_zeros_flipped)), non_zeros_flipped, 4)
        )
        non_zeros_flipped = model_fitted_bottom(range(len(non_zeros_flipped)))
        non_zeros_flipped = non_zeros_flipped.round().astype(np.int64)

    non_zeros = polar_image.shape[0] - non_zeros_flipped

    # Find the middle of the width of the image
    width = polar_image.shape[1]
    width_middle = round(width / 2)

    # For every vertical line in the image
    for x_i in range(width):
        # Move the flipped first non-zero element to the bottom of the image
        polar_image[non_zeros_flipped[x_i] :, x_i] = image_sc[: non_zeros[x_i], x_i]

    # Find indices of first and last non-zero element along x axis (for every horizontal line)
    non_zeros_left = find_first_nonzero_index(polar_image, 1)
    non_zeros_right = width - find_first_nonzero_index(
        np.flip(polar_image, 1), 1, width_middle
    )

    # Remove any black horizontal lines (rows) that do not contain image data
    remove_horizontal_lines = np.max(np.where(non_zeros_left == -1)) + 1
    polar_image = polar_image[remove_horizontal_lines:, :]
    non_zeros_left = non_zeros_left[remove_horizontal_lines:]
    non_zeros_right = non_zeros_right[remove_horizontal_lines:]

    if fit_outline:
        model_fitted_left = np.poly1d(
            np.polyfit(range(len(non_zeros_left)), non_zeros_left, 2)
        )
        non_zeros_left = model_fitted_left(range(len(non_zeros_left)))
        non_zeros_left = non_zeros_left.round().astype(np.int64)

        model_fitted_right = np.poly1d(
            np.polyfit(range(len(non_zeros_right)), non_zeros_right, 2)
        )
        non_zeros_right = model_fitted_right(range(len(non_zeros_right)))
        non_zeros_right = non_zeros_right.round().astype(np.int64)

    # For every horizontal line in the image
    for y_i in range(polar_image.shape[0]):
        small_array = polar_image[y_i, non_zeros_left[y_i] : non_zeros_right[y_i]]

        if len(small_array) <= 1:
            # If the array is too small for interpolation, set it to the middle value.
            polar_image[y_i, :] = polar_image[y_i, width_middle]
        else:
            # Perform linear interpolation to stretch the line to the desired width.
            array_interp = scipy.interpolate.interp1d(
                np.arange(small_array.size), small_array
            )
            polar_image[y_i, :] = array_interp(
                np.linspace(0, small_array.size - 1, width)
            )

    # Resize image to output_size
    return resize(polar_image, output_size, preserve_range=True)


def frustum_convert_rtp2xyz(rho, theta, phi):
    """Convert coordinates from (rho, theta, phi) space to (X,Y,Z) space using
    the frustum coordinate conversion.

    Angles are defined in radians.

    Args:
        rho (ndarray): Radial coordinates of the points to convert.
        theta (ndarray): Theta coordinates of the points to convert.
        phi (ndarray): Phi coordinates of the points to convert.

    Returns:
        x (ndarray): X coordinates of the converted points.
        y (ndarray): Y coordinates of the converted points.
        z (ndarray): Z coordinates of the converted points.
    """
    if ops.size(rho) != ops.size(theta) or ops.size(rho) != ops.size(phi):
        raise ValueError("Number of elements in rho, theta, and phi should be the same")

    z = rho / ops.sqrt(1 + ops.tan(theta) ** 2 + ops.tan(phi) ** 2)
    x = z * ops.tan(theta)
    y = z * ops.tan(phi)

    return x, y, z


def frustum_convert_rt2xz(rho, theta):
    """Convert coordinates from (rho, theta) space to (X,Z) space using
    the frustum coordinate conversion.

    Angles are defined in radians.

    Args:
        rho (ndarray): Radial coordinates of the points to convert.
        theta (ndarray): Theta coordinates of the points to convert.

    Returns:
        x (ndarray): X coordinates of the converted points.
        z (ndarray): Z coordinates of the converted points.
    """
    if ops.size(rho) != ops.size(theta):
        raise ValueError("Number of elements in rho and theta should be the same")

    z = rho / ops.sqrt(1 + ops.tan(theta) ** 2)
    x = z * ops.tan(theta)

    return x, z


def frustum_convert_xz2rt(x, z, theta_limits):
    """Convert coordinates from (X,Z) space to (rho, theta) space using
    the frustum coordinate conversion.

    Angles are defined in radians.

    Args:
        x (ndarray): X coordinates of the points to convert.
        z (ndarray): Z coordinates of the points to convert.
        theta_limits (list): Theta limits of the original volume. Any
            point that resides outside of these limits is potentially
            undefined, and therefore, the radial value for these points is
            made to be -1.

    Returns:
        rho (ndarray): Radial coordinates of the converted points.
        theta (ndarray): Theta coordinates of the converted points.
    """
    if ops.size(x) != ops.size(z):
        raise ValueError("Number of elements in x and z should be the same")

    rho = ops.sqrt(x**2 + z**2)
    theta = ops.arctan2(x, z)

    rho = ops.where(
        (rho < 0) | (theta < theta_limits[0]) | (theta > theta_limits[1]),
        -1,
        rho,
    )

    return rho, theta


def frustum_convert_xyz2rtp(x, y, z, theta_limits, phi_limits):
    """Convert coordinates from (X,Y,Z) space to (rho, theta, phi) space using
    the frustum coordinate conversion.

    Angles are defined in radians.

    Args:
        x (ndarray): X coordinates of the points to convert.
        y (ndarray): Y coordinates of the points to convert.
        z (ndarray): Z coordinates of the points to convert.
        tlimits, plimits:
            Theta and phi limits, respectively, of the original volume. Any
            point that resides outside of these limits is potentially
            undefined, and therefore, the radial value for these points is
            made to be -1.

    Returns:
        rho (ndarray): Radial coordinates of the converted points.
        theta (ndarray): Theta coordinates of the converted points.
        phi (ndarray): Phi coordinates of the converted points.
    """
    if ops.size(x) != ops.size(y) or ops.size(x) != ops.size(z):
        raise ValueError("Number of elements in x, y, and z should be the same")

    rho = ops.sqrt(x**2 + y**2 + z**2)
    theta = ops.arctan2(x, z)
    phi = ops.arctan2(y, z)

    rho = ops.where(
        (rho < 0)
        | (theta < theta_limits[0])
        | (theta > theta_limits[1])
        | (phi < phi_limits[0])
        | (phi > phi_limits[1]),
        -1,
        rho,
    )

    return rho, theta, phi
