"""Display functionality for USBMD.

All functionality related to displaying ultrasound images.

- **Author(s)**     : Tristan Stevens
- **Date**          : 02/11/2023
"""

from functools import partial
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


def compute_scan_convert_2d_coordinates(
    image_shape,
    rho_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    resolution: Union[float, None] = None,
    dtype: str = "float32",
):
    """Precompute coordinates for 2d scan conversion from polar coordinates"""
    assert len(rho_range) == 2, "rho_range should be a tuple of length 2"
    assert len(theta_range) == 2, "theta_range should be a tuple of length 2"
    assert rho_range[0] < rho_range[1], "min_rho should be less than max_rho"

    rho = ops.linspace(rho_range[0], rho_range[1], image_shape[-2], dtype=dtype)
    theta = ops.linspace(theta_range[0], theta_range[1], image_shape[-1], dtype=dtype)

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
        resolution = ops.mean([sRT, d_rho])  # mm per pixel

    x_vec = ops.arange(x_lim[0], x_lim[1], resolution)
    z_vec = ops.arange(z_lim[0], z_lim[1], resolution)

    z_grid, x_grid = ops.meshgrid(z_vec, x_vec)

    rho_grid_interp, theta_grid_interp = frustum_convert_xz2rt(
        x_grid, z_grid, theta_limits=[theta[0], theta[-1]]
    )

    # Map rho and theta interpolation points to grid indices
    rho_min, rho_max = ops.min(rho), ops.max(rho)
    theta_min, theta_max = ops.min(theta), ops.max(theta)
    rho_idx = (rho_grid_interp - rho_min) / (rho_max - rho_min) * (image_shape[-2] - 1)
    theta_idx = (
        (theta_grid_interp - theta_min)
        / (theta_max - theta_min)
        * (image_shape[-1] - 1)
    )
    # Stack coordinates as required for map_coordinates
    coordinates = ops.stack([rho_idx, theta_idx], axis=0)
    return coordinates


def scan_convert_2d(
    image,
    rho_range: Tuple[float, float] = None,
    theta_range: Tuple[float, float] = None,
    resolution: Union[float, None] = None,
    coordinates: Union[None, np.ndarray] = None,
    fill_value: float = 0.0,
    order: int = 1,
):
    """
    Perform scan conversion on a 2D ultrasound image from polar coordinates
    (rho, theta) to Cartesian coordinates (x, z).

    Args:
        image (ndarray): The input 2D ultrasound image in polar coordinates.
            Has dimensions (n_rho, n_theta) with optional batch.
        rho_range (tuple): A tuple specifying the range of rho values
            (min_rho, max_rho). Defined in mm.
        theta_range (tuple): A tuple specifying the range of theta values
            (min_theta, max_theta). Defined in radians.
        resolution (float, optional): The resolution for the Cartesian grid.
            If None, it is calculated based on the input image. In mm / pixel.
        coordinates (ndarray, optional): Precomputed coordinates for scan conversion.
            If provided, it will be used instead of computing new coordinates based on
            the input image shape and ranges.
        fill_value (float, optional): The value to fill in for coordinates
            outside the input image ranges. Defaults to 0.0. When set to NaN,
            no interpolation at the edges will happen.
        order (int, optional): The order of the spline interpolation. Defaults to 1.

    Returns:
        ndarray: The scan-converted 2D ultrasound image in Cartesian coordinates.
            Has dimensions (n_z, n_x). Coordinates outside the input image
            ranges are filled with NaNs.

    Note:
        Polar grid is inferred from the input image shape and the supplied
        rho and theta ranges. Cartesian grid is computed based on polar grid
        with resolutions specified by resolution parameter.

    """
    assert "float" in ops.dtype(image), "Image must be float type"

    if coordinates is None:
        coordinates = compute_scan_convert_2d_coordinates(
            image.shape, rho_range, theta_range, resolution, dtype=image.dtype
        )

    images_sc = _interpolate_batch(image, coordinates, fill_value, order=order)

    # swap axis to match z, x
    images_sc = ops.swapaxes(images_sc, -1, -2)

    return images_sc


def compute_scan_convert_3d_coordinates(
    image_shape,
    rho_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    phi_range: Tuple[float, float],
    resolution: Union[float, None] = None,
    dtype: str = "float32",
):
    """Precompute coordinates for 3d scan conversion from polar coordinates"""
    assert len(rho_range) == 2, "rho_range should be a tuple of length 2"
    assert len(theta_range) == 2, "theta_range should be a tuple of length 2"
    assert len(phi_range) == 2, "phi_range should be a tuple of length 2"
    assert rho_range[0] < rho_range[1], "min_rho should be less than max_rho"

    rho = ops.linspace(rho_range[0], rho_range[1], image_shape[-3], dtype=dtype)
    theta = ops.linspace(theta_range[0], theta_range[1], image_shape[-2], dtype=dtype)
    phi = ops.linspace(phi_range[0], phi_range[1], image_shape[-1], dtype=dtype)

    rho_grid, theta_grid, phi_grid = ops.meshgrid(rho, theta, phi, indexing="ij")

    x_grid, y_grid, z_grid = frustum_convert_rtp2xyz(rho_grid, theta_grid, phi_grid)

    x_lim = [ops.min(x_grid), ops.max(x_grid)]
    y_lim = [ops.min(y_grid), ops.max(y_grid)]
    z_lim = [ops.min(z_grid), ops.max(z_grid)]

    if resolution is None:
        d_rho = rho[1] - rho[0]
        d_theta = theta[1] - theta[0]
        d_phi = phi[1] - phi[0]

        # arc length along constant phi at 1/4 depth
        sRT = 0.25 * (rho[0] + rho[-1]) * d_theta
        # arc length along constant theta at 1/4 depth
        sRP = 0.25 * (rho[0] + rho[-1]) * d_phi
        # average of arc lengths and radial step
        resolution = ops.mean([sRT, sRP, d_rho])  # mm per pixel

    z_vec = ops.arange(z_lim[0], z_lim[1], resolution)
    x_vec = ops.arange(x_lim[0], x_lim[1], resolution)
    y_vec = ops.arange(y_lim[0], y_lim[1], resolution)

    z_grid, x_grid, y_grid = ops.meshgrid(z_vec, x_vec, y_vec)

    rho_grid_interp, theta_grid_interp, phi_grid_interp = frustum_convert_xyz2rtp(
        x_grid,
        y_grid,
        z_grid,
        theta_limits=[theta[0], theta[-1]],
        phi_limits=[phi[0], phi[-1]],
    )

    # return volume
    rho_min, rho_max = ops.min(rho), ops.max(rho)
    theta_min, theta_max = ops.min(theta), ops.max(theta)
    phi_min, phi_max = ops.min(phi), ops.max(phi)
    rho_idx = (rho_grid_interp - rho_min) / (rho_max - rho_min) * (image_shape[-3] - 1)
    theta_idx = (
        (theta_grid_interp - theta_min)
        / (theta_max - theta_min)
        * (image_shape[-2] - 1)
    )
    phi_idx = (phi_grid_interp - phi_min) / (phi_max - phi_min) * (image_shape[-1] - 1)

    # Stack coordinates as required for map_coordinates
    return ops.stack([rho_idx, theta_idx, phi_idx], axis=0)


def scan_convert_3d(
    image,
    rho_range: Tuple[float, float] = None,
    theta_range: Tuple[float, float] = None,
    phi_range: Tuple[float, float] = None,
    resolution: Union[float, None] = None,
    coordinates: Union[None, np.ndarray] = None,
    fill_value: float = 0.0,
    order: int = 1,
):
    """
    Perform scan conversion on a 3D ultrasound image from polar coordinates
    (rho, theta, phi) to Cartesian coordinates (z, x, y).

    Args:
        image (ndarray): The input 3D ultrasound image in polar coordinates.
            Has dimensions (n_rho, n_theta, n_phi) with optional batch.
        rho_range (tuple): A tuple specifying the range of rho values
            (min_rho, max_rho). Defined in mm.
        theta_range (tuple): A tuple specifying the range of theta values
            (min_theta, max_theta). Defined in radians.
        phi_range (tuple): A tuple specifying the range of phi values
            (min_phi, max_phi). Defined in radians.
        resolution (float, optional): The resolution for the Cartesian grid.
            If None, it is calculated based on the input image. In mm / pixel.
        coodinates (ndarray, optional): Precomputed coordinates for scan conversion.
            If provided, it will be used instead of computing new coordinates based on
            the input image shape and ranges.
        fill_value (float, optional): The value to fill in for coordinates
            outside the input image ranges. Defaults to 0.0. When set to NaN,
            no interpolation at the edges will happen.
        order (int, optional): The order of the spline interpolation. Defaults to 1.

    Returns:
        ndarray: The scan-converted 3D ultrasound image in Cartesian coordinates.
            Has dimensions (n_z, n_x, n_y). Coordinates outside the input image
            ranges are filled with NaNs.

    Note:
        Polar grid is inferred from the input image shape and the supplied
        rho, theta and phi ranges. Cartesian grid is computed based on polar grid
        with resolutions specified by resolution parameter.
    """
    assert "float" in ops.dtype(image), "Image must be float type"

    if coordinates is None:
        coordinates = compute_scan_convert_3d_coordinates(
            image.shape,
            rho_range,
            theta_range,
            phi_range,
            resolution,
            dtype=image.dtype,
        )

    images_sc = _interpolate_batch(image, coordinates, fill_value, order=order)

    # swap axis to match z, x, y
    images_sc = ops.swapaxes(images_sc, -2, -3)
    return images_sc


def scan_convert(
    image,
    rho_range: Tuple[float, float] = None,
    theta_range: Tuple[float, float] = None,
    phi_range: Tuple[float, float] = None,
    resolution: Union[float, None] = None,
    coordinates: Union[None, np.ndarray] = None,
    fill_value: float = 0.0,
    order: int = 1,
    with_batch_dim: bool = False,
):
    """Scan convert image based on number of dimensions."""
    if len(image.shape) == 2 + int(with_batch_dim):
        return scan_convert_2d(
            image,
            rho_range,
            theta_range,
            resolution,
            coordinates,
            fill_value,
            order,
        )
    elif len(image.shape) == 3 + int(with_batch_dim):
        return scan_convert_3d(
            image,
            rho_range,
            theta_range,
            phi_range,
            resolution,
            coordinates,
            fill_value,
            order,
        )
    else:
        raise ValueError(
            "Image must be 2D or 3D (with optional batch dim). "
            f"Got shape: {image.shape}"
        )


def map_coordinates(inputs, coordinates, order, fill_mode="constant", fill_value=0):
    """map_coordinates using keras.ops or scipy.ndimage when order > 1."""
    if order > 1:
        inputs = ops.convert_to_numpy(inputs)
        out = scipy.ndimage.map_coordinates(
            inputs, coordinates, order=order, mode=fill_mode, cval=fill_value
        )
        return ops.convert_to_tensor(out)
    else:
        return ops.image.map_coordinates(
            inputs, coordinates, order=order, fill_mode=fill_mode, fill_value=fill_value
        )


def _interpolate_batch(images, coordinates, fill_value=0.0, order=1):
    """Interpolate a batch of images."""
    image_shape = images.shape
    num_image_dims = coordinates.shape[0]

    batch_dims = images.shape[:-num_image_dims]

    images = ops.reshape(images, (-1, *image_shape[-num_image_dims:]))

    map_coordinates_fn = partial(
        map_coordinates,
        coordinates=coordinates,
        order=order,
        fill_mode="constant",
        fill_value=fill_value,
    )

    if order > 1:
        # cpu bound
        images_sc = ops.stack(list(map(map_coordinates_fn, images)))
    else:
        # gpu bound
        images_sc = ops.vectorized_map(map_coordinates_fn, images)

    # ignore batch dim to get image shape
    image_sc_shape = ops.shape(images_sc)[1:]
    images_sc = ops.reshape(images_sc, (*batch_dims, *image_sc_shape))

    return images_sc


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
        non_zeros_flipped = np.clip(non_zeros_flipped, 0, None)

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
