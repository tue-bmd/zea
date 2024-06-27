"""Display functionality for USBMD.

All functionality related to displaying ultrasound images.

- **Author(s)**     : Tristan Stevens
- **Date**          : 02/11/2023
"""

import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.ndimage import map_coordinates
from skimage.transform import resize

from usbmd.utils import find_first_nonzero_index, translate


def to_8bit(image, dynamic_range: tuple = None, pillow: bool = True):
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

    image = np.clip(image, *dynamic_range)
    image = translate(image, dynamic_range, (0, 255))
    image = np.array(image, dtype=np.float32)
    image = image.astype(np.uint8)
    if pillow:
        image = Image.fromarray(image)
    return image


def scan_convert(image, x_axis, z_axis, n_pixels=500, spline_order=1, fill_value=0):
    """Scan conversion method for ultrasound data.

    Args:
        image (ndarray): Input image (in polar coordinates).
        x_axis (ndarray): linspace of the angles
        z_axis (ndarray): linspace of the depth
        n_pixels (int): resolution width of the image
        spline_order (int, optional): Order of spline interpolation.
            Defaults to 1.
        fill_value (float, optional): Value of the points that cannot be
            mapped from sample_points to grid. Defaults to 0.

    Returns:
        image_sc (ndarray): Output image (converted to cartesian coordinates).

    """
    image_shape = image.shape
    assert len(image_shape) >= 2, "function requires 2D data or more (batch dims)"

    z_min = np.min(z_axis)
    z_max = np.max(z_axis)
    # aspect ratio of image
    x_max = np.sqrt(2) * z_max

    # dx = dz is the pixel size
    dx = x_max / n_pixels

    x_grid_points = np.arange(-x_max / 2, x_max / 2, dx)
    z_grid_points = np.arange(z_min, z_max, dx)

    x_sample_points = np.deg2rad(x_axis) + np.pi / 2
    z_sample_points = z_axis

    # to allow for batch processing
    if len(image_shape) != 2:
        images = np.reshape(image, (-1, *image_shape[-2:]))
    else:
        images = np.expand_dims(image, axis=0)

    batch_dims = list(image_shape[:-2])

    images_sc = []
    for _image in images:
        image_sc = project_to_cartesian_grid(
            _image,
            (x_sample_points, z_sample_points),
            (x_grid_points, z_grid_points),
            spline_order=spline_order,
            fill_value=fill_value,
        )
        images_sc.append(image_sc)
    image_sc = np.stack(images_sc).reshape(*batch_dims, *image_sc.shape)
    return image_sc


def project_to_cartesian_grid(
    image, sample_points, grid, spline_order=1, fill_value=None
):
    """Project polar data onto a cartesian grid.

    Args:
        image (ndarray): Input image (polar) with 2 dimensions.
        sample_points (Tuple of 2 2D arrays): Meshgrid of x (2D) and z (2D)
            points. Define the coordinates on which the input image are sampled.
        grid (Tuple of 2 2D arrays): Meshgrid of x (2D) and z (2D)
            points. Define the coordinates on which image should be mapped to.
        spline_order (int, optional): spline order. Defaults to 1.
        fill_value (float, optional): Value of the points that cannot be
            mapped from sample_points to grid. Defaults to None.

    Returns:
        image_sc (ndarray): Scan converted image (cartesian).

    Raises:
        AssertionError('function only allows for 2D data')

    """
    assert len(image.shape) == 2, "function only allows for 2D data"

    x_sample_points, z_sample_points = sample_points
    x_grid_points, z_grid_points = grid
    [grid_x, grid_z] = np.meshgrid(x_grid_points, z_grid_points)
    [theta, radius] = cart2pol(grid_z, grid_x)

    image_sc = interp2(
        x_sample_points,
        z_sample_points,
        image,
        theta,
        radius,
        spline_order=spline_order,
        fill_value=fill_value,
    )
    return image_sc


def interp2(x, y, z, xq, yq, spline_order=1, fill_value=None):
    """Interpolate to target grid.

    In MatLab: Vq = interp2(X, Y, V, Xq, Yq)

    Args:
        x (2darray): x meshgrid of input image.
        y (2darray): y meshgrid of input image.
        z (2darray): input image.
        xq (2darray): x meshgrid of output image.
        yq (2darray): y meshgrid of output image.
        spline_order (int, optional): spline order. Defaults to 1.
        fill_value (float, optional): fill value for values outside
            interpolation region. Defaults to None.

    Returns:
        zq (2darray): Interpolated output array defined on grid by xq and yq.

    """
    rows, cols = z.shape

    output_shape = xq.shape

    xq = xq.flatten()
    yq = yq.flatten()

    s = 1 + (xq - x[0]) / (x[-1] - x[0]) * (cols - 1)
    t = 1 + (yq - y[0]) / (y[-1] - y[0]) * (rows - 1)

    if fill_value is None:
        fill_value = -np.infty
    zq = map_coordinates(z, [t, s], order=spline_order, cval=fill_value)
    zq = zq.reshape(output_shape)
    return zq


def cart2pol(x, y):
    """Convert x, y cartesian coordinates to polar coordinates theta, rho."""
    theta = np.mod(np.arctan2(x, -y), np.pi * 2)
    rho = np.sqrt(x**2 + y**2)
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
            array_interp = interpolate.interp1d(
                np.arange(small_array.size), small_array
            )
            polar_image[y_i, :] = array_interp(
                np.linspace(0, small_array.size - 1, width)
            )

    # Resize image to output_size
    return resize(polar_image, output_size, preserve_range=True)
