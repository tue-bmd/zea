"""Display functionality, including scan conversion frustrum conversion, etc."""

from functools import partial
from typing import Tuple, Union

import numpy as np
import scipy
from keras import ops
from PIL import Image

from zea.func.tensor import translate
from zea.tools.fit_scan_cone import fit_and_crop_around_scan_cone


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

    image = ops.convert_to_numpy(image)
    image = np.clip(image, *dynamic_range)
    image = translate(image, dynamic_range, (0, 255))
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

    d_rho = rho[1] - rho[0]
    d_theta = theta[1] - theta[0]

    if resolution is None:
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
    theta_idx = (theta_grid_interp - theta_min) / (theta_max - theta_min) * (image_shape[-1] - 1)
    # Stack coordinates as required for map_coordinates
    coordinates = ops.stack([rho_idx, theta_idx], axis=0)
    parameters = {
        "resolution": resolution,
        "x_lim": x_lim,
        "z_lim": z_lim,
        "rho_range": rho_range,
        "theta_range": theta_range,
        "d_rho": d_rho,
        "d_theta": d_theta,
    }
    return coordinates, parameters


def scan_convert_2d(
    image,
    rho_range: Tuple[float, float] = None,
    theta_range: Tuple[float, float] = None,
    resolution: Union[float, None] = None,
    coordinates: Union[None, np.ndarray] = None,
    fill_value: float = 0.0,
    order: int = 1,
    **kwargs,
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
            Has dimensions (grid_size_z, grid_size_x). Coordinates outside the input image
            ranges are filled with NaNs.
        parameters (dict): A dictionary containing information about the scan conversion.
            Contains the resolution, x, and z limits, rho and theta ranges.

    Note:
        Polar grid is inferred from the input image shape and the supplied
        rho and theta ranges. Cartesian grid is computed based on polar grid
        with resolutions specified by resolution parameter.

    """
    assert "float" in ops.dtype(image), "Image must be float type"

    parameters = {}
    if coordinates is None:
        coordinates, parameters = compute_scan_convert_2d_coordinates(
            image.shape, rho_range, theta_range, resolution, dtype=image.dtype
        )

    images_sc = _interpolate_batch(image, coordinates, fill_value, order=order, **kwargs)

    # swap axis to match z, x
    images_sc = ops.swapaxes(images_sc, -1, -2)

    return images_sc, parameters


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

    d_rho = rho[1] - rho[0]
    d_theta = theta[1] - theta[0]
    d_phi = phi[1] - phi[0]

    if resolution is None:
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
    theta_idx = (theta_grid_interp - theta_min) / (theta_max - theta_min) * (image_shape[-2] - 1)
    phi_idx = (phi_grid_interp - phi_min) / (phi_max - phi_min) * (image_shape[-1] - 1)

    # Stack coordinates as required for map_coordinates
    coordinates = ops.stack([rho_idx, theta_idx, phi_idx], axis=0)
    parameters = {
        "resolution": resolution,
        "x_lim": x_lim,
        "y_lim": y_lim,
        "z_lim": z_lim,
        "rho_range": rho_range,
        "theta_range": theta_range,
        "phi_range": phi_range,
        "d_rho": d_rho,
        "d_theta": d_theta,
        "d_phi": d_phi,
    }
    return coordinates, parameters


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
            Has dimensions (grid_size_z, grid_size_x, n_y). Coordinates outside the input image
            ranges are filled with NaNs.
        parameters (dict): A dictionary containing information about the scan conversion.
            Contains the resolution, x, y, and z limits, rho, theta, and phi ranges.

    Note:
        Polar grid is inferred from the input image shape and the supplied
        rho, theta and phi ranges. Cartesian grid is computed based on polar grid
        with resolutions specified by resolution parameter.
    """
    assert "float" in ops.dtype(image), "Image must be float type"

    parameters = {}
    if coordinates is None:
        coordinates, parameters = compute_scan_convert_3d_coordinates(
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
    return images_sc, parameters


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
            f"Image must be 2D or 3D (with optional batch dim). Got shape: {image.shape}"
        )


def map_coordinates(inputs, coordinates, order, fill_mode="constant", fill_value=0):
    """map_coordinates using keras.ops or scipy.ndimage when order > 1."""
    if order > 1:
        # Preserve original dtype before conversion
        original_dtype = ops.dtype(inputs)
        inputs_np = ops.convert_to_numpy(inputs).astype(np.float32)
        coordinates_np = ops.convert_to_numpy(coordinates).astype(np.float32)
        out = scipy.ndimage.map_coordinates(
            inputs_np, coordinates_np, order=order, mode=fill_mode, cval=fill_value
        )
        return ops.convert_to_tensor(out.astype(original_dtype))
    else:
        return ops.image.map_coordinates(
            inputs,
            coordinates,
            order=order,
            fill_mode=fill_mode,
            fill_value=fill_value,
        )


def _interpolate_batch(images, coordinates, fill_value=0.0, order=1, vectorize=True):
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
    elif not vectorize:
        images_sc = ops.map(map_coordinates_fn, images)
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


def rotate_coordinates(coords, angle_deg):
    """Rotate (x, y) coordinates by a given angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = ops.array(
        [
            [ops.cos(angle_rad), -ops.sin(angle_rad)],
            [ops.sin(angle_rad), ops.cos(angle_rad)],
        ],
        dtype=coords.dtype,
    )
    return coords @ ops.transpose(rotation_matrix)


def cartesian_to_polar_matrix(
    cartesian_matrix,
    fill_value=0.0,
    polar_shape=None,
    tip=None,
    r_max=None,
    angle=None,
    interpolation_order=1,
):
    """
    Convert a Cartesian image matrix to a polar coordinate representation.

    Args:
        cartesian_matrix (tensor): Input 2D image array in Cartesian coordinates.
        fill_value (float): Value to use for points sampled outside the input image.
        polar_shape (tuple, optional): Desired shape of the polar output (rows, cols).
            Defaults to the shape of the input image.
        tip (tuple, optional): (x, y) coordinates of the origin for the polar
            transformation (typically the probe tip). Defaults to the center-top of the image.
        r_max (float, optional): Maximum radius to consider in the polar transform.
            Defaults to the height of the input image.
        angle (float, optional): Total angular field of view (in radians) centered at 0.
            The polar grid spans from -angle to +angle. Defaults to π/4 radians (45 degrees).
        interpolation_order (int): Order of interpolation to use (0 = nearest-neighbor,
            1 = linear, 2+ = spline). Matches the convention of `scipy.ndimage.map_coordinates`.

    Returns:
        polar_matrix (Array): The image re-sampled in polar coordinates with shape `polar_shape`.
    """
    assert "float" in ops.dtype(cartesian_matrix), "Input image must be float type"

    # Default angle to π/4 radians (45 degrees)
    if angle is None:
        angle = np.deg2rad(45)

    # Assume that polar grid is same shape as cartesian grid unless specified
    cartesian_rows, cartesian_cols = ops.shape(cartesian_matrix)
    if polar_shape is None:
        polar_rows, polar_cols = cartesian_rows, cartesian_cols
    else:
        polar_rows, polar_cols = polar_shape

    # assume tip is at center top unless specified
    if tip is None:
        center_x = cartesian_cols / 2  # center_x can be between two pixels
        tip_y = 0
        tip = (center_x, tip_y)

    # assume r_max is the total height of the input image unless specified
    if r_max is None:
        r_max = cartesian_rows

    center_x, center_y = tip

    # Interpolation grid in polar coordinates
    r = ops.linspace(0, r_max, polar_rows, dtype="float32")
    theta = ops.linspace(-angle, angle, polar_cols, dtype="float32")
    r_grid, theta_grid = ops.meshgrid(r, theta)

    # convert discretized radii and angle intervals to polar coordinates
    x_polar = r_grid * ops.cos(theta_grid)
    y_polar = r_grid * ops.sin(theta_grid)

    # Inverse rotation to match original orientation
    polar_coords = ops.stack([ops.ravel(x_polar), ops.ravel(y_polar)], axis=0)
    polar_coords_rotated = ops.transpose(rotate_coordinates(ops.transpose(polar_coords), 90))

    # Shift to image indices
    yq = polar_coords_rotated[1, :] + center_y
    xq = polar_coords_rotated[0, :] + center_x
    coords_for_interp = ops.stack([yq, xq])

    polar_values = map_coordinates(
        cartesian_matrix,
        coords_for_interp,
        order=interpolation_order,
        fill_mode="constant",
        fill_value=fill_value,
    )

    polar_matrix = ops.rot90(ops.reshape(polar_values, (polar_cols, polar_rows)), k=-1)
    return polar_matrix


def inverse_scan_convert_2d(
    cartesian_image,
    fill_value=0.0,
    angle=None,
    output_size=None,
    interpolation_order=1,
    find_scan_cone=True,
    image_range: tuple | None = None,
):
    """
    Convert a Cartesian-format ultrasound image to a polar representation.

    This function can be used to recover a sector-shaped scan (polar format)
    from a Cartesian representation of an image.
    Optionally, it can detect and crop around the scan cone before conversion.

    Args:
        cartesian_image (tensor): 2D image array in Cartesian coordinates of type float.
        fill_value (float): Value used to fill regions outside the original image
            during interpolation.
        angle (float, optional): Angular field of view (in radians) used for the polar
            transformation. The polar output will span from -angle to +angle.
            Defaults to π/4 radians (45 degrees).
        output_size (tuple, optional): Shape (rows, cols) of the resulting polar image.
            If None, the shape of the input image is used.
        interpolation_order (int): Order of interpolation used in resampling
            (0 = nearest-neighbor, 1 = linear, etc.).
        find_scan_cone (bool): If True, automatically detects and crops around the scan cone
            in the Cartesian image before polar conversion, ensuring that the scan cone is
            centered without padding. Can be set to False if the image is already cropped
            and centered.
        image_range (tuple, optional): Tuple (vmin, vmax) for display scaling
            when detecting the scan cone.

    Returns:
        polar_image (Array): 2D image in polar coordinates (sector-shaped scan).
    """

    if find_scan_cone:
        assert image_range is not None, "image_range must be provided when find_scan_cone is True"
        cartesian_image = fit_and_crop_around_scan_cone(cartesian_image, image_range)

    polar_image = cartesian_to_polar_matrix(
        cartesian_image,
        fill_value=fill_value,
        angle=angle,
        polar_shape=output_size,
        interpolation_order=interpolation_order,
    )
    return polar_image


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
