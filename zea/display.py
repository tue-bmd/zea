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

    .. note::
        If dynamic_range is None, it is assumed that the input image is already in the range
        [-60, 0] dB, which is a common range for ultrasound images.

    .. note::
        NaN values in the input image are replaced with the minimum value of the dynamic range
        before scaling, which ensures that they are represented as black (0) in the output image.
        +/- inf values are replaced with the min and max values of the dynamic range.

    Example:
        .. doctest::

            >>> import numpy as np

            >>> import zea

            >>> file_path = (
            ...     "hf://zeahub/camus-sample/val/patient0401/patient0401_4CH_half_sequence.hdf5"
            ... )

            >>> with zea.File(file_path, mode="r") as file:
            ...     data = file.data.image[0]

            >>> image, _ = zea.display.scan_convert(
            ...     data,
            ...     rho_range=(0, 1),
            ...     theta_range=(-0.78, 0.78),
            ...     fill_value=np.nan,
            ... )
            >>> image = zea.display.to_8bit(image, dynamic_range=(-60, 0))
            >>> image.save("image.png")  # DOCTEST: +SKIP

    """
    if dynamic_range is None:
        dynamic_range = (-60, 0)

    image = ops.nan_to_num(image, nan=dynamic_range[0])
    image = ops.convert_to_numpy(image)
    image = np.clip(image, *dynamic_range)
    image = translate(image, dynamic_range, (0, 255))
    image = image.astype(np.uint8)
    if pillow:
        image = Image.fromarray(image)
    return image


def overlay_masks(
    image,
    masks,
    alpha: float = 0.5,
    colors=None,
):
    """Overlay segmentation masks on top of an image using PIL.

    Args:
        image (PIL.Image or ndarray): Base image. If grayscale, it is converted
            to RGB. If ndarray, it is converted to a PIL Image first.
        masks (list of PIL.Image or ndarray): Segmentation masks to overlay.
            Each mask should be an 8-bit single-channel image where non-zero
            pixels indicate the masked region.
        alpha (float, optional): Opacity of the mask overlays in [0, 1].
            Defaults to 0.5.
        colors (list of tuple, optional): RGB colors for each mask. If None,
            a default palette is used. If provided, must contain at least as
            many entries as masks (extra entries are ignored).

    Returns:
        PIL.Image: RGB image with masks overlaid.
    """
    # Validate alpha parameter before conversion to uint8
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in the range [0.0, 1.0], got {alpha}")

    _DEFAULT_COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Validate colors list has enough entries if provided
    if colors is not None and len(colors) < len(masks):
        raise ValueError(
            f"colors must have at least as many entries as masks: "
            f"got {len(colors)} colors for {len(masks)} masks"
        )

    result = image.copy()

    for i, mask in enumerate(masks):
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.asarray(mask))

        if mask.size != image.size:
            raise ValueError(f"Mask {i} size {mask.size} does not match image size {image.size}")

        if mask.mode != "L":
            mask = mask.convert("L")

        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] if colors is None else colors[i]

        # Create a solid color layer the same size as the image
        color_layer = Image.new("RGB", image.size, color)

        # Build alpha channel from the mask: scale mask values by alpha
        mask_np = (np.asarray(mask) > 0).astype(np.uint8)
        alpha_channel = Image.fromarray((mask_np * int(alpha * 255)).astype(np.uint8))

        result.paste(color_layer, mask=alpha_channel)

    return result


def compute_scan_convert_2d_coordinates(
    image_shape,
    rho_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    resolution: Union[float, None] = None,
    dtype: str = "float32",
    distance_to_apex: float = 0.0,
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
    z_vec = ops.arange(z_lim[0] + distance_to_apex, z_lim[1], resolution)

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
        "distance_to_apex": distance_to_apex,
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
    distance_to_apex: float = 0.0,
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
        distance_to_apex (float, optional): Distance from the apex to the
            start of the z-axis in Cartesian grid. Defaults to 0.0.

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
            image.shape,
            rho_range,
            theta_range,
            resolution,
            dtype=image.dtype,
            distance_to_apex=distance_to_apex,
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


def _polar_to_cartesian_coordinates(polar_shape, cartesian_shape, tip, r_max, theta_range):
    cart_rows, cart_cols = cartesian_shape
    polar_rows, polar_cols = polar_shape
    theta_min, theta_max = theta_range
    center_x, center_y = tip

    # Cartesian pixel coordinates
    y, x = ops.meshgrid(
        ops.arange(cart_rows),
        ops.arange(cart_cols),
        indexing="ij",
    )

    # Coordinates relative to probe tip
    dx = x - center_x
    dy = y - center_y

    # Undo the +90° rotation used in cartesian_to_polar_matrix
    x_unrot = dy
    y_unrot = -dx

    # Convert back to polar coordinates
    r = ops.sqrt(x_unrot**2 + y_unrot**2)
    theta = ops.arctan2(y_unrot, x_unrot)

    # Convert physical coordinates -> polar image indices
    r_idx = (r / r_max) * (polar_rows - 1)
    theta_idx = (theta - theta_min) / (theta_max - theta_min) * (polar_cols - 1)

    # Sample polar image
    return ops.stack([ops.ravel(r_idx), ops.ravel(theta_idx)], axis=0)


# TODO: we might be able to merge this with scan_convert_2d
# TODO: round-trip test
def polar_to_cartesian_matrix(
    polar_matrix,
    cartesian_shape,
    fill_value=0.0,
    tip=None,
    r_max=None,
    angle=None,
    theta_range=None,
    interpolation_order=1,
):
    """
    Approximate inverse of cartesian_to_polar_matrix.

    Parameters
    ----------
    polar_matrix : ndarray
        Polar image.
    cartesian_shape : tuple
        Desired output shape (rows, cols).
    fill_value : float
        Value assigned outside the polar domain.
    tip : tuple, optional
        (x, y) origin used in the forward transform.
    r_max : float, optional
        Maximum radius used in the forward transform.
    angle : float, optional
        Symmetric half-angle in radians.
    theta_range : tuple, optional
        (theta_min, theta_max).
    interpolation_order : int
        Passed to scipy.ndimage.map_coordinates.

    Returns
    -------
    cartesian_matrix : ndarray
    """

    assert angle is None or theta_range is None

    if theta_range is None and angle is None:
        theta_range = (-np.deg2rad(45), np.deg2rad(45))

    cart_rows, cart_cols = cartesian_shape

    if tip is None:
        tip = (cart_cols / 2, 0)

    if r_max is None:
        r_max = cart_rows

    coords = _polar_to_cartesian_coordinates(
        polar_matrix.shape, cartesian_shape, tip, r_max, theta_range
    )

    cartesian = map_coordinates(
        polar_matrix,
        coords,
        order=interpolation_order,
        fill_mode="constant",
        fill_value=fill_value,
    )

    return ops.reshape(cartesian, cartesian_shape)


def cartesian_to_polar_matrix(
    cartesian_matrix,
    fill_value=0.0,
    polar_shape=None,
    tip=None,
    r_max=None,
    angle=None,
    theta_range=None,
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
        angle (float, optional): Symmetric shorthand for ``theta_range=(-angle, angle)``,
            in radians. Mutually exclusive with ``theta_range``. Defaults to π/4 radians
            (45 degrees) when both ``angle`` and ``theta_range`` are None.
        theta_range (tuple, optional): ``(theta_min, theta_max)`` angular extent of the polar
            grid in radians, allowing asymmetric cones. Use this when the left and right
            cone boundaries do not have equal half-angles. Mutually exclusive with ``angle``.
        interpolation_order (int): Order of interpolation to use (0 = nearest-neighbor,
            1 = linear, 2+ = spline). Matches the convention of `scipy.ndimage.map_coordinates`.

    Returns:
        polar_matrix (Array): The image re-sampled in polar coordinates with shape `polar_shape`,
        coordinates (Array): The Cartesian coordinates corresponding to each pixel in the
            polar output.
    """
    assert "float" in ops.dtype(cartesian_matrix), "Input image must be float type"
    assert angle is None or theta_range is None, (
        "Specify either `angle` (symmetric) or `theta_range` (asymmetric), not both"
    )

    if theta_range is None:
        if angle is None:
            angle = np.deg2rad(45)
        theta_min, theta_max = -angle, angle
    else:
        theta_min, theta_max = theta_range

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
    theta = ops.linspace(theta_min, theta_max, polar_cols, dtype="float32")
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
    return polar_matrix, coords_for_interp


def polar_geometry_from_coordinates(coordinates):
    """Recover the polar scan geometry from a per-pixel Cartesian coordinate grid.

    Inverts the geometry produced by :func:`~zea.beamform.pixelgrid.polar_pixel_grid`:
    a polar :class:`~zea.data.spec.Map` stores, for every pixel ``[radial, polar]``, its
    Cartesian position ``[x, y, z]``. This recovers the apex (the common ray origin), the
    radial extent and the angular extent from that grid, which is exactly what is needed to
    resample the map back onto a regular Cartesian grid (see
    :func:`map_polar_to_cartesian`).

    The grid is assumed to lie in the x-z imaging plane (the ``y`` component is ignored), with
    the radial axis first and the polar/angular axis second -- the layout of ``polar_pixel_grid``
    and of ``Map`` values shaped ``(z, x, ...)``. Angles are measured from the +z (depth) axis,
    matching ``polar_pixel_grid`` where ``x = r·sin(theta)`` and ``z = r·cos(theta)``.

    Args:
        coordinates: float array of shape ``(num_radial, num_polar, 3)``. A leading frame axis
            (shape ``(n_frames, num_radial, num_polar, 3)``) is accepted; the first frame is used,
            assuming the geometry is shared across frames.

    Returns:
        apex (np.ndarray): ``(x, z)`` Cartesian position of the cone apex (common ray origin).
        r_range (tuple): ``(r_min, r_max)`` radial distances from the apex spanned by the grid.
        theta_range (tuple): ``(theta_min, theta_max)``, the angles in radians of the first and
            last polar columns. Ordered by column index, not by magnitude, so it can be fed
            directly into the inverse resampling.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim == 4:
        coordinates = coordinates[0]
    assert coordinates.ndim == 3 and coordinates.shape[-1] == 3, (
        f"Expected coordinates of shape (num_radial, num_polar, 3), got {coordinates.shape}"
    )

    # Work in the x-z imaging plane.
    xz = coordinates[..., [0, 2]]  # (num_radial, num_polar, 2)

    # Each polar column is a straight ray; recover its origin (point) and direction.
    point = xz[0]  # (num_polar, 2): nearest sample on each ray
    direction = xz[-1] - xz[0]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)

    # Apex = the point closest to all rays (least-squares ray intersection):
    #   minimise sum_i || (I - d_i d_iᵀ)(apex - p_i) ||²  =>  A apex = b.
    eye = np.eye(2)
    projectors = eye - direction[:, :, None] * direction[:, None, :]  # (num_polar, 2, 2)
    a_mat = projectors.sum(axis=0)
    b_vec = np.einsum("nij,nj->i", projectors, point)
    apex = np.linalg.solve(a_mat, b_vec)

    # Radial extent: distance of every pixel from the apex.
    radii = np.linalg.norm(xz - apex, axis=-1)
    r_range = (float(radii.min()), float(radii.max()))

    # Angular extent of the first/last column (theta measured from +z).
    rel = xz - apex
    theta = np.arctan2(rel[..., 0], rel[..., 1])  # atan2(dx, dz)
    theta_per_column = theta.mean(axis=0)  # average along the ray for robustness
    theta_range = (float(theta_per_column[0]), float(theta_per_column[-1]))

    return apex, r_range, theta_range


def polar_geometry_from_coords_for_interp(coords_for_interp, polar_shape):
    """Recover the pixel-space polar geometry from the sampling map of a forward transform.

    :func:`cartesian_to_polar_matrix` returns ``coords_for_interp``:
    the ``(2, polar_rows*polar_cols)``
    array of ``[row, col]`` pixel locations it sampled the Cartesian image at, one per polar grid
    point. That array fully embeds the geometry used in the forward call, so this recovers the
    ``tip``, ``r_max`` and ``theta_range`` needed to invert it with
    :func:`polar_to_cartesian_matrix`
    -- *without* having to keep the original parameters around.

    This is the pixel-space counterpart to :func:`polar_geometry_from_coordinates` (which instead
    works on a physical ``[x, y, z]`` metre grid). The two conventions are not interchangeable.

    .. note::
        This relies on the exact ravel/rotation convention of :func:`cartesian_to_polar_matrix`
        (radius is the fast axis, angle the slow axis). If that function changes, update this too.

    Args:
        coords_for_interp: The second return value of :func:`cartesian_to_polar_matrix`, shape
            ``(2, polar_rows*polar_cols)`` with rows ``[row (y), col (x)]``.
        polar_shape (tuple): ``(polar_rows, polar_cols)`` of the polar image, i.e.
            ``(num_radial, num_angular)``.

    Returns:
        tip (tuple): ``(x, y)`` pixel coordinates of the polar origin (probe tip / apex).
        r_max (float): Maximum radius in pixels.
        theta_range (tuple): Angular extent ordered to match the *columns of the returned polar
            image*, so it can be passed straight back to :func:`polar_to_cartesian_matrix` for a
            flip-free round-trip. Because :func:`cartesian_to_polar_matrix` ends with a
            ``rot90(k=-1)`` that reverses the angular axis, the polar image's columns run from the
            larger to the smaller angle, so this is ``(theta_max, theta_min)`` in geometric terms.
    """
    polar_rows, polar_cols = polar_shape
    coords_for_interp = np.asarray(ops.convert_to_numpy(coords_for_interp), dtype=np.float64)

    # Ravel order in cartesian_to_polar_matrix is (polar_cols, polar_rows): angle is the slow
    # axis, radius the fast axis. Row 0 of coords is y (image row), row 1 is x (image col).
    yq = coords_for_interp[0].reshape(polar_cols, polar_rows)
    xq = coords_for_interp[1].reshape(polar_cols, polar_rows)

    # Radius 0 (first column) collapses to the tip for every angle.
    center_x = float(xq[:, 0].mean())
    center_y = float(yq[:, 0].mean())

    # The farthest radius column sits at distance r_max from the tip (rotation preserves distance).
    r_max = float(np.sqrt((xq[:, -1] - center_x) ** 2 + (yq[:, -1] - center_y) ** 2).mean())

    # Per-column angle, inverting the +90 deg rotation: dx = -r sin(theta), dy = r cos(theta).
    theta = np.arctan2(-(xq[:, -1] - center_x), yq[:, -1] - center_y)
    # Reverse the order so theta_range matches the polar image columns (rot90(k=-1) in the forward
    # transform flips the angular axis); this lets polar_to_cartesian_matrix invert without a flip.
    theta_range = (float(theta[-1]), float(theta[0]))

    return (center_x, center_y), r_max, theta_range


# def map_polar_to_cartesian(
#     coordinates,
#     values,
#     cartesian_shape=None,
#     fill_value=0.0,
#     interpolation_order=1,
# ):
#     """Resample a polar :class:`~zea.data.spec.Map` onto a regular Cartesian grid.

#     Round-trip companion that takes a polar map's per-pixel ``coordinates`` and ``values`` and
#     produces a Cartesian image, by recovering the scan geometry with
#     :func:`polar_geometry_from_coordinates` and inverting the polar sampling analytically (no
#     lossy scatter, no holes -- only the small interpolation blur inherent to resampling).

#     Args:
#         coordinates: Per-pixel Cartesian positions of the polar grid, shape
#             ``(num_radial, num_polar, 3)`` (a leading frame axis is accepted).
#         values: Polar map values, spatial shape ``(num_radial, num_polar)`` matching
#             ``coordinates``.
#         cartesian_shape (tuple, optional): Output ``(rows, cols)`` = ``(n_z, n_x)``. Defaults to
#             the spatial shape of ``values``.
#         fill_value (float): Value assigned to Cartesian pixels outside the polar cone.
#         interpolation_order (int): Passed to :func:`map_coordinates`.

#     Returns:
#         cartesian_values (Array): The map resampled onto a regular Cartesian grid, shape
#             ``cartesian_shape``.
#         cartesian_coordinates (np.ndarray): Per-pixel ``[x, y, z]`` positions of that grid
#             (``y = 0``), shape ``(*cartesian_shape, 3)``, ready to build a Cartesian ``Map``.
#     """
#     apex, (r_min, r_max), (theta_min, theta_max) = polar_geometry_from_coordinates(coordinates)

#     num_radial, num_polar = ops.shape(values)
#     if cartesian_shape is None:
#         cart_rows, cart_cols = num_radial, num_polar
#     else:
#         cart_rows, cart_cols = cartesian_shape

#     # Target regular grid spanning the bounding box of the polar coordinates (x-z plane).
#     coords_np = np.asarray(coordinates, dtype=np.float64)
#     if coords_np.ndim == 4:
#         coords_np = coords_np[0]
#     x_min, x_max = coords_np[..., 0].min(), coords_np[..., 0].max()
#     z_min, z_max = coords_np[..., 2].min(), coords_np[..., 2].max()

#     xs = np.linspace(x_min, x_max, cart_cols)
#     zs = np.linspace(z_min, z_max, cart_rows)
#     z_grid, x_grid = np.meshgrid(zs, xs, indexing="ij")  # (rows, cols), rows=z, cols=x

#     # Cartesian pixel -> polar (r, theta) -> fractional polar indices.
#     dx = x_grid - apex[0]
#     dz = z_grid - apex[1]
#     r = np.sqrt(dx**2 + dz**2)
#     theta = np.arctan2(dx, dz)

#     r_idx = (r - r_min) / (r_max - r_min) * (num_radial - 1)
#     theta_idx = (theta - theta_min) / (theta_max - theta_min) * (num_polar - 1)

#     coords_for_interp = ops.stack(
#         [ops.ravel(ops.array(r_idx, "float32")), ops.ravel(ops.array(theta_idx, "float32"))]
#     )
#     cartesian_values = map_coordinates(
#         values,
#         coords_for_interp,
#         order=interpolation_order,
#         fill_mode="constant",
#         fill_value=fill_value,
#     )
#     cartesian_values = ops.reshape(cartesian_values, (cart_rows, cart_cols))

#     cartesian_coordinates = np.stack([x_grid, np.zeros_like(x_grid), z_grid], axis=-1).astype(
#         np.float32
#     )

#     return cartesian_values, cartesian_coordinates


def inverse_scan_convert_2d(
    cartesian_image,
    fill_value=0.0,
    angle=None,
    theta_range=None,
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
        angle (float, optional): Symmetric shorthand for ``theta_range=(-angle, angle)``,
            in radians. Mutually exclusive with ``theta_range``. Defaults to π/4 radians
            (45 degrees) when both are None.
        theta_range (tuple, optional): ``(theta_min, theta_max)`` angular extent of the polar
            grid in radians, allowing asymmetric cones. Mutually exclusive with ``angle``.
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

    polar_image, _ = cartesian_to_polar_matrix(
        cartesian_image,
        fill_value=fill_value,
        angle=angle,
        theta_range=theta_range,
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
