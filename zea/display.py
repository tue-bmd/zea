"""Display functionality, including scan conversion frustrum conversion, etc."""

from functools import partial
from typing import Tuple, Union

import numpy as np
import scipy
from keras import ops
from PIL import Image

from zea.func.tensor import translate
from zea.tools.fit_scan_cone import fit_and_crop_around_scan_cone

# `zea.func.ultrasound.log_compress` maps zeros to 20 * log10(1e-16) = -320 dB. Such pixels
# hold no measurement, so they are excluded when fitting a histogram transform.
DEAD_PIXEL_DB: float = -300.0


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
            ...     data = file.data.image.values[0]

            >>> image, _ = zea.display.scan_convert(
            ...     data,
            ...     rho_range=(0, 1),
            ...     theta_range=(-0.78, 0.78),
            ...     fill_value=np.nan,
            ... )
            >>> image = zea.display.to_8bit(image, dynamic_range=(-60, 0))
            >>> image.save("image.png")  # doctest: +SKIP

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


def _masked_pixels(image, roi=None, minimum=2):
    """Mask out the pixels a transform may not use: unmeasured, or outside ``roi`` if given."""
    mask = np.isfinite(image) & (image > DEAD_PIXEL_DB)
    if roi is not None:
        mask &= roi
    pixels = image[mask].astype(np.float64)
    if pixels.size < minimum:
        raise ValueError(
            f"Too few valid pixels to fit a transform: {pixels.size}, need at least {minimum}."
        )
    if pixels.min() == pixels.max():
        raise ValueError("Cannot fit a transform: the valid pixels all have the same value.")
    return pixels


def _quantile_pairs(x, y, n_bins):
    """One quantile of each distribution per bin, i.e. eq. (13): ``h[i] = j`` such that
    ``F_X[i] = F_Y[j]``. The pairs are the knots the mapping is interpolated through.

    The reference implementation bins equispaced in *amplitude*, which on log-compressed
    data spends nearly all of its resolution on the tail towards the noise floor. Binning
    equispaced in *probability* instead puts equal mass in every bin, so the resolution
    follows the data (the paper's own remark in Sec. V that bins should be chosen after
    dynamic range compression). Hence the bins are counted out in probability, as the
    ``n_bins`` levels at which both CDFs are read.

    At ``n_bins = x.size`` the quantiles of ``x`` are its order statistics, so the pairs
    become the rank transport of Sec. III.C.1: one pixel per bin, with the quantile function
    of ``y`` resampled onto the ranks of ``x`` when the two differ in size.
    """
    levels = np.linspace(0, 1, n_bins)

    def quantiles(v):
        # `np.quantile` selects rather than sorts, which is quadratic in the number of
        # levels; this is its linear method, interpolating between the order statistics.
        v = np.sort(v)
        return np.interp(levels, np.linspace(0, 1, v.size), v)

    return quantiles(x), quantiles(y)


def _merge_ties(xs, ys):
    """Collapse knots that share an ``x``, which PCHIP cannot accept, onto a single ``y``.

    Knots collapse wherever the image distribution has an atom, i.e. a value carrying a
    finite fraction of the mass: a pile-up at the edge of the range the image was clipped
    to, or a repeated level in a quantized image. A monotone map has to send an atom to one
    value, and this takes the lowest of the reference values it spans, which is the CDF read
    just below the atom. A clipping floor then stays at the bottom of the matched range
    rather than being lifted into the middle of it.
    """
    xs, first = np.unique(xs, return_index=True)
    return xs, ys[first]


def _end_slopes(xs, ys, frac=0.125, max_ratio=4.0):
    """Slopes for linear extension of the mapping past the first and last knot."""
    # Secants between adjacent knots are unusable: an atom collapses knots onto each other
    # (see `_merge_ties`), which makes the secant explode. Measure across the outer `frac`
    # of the knots instead, which because they are equispaced in probability is the outer
    # `frac` of the mass, and cap the result relative to the overall slope.
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    k = max(2, round(frac * xs.size))

    def secant(dx, dy):
        if dx < 1e-3:  # degenerate span, nothing local to learn from
            return slope
        return float(np.clip(dy / dx, slope / max_ratio, slope * max_ratio))

    return (
        secant(xs[k - 1] - xs[0], ys[k - 1] - ys[0]),
        secant(xs[-1] - xs[-k], ys[-1] - ys[-k]),
    )


def _monotone_map(image, xs, ys):
    """Monotone interpolation through the knots, one per bin, extended linearly at the ends."""
    # The end slopes are read off the knots before merging, which loses how much mass each
    # of them carries and with it the meaning of "the outer `frac` of the distribution".
    slope_low, slope_high = _end_slopes(xs, ys)
    xs, ys = _merge_ties(xs, ys)
    out = scipy.interpolate.PchipInterpolator(xs, ys)(image)
    # Cubic extrapolation can fold back on itself and break monotonicity, so replace it by
    # a linear extension (Sec. III.C.3): a point target brighter than anything in the ROI
    # should stay the brightest instead of piling onto the top knot.
    out = np.where(image < xs[0], ys[0] + slope_low * (image - xs[0]), out)
    return np.where(image > xs[-1], ys[-1] + slope_high * (image - xs[-1]), out)


def histogram_match(
    image,
    reference,
    mode: str = "full",
    n_bins: Union[int, str] = 256,
    roi=None,
) -> np.ndarray:
    """Match the histogram of an image to a reference image, for fair visual comparison.

    Image formation methods apply different dynamic range transformations, which biases
    visual as well as quantitative comparison. Matching the histogram of an image to a
    reference (typically a conventional B-mode image) puts the two on a common scale.

    Args:
        image (ndarray): Image to transform, typically log-compressed (dB).
        reference (ndarray): Image to match to. Its shape need not equal that of ``image``,
            as only the distributions are compared, unless an ``roi`` indexes both.
        mode (str, optional): Class of transform to match with. Defaults to ``"full"``.

            - ``"partial"``: affine match (Sec. III.A), the scale and offset that match
              mean and variance. It leaves the *shape* of the image's own distribution
              intact, so pick it when that shape is the thing under comparison, when the
              fitted region contains structure rather than plain speckle, or when the CNR
              must survive the matching.
            - ``"full"``: any monotone mapping (Sec. III.B), the one that carries the
              histogram of the image onto that of the reference, estimated on ``n_bins``
              bins. It matches the shape of the distribution too, hence undoes non-linear
              compression, so pick it to compare methods whose transformations differ in
              more than scale and offset, or are unknown ("black box" images, Sec. III.C.2).
              Together with a speckle ``roi``, the paper's expected common use (Sec. V).

        n_bins (int or str, optional): Number of bins the mapping of a ``"full"`` match is
            estimated on; ignored by ``"partial"``. Defaults to 256, the number of bins used
            in the paper, which it found enough for consistent results.
            Pass ``"all"`` for one bin per fitted pixel, the limit of the binning process
            (Sec. III.C.1): it reproduces the reference distribution down to its outliers,
            which replicates them in the matched image. This is what
            `skimage.exposure.match_histograms` does.
        roi (ndarray, optional): Boolean mask, of the shape of both images, selecting a
            homogeneous speckle region (Sec. III.C.3, the paper's expected common use).
            The mapping is fit inside the mask and applied to the whole image, extended
            linearly outside the fitted range. Defaults to None (fit on the whole image).

    Returns:
        ndarray: Transformed image (float64), unclipped: clipping to a display range is
        left to the caller (`to_8bit`, or ``vmin``/``vmax``).

    .. note::
        Every mode fits on finite pixels above `DEAD_PIXEL_DB` only, so that zea's
        ``log(0)`` sentinel neither anchors the bottom of the mapping nor is lifted into
        the displayed range. Sec. III.C.2 needs no such exclusion for clipped values: under
        a full match the same fraction of the distribution maps to the clipped value.

    .. note::
        Which image is matched to which is a choice (Sec. V): matching to B-mode is natural
        for a reader used to B-mode, but features outside its dynamic range are lost in the
        process, and matching in the other direction then shows them off better.

    Example:
        .. doctest::

            >>> import numpy as np

            >>> from zea.display import histogram_match

            >>> rng = np.random.default_rng(0)
            >>> image = 20 * np.log10(rng.rayleigh(0.1, (64, 64)))
            >>> reference = 20 * np.log10(rng.rayleigh(1.0, (64, 64)))

            >>> round(image.mean() - reference.mean())  # the image is 20 dB darker
            -20

            >>> matched = histogram_match(image, reference)
            >>> round(matched.mean() - reference.mean())
            0

    .. admonition:: Reference

        N. Bottenus, B. Byram, and D. Hyun, "Histogram Matching for Visual Ultrasound
        Image Comparison," *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency
        Control*, vol. 68, no. 5, pp. 1487-1495, 2021.
        https://doi.org/10.1109/TUFFC.2020.3035965

        Their MATLAB implementation, referred to above as the reference implementation:
        https://github.com/nbottenus/histogram_matching/blob/main/histmatch.m

    """
    if mode not in ("full", "partial"):
        raise ValueError(f"Unknown mode {mode!r}, expected 'full' or 'partial'.")
    every_pixel = n_bins == "all"
    if not every_pixel and not (isinstance(n_bins, (int, np.integer)) and n_bins >= 2):
        raise ValueError(f"Invalid n_bins {n_bins!r}, expected an integer >= 2 or 'all'.")

    image = ops.convert_to_numpy(image)
    reference = ops.convert_to_numpy(reference)
    if roi is not None:
        roi = ops.convert_to_numpy(roi).astype(bool)

    # A bin has to be worth estimating: ask for as many pixels as there are bins.
    minimum = 2 if mode == "partial" or every_pixel else n_bins
    x = _masked_pixels(image, roi, minimum)
    y = _masked_pixels(reference, roi, minimum)

    image = image.astype(np.float64)
    if mode == "partial":
        # eq. (8)-(9): the scale and offset that match mean and variance.
        slope = y.std() / x.std()
        return slope * image + (y.mean() - slope * x.mean())

    xs, ys = _quantile_pairs(x, y, x.size if every_pixel else n_bins)
    return _monotone_map(image, xs, ys)


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


def _polar_sampling_coordinates(x_vec, z_vec, rho, theta):
    """Map an explicit Cartesian (x, z) output grid onto fractional (rho, theta) indices
    into a polar image, using the frustum convention.

    Args:
        x_vec, z_vec: 1D Cartesian output axes (x is lateral, z is depth).
        rho, theta: 1D polar axes of the input image (``rho`` radial, ``theta`` angular).
            Their lengths set the polar image dimensions the indices address.

    Returns:
        coordinates (Array): ``(2, len(x_vec), len(z_vec))`` stack of ``[rho_idx, theta_idx]``
        ready for :func:`map_coordinates`.
    """
    # The index mapping is endpoint-based (rho[0]/rho[-1], theta[0]/theta[-1]), so the *order*
    # of the axes is preserved: a descending ``theta`` maps the angular axis in reverse. The
    # masking limits, by contrast, are sorted so out-of-cone pixels are dropped regardless of
    # axis direction.
    rho_start, rho_end = rho[0], rho[-1]
    theta_start, theta_end = theta[0], theta[-1]
    theta_min = ops.minimum(theta_start, theta_end)
    theta_max = ops.maximum(theta_start, theta_end)

    z_grid, x_grid = ops.meshgrid(z_vec, x_vec)
    rho_grid_interp, theta_grid_interp = frustum_convert_xz2rt(
        x_grid, z_grid, theta_limits=[theta_min, theta_max]
    )

    n_rho, n_theta = ops.shape(rho)[0], ops.shape(theta)[0]
    rho_idx = (rho_grid_interp - rho_start) / (rho_end - rho_start) * (n_rho - 1)
    theta_idx = (theta_grid_interp - theta_start) / (theta_end - theta_start) * (n_theta - 1)
    return ops.stack([rho_idx, theta_idx], axis=0)


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

    coordinates = _polar_sampling_coordinates(x_vec, z_vec, rho, theta)
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
    rho_range: Tuple[float, float] | None = None,
    theta_range: Tuple[float, float] | None = None,
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
        if rho_range is None or theta_range is None:
            raise ValueError(
                "rho_range and theta_range are required when coordinates is not provided."
            )
        coordinates, parameters = compute_scan_convert_2d_coordinates(
            image.shape,
            rho_range,
            theta_range,
            resolution,
            dtype=image.dtype,
            distance_to_apex=distance_to_apex,
        )

    scan_converted = _interpolate_batch(image, coordinates, fill_value, order=order, **kwargs)

    # swap axis to match z, x
    scan_converted = ops.swapaxes(scan_converted, -1, -2)

    return scan_converted, parameters


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
    rho_range: Tuple[float, float] | None = None,
    theta_range: Tuple[float, float] | None = None,
    phi_range: Tuple[float, float] | None = None,
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
        if rho_range is None or theta_range is None or phi_range is None:
            raise ValueError(
                "rho_range, theta_range, and phi_range are required "
                "when coordinates is not provided."
            )
        coordinates, parameters = compute_scan_convert_3d_coordinates(
            image.shape,
            rho_range,
            theta_range,
            phi_range,
            resolution,
            dtype=image.dtype,
        )

    scan_converted = _interpolate_batch(image, coordinates, fill_value, order=order)

    # swap axis to match z, x, y
    scan_converted = ops.swapaxes(scan_converted, -2, -3)
    return scan_converted, parameters


def scan_convert(
    image,
    rho_range: Tuple[float, float] | None = None,
    theta_range: Tuple[float, float] | None = None,
    phi_range: Tuple[float, float] | None = None,
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
        scan_converted = ops.stack(list(map(map_coordinates_fn, images)))
    elif not vectorize:
        scan_converted = ops.map(map_coordinates_fn, images)
    else:
        # gpu bound
        scan_converted = ops.vectorized_map(map_coordinates_fn, images)

    # ignore batch dim to get image shape
    scan_converted_shape = ops.shape(scan_converted)[1:]
    scan_converted = ops.reshape(scan_converted, (*batch_dims, *scan_converted_shape))

    return scan_converted


def cartesian_to_polar_matrix(
    cartesian_matrix,
    fill_value=0.0,
    polar_shape=None,
    tip=None,
    r_max=None,
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
        theta_range (tuple, optional): ``(theta_min, theta_max)`` angular extent of the polar
            grid in radians, allowing asymmetric cones. Use this when the left and right
            cone boundaries do not have equal half-angles. Defaults to (-45, 45) degrees.
        interpolation_order (int): Order of interpolation to use (0 = nearest-neighbor,
            1 = linear, 2+ = spline). Matches the convention of `scipy.ndimage.map_coordinates`.

    Returns:
        polar_matrix (Array): The image re-sampled in polar coordinates with shape
            `polar_shape`.
        coordinates (Array): The Cartesian coordinates corresponding to each pixel in
            the polar output.
    """
    assert "float" in ops.dtype(cartesian_matrix), "Input image must be float type"

    if theta_range is None:
        theta_min, theta_max = -np.deg2rad(45), np.deg2rad(45)
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

    # Polar sampling grid: rows are radius (0..r_max), columns are angle. The columns run from
    # theta_max down to theta_min so the lateral axis of the polar image keeps the same
    # left-right orientation as the Cartesian image.
    r = ops.linspace(0, r_max, polar_rows, dtype="float32")
    theta = ops.linspace(theta_max, theta_min, polar_cols, dtype="float32")
    r_grid, theta_grid = ops.meshgrid(r, theta, indexing="ij")

    # Cartesian pixel each (r, theta) samples from. The probe images "downwards" (+depth = +y),
    # so depth = center_y + r cos(theta) and lateral = center_x - r sin(theta).
    xq = center_x - r_grid * ops.sin(theta_grid)
    yq = center_y + r_grid * ops.cos(theta_grid)
    coords_for_interp = ops.stack([ops.ravel(yq), ops.ravel(xq)])

    polar_values = map_coordinates(
        cartesian_matrix,
        coords_for_interp,
        order=interpolation_order,
        fill_mode="constant",
        fill_value=fill_value,
    )

    polar_matrix = ops.reshape(polar_values, (polar_rows, polar_cols))
    return polar_matrix


def polar_to_cartesian_matrix(
    polar_matrix,
    cartesian_shape: Tuple[int, int],
    tip: Union[Tuple[float, float], None] = None,
    r_max: Union[float, None] = None,
    theta_range: Union[Tuple[float, float], None] = None,
    pitch: float = 1.0,
    fill_value: float = 0.0,
    order: int = 1,
):
    """Resample a polar image onto a fixed Cartesian canvas with the apex at a chosen pixel.

    Faithful inverse of :func:`cartesian_to_polar_matrix`. Where :func:`scan_convert_2d`
    fits the cone bounding box into the output (losing absolute position and scale), this
    pins the cone apex at pixel ``tip`` on a canvas of shape ``cartesian_shape`` and samples
    at ``pitch`` units per pixel, so a forward/inverse round-trip reproduces the original
    frame. Both share the :func:`_polar_sampling_coordinates` core; they differ only in how
    the output grid is built.

    Args:
        polar_matrix (tensor): Input polar image of shape ``(n_rho, n_theta)``, float type.
        cartesian_shape (tuple): Output ``(rows, cols)`` = ``(n_z, n_x)``.
        tip (tuple, optional): ``(col, row)`` pixel location of the cone apex in the output.
            Defaults to the centre-top ``(cols / 2, 0)``.
        r_max (float, optional): Radius spanned by the polar image, in the same units as
            ``pitch`` (pixels by default). Defaults to ``rows``.
        theta_range (tuple, optional): angular extent in radians. The *order* is significant:
            it must match the column order of ``polar_matrix``. :func:`cartesian_to_polar_matrix`
            lays its columns out from the larger to the smaller angle, so to invert it pass the
            range reversed, i.e. ``(theta_max, theta_min)`` -- which is exactly what
            :func:`polar_geometry_from_coords_for_interp` returns. Defaults to (45, -45) degrees.
        pitch (float, optional): Output units per pixel (radial units of ``r_max``).
            Defaults to 1.0, i.e. ``tip`` and ``r_max`` are in pixels.
        fill_value (float, optional): Value for pixels outside the polar domain.
        order (int, optional): Spline interpolation order. Defaults to 1.

    Returns:
        cartesian_matrix (Array): The polar image resampled onto the Cartesian canvas, with
            shape ``cartesian_shape``.
    """
    assert "float" in ops.dtype(polar_matrix), "Input image must be float type"

    dtype = ops.dtype(polar_matrix)
    n_rho, n_theta = ops.shape(polar_matrix)[-2], ops.shape(polar_matrix)[-1]

    coordinates = polar_to_cartesian_coordinates(
        cartesian_shape,
        n_rho,
        n_theta,
        tip=tip,
        r_max=r_max,
        theta_range=theta_range,
        pitch=pitch,
        dtype=dtype,
    )
    cartesian, _ = scan_convert_2d(
        polar_matrix, coordinates=coordinates, fill_value=fill_value, order=order
    )
    return cartesian


def polar_to_cartesian_coordinates(
    cartesian_shape: Tuple[int, int],
    n_rho: int,
    n_theta: int,
    tip: Union[Tuple[float, float], None] = None,
    r_max: Union[float, None] = None,
    theta_range: Union[Tuple[float, float], None] = None,
    pitch: float = 1.0,
    dtype: str = "float32",
):
    """(rho, theta) sampling grid that pins a polar cone's apex at pixel ``tip``.

    Shared core of :func:`polar_to_cartesian_matrix`: builds the fractional index grid that
    resamples an ``(n_rho, n_theta)`` polar image onto a ``cartesian_shape`` canvas, with the
    apex at pixel ``tip`` and ``pitch`` units per pixel. Pass the result as ``coordinates`` to
    :func:`scan_convert_2d` (or the :class:`~zea.ops.ScanConvert` operation) to run the same
    resampling inside a pipeline. Unlike :func:`compute_scan_convert_2d_coordinates`, this uses
    the ``arctan2`` mapping in :func:`_polar_sampling_coordinates`, so it covers a full circle
    rather than only a narrow (``|theta| < pi/2``) sector.

    Args:
        cartesian_shape (tuple): Output ``(rows, cols)`` = ``(n_z, n_x)``.
        n_rho (int): Radial samples of the polar image the indices address.
        n_theta (int): Angular samples of the polar image the indices address.
        tip (tuple, optional): ``(col, row)`` pixel location of the cone apex in the output.
            Defaults to the centre-top ``(cols / 2, 0)``.
        r_max (float, optional): Radius spanned by the polar image, in the same units as
            ``pitch`` (pixels by default). Defaults to ``rows``.
        theta_range (tuple, optional): angular extent in radians; the order must match the
            column order of the polar image. Defaults to (45, -45) degrees.
        pitch (float, optional): Output units per pixel (radial units of ``r_max``).
            Defaults to 1.0, i.e. ``tip`` and ``r_max`` are in pixels.
        dtype (str, optional): Compute dtype for the axes. Defaults to ``"float32"``.

    Returns:
        coords (Array): ``(2, cols, rows)`` stack of ``[rho_idx, theta_idx]`` sampling indices.
    """
    cart_rows, cart_cols = cartesian_shape
    if tip is None:
        tip = (cart_cols / 2, 0.0)
    if r_max is None:
        r_max = cart_rows
    if theta_range is None:
        theta_range = (np.deg2rad(45), -np.deg2rad(45))

    tip_x, tip_y = tip
    rho = ops.linspace(0.0, r_max, n_rho, dtype=dtype)
    theta = ops.linspace(theta_range[0], theta_range[1], n_theta, dtype=dtype)

    x_vec = (tip_x - ops.cast(ops.arange(cart_cols), dtype)) * pitch
    z_vec = (ops.cast(ops.arange(cart_rows), dtype) - tip_y) * pitch

    return _polar_sampling_coordinates(x_vec, z_vec, rho, theta)


def inverse_scan_convert_2d(
    cartesian_image,
    fill_value=0.0,
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
        theta_range (tuple, optional): ``(theta_min, theta_max)`` angular extent of the polar
            grid in radians, allowing asymmetric cones. Defaults to (-45, 45) degrees.
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
        cartesian_image, _ = fit_and_crop_around_scan_cone(cartesian_image, image_range)

    polar_image = cartesian_to_polar_matrix(
        cartesian_image,
        fill_value=fill_value,
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
