"""
Identifies the scan cone of an ultrasound image and crops it such that the apex of the cone
is at the top center of the image. In this form, it can be scan converted to polar coordinates.

This module provides functionality to:
1. Detect the scan cone boundaries in ultrasound images
2. Fit lines to the cone edges
3. Calculate cone parameters (apex position, opening angle, etc.)
4. Crop and center the image around the cone
5. Visualize the detected cone and its parameters
6. Detect the scan frustum of a 3-D volume (:func:`detect_frustum_parameters`), so it can be
   resampled onto a polar ``(rho, theta, phi)`` grid, and tell whether the sector is a pyramid
   or a fan of tilted 2-D sectors
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zea import log
from zea.func.tensor import translate


def filter_edge_points_by_boundary(edge_points, is_left=True, min_cone_half_angle_deg=20):
    """
    Filter edge points to keep only those that form the actual boundary of the scan cone.
    Enforces minimum cone angle constraint to ensure valid cone shapes.

    Args:
        edge_points: Array of shape (N, 2) containing (x, y) coordinates of edge points
        is_left: Boolean indicating whether these are left (True) or right (False) edge points
        min_cone_half_angle_deg: Minimum expected half-angle of the cone in degrees

    Returns:
        Array of shape (M, 2) containing filtered edge points that satisfy the boundary constraints
    """
    if len(edge_points) == 0:
        return edge_points

    # Convert minimum angle to slope
    min_slope = np.tan(np.radians(min_cone_half_angle_deg))

    # Sort by y coordinate (top to bottom)
    sorted_indices = np.argsort(edge_points[:, 1])
    sorted_points = edge_points[sorted_indices]

    filtered_points = []

    for i, point in enumerate(sorted_points):
        x, y = point
        is_boundary_point = True

        # Check all points above this one
        for j in range(i):
            above_x, above_y = sorted_points[j]
            dy = y - above_y
            min_dx_required = min_slope * dy

            if is_left:
                # For left boundary
                if above_x < x or above_x - x < min_dx_required:
                    is_boundary_point = False
                    break
            else:
                # For right boundary
                if above_x > x or x - above_x < min_dx_required:
                    is_boundary_point = False
                    break

        if is_boundary_point:
            filtered_points.append(point)

    return np.array(filtered_points) if filtered_points else np.zeros((0, 2))


def detect_cone_parameters(image, image_range, min_cone_half_angle_deg=20, threshold=15):
    """Detect the ultrasound cone parameters from a grayscale image.

    This function performs the following steps:
    1. Thresholds the image to create a binary mask
    2. Detects left and right edge points of the cone
    3. Filters edge points to ensure physical constraints
    4. Fits lines to the cone boundaries
    5. Calculates cone parameters including apex position, opening angle, and crop boundaries

    Args:
        image: 2D numpy array (grayscale image)
        image_range: Tuple (vmin, vmax) for display scaling
        min_cone_half_angle_deg: Minimum expected half-angle of the cone in degrees
        threshold: Threshold for binary image (pixels above this are considered data),
            on a scale of 0-255

    Returns:
        Dictionary containing cone parameters including:
        - apex_x, apex_y: Coordinates of the cone apex
        - crop_left, crop_right, crop_top, crop_bottom: Crop boundaries
        - cone_height: Height of the cone
        - opening_angle: Opening angle of the cone
        - symmetry_ratio: Measure of cone symmetry
        - data_coverage: Fraction of crop region containing data
        - And other geometric parameters

    Raises:
        ValueError: If input image is not 2D or cone detection fails
    """

    if len(image.shape) != 2:
        raise ValueError("Input image must be 2D (grayscale)")

    threshold = translate(threshold, range_from=(0, 255), range_to=image_range)

    h, w = image.shape

    # Apply threshold
    thresh = np.where(image > threshold, 255, 0)

    # Find non-zero pixel bounds
    non_zero_indices = np.argwhere(thresh > 0)
    if len(non_zero_indices) == 0:
        raise ValueError(
            f"Failed to detect ultrasound cone: no pixels above threshold ({threshold:.4g}). "
            "The image may be empty or the threshold too high."
        )

    min_y = np.min(non_zero_indices[:, 0])
    max_y = np.max(non_zero_indices[:, 0])

    # Collect left and right edge points
    left_edge_points = []
    right_edge_points = []

    for y in range(min_y, max_y + 1):
        row = thresh[y, :]
        non_zero_x = np.where(row > 0)[0]

        if len(non_zero_x) > 0:
            width = non_zero_x[-1] - non_zero_x[0]
            if width > 5:  # Minimum width to avoid noise
                left_edge_points.append([non_zero_x[0], y])
                right_edge_points.append([non_zero_x[-1], y])

    if len(left_edge_points) < 10:
        raise ValueError(
            f"Failed to detect ultrasound cone: only {len(left_edge_points)} row(s) wide enough "
            "to be cone edges (need at least 10). The thresholded region is too small or too noisy."
        )

    left_edge_points = np.array(left_edge_points, dtype=np.float32)
    right_edge_points = np.array(right_edge_points, dtype=np.float32)

    # Filter edge points to keep only boundary points
    filtered_left_points = filter_edge_points_by_boundary(
        left_edge_points, is_left=True, min_cone_half_angle_deg=min_cone_half_angle_deg
    )
    filtered_right_points = filter_edge_points_by_boundary(
        right_edge_points,
        is_left=False,
        min_cone_half_angle_deg=min_cone_half_angle_deg,
    )

    if len(filtered_left_points) < 3 or len(filtered_right_points) < 3:
        raise ValueError(
            "Failed to detect ultrasound cone: too few boundary points after filtering "
            f"(left={len(filtered_left_points)}, right={len(filtered_right_points)}, need at "
            "least 3 each). The cone edges may not be straight or the half-angle constraint "
            f"(min_cone_half_angle_deg={min_cone_half_angle_deg}) may be too strict."
        )

    # Fit lines using least squares: x = a + b*y
    # Left line
    A_left = np.vstack([np.ones(len(filtered_left_points)), filtered_left_points[:, 1]]).T
    left_coeffs, _, _, _ = np.linalg.lstsq(A_left, filtered_left_points[:, 0], rcond=None)
    left_a, left_b = left_coeffs

    # Right line
    A_right = np.vstack([np.ones(len(filtered_right_points)), filtered_right_points[:, 1]]).T
    right_coeffs, _, _, _ = np.linalg.lstsq(A_right, filtered_right_points[:, 0], rcond=None)
    right_a, right_b = right_coeffs

    # Calculate apex as intersection of fitted lines
    if np.abs(left_b - right_b) < 1e-6:  # Lines are parallel
        raise ValueError(
            "Failed to detect ultrasound cone: the fitted left and right edges are parallel "
            "(no apex intersection). The region may be rectangular rather than cone-shaped."
        )

    apex_y = (right_a - left_a) / (left_b - right_b)
    apex_x = left_a + left_b * apex_y

    # Calculate cone height
    cone_height = max_y - apex_y

    # Calculate opening angle from the line slopes
    left_angle = np.arctan(left_b)  # angle of left line from horizontal
    right_angle = np.arctan(right_b)  # angle of right line from horizontal
    opening_angle = np.abs(left_angle - right_angle)

    min_non_zero_pixel_idx = (0, 0)
    for i in reversed(range(0, thresh.shape[0])):
        row = thresh[i]
        non_zero_pixel_col = np.where(row > 0)[0]
        if np.any(non_zero_pixel_col):
            min_non_zero_pixel_idx = (i, non_zero_pixel_col[0])
            break

    circle_radius = float(
        np.sqrt(
            (min_non_zero_pixel_idx[1] - apex_x) ** 2 + (min_non_zero_pixel_idx[0] - apex_y) ** 2
        )
    )
    circle_center_x = float(apex_x)
    circle_center_y = float(apex_y)

    # Calculate where the circle intersects with the cone lines
    def line_circle_intersection(a, b, cx, cy, r):
        """Find intersection of line x = a + b*y with circle centered at (cx, cy) with radius r"""
        # Substitute line equation into circle equation
        # (a + b*y - cx)^2 + (y - cy)^2 = r^2
        # Expand and collect terms to get quadratic in y
        A = 1 + b**2
        B = 2 * b * (a - cx) - 2 * cy
        C = (a - cx) ** 2 + cy**2 - r**2

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            raise ValueError(
                "Failed to detect ultrasound cone: a fitted edge line does not intersect the "
                "bottom-boundary circle. The cone geometry is inconsistent (bad edge/radius fit)."
            )

        # Two solutions
        y1 = (-B + np.sqrt(discriminant)) / (2 * A)
        y2 = (-B - np.sqrt(discriminant)) / (2 * A)

        # Return the solution closest to the bottom (larger y)
        if y1 > y2:
            return y1, a + b * y1
        else:
            return y2, a + b * y2

    # Find intersection points
    left_intersect = line_circle_intersection(
        left_a, left_b, circle_center_x, circle_center_y, circle_radius
    )
    right_intersect = line_circle_intersection(
        right_a, right_b, circle_center_x, circle_center_y, circle_radius
    )

    if left_intersect is None or right_intersect is None:
        # Fallback to line endpoints at max_y
        left_y_bottom, left_x_bottom = max_y, left_a + left_b * max_y
        right_y_bottom, right_x_bottom = max_y, right_a + right_b * max_y
    else:
        left_y_bottom, left_x_bottom = left_intersect
        right_y_bottom, right_x_bottom = right_intersect

    sector_bottom = int(circle_radius + apex_y)  # this is the coordinate on the unpadded image

    # Calculate crop boundaries (can be negative)
    padding_x = 0
    padding_y = 0

    # convservative cropping boundaries to avoid empty space
    crop_left = math.ceil(left_x_bottom) - padding_x
    crop_right = math.floor(right_x_bottom) + padding_x
    crop_top = math.ceil(apex_y) - padding_y
    crop_bottom = math.floor(sector_bottom)

    # Calculate final dimensions
    new_width = crop_right - crop_left
    new_height = crop_bottom - crop_top

    # Store the intersection points for visualization
    sector_left_x = left_x_bottom
    sector_left_y = left_y_bottom
    sector_right_x = right_x_bottom
    sector_right_y = right_y_bottom

    # Calculate symmetry ratio (how symmetric the cone is)
    symmetry_ratio = float(np.abs(left_b + right_b) / (np.abs(left_b) + np.abs(right_b) + 1e-8))

    # Calculate data coverage in the crop region
    crop_left_clipped = max(0, crop_left)
    crop_right_clipped = min(w, crop_right)
    crop_top_clipped = max(0, crop_top)
    crop_bottom_clipped = min(h, crop_bottom)

    data_coverage = 0.0
    assert crop_left_clipped < crop_right_clipped and crop_top_clipped < crop_bottom_clipped
    crop_region = thresh[crop_top_clipped:crop_bottom_clipped, crop_left_clipped:crop_right_clipped]
    data_coverage = float(np.sum(crop_region > 0) / crop_region.size)

    return {
        "apex_x": float(apex_x),
        "apex_y": float(apex_y),
        "crop_left": crop_left,
        "crop_right": crop_right,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "original_width": int(w),
        "original_height": int(h),
        # Additional parameters for debug and analysis
        "cone_height": float(cone_height),
        "opening_angle": float(opening_angle),
        "new_width": new_width,
        "new_height": new_height,
        "symmetry_ratio": symmetry_ratio,
        "first_data_row": int(min_y),
        "data_coverage": data_coverage,
        "apex_above_image": bool(apex_y < 0),
        # Line parameters for reconstruction if needed
        "left_slope": float(left_b),
        "right_slope": float(right_b),
        "left_intercept": float(left_a),
        "right_intercept": float(right_a),
        # Circle parameters for the bottom boundary
        "circle_center_x": circle_center_x,
        "circle_center_y": circle_center_y,
        "circle_radius": circle_radius,
        # Sector intersection points for visualization
        "sector_left_x": sector_left_x,
        "sector_left_y": sector_left_y,
        "sector_right_x": sector_right_x,
        "sector_right_y": sector_right_y,
        "sector_bottom": sector_bottom,
    }


def crop_and_center_cone(image, cone_params):
    """
    Crop the image to the sector bounding box and pad as needed to center the apex.

    This function:
    1. Crops the image to the detected cone boundaries
    2. Adds padding if the apex is above the image
    3. Centers the apex horizontally in the final image

    Args:
        image: 3D array (batched grayscale images) of shape (B, H, W)
        cone_params: Dictionary of cone parameters from detect_cone_parameters()
        backend: Numpy backend to use for padding and cropping. Defaults to numpy.
            Can be set to ``jax.numpy``, ``keras.ops`` or ``numpy``.

    Returns:
        array of the cropped and centered image with the cone apex at the top center,
        shape (B, H', W')
    """
    # Get crop boundaries
    crop_left = int(cone_params["crop_left"])
    crop_right = int(cone_params["crop_right"])
    crop_top = int(cone_params["crop_top"])
    crop_bottom = int(cone_params["crop_bottom"])

    # Crop the image (handle negative crop_top)
    if crop_top < 0:
        cropped = image[:, 0:crop_bottom, crop_left:crop_right]
        # Add top padding
        top_padding = -crop_top
        batch_size = cropped.shape[0]
        top_pad = np.zeros((batch_size, top_padding, cropped.shape[2]), dtype=cropped.dtype)
        cropped = np.concatenate([top_pad, cropped], axis=1)
    else:
        cropped = image[:, crop_top:crop_bottom, crop_left:crop_right]

    # Now handle horizontal centering
    # Calculate where the apex is in the cropped image
    apex_x_in_crop = cone_params["apex_x"] - crop_left
    cropped_height, cropped_width = cropped.shape[1], cropped.shape[2]
    batch_size = cropped.shape[0]

    # Calculate the target center position
    target_center_x = cropped_width / 2

    # Calculate how much padding we need on each side
    left_padding_needed = target_center_x - apex_x_in_crop

    # Ensure we have non-negative padding
    left_padding = max(0, int(left_padding_needed))
    right_padding = max(0, int(-left_padding_needed))

    # Apply horizontal padding if needed
    if left_padding > 0 or right_padding > 0:
        if left_padding > 0:
            left_pad = np.zeros((batch_size, cropped_height, left_padding), dtype=cropped.dtype)
            cropped = np.concatenate([left_pad, cropped], axis=2)

        if right_padding > 0:
            right_pad = np.zeros((batch_size, cropped_height, right_padding), dtype=cropped.dtype)
            cropped = np.concatenate([cropped, right_pad], axis=2)

    return cropped


def fit_and_crop_around_scan_cone(
    image, image_range=(0, 255), min_cone_half_angle_deg=20, threshold=15
):
    """
    Detect scan cone in ultrasound image and return cropped/padded image with centered apex.

    Args:
        image: numpy array (2D grayscale image)
        image_range: tuple (vmin, vmax) for display scaling (default: (0, 255))
        min_cone_half_angle_deg: Minimum expected half-angle of the cone in degrees (default: 20)
        threshold: Threshold for binary image - pixels above this are considered data.
            This is always on a scale of 0-255 (default: 15).

    Returns:
        Tuple of (cropped_array, cone_parameters_dict)

    Raises:
        ValueError: If cone detection fails or image is not 2D
    """

    cone_params = detect_cone_parameters(
        image,
        image_range=image_range,
        min_cone_half_angle_deg=min_cone_half_angle_deg,
        threshold=threshold,
    )

    # Crop and center the image
    cropped_image = crop_and_center_cone(image[None], cone_params)
    cropped_image = np.squeeze(cropped_image, axis=0)

    return cropped_image, cone_params


# ---------------------------------------------------------------------------
# 3-D scan frustum
# ---------------------------------------------------------------------------


def _largest_connected_component(mask):
    """Keep only the largest connected component of a boolean mask."""
    from scipy import ndimage

    labels, n_labels = ndimage.label(mask)
    if n_labels <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    return labels == np.argmax(sizes)


def _silhouette_edges(silhouette):
    """Outermost data index on the low and high side of each row of a 2-D silhouette.

    Rows without data are NaN.
    """
    has_data = silhouette.any(axis=1)
    n_lateral = silhouette.shape[1]
    low = np.argmax(silhouette, axis=1).astype(np.float64)
    high = (n_lateral - 1 - np.argmax(silhouette[:, ::-1], axis=1)).astype(np.float64)
    low[~has_data] = np.nan
    high[~has_data] = np.nan
    return low, high


def _straight_edge_rows(edge, n_lateral, is_low):
    """Rows in which a silhouette edge lies on the straight side of the sector.

    Going down from the apex, an edge moves outward along a straight line until the far cap of
    the sector bends it back in. Rows below the outermost point are therefore dropped, as are
    rows where the edge touches the volume border, because the sector is clipped there.
    """
    valid = np.isfinite(edge)
    outward = np.where(valid, -edge if is_low else edge, -np.inf)
    last_row = np.argmax(outward)  # first row where the edge is furthest out
    rows = np.arange(len(edge))
    on_border = np.zeros_like(valid)
    on_border[valid] = (edge[valid] <= 0) if is_low else (edge[valid] >= n_lateral - 1)
    return valid & (rows <= last_row) & ~on_border


def _fit_line_robust(z, e, min_scale, threshold, n_iter=5):
    """Fit ``e = a + b * z`` by least squares, iteratively discarding outliers.

    Points further than ``threshold`` robust standard deviations (estimated from the median
    absolute residual, but at least ``min_scale``) from the line are excluded from the next fit.

    Returns:
        Tuple ``(a, b, inliers)``.
    """
    inliers = np.ones(z.shape, dtype=bool)
    for _ in range(n_iter):
        design = np.stack([np.ones(inliers.sum()), z[inliers]], axis=1)
        (a, b), *_ = np.linalg.lstsq(design, e[inliers], rcond=None)
        residual = e - (a + b * z)
        scale = max(1.4826 * np.median(np.abs(residual[inliers])), min_scale)
        new_inliers = np.abs(residual) <= threshold * scale
        if new_inliers.sum() < 2 or np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    return a, b, inliers


def _fit_edges_through_apex(edges, z_apex):
    """Fit each pair of opposite edges as two lines through one apex at depth ``z_apex``.

    For each lateral axis the low edge is modelled as ``e = c + s_low * (z - z_apex)`` and the
    high edge as ``e = c + s_high * (z - z_apex)``, which is linear in ``(c, s_low, s_high)``.

    Args:
        edges: Per lateral axis, a pair ``((z_low, e_low), (z_high, e_high))`` of edge points.
        z_apex: Depth of the apex.

    Returns:
        Tuple of the per-axis parameters ``[(c, s_low, s_high), ...]``, the per-edge residuals
        ``[r_low, r_high, ...]`` and the total sum of squared residuals.
    """
    params, residuals, sse = [], [], 0.0
    for (z_low, e_low), (z_high, e_high) in edges:
        n_low = len(z_low)
        design = np.zeros((n_low + len(z_high), 3))
        design[:, 0] = 1.0
        design[:n_low, 1] = z_low - z_apex
        design[n_low:, 2] = z_high - z_apex
        target = np.concatenate([e_low, e_high])
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = design @ solution - target
        params.append(solution)
        residuals.extend([residual[:n_low], residual[n_low:]])
        sse += float(np.sum(residual**2))
    return params, residuals, sse


def _distance_range(mask, apex, spacing):
    """Distance from ``apex`` to the nearest and farthest ``True`` voxel of ``mask``.

    Works one depth slice at a time to avoid materialising every voxel's coordinates.
    """
    z = (np.arange(mask.shape[0]) - apex[0]) * spacing[0]
    x = (np.arange(mask.shape[1]) - apex[1]) * spacing[1]
    y = (np.arange(mask.shape[2]) - apex[2]) * spacing[2]
    lateral_sq = x[:, None] ** 2 + y[None, :] ** 2
    r_sq_min, r_sq_max = np.inf, -np.inf
    for z_slice, plane in zip(z, mask):
        if plane.any():
            r_sq = lateral_sq[plane]
            r_sq_min = min(r_sq_min, z_slice**2 + r_sq.min())
            r_sq_max = max(r_sq_max, z_slice**2 + r_sq.max())
    return float(np.sqrt(r_sq_min)), float(np.sqrt(r_sq_max))


def _edge_widening(mask, apex, spacing, rho_max, axis, min_points, n_bins=16):
    """How much the edges along one lateral axis widen towards the edges of the other axis.

    Every line of voxels along ``axis`` (one per depth and position on the other lateral axis)
    has an outermost data voxel on both sides. Seen from the apex, it lies at an edge angle with
    ``tan(edge) = lateral / depth`` on a line at ``tan(across) = other / depth``. The lines are
    grouped into ``n_bins`` bins of ``across``, and the outermost edge of each bin is fitted as
    ``|tan(edge)| = a * sec(across) ** k`` with the Theil-Sen estimator. A pyramid has the same
    edge angle on every line, so ``k = 0``. A fan of 2-D sectors in the plane of ``axis``,
    tilted along the other axis, has ``tan(edge) = tan(edge_in_plane) * sec(across)``, so
    ``k = 1``.

    Taking the outermost edge per bin, rather than fitting every line, keeps dark regions at the
    sector edge from pulling the edge inward: one line per bin that reaches the edge suffices.
    Lines whose outermost voxel lies on the volume border (where the sector is clipped), near
    the far cap (``rho > 0.9 * rho_max``) or close to the apex (``rho < 0.3 * rho_max``, where
    the voxels are too coarse to resolve the angle) are not used.

    Returns:
        The exponent ``k`` averaged over both edges, or NaN if neither edge has at least
        ``min_points`` usable lines in four or more bins.
    """
    from scipy.stats import theilslopes

    other = 3 - axis
    n_lateral = mask.shape[axis]
    has_data = mask.any(axis=axis)  # (n_z, n_other)
    first = np.argmax(mask, axis=axis)
    last = n_lateral - 1 - np.argmax(np.flip(mask, axis=axis), axis=axis)

    depth = ((np.arange(mask.shape[0]) - apex[0]) * spacing[0])[:, None]
    across = ((np.arange(mask.shape[other]) - apex[other]) * spacing[other])[None, :]
    safe_depth = np.where(depth > 0, depth, np.inf)
    tan_across = np.broadcast_to(across / safe_depth, has_data.shape)

    exponents = []
    for index, on_border in ((first, first == 0), (last, last == n_lateral - 1)):
        lateral = (index - apex[axis]) * spacing[axis]
        rho = np.sqrt(depth**2 + lateral**2 + across**2)
        tan_edge = np.abs(lateral) / safe_depth
        usable = (
            has_data
            & (depth > 0)
            & ~on_border
            & (rho > 0.3 * rho_max)
            & (rho < 0.9 * rho_max)
            & (tan_edge > 0.05)  # an edge near the depth axis has no measurable widening
        )
        if usable.sum() < min_points:
            continue
        t_across, t_edge = tan_across[usable], tan_edge[usable]
        bin_edges = np.linspace(t_across.min(), t_across.max(), n_bins + 1)
        which = np.clip(np.digitize(t_across, bin_edges) - 1, 0, n_bins - 1)
        x, y = [], []
        for b in range(n_bins):
            in_bin = np.flatnonzero(which == b)
            if len(in_bin) >= 5:
                outermost = in_bin[np.argmax(t_edge[in_bin])]
                x.append(0.5 * np.log1p(t_across[outermost] ** 2))  # log(sec(across))
                y.append(np.log(t_edge[outermost]))
        if len(x) < 4 or np.ptp(x) < 1e-3:
            continue
        # The median of the pairwise slopes, so that a few bins with a protrusion beyond the
        # sector, or a dark region at its edge, do not tilt the fit. These bins are often the
        # outermost ones, which would dominate a least-squares fit.
        exponents.append(float(theilslopes(y, x).slope))
    return float(np.mean(exponents)) if exponents else float("nan")


def _classify_sector(theta_widening, phi_widening, tolerance=0.5):
    """Name the sector shape from the edge widening exponents of :func:`_edge_widening`.

    A pyramid has exponents ``(0, 0)`` and a fan of azimuth sectors tilted in elevation
    ``(1, 0)``. Anything else, or an exponent that could not be measured, is ``"unknown"``.
    """
    if not (np.isfinite(theta_widening) and np.isfinite(phi_widening)):
        return "unknown"
    if abs(phi_widening) < tolerance:
        if abs(theta_widening) < tolerance:
            return "pyramid"
        if abs(theta_widening - 1) < tolerance:
            return "fan"
    return "unknown"


def detect_frustum_parameters(
    mask,
    spacing=(1.0, 1.0, 1.0),
    keep_largest_component=True,
    min_edge_points=10,
    outlier_threshold=3.0,
):
    """Detect the scan frustum of a scan-converted 3-D ultrasound volume.

    This is the 3-D counterpart of :func:`detect_cone_parameters`. Seen from its apex, the
    imaging sector of a volumetric scan spans a range of azimuth angles ``theta`` and elevation
    angles ``phi`` and is closed at depth by a spherical cap. This function estimates the apex
    and these ranges in the frustum convention of :func:`zea.display.frustum_convert_rtp2xyz`
    (``tan(theta) = x / z``, ``tan(phi) = y / z``), so the volume can be resampled onto a polar
    ``(rho, theta, phi)`` grid and scan converted back with :func:`zea.display.scan_convert_3d`.

    The volume is indexed ``(z, x, y)``, as for 3-D maps in the zea format: depth along the first
    axis, with the apex at the top (towards index 0, possibly above the volume), azimuth along
    the second axis and elevation along the third.

    The detection works on silhouettes, which keeps it as simple as the 2-D case:

    1. Optionally keep only the largest connected component of ``mask``, which drops stray
       voxels outside the sector.
    2. Project the mask along each lateral axis. Each projection is a 2-D sector whose straight
       sides are rays from the apex.
    3. In every depth row of a projection, take the outermost data voxel on both sides. Rows
       where the sector is clipped by the volume border, and rows below the point where an edge
       is furthest out (where the far cap begins), are not used.
    4. Fit a line to each of the four edges, iteratively rejecting outliers.
    5. Refit the four lines so that they all pass through a single apex.
    6. Take the radial range from the nearest and farthest data voxel seen from the apex.
    7. Classify the sector as a pyramid or a fan (see below) from how its edges widen.

    The fitted faces pass through the centres of the outermost data voxels, as in the 2-D case,
    so samples on the boundary of the polar grid land on data rather than on background.

    Not every scanner acquires a true frustum. Some acquire a fan of 2-D azimuth sectors, each
    tilted to a different elevation, with the azimuth angle measured within each tilted plane.
    Seen in the frustum convention, such a sector is narrowest in azimuth at zero elevation and
    widens towards the elevation edges (``tan(theta) = tan(theta_in_plane) / cos(phi)``), so its
    constant-depth cross-sections are pincushion-shaped instead of rectangular. The silhouette
    edges are then the corner rays of the sector, so the result is a frustum that encloses the
    whole sector, and parts of the polar grid near the middle of its azimuth sides fall outside
    the sector.

    To tell the two apart, the outermost data voxels along each lateral axis are fitted as
    ``|tan(edge)| = a / cos(across) ** k``, where ``across`` is the angle along the other lateral
    axis, using the outermost voxel in each range of ``across`` so that dark regions at the
    sector edge do not pull the edge inward. A pyramid has ``k = 0`` for both axes. A fan of
    azimuth sectors tilted in elevation has ``k = 1`` for the azimuth edges and ``k = 0`` for
    the elevation edges.

    Args:
        mask (np.ndarray): Boolean array of shape ``(n_z, n_x, n_y)`` that is ``True`` for
            voxels inside the imaging sector, i.e. voxels that are not background padding.
        spacing (tuple, optional): Voxel size along ``(z, x, y)``. Distances in the output are
            in the same unit. Defaults to ``(1.0, 1.0, 1.0)`` (voxels).
        keep_largest_component (bool, optional): Only use the largest connected component of
            ``mask``. Defaults to True.
        min_edge_points (int, optional): Minimum number of depth rows needed to fit each of the
            four edges. Defaults to 10.
        outlier_threshold (float, optional): Edge points further than this many robust standard
            deviations from a fitted edge are rejected. Defaults to 3.0.

    Returns:
        dict: Frustum parameters, as plain floats:

        - ``apex``: ``(z, x, y)`` position of the apex as a fractional voxel index. ``z`` is
          negative when the apex lies above the volume.
        - ``theta_range``: ``(theta_min, theta_max)`` azimuth range in radians, along ``x``.
        - ``phi_range``: ``(phi_min, phi_max)`` elevation range in radians, along ``y``.
        - ``rho_range``: ``(rho_min, rho_max)`` distance from the apex to the nearest and the
          farthest data voxel, in units of ``spacing``.
        - ``edge_rms``: RMS distance in voxels of the edge points to the fitted edges, in the
          order ``(x_min, x_max, y_min, y_max)``. Values around 0.5 indicate a clean fit.
        - ``apex_depth_spread``: Difference in voxels between the apex depths found from the
          ``x`` and ``y`` silhouettes separately. Large values mean that the edges do not meet
          in one point, so the volume is not well described by a frustum.
        - ``sector_type``: Shape of the sector. ``"pyramid"`` if it is the frustum itself, so
          the polar grid lies inside the sector. ``"fan"`` if it is a fan of azimuth sectors
          tilted in elevation, which the frustum encloses, so the polar grid has empty regions
          at its azimuth edges. ``"unknown"`` if it is neither, or the widening could not be
          measured.
        - ``edge_widening``: The exponents ``(k_theta, k_phi)`` from which ``sector_type`` is
          derived: about ``(0, 0)`` for a pyramid and ``(1, 0)`` for a fan.

    Raises:
        ValueError: If ``mask`` is not 3-D, contains no data, or no frustum can be fitted, for
            instance because an edge has too few usable points or the sector does not widen
            with depth.
    """
    from scipy.optimize import minimize_scalar

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D (z, x, y), got shape {mask.shape}")
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(f"spacing must contain three positive values, got {spacing}")

    if keep_largest_component:
        mask = _largest_connected_component(mask)
    if not mask.any():
        raise ValueError("Failed to detect scan frustum: the mask contains no data voxels.")

    depth = np.arange(mask.shape[0]) * spacing[0]
    edges, axis_apex_depths = [], []
    for axis, name in ((1, "x"), (2, "y")):
        silhouette = mask.any(axis=3 - axis)  # project along the other lateral axis
        n_lateral = silhouette.shape[1]
        pair, lines = [], []
        for edge, is_low in zip(_silhouette_edges(silhouette), (True, False)):
            side = f"{name}_{'min' if is_low else 'max'}"
            rows = _straight_edge_rows(edge, n_lateral, is_low)
            if rows.sum() < min_edge_points:
                raise ValueError(
                    f"Failed to detect scan frustum: only {rows.sum()} usable depth rows on the "
                    f"{side} edge (need at least {min_edge_points}). The sector may be clipped "
                    "by the volume border almost immediately, or the apex is not at the top."
                )
            z, e = depth[rows], edge[rows] * spacing[axis]
            a, b, inliers = _fit_line_robust(
                z, e, min_scale=0.5 * spacing[axis], threshold=outlier_threshold
            )
            if inliers.sum() < min_edge_points:
                raise ValueError(
                    f"Failed to detect scan frustum: the {side} edge is not straight "
                    f"({inliers.sum()} inliers, need at least {min_edge_points})."
                )
            pair.append((z[inliers], e[inliers]))
            lines.append((a, b))
        (a_low, b_low), (a_high, b_high) = lines
        if b_high - b_low < 1e-3:
            raise ValueError(
                f"Failed to detect scan frustum: the {name} edges do not diverge with depth "
                f"(slopes {b_low:.3g} and {b_high:.3g}). The apex must be at the top of the "
                "volume (towards index 0 of the first axis)."
            )
        axis_apex_depths.append((a_high - a_low) / (b_low - b_high))
        edges.append(pair)

    # All four edges should meet in one apex. The x and y silhouettes each give an apex depth;
    # find the common depth that fits all edges best. For a fixed depth this is linear.
    margin = 5 * spacing[0]
    result = minimize_scalar(
        lambda z_apex: _fit_edges_through_apex(edges, z_apex)[2],
        bounds=(min(axis_apex_depths) - margin, max(axis_apex_depths) + margin),
        method="bounded",
    )
    z_apex = float(result.x)
    params, residuals, _ = _fit_edges_through_apex(edges, z_apex)
    (x_apex, sx_low, sx_high), (y_apex, sy_low, sy_high) = params

    apex = np.array([z_apex / spacing[0], x_apex / spacing[1], y_apex / spacing[2]])
    rho_range = _distance_range(mask, apex, spacing)
    lateral_spacing = (spacing[1], spacing[1], spacing[2], spacing[2])
    edge_rms = tuple(
        float(np.sqrt(np.mean((r / s) ** 2))) for r, s in zip(residuals, lateral_spacing)
    )
    edge_widening = tuple(
        _edge_widening(mask, apex, spacing, rho_range[1], axis, min_edge_points) for axis in (1, 2)
    )

    return {
        "apex": tuple(float(v) for v in apex),
        "theta_range": (float(np.arctan(sx_low)), float(np.arctan(sx_high))),
        "phi_range": (float(np.arctan(sy_low)), float(np.arctan(sy_high))),
        "rho_range": rho_range,
        "edge_rms": edge_rms,
        "apex_depth_spread": float(abs(axis_apex_depths[0] - axis_apex_depths[1]) / spacing[0]),
        "sector_type": _classify_sector(*edge_widening),
        "edge_widening": edge_widening,
    }


def visualize_scan_cone(image, cone_params, output_dir="output"):
    """
    Create visualization plots for the scan cone detection.

    Args:
        image: Original grayscale image
        cone_params: Dictionary of cone parameters from detect_cone_parameters()
        output_dir: Directory to save output plots (default: "output")

    The visualization is saved as 'scan_cone_visualization.png' in the output directory.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create figure
    _, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(image, cmap="gray" if len(image.shape) == 2 else None)

    # Extract parameters
    apex_x, apex_y = cone_params["apex_x"], cone_params["apex_y"]
    left_a = cone_params["left_intercept"]
    left_b = cone_params["left_slope"]
    right_a = cone_params["right_intercept"]
    right_b = cone_params["right_slope"]

    # Get bottom intersection points
    bottom_y = image.shape[0] - 1
    left_x_bottom = left_a + left_b * bottom_y
    right_x_bottom = right_a + right_b * bottom_y

    # Draw cone lines
    ax.plot(
        [apex_x, left_x_bottom],
        [apex_y, bottom_y],
        color="#FF0000",  # Bright red
        linewidth=2,
        label="Cone boundary",
    )
    ax.plot(
        [apex_x, right_x_bottom],
        [apex_y, bottom_y],
        color="#FF0000",  # Bright red
        linewidth=2,
    )

    # Draw circular arc between the fitted lines
    if "circle_center_x" in cone_params:
        circle_center_x = cone_params["circle_center_x"]
        circle_center_y = cone_params["circle_center_y"]
        circle_radius = cone_params["circle_radius"]

        # Calculate angles for left and right intersection points
        left_angle = np.arctan2(left_x_bottom - circle_center_x, bottom_y - circle_center_y)
        right_angle = np.arctan2(right_x_bottom - circle_center_x, bottom_y - circle_center_y)

        # Ensure angles are in the correct order
        if right_angle < left_angle:
            right_angle += 2 * np.pi

        # Create arc points
        theta = np.linspace(left_angle, right_angle, 100)
        arc_x = circle_center_x + circle_radius * np.sin(theta)
        arc_y = circle_center_y + circle_radius * np.cos(theta)

        # Only draw the part that's visible in the image
        visible_mask = (
            (arc_y >= 0) & (arc_y < image.shape[0]) & (arc_x >= 0) & (arc_x < image.shape[1])
        )
        if np.any(visible_mask):
            ax.plot(
                arc_x[visible_mask],
                arc_y[visible_mask],
                color="#00FF00",  # Bright green
                linestyle="--",
                linewidth=2,
                label="Sector arc",
            )

    # Draw apex as a star
    ax.plot(
        apex_x,
        apex_y,
        marker="*",  # Star marker
        markersize=15,
        color="#FFD700",  # Gold
        markeredgewidth=2,
        label="Cone apex",
    )

    # Draw crop rectangle
    rect = plt.Rectangle(
        (cone_params["crop_left"], cone_params["crop_top"]),
        cone_params["crop_right"] - cone_params["crop_left"],
        cone_params["crop_bottom"] - cone_params["crop_top"],
        fill=False,
        color="#00FFFF",  # Cyan
        linewidth=2,
        label="Crop region",
        linestyle="dotted",
    )
    ax.add_patch(rect)

    # Add info text
    opening_angle_deg = np.degrees(cone_params["opening_angle"])
    info_text = (
        f"Opening Angle: {opening_angle_deg:.1f}°\n"
        f"Symmetry Ratio: {cone_params.get('symmetry_ratio', 0.0):.3f}"
    )

    if "circle_radius" in cone_params:
        info_text += f"\nCircle Radius: {cone_params['circle_radius']:.1f} px"

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "#FFFF00",
            "alpha": 0.9,
        },  # Bright yellow
    )

    # Add crop info
    crop_info = (
        f"Crop: ({cone_params['crop_left']}, {cone_params['crop_top']}) to "
        f"({cone_params['crop_right']}, {cone_params['crop_bottom']})\n"
        f"Final Size: {cone_params['new_width']} x {cone_params['new_height']}"
    )

    ax.text(
        0.02,
        0.02,
        crop_info,
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "#00FFFF",
            "alpha": 0.9,
        },
    )

    # Set up the plot
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.legend(loc="upper right")
    ax.set_title("Scan Cone Detection", fontsize=16)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path / "scan_cone_visualization.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.5,
    )
    log.info(
        f"Saved cone visualization to {log.yellow(output_path / 'scan_cone_visualization.png')}"
    )
    plt.close()


def _load_first_frame(avi_file):
    """
    Load only the first frame of a video file.

    Args:
        avi_file: Path to the video file

    Returns:
        First frame as numpy array of shape (H, W) and dtype np.uint8 (grayscale)
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for loading video files. "
            "Please install it with 'pip install opencv-python' or "
            "'pip install opencv-python-headless'."
        ) from exc

    cap = cv2.VideoCapture(str(avi_file))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read first frame from {avi_file}")

    # Convert BGR to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


def main(avi_path):
    """Demonstrate scan cone fitting on a sample AVI file."""
    frame = _load_first_frame(avi_path)

    try:
        # Fit scan cone
        cone_params = detect_cone_parameters(
            frame, image_range=(0, 255), min_cone_half_angle_deg=20, threshold=15
        )

        # Create visualization
        visualize_scan_cone(frame, cone_params)

    except ValueError as e:
        print(f"Error fitting scan cone: {e}")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute cone parameters for EchoNet-LVH dataset"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input AVI file for visualization",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_file = args.input_file
    main(input_file)
