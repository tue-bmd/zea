"""
Identifies the scan cone of an ultrasound image and crops it such that the apex of the cone
is at the top center of the image. In this form, it can be scan converted to polar coordinates.

This module provides functionality to:
1. Detect the scan cone boundaries in ultrasound images
2. Fit lines to the cone edges
3. Calculate cone parameters (apex position, opening angle, etc.)
4. Crop and center the image around the cone
5. Visualize the detected cone and its parameters
"""

import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = (
        "numpy"  # recommend using numpy for this since some line fitting is performed on CPU
    )

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import ops

from zea import log


def filter_edge_points_by_boundary(edge_points, is_left=True, min_cone_half_angle_deg=20):
    """
    Filter edge points to keep only those that form the actual boundary of the scan cone.
    Enforces minimum cone angle constraint to ensure valid cone shapes.

    Args:
        edge_points: Tensor of shape (N, 2) containing (x, y) coordinates of edge points
        is_left: Boolean indicating whether these are left (True) or right (False) edge points
        min_cone_half_angle_deg: Minimum expected half-angle of the cone in degrees

    Returns:
        Tensor of shape (M, 2) containing filtered edge points that satisfy the boundary constraints
    """
    if ops.shape(edge_points)[0] == 0:
        return edge_points

    # Convert minimum angle to slope
    min_slope = ops.tan(np.radians(min_cone_half_angle_deg))

    # Sort by y coordinate (top to bottom)
    sorted_indices = ops.argsort(edge_points[:, 1])
    sorted_points = ops.take(edge_points, sorted_indices, axis=0)

    filtered_points = []

    # Convert to numpy for the iterative logic (this part is hard to vectorize)
    sorted_points_np = ops.convert_to_numpy(sorted_points)

    for i, point in enumerate(sorted_points_np):
        x, y = point
        is_boundary_point = True

        # Check all points above this one
        for j in range(i):
            above_x, above_y = sorted_points_np[j]
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

    return ops.convert_to_tensor(filtered_points) if filtered_points else ops.zeros((0, 2))


def detect_cone_parameters(image, min_cone_half_angle_deg=20, threshold=15):
    """
    Detect the ultrasound cone parameters from a grayscale image.

    This function performs the following steps:
    1. Thresholds the image to create a binary mask
    2. Detects left and right edge points of the cone
    3. Filters edge points to ensure physical constraints
    4. Fits lines to the cone boundaries
    5. Calculates cone parameters including apex position, opening angle, and crop boundaries

    Args:
        image: 2D Keras tensor (grayscale image)
        min_cone_half_angle_deg: Minimum expected half-angle of the cone in degrees
        threshold: Threshold for binary image (pixels above this are considered data)

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
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for cone detection and visualization. "
            "Please install it with 'pip install opencv-python' or "
            "'pip install opencv-python-headless'."
        ) from exc

    if len(ops.shape(image)) != 2:
        raise ValueError("Input image must be 2D (grayscale)")

    # Ensure image is in proper range for cv2.threshold
    if image.dtype != "uint8":
        image = ops.cast(image * 255, "uint8")

    h, w = ops.shape(image)

    # OpenCV threshold requires numpy array
    image_np = ops.convert_to_numpy(image)
    _, thresh_np = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
    thresh = ops.convert_to_tensor(thresh_np)

    # Find non-zero pixel bounds
    non_zero_indices = ops.convert_to_tensor(ops.where(thresh > 0))
    if ops.shape(non_zero_indices)[0] == 0:
        return None

    min_y = ops.min(non_zero_indices[:, 0])
    max_y = ops.max(non_zero_indices[:, 0])

    # Collect left and right edge points
    left_edge_points = []
    right_edge_points = []

    # Convert back to numpy for row-wise processing (this is hard to vectorize efficiently)
    thresh_np = ops.convert_to_numpy(thresh)
    min_y_np = int(ops.convert_to_numpy(min_y))
    max_y_np = int(ops.convert_to_numpy(max_y))

    for y in range(min_y_np, max_y_np + 1):
        row = thresh_np[y, :]
        non_zero_x = np.where(row > 0)[0]

        if len(non_zero_x) > 0:
            width = non_zero_x[-1] - non_zero_x[0]
            if width > 5:  # Minimum width to avoid noise
                left_edge_points.append([non_zero_x[0], y])
                right_edge_points.append([non_zero_x[-1], y])

    if len(left_edge_points) < 10:
        return None

    left_edge_points = ops.convert_to_tensor(left_edge_points, dtype="float32")
    right_edge_points = ops.convert_to_tensor(right_edge_points, dtype="float32")

    # Filter edge points to keep only boundary points
    filtered_left_points = filter_edge_points_by_boundary(
        left_edge_points, is_left=True, min_cone_half_angle_deg=min_cone_half_angle_deg
    )
    filtered_right_points = filter_edge_points_by_boundary(
        right_edge_points,
        is_left=False,
        min_cone_half_angle_deg=min_cone_half_angle_deg,
    )

    if ops.shape(filtered_left_points)[0] < 3 or ops.shape(filtered_right_points)[0] < 3:
        return None

    # Fit lines using least squares: x = a + b*y
    # Convert to numpy for lstsq
    filtered_left_np = ops.convert_to_numpy(filtered_left_points)
    filtered_right_np = ops.convert_to_numpy(filtered_right_points)

    # Left line
    A_left = np.vstack([np.ones(len(filtered_left_np)), filtered_left_np[:, 1]]).T
    left_coeffs, _, _, _ = np.linalg.lstsq(A_left, filtered_left_np[:, 0], rcond=None)
    left_a, left_b = left_coeffs

    # Right line
    A_right = np.vstack([np.ones(len(filtered_right_np)), filtered_right_np[:, 1]]).T
    right_coeffs, _, _, _ = np.linalg.lstsq(A_right, filtered_right_np[:, 0], rcond=None)
    right_a, right_b = right_coeffs

    # Convert back to tensors
    left_a = ops.convert_to_tensor(left_a)
    left_b = ops.convert_to_tensor(left_b)
    right_a = ops.convert_to_tensor(right_a)
    right_b = ops.convert_to_tensor(right_b)

    # Calculate apex as intersection of fitted lines
    if ops.abs(left_b - right_b) < 1e-6:  # Lines are parallel
        return None

    apex_y = (right_a - left_a) / (left_b - right_b)
    apex_x = left_a + left_b * apex_y

    # Calculate cone height
    cone_height = max_y - apex_y

    # Calculate opening angle from the line slopes
    # Convert slopes to angles and calculate opening angle
    left_angle = ops.arctan(left_b)  # angle of left line from horizontal
    right_angle = ops.arctan(right_b)  # angle of right line from horizontal
    opening_angle = ops.abs(left_angle - right_angle)

    min_non_zero_pixel_idx = (0, 0)
    for i in reversed(range(0, thresh_np.shape[0])):
        row = thresh_np[i]
        non_zero_pixel_col = np.where(row > 0)[0]
        if np.any(non_zero_pixel_col):
            min_non_zero_pixel_idx = (i, non_zero_pixel_col[0])
            break

    circle_radius = float(
        np.sqrt(
            (min_non_zero_pixel_idx[1] - ops.convert_to_numpy(apex_x)) ** 2
            + (min_non_zero_pixel_idx[0] - ops.convert_to_numpy(apex_y)) ** 2
        )
    )
    circle_center_x = float(ops.convert_to_numpy(apex_x))
    circle_center_y = float(ops.convert_to_numpy(apex_y))

    # Calculate where the circle intersects with the cone lines
    # For line: x = a + b*y and circle: (x - cx)^2 + (y - cy)^2 = r^2
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
            return None

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
        ops.convert_to_numpy(left_a),
        ops.convert_to_numpy(left_b),
        circle_center_x,
        circle_center_y,
        circle_radius,
    )
    right_intersect = line_circle_intersection(
        ops.convert_to_numpy(right_a),
        ops.convert_to_numpy(right_b),
        circle_center_x,
        circle_center_y,
        circle_radius,
    )

    if left_intersect is None or right_intersect is None:
        # Fallback to line endpoints at max_y
        left_y_bottom, left_x_bottom = (
            ops.convert_to_numpy(max_y),
            ops.convert_to_numpy(left_a + left_b * max_y),
        )
        right_y_bottom, right_x_bottom = (
            ops.convert_to_numpy(max_y),
            ops.convert_to_numpy(right_a + right_b * max_y),
        )
    else:
        left_y_bottom, left_x_bottom = left_intersect
        right_y_bottom, right_x_bottom = right_intersect

    sector_bottom = int(circle_radius + apex_y)  # this is the coordinate on the unpadded image

    # Calculate crop boundaries (can be negative)
    padding_x = 0
    padding_y = 0

    crop_left = int(left_x_bottom) - padding_x
    crop_right = int(right_x_bottom) + padding_x
    crop_top = int(ops.convert_to_numpy(apex_y)) - padding_y
    crop_bottom = int(sector_bottom)

    # Calculate final dimensions
    new_width = crop_right - crop_left
    new_height = crop_bottom - crop_top

    # Store the intersection points for visualization
    sector_left_x = left_x_bottom
    sector_left_y = left_y_bottom
    sector_right_x = right_x_bottom
    sector_right_y = right_y_bottom

    # Calculate symmetry ratio (how symmetric the cone is)
    symmetry_ratio = float(
        ops.convert_to_numpy(
            ops.abs(left_b + right_b) / (ops.abs(left_b) + ops.abs(right_b) + 1e-8)
        )
    )

    # Calculate data coverage in the crop region
    h_np = int(ops.convert_to_numpy(h))
    w_np = int(ops.convert_to_numpy(w))
    crop_left_clipped = max(0, crop_left)
    crop_right_clipped = min(w_np, crop_right)
    crop_top_clipped = max(0, crop_top)
    crop_bottom_clipped = min(h_np, crop_bottom)

    data_coverage = 0.0
    assert crop_left_clipped < crop_right_clipped and crop_top_clipped < crop_bottom_clipped
    crop_region = thresh_np[
        crop_top_clipped:crop_bottom_clipped, crop_left_clipped:crop_right_clipped
    ]
    data_coverage = float(np.sum(crop_region > 0) / crop_region.size)

    return {
        "apex_x": float(ops.convert_to_numpy(apex_x)),
        "apex_y": float(ops.convert_to_numpy(apex_y)),
        "crop_left": crop_left,
        "crop_right": crop_right,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "original_width": int(ops.convert_to_numpy(w)),
        "original_height": int(ops.convert_to_numpy(h)),
        # Additional parameters for debug and analysis
        "cone_height": float(ops.convert_to_numpy(cone_height)),
        "opening_angle": float(ops.convert_to_numpy(opening_angle)),
        "new_width": new_width,
        "new_height": new_height,
        "symmetry_ratio": symmetry_ratio,
        "first_data_row": int(ops.convert_to_numpy(min_y)),
        "data_coverage": data_coverage,
        "apex_above_image": bool(ops.convert_to_numpy(apex_y) < 0),
        # Line parameters for reconstruction if needed
        "left_slope": float(ops.convert_to_numpy(left_b)),
        "right_slope": float(ops.convert_to_numpy(right_b)),
        "left_intercept": float(ops.convert_to_numpy(left_a)),
        "right_intercept": float(ops.convert_to_numpy(right_a)),
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
        image: 2D Keras tensor (grayscale image)
        cone_params: Dictionary of cone parameters from detect_cone_parameters()

    Returns:
        Keras tensor of the cropped and centered image with the cone apex at the top center
    """
    # Get crop boundaries
    crop_left = cone_params["crop_left"]
    crop_right = cone_params["crop_right"]
    crop_top = cone_params["crop_top"]
    crop_bottom = cone_params["crop_bottom"]

    # Crop the image (handle negative crop_top)
    if crop_top < 0:
        cropped = image[0:crop_bottom, crop_left:crop_right]
        # Add top padding
        top_padding = -crop_top
        cropped_width = ops.shape(cropped)[1]
        top_pad = ops.zeros((top_padding, cropped_width), dtype=cropped.dtype)
        cropped = ops.concatenate([top_pad, cropped], axis=0)
    else:
        cropped = image[crop_top:crop_bottom, crop_left:crop_right]

    # Now handle horizontal centering
    # Calculate where the apex is in the cropped image
    apex_x_in_crop = cone_params["apex_x"] - crop_left
    cropped_height = ops.shape(cropped)[0]
    cropped_width = ops.shape(cropped)[1]

    # Calculate the target center position
    target_center_x = ops.cast(cropped_width / 2, "float32")

    # Calculate how much padding we need on each side
    # We want: left_padding + apex_x_in_crop = final_width / 2
    # And: final_width = cropped_width + left_padding + right_padding
    # For symmetric padding: left_padding = right_padding
    # So: left_padding + apex_x_in_crop = (cropped_width + 2*left_padding) / 2
    # Solving: left_padding = cropped_width/2 - apex_x_in_crop

    left_padding_needed = target_center_x - apex_x_in_crop

    # Ensure we have non-negative padding
    left_padding = ops.maximum(0, ops.cast(left_padding_needed, "int32"))
    right_padding = ops.maximum(0, ops.cast(-left_padding_needed, "int32"))

    # Apply horizontal padding if needed
    if left_padding > 0 or right_padding > 0:
        if left_padding > 0:
            left_pad = ops.zeros((cropped_height, left_padding), dtype=cropped.dtype)
            cropped = ops.concatenate([left_pad, cropped], axis=1)

        if right_padding > 0:
            right_pad = ops.zeros((cropped_height, right_padding), dtype=cropped.dtype)
            cropped = ops.concatenate([cropped, right_pad], axis=1)

    return cropped


def fit_and_crop_around_scan_cone(
    image_tensor, min_cone_half_angle_deg=20, threshold=15, return_params=False
):
    """
    Detect scan cone in ultrasound image and return cropped/padded image with centered apex.

    Args:
        image_tensor: Keras tensor (2D grayscale image)
        min_cone_half_angle_deg: Minimum expected half-angle of the cone in degrees (default: 20)
        threshold: Threshold for binary image - pixels above this are considered data (default: 15)
        return_params: If True, also return cone parameters (default: False)

    Returns:
        - If return_params is False: Keras tensor (cropped and padded image with apex at center)
        - If return_params is True: Tuple of (cropped_tensor, cone_parameters_dict)

    Raises:
        ValueError: If cone detection fails or image is not 2D
    """
    if keras.backend.backend() != "numpy":
        log.info(f"❗️ It is recommended to use {log.blue('numpy')} backend for `fit_scan_cone()`.")

    # Ensure image is 2D
    if len(ops.shape(image_tensor)) != 2:
        raise ValueError(f"Input must be 2D grayscale image, got shape {ops.shape(image_tensor)}")

    # Detect cone parameters
    cone_params = detect_cone_parameters(
        image_tensor,
        min_cone_half_angle_deg=min_cone_half_angle_deg,
        threshold=threshold,
    )

    if cone_params is None:
        raise ValueError("Failed to detect ultrasound cone in image")

    # Crop and center the image
    cropped_image = crop_and_center_cone(image_tensor, cone_params)

    if return_params:
        return cropped_image, cone_params
    else:
        return cropped_image


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
        # markeredgecolor="white",
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


def main(avi_path):
    """Demonstrate scan cone fitting on a sample AVI file."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for cone detection and visualization. "
            "Please install it with 'pip install opencv-python' or "
            "'pip install opencv-python-headless'."
        ) from exc

    # Load first frame
    cap = cv2.VideoCapture(avi_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Failed to read video file: {avi_path}")
        return

    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert to tensor
    frame_tensor = ops.convert_to_tensor(frame)

    try:
        # Fit scan cone
        _, cone_params = fit_and_crop_around_scan_cone(
            frame_tensor, min_cone_half_angle_deg=20, threshold=15, return_params=True
        )

        # Create visualization
        visualize_scan_cone(frame, cone_params)
        print("Visualization saved to output/scan_cone_visualization.png")

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
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_file = args.input_file
    main(input_file)
