"""
Script to analyze EchoNet-LVH AVI files for automatic cropping detection.
Uses geometric cone fitting to detect scan parameters.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from usbmd import log
from usbmd.utils.fit_scan_cone import (
    detect_cone_parameters,
    crop_and_center_cone,
    fit_scan_cone,
)
from keras import ops


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze EchoNet-LVH cropping patterns using cone fitting"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/_RAW/echonetlvh/Batch1",
        help="Directory containing AVI files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./echonetlvh_cropping_analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )
    parser.add_argument(
        "--min_cone_half_angle",
        type=float,
        default=20.0,
        help="Minimum expected half-angle of the cone in degrees (default: 20.0)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="Threshold for binary image (default: 15)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging",
    )
    return parser.parse_args()


def load_first_frame_only(avi_path):
    """Load only the first frame from an AVI file for efficiency."""
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {avi_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read first frame from: {avi_path}")

    # Convert BGR to RGB if needed
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def fit_cone_geometry_wrapper(
    image, min_cone_half_angle_deg=20, threshold=15, verbose=False
):
    """
    Wrapper for detect_cone_parameters to maintain compatibility with existing code.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Convert to Keras tensor
    gray_tensor = ops.convert_to_tensor(gray, dtype="uint8")

    # Use the generic function - this does ALL the calculations
    cone_params = detect_cone_parameters(
        gray_tensor,
        min_cone_half_angle_deg=min_cone_half_angle_deg,
        threshold=threshold,
    )

    if cone_params is None:
        return None

    # Only add minimal compatibility fields - don't recalculate anything
    h, w = gray.shape

    # Add backward compatibility fields (using values from cone_params, not recalculating)
    cone_params.update(
        {
            # These are just for backward compatibility - use values already calculated
            "symmetry_ratio": cone_params.get("symmetry_ratio", 0.0),
            "first_data_row": cone_params.get("first_data_row", 0),
            "data_coverage": cone_params.get("data_coverage", 0.0),
            "apex_above_image": cone_params.get("apex_above_image", False),
        }
    )

    if verbose:
        print(f"Apex: ({cone_params['apex_x']:.1f}, {cone_params['apex_y']:.1f})")
        print(f"Opening angle: {np.degrees(cone_params['opening_angle']):.1f}°")
        print(
            f"Crop: ({cone_params['crop_left']}, {cone_params['crop_top']}) to ({cone_params['crop_right']}, {cone_params['crop_bottom']})"
        )

    return cone_params


def analyze_frames(
    source_dir, max_files=None, min_cone_half_angle_deg=20, threshold=15, verbose=False
):
    """Analyze first frames from AVI files and extract cone parameters."""
    source_path = Path(source_dir)
    avi_files = list(source_path.glob("*.avi"))

    if max_files:
        avi_files = avi_files[:max_files]

    if verbose:
        print(f"Found {len(avi_files)} AVI files to process")

    results = []
    for avi_file in tqdm(avi_files, desc="Analyzing AVI files", disable=not verbose):
        try:
            # Load only the first frame
            first_frame = load_first_frame_only(avi_file)

            # Convert to uint8 if needed
            if first_frame.dtype != np.uint8:
                first_frame = (first_frame * 255).astype(np.uint8)

            # Fit cone geometry using wrapper
            cone_params = fit_cone_geometry_wrapper(
                first_frame, min_cone_half_angle_deg, threshold, verbose
            )

            if cone_params:
                result = {
                    "filename": avi_file.stem,
                    "avi_path": str(avi_file),
                    "first_frame": first_frame,
                    **cone_params,
                }
                results.append(result)
            elif verbose:
                print(f"Warning: Could not fit cone to {avi_file.name}")

        except Exception as e:
            if verbose:
                print(f"Error processing {avi_file.name}: {e}")
            continue

    return results


def plot_cone_analysis(image, cone_params, filename="", show_crosshairs=False):
    """
    Shared function to plot cone analysis visualization.
    Used by both save_analysis_plots and debug_cone_detection.
    """
    h, w = image.shape[:2]

    # Extract parameters directly from cone_params (no recalculation)
    apex_x, apex_y = cone_params["apex_x"], cone_params["apex_y"]
    left_a = cone_params["left_intercept"]
    left_b = cone_params["left_slope"]
    right_a = cone_params["right_intercept"]
    right_b = cone_params["right_slope"]

    # Use sector intersection points if available
    if "sector_left_x" in cone_params and "sector_right_x" in cone_params:
        left_x_bottom = cone_params["sector_left_x"]
        left_y_bottom = cone_params["sector_left_y"]
        right_x_bottom = cone_params["sector_right_x"]
        right_y_bottom = cone_params["sector_right_y"]
    else:
        # Fallback
        bottom_y = h - 1
        left_x_bottom = left_a + left_b * bottom_y
        right_x_bottom = right_a + right_b * bottom_y
        left_y_bottom = right_y_bottom = bottom_y

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(image, cmap="gray" if len(image.shape) == 2 else None)

    # Draw cone lines
    ax.plot(
        [apex_x, left_x_bottom],
        [apex_y, left_y_bottom],
        "r-",
        linewidth=2,
        label="Detected cone",
    )
    ax.plot([apex_x, right_x_bottom], [apex_y, right_y_bottom], "r-", linewidth=2)

    # Draw circular arc if available
    if "circle_center_x" in cone_params:
        circle_center_x = cone_params["circle_center_x"]
        circle_center_y = cone_params["circle_center_y"]
        circle_radius = cone_params["circle_radius"]

        # Calculate the angular extent of the sector
        left_angle = np.arctan2(
            left_x_bottom - circle_center_x, left_y_bottom - circle_center_y
        )
        right_angle = np.arctan2(
            right_x_bottom - circle_center_x, right_y_bottom - circle_center_y
        )

        # Draw the circular arc
        theta = np.linspace(left_angle, right_angle, 100)
        arc_x = circle_center_x + circle_radius * np.sin(theta)
        arc_y = circle_center_y + circle_radius * np.cos(theta)

        # Only draw the part that's visible in the image
        visible_mask = (arc_y >= 0) & (arc_y < h) & (arc_x >= 0) & (arc_x < w)
        if np.any(visible_mask):
            ax.plot(
                arc_x[visible_mask],
                arc_y[visible_mask],
                "g--",
                linewidth=2,
                label="Circular bottom",
            )

    # Draw apex
    ax.plot(
        apex_x,
        apex_y,
        "ro",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Cone apex",
    )

    # Highlight intersection points if available
    if "sector_left_x" in cone_params:
        ax.plot(
            cone_params["sector_left_x"],
            cone_params["sector_left_y"],
            "yo",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Intersection points",
        )
        ax.plot(
            cone_params["sector_right_x"],
            cone_params["sector_right_y"],
            "yo",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

    # Add crosshairs if requested (for debug mode)
    if show_crosshairs:
        ax.axhline(y=h / 2, color="yellow", linestyle="--", alpha=0.5)
        ax.axvline(x=w / 2, color="yellow", linestyle="--", alpha=0.5)

    # Add info text
    opening_angle_deg = np.degrees(cone_params["opening_angle"])
    info_text = (
        f"Opening Angle: {opening_angle_deg:.1f}°\n"
        f'Symmetry Ratio: {cone_params.get("symmetry_ratio", 0.0):.3f}'
    )

    if "circle_radius" in cone_params:
        info_text += f'\nCircle Radius: {cone_params["circle_radius"]:.1f} px'

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    # Simpler crop info without showing the bounding box
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
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.9),
    )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Ultrasound Cone Detection: {filename}", fontsize=16, fontweight="bold"
    )
    ax.axis("off")

    plt.tight_layout()
    return fig


def create_cropped_image_wrapper(image, cone_params):
    """
    Wrapper for crop_and_center_cone to maintain compatibility.
    All cropping logic comes from fit_scan_cone.py
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Convert to Keras tensor
    gray_tensor = ops.convert_to_tensor(gray, dtype="uint8")

    # Use the generic function - all cropping logic is in fit_scan_cone.py
    cropped_tensor = crop_and_center_cone(gray_tensor, cone_params)

    # Convert back to numpy
    cropped_image = ops.convert_to_numpy(cropped_tensor)

    # Add centering information for compatibility (these are just informational)
    cone_params["final_apex_x"] = cropped_image.shape[1] / 2
    cone_params["final_apex_y"] = cropped_image.shape[0] / 2
    cone_params["apex_centered"] = True

    return cropped_image


def save_analysis_plots(results, output_dir, verbose=False):
    """Save analysis plots for all images."""
    output_path = Path(output_dir)
    plots_dir = output_path / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)

    for result in tqdm(results, desc="Saving analysis plots", disable=not verbose):
        base_filename = result["filename"]
        size_str = f"{result['original_height']}x{result['original_width']}"

        # Create combined figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Original image
        axes[0].imshow(
            result["first_frame"],
            cmap="gray" if len(result["first_frame"].shape) == 2 else None,
        )
        axes[0].set_title(f"Original: {base_filename}")
        axes[0].axis("off")

        # Plot 2: Analysis with cone detection
        # Use shared plotting function
        analysis_fig = plot_cone_analysis(
            result["first_frame"], result, filename=base_filename
        )

        # Copy the analysis plot to our subplot
        analysis_ax = analysis_fig.gca()
        axes[1].imshow(
            analysis_ax.get_images()[0].get_array(),
            cmap=analysis_ax.get_images()[0].get_cmap(),
        )

        # Copy all the lines and patches
        for line in analysis_ax.get_lines():
            axes[1].plot(
                *line.get_data(),
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                markeredgecolor=line.get_markeredgecolor(),
                markeredgewidth=line.get_markeredgewidth(),
                label=line.get_label(),
            )

        for patch in analysis_ax.patches:
            axes[1].add_patch(
                type(patch)(
                    patch.get_xy(),
                    patch.get_width(),
                    patch.get_height(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor=patch.get_facecolor(),
                    label=patch.get_label(),
                )
            )

        axes[1].set_title("Analysis & Detection")
        axes[1].legend()
        axes[1].set_xlim(0, result["original_width"])
        axes[1].set_ylim(result["original_height"], 0)
        axes[1].axis("off")

        plt.close(analysis_fig)  # Clean up the temporary figure

        # Plot 3: Cropped and centered result
        cropped_image = create_cropped_image_wrapper(result["first_frame"], result)
        axes[2].imshow(cropped_image, cmap="gray")

        # Draw apex point (should be at center)
        final_apex_x = cropped_image.shape[1] / 2
        final_apex_y = cropped_image.shape[0] / 2
        axes[2].plot(
            final_apex_x,
            final_apex_y,
            "ro",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
        )

        # Draw crosshairs for verification
        axes[2].axhline(
            y=cropped_image.shape[0] / 2, color="yellow", linestyle="--", alpha=0.7
        )
        axes[2].axvline(
            x=cropped_image.shape[1] / 2, color="yellow", linestyle="--", alpha=0.7
        )

        axes[2].set_title(
            f"Cropped & Centered ({cropped_image.shape[1]}x{cropped_image.shape[0]})"
        )
        axes[2].axis("off")

        plt.tight_layout()

        # Save combined plot
        filename = f"{base_filename}_{size_str}_analysis.png"
        fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Store the plot path
        result["analysis_plot"] = str(plots_dir / filename)


def create_cropping_csv(results, output_dir):
    """Create CSV file with cropping information for all files."""
    output_path = Path(output_dir)

    # Prepare data for CSV (exclude first_frame array)
    csv_data = [
        {k: v for k, v in result.items() if k != "first_frame"} for result in results
    ]

    # Save full CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path / "cropping_parameters.csv", index=False)

    # Save essential CSV
    essential_columns = [
        "filename",
        "original_width",
        "original_height",
        "crop_left",
        "crop_right",
        "crop_top",
        "crop_bottom",
        "new_width",
        "new_height",
        "opening_angle",
        "circle_radius",
        "data_coverage",
        "apex_above_image",
    ]

    available_columns = [col for col in essential_columns if col in df.columns]
    df[available_columns].to_csv(
        output_path / "cropping_parameters_essential.csv", index=False
    )


def generate_report(results, output_dir):
    """Generate analysis report and save results."""
    output_path = Path(output_dir)

    # Calculate some summary statistics
    opening_angles = [np.degrees(r["opening_angle"]) for r in results]
    circle_radii = [r.get("circle_radius", 0) for r in results]
    sizes = [(r["original_height"], r["original_width"]) for r in results]
    unique_sizes = list(set(sizes))

    report_lines = [
        "EchoNet-LVH Cropping Analysis Report",
        "=" * 40,
        f"Total files processed: {len(results)}",
        f"Unique image sizes: {len(unique_sizes)}",
        "",
        "Summary Statistics:",
        f"  Opening angles: {np.mean(opening_angles):.1f}° ± {np.std(opening_angles):.1f}°",
        f"    Range: {np.min(opening_angles):.1f}° to {np.max(opening_angles):.1f}°",
        f"  Circle radii: {np.mean(circle_radii):.1f} ± {np.std(circle_radii):.1f} px",
        f"    Range: {np.min(circle_radii):.1f} to {np.max(circle_radii):.1f} px",
        "",
        "Image sizes found:",
    ]

    # Group by size
    size_counts = {}
    for size in sizes:
        size_counts[size] = size_counts.get(size, 0) + 1

    for size, count in sorted(size_counts.items()):
        report_lines.append(f"  {size[0]}x{size[1]}: {count} files")

    report_lines.extend(
        [
            "",
            "All analysis plots saved to: analysis_plots/",
            "Parameter CSV files saved to: cropping_parameters.csv",
        ]
    )

    # Save report
    with open(output_path / "cropping_analysis_report.txt", "w") as f:
        f.write("\n".join(report_lines))


def debug_cone_detection(image_path, min_cone_half_angle_deg=20, threshold=15):
    """
    Debug function to visualize cone detection steps for a single image.
    Uses the same logic as the main analysis.
    """
    # Load first frame
    frame = load_first_frame_only(image_path)

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.copy()

    # Use the generic fit_scan_cone function with return_params=True
    gray_tensor = ops.convert_to_tensor(gray, dtype="uint8")

    try:
        cropped_tensor, cone_params = fit_scan_cone(
            gray_tensor,
            min_cone_half_angle_deg=min_cone_half_angle_deg,
            threshold=threshold,
            return_params=True,
        )
        cropped_image = ops.convert_to_numpy(cropped_tensor)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create visualization with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Original image
    axes[0].imshow(frame, cmap="gray" if len(frame.shape) == 2 else None)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot 2: Analysis with cone detection - use shared function
    analysis_fig = plot_cone_analysis(
        frame, cone_params, filename=Path(image_path).stem, show_crosshairs=False
    )

    # Copy to subplot (same as above)
    analysis_ax = analysis_fig.gca()
    axes[1].imshow(
        analysis_ax.get_images()[0].get_array(),
        cmap=analysis_ax.get_images()[0].get_cmap(),
    )

    # Copy all the lines and patches
    for line in analysis_ax.get_lines():
        axes[1].plot(
            *line.get_data(),
            color=line.get_color(),
            linewidth=line.get_linewidth(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            markeredgecolor=line.get_markeredgecolor(),
            markeredgewidth=line.get_markeredgewidth(),
            label=line.get_label(),
        )

    for patch in analysis_ax.patches:
        axes[1].add_patch(
            type(patch)(
                patch.get_xy(),
                patch.get_width(),
                patch.get_height(),
                linewidth=patch.get_linewidth(),
                edgecolor=patch.get_edgecolor(),
                facecolor=patch.get_facecolor(),
                label=patch.get_label(),
            )
        )

    axes[1].set_title("Cone Detection & Crop")
    axes[1].legend()
    axes[1].set_xlim(0, gray.shape[1])
    axes[1].set_ylim(gray.shape[0], 0)
    axes[1].axis("off")

    plt.close(analysis_fig)  # Clean up

    # Plot 3: Cropped and centered result
    axes[2].imshow(cropped_image, cmap="gray")

    # Draw apex point in the cropped image (should be at center)
    final_apex_x = cropped_image.shape[1] / 2
    final_apex_y = cropped_image.shape[0] / 2

    axes[2].plot(
        final_apex_x,
        final_apex_y,
        "ro",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=2,
    )

    # Draw crosshairs at center for verification
    axes[2].axhline(
        y=cropped_image.shape[0] / 2, color="yellow", linestyle="--", alpha=0.7
    )
    axes[2].axvline(
        x=cropped_image.shape[1] / 2, color="yellow", linestyle="--", alpha=0.7
    )

    axes[2].set_title(
        f"Cropped & Centered ({cropped_image.shape[1]}x{cropped_image.shape[0]})"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("cone_debug.png", dpi=150, bbox_inches="tight")
    log.info(f"Debug plot saved to {log.yellow('cone_debug.png')}")

    # Save the cropped image separately
    Image.fromarray(cropped_image).save("cropped_debug.png")
    log.info(f"Cropped image saved to {log.yellow('cropped_debug.png')}")

    # Print detailed statistics (using values from fit_scan_cone.py)
    print(f"\n--- Cone Detection Results ---")
    print(f"Apex: ({cone_params['apex_x']:.1f}, {cone_params['apex_y']:.1f})")
    print(f"Opening angle: {np.degrees(cone_params['opening_angle']):.1f}°")
    print(f"Cone height: {cone_params['cone_height']:.1f} pixels")
    print(f"Circle radius: {cone_params.get('circle_radius', 'N/A'):.1f} pixels")
    print(
        f"Logical crop: ({cone_params['logical_crop_left']}, {cone_params['logical_crop_top']}) to ({cone_params['logical_crop_right']}, {cone_params['logical_crop_bottom']})"
    )
    print(
        f"Physical crop: ({cone_params['physical_crop_left']}, {cone_params['physical_crop_top']}) to ({cone_params['physical_crop_right']}, {cone_params['physical_crop_bottom']})"
    )
    print(f"Top padding: {cone_params['top_padding']} pixels")
    print(f"Original size: {gray.shape[1]}x{gray.shape[0]}")
    print(f"Final size: {cone_params['final_width']}x{cone_params['final_height']}")
    print(f"Cropped size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    print(f"Apex above image: {cone_params.get('apex_above_image', False)}")
    print(f"Data coverage: {cone_params.get('data_coverage', 0.0):.3f}")


def main():
    args = get_args()

    if args.verbose:
        print(f"Analyzing AVI files in: {args.source}")
        print(f"Output directory: {args.output}")
        if args.max_files:
            print(f"Processing max {args.max_files} files")
        print(f"Min cone half-angle: {args.min_cone_half_angle}°")
        print(f"Threshold: {args.threshold}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Analyze first frames
    results = analyze_frames(
        args.source,
        args.max_files,
        args.min_cone_half_angle,
        args.threshold,
        args.verbose,
    )

    if not results:
        print("No valid results found. Exiting.")
        return

    # Save analysis plots for all images
    save_analysis_plots(results, args.output, args.verbose)

    # Create CSV with parameters
    create_cropping_csv(results, args.output)

    # Generate summary report
    generate_report(results, args.output)

    if args.verbose:
        print(f"Analysis complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
    # Uncomment the line below to debug a specific file:
    # debug_cone_detection(
    #     "/mnt/z/Ultrasound-BMd/data/USBMD_datasets/_RAW/echonetlvh/Batch1/0X1027077445AE5512.avi",
    #     min_cone_half_angle_deg=20,
    # )
