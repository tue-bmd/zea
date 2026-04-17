"""
Script to visualize a sample from the EchoNet-LVH dataset in zea format.
Creates a GIF of the video frames and images with measurement overlays.
"""

import argparse
import csv
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from zea.func.tensor import translate
from zea.io_lib import save_video


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot a sample from EchoNet-LVH dataset")
    parser.add_argument(
        "--input_h5",
        type=str,
        # e.g. {data_root}/echonetlvh_v2025/train/0X1017398D3C3F5FF9.hdf5
        required=True,
        help="Path to input HDF5 file",
    )
    parser.add_argument(
        "--measurements_csv",
        type=str,
        required=True,
        # e.g. {data_root}/echonetlvh_v2025/MeasurementsList.csv
        help="Path to measurements CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output files",
    )
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for the GIF")
    return parser.parse_args()


def load_video_from_h5(h5_path):
    """
    Load video frames from HDF5 file.

    Args:
        h5_path: Path to the HDF5 file

    Returns:
        Numpy array of video frames
    """
    with h5py.File(h5_path, "r") as f:
        if "data/image_sc" in f:
            # Load all frames
            frames = np.array(f["data/image_sc"])
            return frames
        else:
            print(f"Error: 'data/image_sc' not found in {h5_path}")
            print(f"Available keys: {list(f.keys())}")
            raise KeyError(f"'data/image_sc' not found in {h5_path}")


def find_measurements(csv_path, video_id):
    """
    Find measurements for a specific video ID.

    Args:
        csv_path: Path to the measurements CSV file
        video_id: ID of the video (HDF5 filename without extension)

    Returns:
        List of dicts with measurements for the video
    """
    video_basename = Path(video_id).stem
    measurements = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("HashedFileName") == video_basename:
                measurements.append(row)

    if len(measurements) == 0:
        print(f"Warning: No measurements found for {video_basename}")

    return measurements


def create_measurement_overlays(frames, measurements, output_dir):
    """
    Create images with measurement line overlays.

    Args:
        frames: Video frames
        measurements: List of dicts with measurements
        output_dir: Directory to save output images
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each measurement
    for idx, row in enumerate(measurements):
        if "Frame" in row and row["Frame"] != "":
            frame_idx = int(row["Frame"])
            if 0 <= frame_idx < len(frames):
                frame = frames[frame_idx]

                # Create figure with two subplots: image and legend area
                fig = plt.figure(figsize=(12, 8))

                # Main image takes 80% of width, legend takes 20%
                ax_image = fig.add_axes([0.05, 0.1, 0.75, 0.8])  # [left, bottom, width, height]
                ax_legend = fig.add_axes([0.82, 0.1, 0.15, 0.8])

                # Normalize frame for better visualization
                vmin, vmax = np.percentile(frame, (1, 99))
                normalized = np.clip((frame - vmin) / (vmax - vmin + 1e-6), 0, 1)

                # Display frame
                ax_image.imshow(normalized, cmap="gray")

                # Extract coordinates and draw the measurement line
                try:
                    # Convert coordinates to float
                    x1, y1 = float(row["X1"]), float(row["Y1"])
                    x2, y2 = float(row["X2"]), float(row["Y2"])

                    # Draw line
                    ax_image.plot([x1, x2], [y1, y2], "r-", linewidth=2, label=f"{row['Calc']}")
                    ax_image.plot([x1], [y1], "go", markersize=5)  # Start point
                    ax_image.plot([x2], [y2], "bo", markersize=5)  # End point

                    # Get measurement type and value
                    measurement_type = row["Calc"]
                    measurement_value = row.get("CalcValue", "N/A")

                    # Hide axes in legend area
                    ax_legend.axis("off")

                    # Add measurement details to legend area
                    ax_legend.text(0.05, 0.9, "Measurement:", fontsize=12, fontweight="bold")
                    ax_legend.text(0.05, 0.8, f"Type: {measurement_type}", fontsize=12)
                    ax_legend.text(0.05, 0.7, f"Value: {measurement_value}", fontsize=12)
                    ax_legend.text(0.05, 0.6, f"Frame: {frame_idx}", fontsize=12)

                    # Add coordinate information
                    ax_legend.text(0.05, 0.45, "Coordinates:", fontsize=12, fontweight="bold")
                    ax_legend.text(0.05, 0.35, f"Start: ({x1:.1f}, {y1:.1f})", fontsize=10)
                    ax_legend.text(0.05, 0.25, f"End: ({x2:.1f}, {y2:.1f})", fontsize=10)

                    # Add color legend
                    ax_legend.text(0.05, 0.1, "● Start point", fontsize=10, color="green")
                    ax_legend.text(0.05, 0.05, "● End point", fontsize=10, color="blue")
                    ax_legend.text(0.05, 0.0, "— Measurement", fontsize=10, color="red")

                    # Set title for the entire figure
                    fig.suptitle(f"Frame {frame_idx}: {measurement_type}", fontsize=14)

                    # Save figure
                    filename = f"frame_{frame_idx}_{measurement_type.replace('/', '_')}.png"
                    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved measurement overlay: {filename}")

                except Exception as e:
                    print(f"Error processing measurement at index {idx}: {e}")
            else:
                print(f"Warning: Frame index {frame_idx} out of bounds (0-{len(frames) - 1})")
        else:
            print(f"Warning: No Frame index for measurement at index {idx}")


def main():
    """
    Main function to process and visualize EchoNet-LVH dataset samples.

    This function:
    1. Loads video frames from an HDF5 file
    2. Creates a GIF of the video
    3. Finds measurements for the video
    4. Creates measurement overlay images with side panels
    """
    args = get_args()

    try:
        # Load video from H5
        print(f"Loading video from {args.input_h5}")
        frames = load_video_from_h5(args.input_h5)
        print(f"Loaded {len(frames)} frames")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Save as GIF
        gif_path = os.path.join(args.output_dir, f"{Path(args.input_h5).stem}.gif")
        frames_as_img = translate(frames, (-60, 0), (0, 255)).astype(np.uint8)
        save_video(frames_as_img, gif_path, fps=args.fps)

        # Find measurements
        print(f"Looking for measurements in {args.measurements_csv}")
        measurements = find_measurements(args.measurements_csv, args.input_h5)
        print(f"Found {len(measurements)} measurements")

        # Create measurement overlays if any measurements found
        if len(measurements) > 0:
            measurement_dir = os.path.join(args.output_dir, "measurements")
            create_measurement_overlays(frames, measurements, measurement_dir)

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
