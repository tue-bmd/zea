"""
Example script for visualizing measurements from EchoNet-LVH dataset.
"""

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from tqdm import tqdm
import h5py
import argparse

from usbmd import init_device, log, set_data_paths
from usbmd.utils.visualize import set_mpl_style


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize EchoNet-LVH measurements")
    parser.add_argument(
        "--limit_frames",
        type=int,
        help="Limit the number of frames to process",
        default=None,
    )
    return parser.parse_args()


def get_measurements_for_file(csv_path, file_id):
    """Get all measurements for a specific file."""
    df = pd.read_csv(csv_path)
    file_measurements = df[df["HashedFileName"] == file_id]
    return file_measurements


def plot_frame_with_measurements(frame, measurements, frame_idx):
    """Plot a single frame with its measurements."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame, cmap="gray")

    # Get measurements for this frame
    frame_measurements = measurements[measurements["Frame"] == frame_idx]
    log.info(f"Frame {frame_idx}: Found {len(frame_measurements)} measurements")

    # Plot each measurement
    for _, measurement in frame_measurements.iterrows():
        # Draw measurement line segment
        ax.plot(
            [measurement["X1"], measurement["X2"]],
            [measurement["Y1"], measurement["Y2"]],
            "r-",
            linewidth=2,
        )

        # Add measurement label at the midpoint of the line
        mid_x = (measurement["X1"] + measurement["X2"]) / 2
        mid_y = (measurement["Y1"] + measurement["Y2"]) / 2

        # Calculate angle for label rotation
        dx = measurement["X2"] - measurement["X1"]
        dy = measurement["Y2"] - measurement["Y1"]
        angle = np.degrees(np.arctan2(dy, dx))

        # Add rotated text label
        ax.text(
            mid_x,
            mid_y,
            f"{measurement['Calc']}: {measurement['CalcValue']:.2f}",
            color="red",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            rotation=angle,
        )

        # Debug: print measurement details
        log.info(
            f"  Drawing measurement: {measurement['Calc']} at ({measurement['X1']}, {measurement['Y1']}) to ({measurement['X2']}, {measurement['Y2']})"
        )

    ax.set_title(f"Frame {frame_idx}")
    ax.axis("off")
    return fig


def create_measurement_gif(video_data, measurements, output_path, fps=30):
    """Create a GIF of the video with measurements."""
    # Create frames
    frames = []
    for frame_idx in tqdm(range(len(video_data)), desc="Creating frames"):
        fig = plot_frame_with_measurements(
            video_data[frame_idx], measurements, frame_idx
        )

        # Convert figure to image
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf.shape = (h, w, 4)

        # Convert RGBA to RGB
        image = buf[:, :, :3]
        frames.append(image)

        plt.close(fig)

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)
    log.info(f"Saved GIF to {log.yellow(output_path)}")


if __name__ == "__main__":
    # Parse arguments
    args = get_args()

    # Set up data paths and device
    data_paths = set_data_paths()
    init_device()

    # Set paths
    base_path = Path("/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonetlvh_v2025")
    csv_path = base_path / "MeasurementsList.csv"
    train_path = base_path / "train"

    # Get list of valid files (768x1024)
    df = pd.read_csv(csv_path)
    valid_files = df[(df["Width"] == 1024) & (df["split"] == "train")][
        "HashedFileName"
    ].unique()

    if len(valid_files) == 0:
        raise ValueError("No valid training files found with width 1024")

    # Filter valid files to only those that exist in the training directory
    existing_files = []
    for file_id in valid_files:
        video_path = train_path / f"{file_id}.hdf5"
        if video_path.exists():
            existing_files.append(file_id)

    if len(existing_files) == 0:
        raise ValueError("No valid training files found in the training directory")

    # Select a random file from existing files
    file_id = random.choice(existing_files)
    log.info(f"Selected file: {file_id}")
    video_path = train_path / f"{file_id}.hdf5"

    # Load video data using h5py
    try:
        with h5py.File(video_path, "r") as f:
            video_data = f["data/image_sc"][
                : args.limit_frames
            ]  # Load only specified number of frames
    except Exception as e:
        log.error(f"Error loading file {video_path}: {str(e)}")
        raise

    # Get measurements for this file
    measurements = get_measurements_for_file(csv_path, file_id)
    log.info(f"Found {len(measurements)} total measurements for file {file_id}")
    log.info(
        f"Measurement frames range: {measurements['Frame'].min()} to {measurements['Frame'].max()}"
    )
    log.info(f"Video has {len(video_data)} frames")

    # Create output directory if it doesn't exist
    output_dir = Path("examples/echonetlvh/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create GIF
    output_path = output_dir / f"{file_id}_measurements.gif"
    create_measurement_gif(video_data, measurements, output_path)
