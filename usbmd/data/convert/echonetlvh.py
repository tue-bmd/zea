"""
Script to convert the EchoNet-LVH database to USBMD format.
Will segment the images and convert them to polar coordinates.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"


if __name__ == "__main__":
    from usbmd import init_device

    init_device("auto:1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from pathlib import Path
import pandas as pd

import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
from jax import jit, vmap, pmap
from jax.scipy.ndimage import map_coordinates

from usbmd.data.convert.echonet import H5Processor, segment
from usbmd.utils.io_lib import load_video
from usbmd.utils.utils import translate
from usbmd.data import generate_usbmd_dataset


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert EchoNet-LVH to USBMD format")
    parser.add_argument(
        "--source",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/_RAW/echonetlvh",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonetlvh_v2025",
    )
    parser.add_argument("--output_numpy", type=str, default=None)
    parser.add_argument("--file_list", type=str, help="Optional path to list of files")
    parser.add_argument(
        "--use_hyperthreading", action="store_true", help="Enable hyperthreading"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Specify which BatchX directory to process, e.g. --batch=Batch2",
    )
    # if neither is specified, both will be converted
    parser.add_argument(
        "--convert_measurements",
        action="store_true",
        help="Only convert measurements CSV file",
    )
    parser.add_argument(
        "--convert_images", action="store_true", help="Only convert image files"
    )
    return parser.parse_args()


def load_splits(source_dir):
    """Load splits from MeasurementsList.csv"""
    csv_path = Path(source_dir) / "MeasurementsList.csv"
    df = pd.read_csv(csv_path)

    # Create dictionary of filename to split mapping
    splits = {"train": [], "val": [], "test": []}

    # Group by HashedFileName to get unique files and their splits
    for filename, group in df.groupby("HashedFileName"):
        # Get the split for this file (should all be the same)
        split = group["split"].iloc[0]
        splits[split].append(filename + ".hdf5")

    return splits


def find_avi_file(source_dir, hashed_filename, batch=None):
    """Find AVI file in the specified batch directory or any batch if not specified."""
    if batch:
        batch_dir = Path(source_dir) / batch
        avi_path = batch_dir / f"{hashed_filename}.avi"
        if avi_path.exists():
            return avi_path
        return None
    else:
        for batch_dir in Path(source_dir).glob("Batch*"):
            avi_path = batch_dir / f"{hashed_filename}.avi"
            if avi_path.exists():
                return avi_path
        return None


def manual_crop(sequence):
    """Manually crop sequence to remove zero padding for two known shapes."""

    """
    File shape counts:
    Unique (Height, Width) pairs with counts and example HashedFileNames:
    (300, 400.0) — 7 files
    (384, 512.0) — 26 files
    (480, 640.0) — 6 files
    (576, 1024.0) — 1 files
    (600, 800.0) — 3532 files --> TODO: there are at least 2 views here, we need to account for this
    (708, 1016.0) — 1 files
    (768, 1024.0) — 8417 files
    (768, 1040.0) — 10 files    
    """

    F, H, W = sequence.shape
    # full scan, new size = [632, 868]
    if (H, W) == (768, 1024):
        return sequence[:, 86:-50, 78:-78]
    # full scan, new size = [490, 680]
    elif (H, W) == (600, 800):
        return sequence[:, 70:-40, 60:-60]
    # scan cropped ~15% on all sides, new size = [344, 452]
    elif (H, W) == (384, 512):
        return sequence[:, 20:-20, 30:-30]
    # scan cropped ~15% on all sides, new size = [267, 356]
    elif (H, W) == (300, 400):
        return sequence[:, 18:-15, 22:-22]
    # full scan, new size = [362, 514]
    elif (H, W) == (480, 640):
        return sequence[:, 90:-28, 85:-40]
    # full scan, new size = [500, 684]
    elif (H, W) == (576, 1024):
        return sequence[:, 50:-26, 175:-165]
    # almost full scan, cropped ~5% at top. new size = [588, 816]
    elif (H, W) == (708, 1016):
        return sequence[:, 70:-50, 100:-100]
    # this one is quite whacky, likely an outlier, new size = [653, 865]
    elif (H, W) == (768, 1040):
        return sequence[:, 50:-65, 15:-160]
    else:
        raise ValueError(f"Unexpected image shape: {sequence.shape}")


def adaptive_segment(tensor, number_erasing=0, min_clip=0):
    """Segments the background of echonet images adaptively based on image size.

    Args:
        tensor (ndarray): Input image with shape (N, H, W)
        number_erasing (float): Value to fill background with
        min_clip (float): Minimum value for edge pixels
    """
    H, W = tensor.shape[1:]
    center_x = W // 2
    center_y = H // 2

    # Find the cone boundaries by detecting intensity changes
    # Sample columns vertically to find top edge
    top_edge = []
    for x in range(W):
        col = tensor[0, :, x]  # Take first frame
        # Find first significant intensity change from top
        edges = np.where(np.diff(col > 0.1) > 0)[0]
        if len(edges) > 0:
            top_edge.append((x, edges[0]))

    # Sample rows horizontally to find side edges
    side_edges = []
    for y in range(center_y, H):
        row = tensor[0, y, :]
        # Find leftmost and rightmost significant intensity
        edges = np.where(row > 0.1)[0]
        if len(edges) > 1:
            side_edges.append((y, edges[0], edges[-1]))

    # Fit curves to detected edges
    if len(top_edge) > 0 and len(side_edges) > 0:
        # Create mask
        mask = np.ones_like(tensor)

        # Apply top edge mask
        top_x, top_y = zip(*top_edge)
        for x, y in zip(top_x, top_y):
            mask[:, :y, x] = number_erasing
            if min_clip > 0:
                mask[:, y, x] = np.clip(tensor[:, y, x], min_clip, 1)

        # Apply side edge mask
        for y, left, right in side_edges:
            mask[:, y, :left] = number_erasing
            mask[:, y, right:] = number_erasing
            if min_clip > 0:
                mask[:, y, left] = np.clip(tensor[:, y, left], min_clip, 1)
                mask[:, y, right] = np.clip(tensor[:, y, right], min_clip, 1)

        return tensor * mask
    return tensor


def adaptive_accept_shape(tensor):
    """Determines whether to accept an image based on content analysis.

    Args:
        tensor (ndarray): Input image with shape (H, W)
    """
    H, W = tensor.shape
    center_x = W // 2
    center_y = H // 2

    # Check bottom corners for valid data
    left_roi = tensor[center_y:, : W // 4]
    right_roi = tensor[center_y:, 3 * W // 4 :]

    # Compute content metrics
    left_content = np.mean(left_roi > 0.1)
    right_content = np.mean(right_roi > 0.1)

    # Accept if both corners have sufficient content
    return left_content > 0.1 and right_content > 0.1


def rotate_coordinates(coords, angle_deg):
    """Rotate (x, y) coordinates by a given angle in degrees."""
    angle_rad = jnp.deg2rad(angle_deg)
    rotation_matrix = jnp.array(
        [
            [jnp.cos(angle_rad), -jnp.sin(angle_rad)],
            [jnp.sin(angle_rad), jnp.cos(angle_rad)],
        ]
    )
    return coords @ rotation_matrix.T


def cartesian_to_polar_matrix_jax(
    cartesian_matrix, tip=(61, 7), r_max=107, angle=0.79, interpolation="linear"
):
    rows, cols = cartesian_matrix.shape
    center_x, center_y = tip

    # Create cartesian coordinate grid
    x = jnp.linspace(-center_x, cols - center_x - 1, cols)
    y = jnp.linspace(-center_y, rows - center_y - 1, rows)
    x_grid, y_grid = jnp.meshgrid(x, y)

    # Flatten and rotate coordinates
    coords = jnp.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Interpolation grid in polar coordinates
    r = jnp.linspace(0, r_max, rows)
    theta = jnp.linspace(-angle, angle, cols)
    r_grid, theta_grid = jnp.meshgrid(r, theta)

    x_polar = r_grid * jnp.cos(theta_grid)
    y_polar = r_grid * jnp.sin(theta_grid)

    # Inverse rotation to match original orientation
    polar_coords = jnp.stack([x_polar.ravel(), y_polar.ravel()], axis=0)
    polar_coords_rotated = rotate_coordinates(polar_coords.T, 90).T

    # Shift to image indices
    yq = polar_coords_rotated[1, :] + center_y
    xq = polar_coords_rotated[0, :] + center_x
    coords_for_interp = jnp.stack([yq, xq])

    order = 0 if interpolation == "nearest" else 1
    polar_values = map_coordinates(
        cartesian_matrix,
        coords_for_interp,
        order=order,
        mode="constant",
        cval=0.0,
    )

    polar_matrix = jnp.rot90(polar_values.reshape(cols, rows), k=-1)
    return polar_matrix


def adaptive_cartesian_to_polar(cartesian_matrix, angle=jnp.deg2rad(42)):
    """Converts cartesian image to polar coordinates adaptively using JAX.

    Args:
        cartesian_matrix (ndarray): Input image with shape (H, W)
    """
    H, W = cartesian_matrix.shape
    # assume tip is at center top
    center_x = W // 2
    tip_y = 0

    # Use JAX-based polar conversion with adapted parameters
    polar_jax = cartesian_to_polar_matrix_jax(
        cartesian_matrix,
        tip=(center_x, tip_y),
        r_max=H,
        angle=angle,
    )
    return polar_jax  # Convert back to numpy for downstream compatibility


class LVHProcessor(H5Processor):
    """Modified H5Processor for EchoNet-LVH dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cart2pol_jit = jit(adaptive_cartesian_to_polar)
        self.cart2pol_batched = vmap(self.cart2pol_jit)

    def get_split(self, hdf5_file: str, sequence):
        for split, files in self.splits.items():
            if hdf5_file in files:
                return split
        raise UserWarning("Unknown split for file: " + hdf5_file)

    def __call__(self, avi_file):
        print(avi_file)
        hdf5_file = avi_file.stem + ".hdf5"
        sequence = jnp.array(load_video(avi_file))

        sequence = translate(sequence, self.range_from, self._process_range)
        sequence = manual_crop(sequence)

        split = self.get_split(hdf5_file, sequence)
        out_h5 = self.path_out_h5 / split / hdf5_file

        polar_im_set = self.cart2pol_batched(sequence)

        usbmd_dataset = {
            "path": out_h5,
            "image_sc": self.translate(np.array(sequence)),
            "probe_name": "generic",
            "description": "EchoNet-LVH dataset converted to USBMD format",
            "image": self.translate(np.array(polar_im_set)),
        }
        return generate_usbmd_dataset(**usbmd_dataset)


def transform_measurement_coordinates(row):
    """Transform measurement coordinates according to the cropping scheme.

    Args:
        row: A pandas Series containing measurement data with X1,X2,Y1,Y2 coordinates

    Returns:
        A new row with transformed coordinates, or None if shape is unexpected
    """
    H, W = row["Height"], row["Width"]

    # Define cropping parameters for each image size
    crop_params = {
        (768, 1024): {"top": 86, "bottom": 50, "left": 78, "right": 78},
        (600, 800): {"top": 70, "bottom": 40, "left": 60, "right": 60},
        (384, 512): {"top": 20, "bottom": 20, "left": 30, "right": 30},
        (300, 400): {"top": 18, "bottom": 15, "left": 22, "right": 22},
        (480, 640): {"top": 90, "bottom": 28, "left": 85, "right": 40},
        (576, 1024): {"top": 50, "bottom": 26, "left": 175, "right": 165},
        (708, 1016): {"top": 70, "bottom": 50, "left": 100, "right": 100},
        (768, 1040): {"top": 50, "bottom": 65, "left": 15, "right": 160},
    }

    # Get cropping parameters for this image size
    params = crop_params.get((H, W))
    if params is None:
        print(
            f"Warning: Skipping file {row['HashedFileName']} due to unexpected image shape: ({H}, {W})"
        )
        return None

    # Transform coordinates
    new_row = row.copy()
    new_row["X1"] = row["X1"] - params["left"]
    new_row["X2"] = row["X2"] - params["left"]
    new_row["Y1"] = row["Y1"] - params["top"]
    new_row["Y2"] = row["Y2"] - params["top"]

    return new_row


def convert_measurements_csv(source_csv, output_csv):
    """Convert measurements CSV file with updated coordinates.

    Args:
        source_csv: Path to source CSV file
        output_csv: Path to output CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(source_csv)

        # Apply coordinate transformation and track skipped rows
        transformed_rows = []
        skipped_files = set()

        for _, row in df.iterrows():
            try:
                transformed_row = transform_measurement_coordinates(row)
                if transformed_row is not None:
                    transformed_rows.append(transformed_row)
                else:
                    skipped_files.add(row["HashedFileName"])
            except Exception as e:
                print(
                    f"Error processing row for file {row['HashedFileName']}: {str(e)}"
                )
                skipped_files.add(row["HashedFileName"])

        # Create new dataframe from transformed rows
        df_transformed = pd.DataFrame(transformed_rows)

        # Save to new CSV file
        df_transformed.to_csv(output_csv, index=False)

        # Print summary
        print(f"\nConversion Summary:")
        print(f"Total rows processed: {len(df)}")
        print(f"Rows successfully converted: {len(df_transformed)}")
        print(f"Rows skipped: {len(df) - len(df_transformed)}")
        if skipped_files:
            print("\nSkipped files:")
            for filename in sorted(skipped_files):
                print(f"  - {filename}")
        print(f"\nConverted measurements saved to {output_csv}")

    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        raise


if __name__ == "__main__":
    args = get_args()

    # If no specific conversion is requested, convert both
    if not (args.convert_measurements or args.convert_images):
        args.convert_measurements = True
        args.convert_images = True

    # Convert measurements if requested
    if args.convert_measurements:
        source_path = Path(args.source)
        measurements_csv = source_path / "MeasurementsList.csv"
        if measurements_csv.exists():
            output_csv = Path(args.output) / "MeasurementsList.csv"
            convert_measurements_csv(measurements_csv, output_csv)
        else:
            print("Warning: MeasurementsList.csv not found in source directory")

    # Convert images if requested
    if args.convert_images:
        source_path = Path(args.source)
        splits = load_splits(source_path)

        files_to_process = []
        for split_files in splits.values():
            for hdf5_file in split_files:
                avi_file = find_avi_file(
                    args.source, hdf5_file[:-5], batch=args.batch
                )  # Pass batch arg
                if avi_file:
                    files_to_process.append(avi_file)
                else:
                    print(
                        f"Warning: Could not find AVI file for {hdf5_file} in batch {args.batch if args.batch else 'any'}"
                    )

        # List files that have already been processed
        files_done = []
        for _, _, filenames in os.walk(args.output):
            for filename in filenames:
                files_done.append(filename.replace(".hdf5", ""))

        # Filter out already processed files
        files_to_process = [f for f in files_to_process if f.stem not in files_done]
        print(f"Files left to process: {len(files_to_process)}")

        # Initialize processor with splits
        processor = LVHProcessor(
            path_out_h5=args.output,
            path_out=args.output_numpy,
            splits=splits,
        )

        print("Starting the conversion process.")

        if args.use_hyperthreading:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(processor, file): file for file in files_to_process
                }
                for future in tqdm(as_completed(futures), total=len(files_to_process)):
                    future.result()
        else:
            for file in tqdm(files_to_process):
                processor(file)

        print("All tasks are completed.")
