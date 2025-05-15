"""
Script to convert the EchoNet-LVH database to USBMD format.
Will segment the images and convert them to polar coordinates.
"""

import os
import pickle

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
from jax import jit, vmap, pmap, lax
from jax.scipy.ndimage import map_coordinates

from usbmd.data.convert.echonet import H5Processor, segment
from usbmd.utils.io_lib import load_video
from usbmd.utils.utils import translate
from usbmd.data import generate_usbmd_dataset
from usbmd.utils.fit_scan_cone import fit_scan_cone


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
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)",
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


# def manual_crop(sequence):
#     """Manually crop sequence to remove zero padding for two known shapes."""

#     """
#     File shape counts:
#     Unique (Height, Width) pairs with counts and example HashedFileNames:
#     (300, 400.0) — 7 files
#     (384, 512.0) — 26 files
#     (480, 640.0) — 6 files
#     (576, 1024.0) — 1 files
#     (600, 800.0) — 3532 files --> TODO: there are at least 2 views here, we need to account for this
#     (708, 1016.0) — 1 files
#     (768, 1024.0) — 8417 files
#     (768, 1040.0) — 10 files
#     """

#     F, H, W = sequence.shape
#     # full scan, new size = [632, 868]
#     if (H, W) == (768, 1024):
#         return sequence[:, 86:-50, 78:-78]
#     # full scan, new size = [490, 680]
#     elif (H, W) == (600, 800):
#         return sequence[:, 70:-40, 60:-60]
#     # scan cropped ~15% on all sides, new size = [344, 452]
#     elif (H, W) == (384, 512):
#         return sequence[:, 20:-20, 30:-30]
#     # scan cropped ~15% on all sides, new size = [267, 356]
#     elif (H, W) == (300, 400):
#         return sequence[:, 18:-15, 22:-22]
#     # full scan, new size = [362, 514]
#     elif (H, W) == (480, 640):
#         return sequence[:, 90:-28, 85:-40]
#     # full scan, new size = [500, 684]
#     elif (H, W) == (576, 1024):
#         return sequence[:, 50:-26, 175:-165]
#     # almost full scan, cropped ~5% at top. new size = [588, 816]
#     elif (H, W) == (708, 1016):
#         return sequence[:, 70:-50, 100:-100]
#     # this one is quite whacky, likely an outlier, new size = [653, 865]
#     elif (H, W) == (768, 1040):
#         return sequence[:, 50:-65, 15:-160]
#     else:
#         raise ValueError(f"Unexpected image shape: {sequence.shape}")


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
        # Store cone parameters for each processed file (one per file, not per frame)
        self.cone_parameters = {}

        # Create JIT-compiled cropping function
        self.crop_frame_jit = jit(self._crop_single_frame_with_params)
        self.crop_sequence_jit = jit(
            vmap(self._crop_single_frame_with_params, in_axes=(0, None))
        )

    def get_split(self, hdf5_file: str, sequence):
        for split, files in self.splits.items():
            if hdf5_file in files:
                return split
        raise UserWarning("Unknown split for file: " + hdf5_file)

    @staticmethod
    def _crop_single_frame_with_params(frame, cone_params_array):
        """JAX-optimized function to crop a single frame with cone parameters.

        Args:
            frame: JAX array of shape (H, W)
            cone_params_array: JAX array containing [crop_left, crop_right, crop_top, crop_bottom,
                              apex_x, left_padding, right_padding, top_padding, crop_height, crop_width]
        """
        (
            crop_left,
            crop_right,
            crop_top,
            crop_bottom,
            apex_x,
            left_padding,
            right_padding,
            top_padding,
            crop_height,
            crop_width,
        ) = cone_params_array

        # Convert to integers
        crop_left = jnp.int32(crop_left)
        crop_right = jnp.int32(crop_right)
        crop_top = jnp.int32(crop_top)
        crop_bottom = jnp.int32(crop_bottom)
        left_padding = jnp.int32(left_padding)
        right_padding = jnp.int32(right_padding)
        top_padding = jnp.int32(top_padding)
        crop_height = jnp.int32(crop_height)
        crop_width = jnp.int32(crop_width)

        # Handle negative crop_top by using max(0, crop_top)
        actual_crop_top = jnp.maximum(0, crop_top)

        # Use dynamic_slice for JAX-compatible cropping
        # dynamic_slice expects (start_indices, slice_sizes)
        start_indices = (actual_crop_top, crop_left)
        slice_sizes = (crop_height, crop_width)
        cropped = lax.dynamic_slice(frame, start_indices, slice_sizes)

        # Apply top padding if needed
        def apply_top_padding(cropped):
            top_pad = jnp.zeros((top_padding, cropped.shape[1]), dtype=cropped.dtype)
            return jnp.concatenate([top_pad, cropped], axis=0)

        cropped = lax.cond(top_padding > 0, apply_top_padding, lambda x: x, cropped)

        # Apply horizontal padding if needed
        def apply_left_padding(cropped):
            left_pad = jnp.zeros((cropped.shape[0], left_padding), dtype=cropped.dtype)
            return jnp.concatenate([left_pad, cropped], axis=1)

        def apply_right_padding(cropped):
            right_pad = jnp.zeros(
                (cropped.shape[0], right_padding), dtype=cropped.dtype
            )
            return jnp.concatenate([cropped, right_pad], axis=1)

        cropped = lax.cond(left_padding > 0, apply_left_padding, lambda x: x, cropped)

        cropped = lax.cond(right_padding > 0, apply_right_padding, lambda x: x, cropped)

        return cropped

    def prepare_cone_params_for_jax(self, cone_params):
        """Convert cone parameters to JAX-compatible format for GPU processing."""
        crop_left = cone_params["crop_left"]
        crop_right = cone_params["crop_right"]
        crop_top = cone_params["crop_top"]
        crop_bottom = cone_params["crop_bottom"]
        apex_x = cone_params["apex_x"]

        # Calculate padding parameters
        apex_x_in_crop = apex_x - crop_left
        original_width = crop_right - crop_left
        target_center_x = original_width / 2
        left_padding_needed = target_center_x - apex_x_in_crop

        left_padding = max(0, int(left_padding_needed))
        right_padding = max(0, int(-left_padding_needed))
        top_padding = max(0, -crop_top) if crop_top < 0 else 0

        # Calculate actual crop dimensions for dynamic_slice
        crop_height = crop_bottom - max(0, crop_top)
        crop_width = crop_right - crop_left

        # Return as JAX array for efficient GPU transfer
        return jnp.array(
            [
                crop_left,
                crop_right,
                crop_top,
                crop_bottom,
                apex_x,
                left_padding,
                right_padding,
                top_padding,
                crop_height,
                crop_width,
            ],
            dtype=jnp.float32,
        )

    def crop_sequence_with_cone_params_jax(self, sequence, cone_params):
        """JAX-optimized version of sequence cropping."""
        # Prepare cone parameters for JAX
        cone_params_array = self.prepare_cone_params_for_jax(cone_params)

        # Apply cropping to all frames using vectorized JAX operations
        cropped_sequence = self.crop_sequence_jit(sequence, cone_params_array)

        return cropped_sequence

    def __call__(self, avi_file):
        print(avi_file)
        hdf5_file = avi_file.stem + ".hdf5"
        sequence = jnp.array(load_video(avi_file))

        sequence = translate(sequence, self.range_from, self._process_range)

        # Fit cone parameters on the first frame only (keep on CPU since it uses OpenCV)
        try:
            first_frame_np = np.array(sequence[0])
            _, cone_params = fit_scan_cone(first_frame_np, return_params=True)

            # Apply the same cropping to all frames using JAX (GPU-optimized)
            sequence = self.crop_sequence_with_cone_params_jax(sequence, cone_params)

            # Store cone parameters for this file
            self.cone_parameters[hdf5_file] = cone_params

        except ValueError as e:
            print(f"Warning: Cone detection failed for {hdf5_file}: {e}")
            # If cone detection fails, use original sequence
            self.cone_parameters[hdf5_file] = None

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

    def save_cone_parameters(self, output_path):
        """Save cone parameters to a pickle file for use in measurement coordinate transformation."""
        cone_params_file = Path(output_path) / "cone_parameters.pkl"
        with open(cone_params_file, "wb") as f:
            pickle.dump(self.cone_parameters, f)
        print(f"Saved cone parameters to {cone_params_file}")


def transform_measurement_coordinates_with_cone_params(row, cone_params):
    """Transform measurement coordinates using cone parameters from fit_scan_cone.

    Args:
        row: A pandas Series containing measurement data with X1,X2,Y1,Y2 coordinates
        cone_params: Dictionary containing cone parameters from fit_scan_cone

    Returns:
        A new row with transformed coordinates, or None if cone_params is None
    """
    if cone_params is None:
        print(f"Warning: No cone parameters for file {row['HashedFileName']}")
        return None

    new_row = row.copy()

    # Apply cropping offset
    crop_left = cone_params["crop_left"]
    crop_top = cone_params["crop_top"]

    # Transform coordinates
    new_row["X1"] = row["X1"] - crop_left
    new_row["X2"] = row["X2"] - crop_left
    new_row["Y1"] = row["Y1"] - crop_top
    new_row["Y2"] = row["Y2"] - crop_top

    # Apply horizontal centering offset
    apex_x_in_crop = cone_params["apex_x"] - crop_left
    original_width = cone_params["crop_right"] - cone_params["crop_left"]
    target_center_x = original_width / 2
    left_padding_needed = target_center_x - apex_x_in_crop
    left_padding = max(0, int(left_padding_needed))

    # Adjust x coordinates for horizontal padding
    new_row["X1"] = new_row["X1"] + left_padding
    new_row["X2"] = new_row["X2"] + left_padding

    # Apply top padding offset if crop_top was negative
    if cone_params["crop_top"] < 0:
        top_padding = -cone_params["crop_top"]
        new_row["Y1"] = new_row["Y1"] + top_padding
        new_row["Y2"] = new_row["Y2"] + top_padding

    # Check if coordinates are within the final image bounds
    final_width = cone_params["new_width"]
    final_height = cone_params["new_height"]

    if (
        new_row["X1"] < 0
        or new_row["X2"] < 0
        or new_row["Y1"] < 0
        or new_row["Y2"] < 0
        or new_row["X1"] >= final_width
        or new_row["X2"] >= final_width
        or new_row["Y1"] >= final_height
        or new_row["Y2"] >= final_height
    ):
        print(
            f"Warning: Transformed coordinates out of bounds for file {row['HashedFileName']}"
        )

    return new_row


def convert_measurements_csv(source_csv, output_csv, cone_params_file=None):
    """Convert measurements CSV file with updated coordinates using cone parameters.

    Args:
        source_csv: Path to source CSV file
        output_csv: Path to output CSV file
        cone_params_file: Path to pickle file with cone parameters
    """
    try:
        # Read the CSV file
        df = pd.read_csv(source_csv)

        # Load cone parameters if available
        cone_parameters = {}
        if cone_params_file and Path(cone_params_file).exists():
            with open(cone_params_file, "rb") as f:
                cone_parameters = pickle.load(f)
        else:
            print(
                "Warning: No cone parameters file found. Measurements will not be transformed."
            )

        # Apply coordinate transformation and track skipped rows
        transformed_rows = []
        skipped_files = set()

        for _, row in df.iterrows():
            try:
                hdf5_file = row["HashedFileName"] + ".hdf5"

                # Get cone parameters for this file (no longer need frame-specific params)
                cone_params = cone_parameters.get(hdf5_file, None)

                transformed_row = transform_measurement_coordinates_with_cone_params(
                    row, cone_params
                )
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

        # Limit files if max_files is specified
        if args.max_files is not None:
            files_to_process = files_to_process[: args.max_files]

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

        # Save cone parameters for measurement coordinate transformation
        processor.save_cone_parameters(args.output)

        print("All image conversion tasks are completed.")

    # Convert measurements if requested (must be done after images to use cone parameters)
    if args.convert_measurements:
        source_path = Path(args.source)
        measurements_csv = source_path / "MeasurementsList.csv"
        if measurements_csv.exists():
            output_csv = Path(args.output) / "MeasurementsList.csv"
            cone_params_file = Path(args.output) / "cone_parameters.pkl"
            convert_measurements_csv(measurements_csv, output_csv, cone_params_file)
        else:
            print("Warning: MeasurementsList.csv not found in source directory")

    print("All tasks are completed.")
