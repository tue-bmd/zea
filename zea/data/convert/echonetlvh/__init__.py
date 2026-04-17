"""
Script to convert the EchoNet-LVH database to zea format.

Each video is cropped so that the scan cone is centered
without padding, such that it can be converted to polar domain.

.. note::
    This cropping requires first computing scan cone parameters
    using :mod:`zea.data.convert.echonetlvh.precompute_crop`, which
    are then passed to this script.

For more information about the dataset, resort to the following links:

- The original dataset can be found at `this link <https://stanfordaimi.azurewebsites.net/datasets/5b7fcc28-579c-4285-8b72-e4238eac7bd1>`_.
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from tqdm import tqdm

from zea import log
from zea.data import generate_zea_dataset
from zea.data.convert.echonet import H5Processor
from zea.data.convert.echonetlvh.precompute_crop import precompute_cone_parameters
from zea.data.convert.utils import load_avi, unzip
from zea.display import cartesian_to_polar_matrix
from zea.func.tensor import translate


def overwrite_splits(source_dir, rejection_path=None):
    """
    Overwrite MeasurementsList.csv splits based on manual_rejections.txt or another
    txt file specifying which hashes to reject.

    Args:
        source_dir: Source directory containing MeasurementsList.csv and manual_rejections.txt
        rejection_path: Path to the rejection txt file. If None, defaults to ./manual_rejections.txt
    Returns:
        None
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if rejection_path is None:
        rejection_path = os.path.join(current_dir, "manual_rejections.txt")
        expected_num_rejections = 278
    else:
        # unknown number of rejections for custom rejection file.
        # NOTE: this is used for testing, where we want to use a dummy rejections file
        expected_num_rejections = -1
    try:
        with open(rejection_path) as f:
            rejected_hashes = [line.strip() for line in f]
    except FileNotFoundError:
        log.warning(f"{rejection_path} not found, skipping rejections.")
        return

    csv_path = Path(source_dir) / "MeasurementsList.csv"
    temp_path = Path(source_dir) / "MeasurementsList_temp.csv"
    try:
        rejection_counter = 0
        with (
            csv_path.open("r", newline="", encoding="utf-8") as infile,
            temp_path.open("w", encoding="utf-8", newline="") as outfile,
        ):
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["HashedFileName"] in rejected_hashes:
                    row["split"] = "rejected"
                    rejection_counter += 1
                writer.writerow(row)
            if expected_num_rejections != -1:
                assert rejection_counter == expected_num_rejections, (
                    f"Expected {expected_num_rejections} rejections, but applied only {rejection_counter}."
                )
    except FileNotFoundError:
        log.warning(f"{csv_path} not found, skipping rejections.")
        return
    temp_path.replace(csv_path)
    log.info(f"Overwritten {rejection_counter}/278 rejections to {csv_path}")
    return


def load_splits(source_dir):
    """
    Load splits from MeasurementsList.csv and return avi filenames

    Args:
        source_dir: Source directory containing MeasurementsList.csv
    Returns:
        Dictionary with keys 'train', 'val', 'test', 'rejected' and values as lists of avi filenames
    """
    csv_path = Path(source_dir) / "MeasurementsList.csv"
    splits = {"train": [], "val": [], "test": [], "rejected": []}
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        file_split_map = {}
        for row in reader:
            filename = row["HashedFileName"]
            split = row["split"]
            file_split_map.setdefault(filename, split)
        for filename, split in file_split_map.items():
            splits[split].append(filename + ".avi")
    return splits


def find_avi_file(source_dir, hashed_filename, batch=None):
    """
    Find AVI file in the specified batch directory or any batch if not specified.

    Args:
        source_dir: Source directory containing BatchX subdirectories
        hashed_filename: Hashed filename (with or without .avi extension)
        batch: Specific batch directory to search in (e.g., "Batch2"), or None to search all batches

    Returns:
        Path to the AVI file if found, else None
    """
    # If filename already has .avi extension, strip it
    if hashed_filename.endswith(".avi"):
        hashed_filename = hashed_filename[:-4]

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


def load_cone_parameters(csv_path):
    """
    Load cone parameters from CSV file into a dictionary.

    Args:
        csv_path: Path to the CSV file containing cone parameters

    Returns:
        Dictionary mapping avi_filename to cone parameters
    """
    cone_params = {}

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["status"] == "success":
                # Convert string values to appropriate types
                params = {}
                for key, value in row.items():
                    if key in ("avi_filename", "status"):
                        params[key] = value
                    elif key == "apex_above_image":
                        params[key] = value.lower() == "true"
                    elif value is not None and value != "":
                        params[key] = float(value)
                    else:
                        params[key] = None

                cone_params[row["avi_filename"]] = params

    return cone_params


def crop_frame_with_params(frame, cone_params):
    """
    Crop a single frame using predetermined cone parameters.

    Args:
        frame: Input frame as numpy array
        cone_params: Dictionary containing cropping parameters

    Returns:
        Cropped and padded frame
    """
    crop_left = int(cone_params["crop_left"])
    crop_right = int(cone_params["crop_right"])
    crop_top = int(cone_params["crop_top"])
    crop_bottom = int(cone_params["crop_bottom"])

    # Handle negative crop_top
    if crop_top < 0:
        cropped = frame[0:crop_bottom, crop_left:crop_right]
        # Add top padding
        top_padding = -crop_top
        top_pad = jnp.zeros((top_padding, cropped.shape[1]), dtype=cropped.dtype)
        cropped = jnp.concatenate([top_pad, cropped], axis=0)
    else:
        cropped = frame[crop_top:crop_bottom, crop_left:crop_right]

    # Apply horizontal centering
    apex_x_in_crop = cone_params["apex_x"] - crop_left
    cropped_height, cropped_width = cropped.shape
    target_center_x = cropped_width / 2
    left_padding_needed = target_center_x - apex_x_in_crop

    left_padding = max(0, int(left_padding_needed))
    right_padding = max(0, int(-left_padding_needed))

    if left_padding > 0 or right_padding > 0:
        if left_padding > 0:
            left_pad = jnp.zeros((cropped_height, left_padding), dtype=cropped.dtype)
            cropped = jnp.concatenate([left_pad, cropped], axis=1)

        if right_padding > 0:
            right_pad = jnp.zeros((cropped_height, right_padding), dtype=cropped.dtype)
            cropped = jnp.concatenate([cropped, right_pad], axis=1)

    return cropped


def crop_sequence_with_params(sequence, cone_params):
    """
    Apply cropping to a sequence of frames using predetermined parameters.

    Args:
        sequence: Input sequence as numpy array of shape (frames, height, width)
        cone_params: Dictionary containing cropping parameters

    Returns:
        Cropped and padded sequence
    """
    crop_sequence = vmap(lambda frame: crop_frame_with_params(frame, cone_params))
    return crop_sequence(sequence)


class LVHProcessor(H5Processor):
    """Modified H5Processor for EchoNet-LVH dataset."""

    def __init__(self, *args, cone_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the pre-computed cone parameters
        self.cart2pol_jit = jit(cartesian_to_polar_matrix)
        self.cart2pol_batched = vmap(
            (lambda matrix, angle: self.cart2pol_jit(matrix, angle=angle)), in_axes=(0, None)
        )  # map over sequence of images, keep the angle fixed since it's constant across a sequence
        self.cone_parameters = cone_params or {}

    def get_split(self, avi_file: str, sequence):
        """
        Get the split (train/val/test) for a given AVI file.

        Args:
            avi_file: Path to the AVI file
            sequence: Video sequence (unused)

        Returns:
            String indicating the split ('train', 'val', or 'test')
        """
        # Extract base filename without extension
        filename = Path(avi_file).stem + ".avi"

        for split, files in self.splits.items():
            if filename in files:
                return split
        raise UserWarning("Unknown split for file: " + filename)

    def __call__(self, avi_file):
        """Takes a single avi_file and generates a zea dataset

        Args:
            avi_file: String or path to avi_file to be processed

        Returns:
            zea dataset
        """

        avi_filename = Path(avi_file).stem + ".avi"
        sequence_np = load_avi(avi_file)
        sequence_processed = jnp.array(sequence_np)
        sequence_processed = translate(sequence_processed, self.range_from, self._process_range)
        # Get pre-computed cone parameters for this file
        cone_params = self.cone_parameters.get(avi_filename)
        if cone_params is not None:
            # Apply pre-computed cropping parameters
            sequence_processed = crop_sequence_with_params(sequence_processed, cone_params)
        else:
            raise UserWarning(f"No cone parameters for {avi_filename}")

        split = self.get_split(avi_file, sequence_processed)
        out_h5 = self.path_out_h5 / split / (Path(avi_file).stem + ".hdf5")

        angle = cone_params["opening_angle"] / 2  # angular field spans (-angle, +angle)
        polar_im_set = self.cart2pol_batched(sequence_processed, angle)
        sequence_processed = translate(sequence_processed, self._process_range, self.range_from)
        sequence_processed_uint8 = jnp.asarray(jnp.floor(sequence_processed + 0.5), dtype=jnp.uint8)
        del sequence_processed

        polar_im_set = translate(polar_im_set, self._process_range, (0, 255))
        polar_im_set_uint8 = jnp.asarray(jnp.floor(polar_im_set + 0.5), dtype=jnp.uint8)
        del polar_im_set

        if jnp.all(sequence_processed_uint8 == 0):
            raise ValueError(f"Processed sequence is all zeros for file {avi_file}")

        if jnp.all(polar_im_set_uint8 == 0):
            raise ValueError(f"Polar sequence is all zeros for file {avi_file}")

        zea_dataset = {
            "path": out_h5,
            "image_sc": sequence_processed_uint8,
            "probe_name": "generic",
            "description": "EchoNet-LVH dataset converted to zea format",
            "image": polar_im_set_uint8,
            "cast_to_float": False,
        }
        return generate_zea_dataset(**zea_dataset)


def transform_measurement_coordinates_with_cone_params(row, cone_params):
    """Transform measurement coordinates using cone parameters from fit_scan_cone.

    Args:
        row: A dict containing measurement data with X1,X2,Y1,Y2 coordinates
        cone_params: Dictionary containing cone parameters from fit_scan_cone

    Returns:
        A new row with transformed coordinates, or None if cone_params is None
    """
    if cone_params is None:
        log.warning(f"No cone parameters for file {row['HashedFileName']}")
        return None

    new_row = dict(row)

    # Apply cropping offset
    crop_left = cone_params["crop_left"]
    crop_top = cone_params["crop_top"]

    # Transform coordinates
    for k in ["X1", "X2", "Y1", "Y2"]:
        # Convert to float if not already
        new_row[k] = float(row[k]) - (crop_left if k.startswith("X") else crop_top)

    # Apply horizontal centering offset
    apex_x_in_crop = cone_params["apex_x"] - crop_left
    original_width = cone_params["crop_right"] - cone_params["crop_left"]
    target_center_x = original_width / 2
    left_padding_needed = target_center_x - apex_x_in_crop
    left_padding = max(0, int(left_padding_needed))

    # Adjust x coordinates for horizontal padding
    new_row["X1"] = new_row["X1"] + left_padding
    new_row["X2"] = new_row["X2"] + left_padding

    # Check if coordinates are within the final image bounds
    final_width = cone_params["new_width"]
    final_height = cone_params["new_height"]

    # Check if coordinates are out of bounds
    is_out_of_bounds = (
        new_row["X1"] < 0
        or new_row["X2"] < 0
        or new_row["Y1"] < 0
        or new_row["Y2"] < 0
        or new_row["X1"] >= final_width
        or new_row["X2"] >= final_width
        or new_row["Y1"] >= final_height
        or new_row["Y2"] >= final_height
    )

    if is_out_of_bounds:
        log.warning(f"Transformed coordinates out of bounds for file {row['HashedFileName']}")

    # Convert back to string if original was string
    for k in ["X1", "X2", "Y1", "Y2"]:
        new_row[k] = str(new_row[k])

    return new_row


def convert_measurements_csv(source_csv, output_csv, cone_params_csv=None):
    """Convert measurements CSV file with updated coordinates using cone parameters.

    Args:
        source_csv: Path to source CSV file
        output_csv: Path to output CSV file
        cone_params_csv: Path to CSV file with cone parameters
    """
    try:
        # Read the CSV file
        with open(source_csv, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            fieldnames = reader.fieldnames

        # Load cone parameters if available
        cone_parameters = {}
        if cone_params_csv and Path(cone_params_csv).exists():
            cone_parameters = load_cone_parameters(cone_params_csv)
        else:
            log.warning("No cone parameters file found. Measurements will not be transformed.")

        # Apply coordinate transformation and track skipped rows
        transformed_rows = []
        skipped_files = set()

        for row in rows:
            try:
                avi_filename = row["HashedFileName"] + ".avi"
                cone_params = cone_parameters.get(avi_filename, None)
                transformed_row = transform_measurement_coordinates_with_cone_params(
                    row, cone_params
                )
                if transformed_row is not None:
                    transformed_rows.append(transformed_row)
                else:
                    skipped_files.add(row["HashedFileName"])
            except Exception as e:
                log.error(f"Error processing row for file {row['HashedFileName']}: {str(e)}")
                skipped_files.add(row["HashedFileName"])

        # Save to new CSV file
        if transformed_rows:
            # Use keys from first row as fieldnames
            out_fieldnames = list(transformed_rows[0].keys())
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=out_fieldnames)
                writer.writeheader()
                writer.writerows(transformed_rows)
        else:
            # Write header only if no rows
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # Print summary
        log.info("Conversion Summary:")
        log.info(f"Total rows processed: {len(rows)}")
        log.info(f"Rows successfully converted: {len(transformed_rows)}")
        log.info(f"Rows skipped: {len(rows) - len(transformed_rows)}")
        if skipped_files:
            log.info("Skipped files:")
            for filename in sorted(skipped_files):
                log.info(f"  - {filename}")
        log.info(f"Converted measurements saved to {output_csv}")

    except Exception as e:
        log.error(f"Error processing CSV file: {str(e)}")
        raise


def _process_file_worker(avi_file, dst, splits, cone_parameters, range_from, process_range):
    """
    Function for a hyperthreading worker to process a single file.

    Args:
        avi_file: Path to the AVI file to process
        dst: Destination directory for output
        splits: Dictionary of splits
        cone_parameters: Dictionary of cone parameters
        range_from: Range from value for processing
        process_range: Process range value for processing
    Returns:
        Result of processing the file
    """

    # create a fresh processor inside the worker process
    proc = LVHProcessor(path_out_h5=dst, splits=splits, cone_params=cone_parameters)
    # if LVHProcessor needs range_from/_process_range set, set them here
    proc.range_from = range_from
    proc._process_range = process_range
    return proc(avi_file)


def convert_echonetlvh(args):
    """
    Conversion script for the EchoNet-LVH dataset.
    Unzips, overwrites splits if needed, precomputes cone parameters,
    and converts images and/or measurements to zea format and saves dataset.
    Is called with argparse arguments through zea/zea/data/convert/__main__.py

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    # Check if unzip is needed
    src = unzip(args.src, "echonetlvh")

    # Overwrite the splits if manual rejections are provided
    if not args.no_rejection:
        overwrite_splits(args.src, getattr(args, "rejection_path", None))

    # Check that cone parameters exist
    cone_params_csv = Path(args.dst) / "cone_parameters.csv"
    if not cone_params_csv.exists():
        precompute_cone_parameters(args)

    # If no specific conversion is requested, convert both
    if not (args.convert_measurements or args.convert_images):
        args.convert_measurements = True
        args.convert_images = True

    # Convert images if requested
    if args.convert_images:
        source_path = Path(src)
        splits = load_splits(source_path)

        # Load precomputed cone parameters
        cone_parameters = load_cone_parameters(cone_params_csv)
        log.info(f"Loaded cone parameters for {len(cone_parameters)} files")

        files_to_process = []
        for split_files in splits.values():
            for avi_filename in split_files:
                # Strip .avi if present
                base_filename = avi_filename[:-4] if avi_filename.endswith(".avi") else avi_filename
                avi_file = find_avi_file(src, base_filename, batch=args.batch)
                if avi_file:
                    files_to_process.append(avi_file)
                else:
                    log.warning(
                        f"Warning: Could not find AVI file for {base_filename} in batch "
                        f"{args.batch if args.batch else 'any'}"
                    )

        # List files that have already been processed
        files_done = []
        for _, _, filenames in os.walk(args.dst):
            for filename in filenames:
                if filename.endswith(".hdf5"):
                    files_done.append(filename.replace(".hdf5", ""))

        # Filter out already processed files
        files_to_process = [f for f in files_to_process if f.stem not in files_done]

        # Limit files if max_files is specified
        if args.max_files is not None:
            files_to_process = files_to_process[: args.max_files]
            log.info(f"Limited to processing {args.max_files} files due to max_files parameter")

        log.info(f"Files left to process: {len(files_to_process)}")

        # Initialize processor with splits and cone parameters
        processor = LVHProcessor(path_out_h5=args.dst, splits=splits, cone_params=cone_parameters)

        log.info("Starting the conversion process.")

        for file in tqdm(files_to_process):
            try:
                processor(file)
            except Exception as e:
                log.error(f"Error processing {file}: {str(e)}")

        log.info("All image conversion tasks are completed.")

    # Convert measurements if requested
    if args.convert_measurements:
        source_path = Path(src)
        measurements_csv = source_path / "MeasurementsList.csv"
        if measurements_csv.exists():
            output_csv = Path(args.dst) / "MeasurementsList.csv"
            convert_measurements_csv(measurements_csv, output_csv, cone_params_csv)
        else:
            log.warning("MeasurementsList.csv not found in source directory")

    log.info("All tasks are completed.")
