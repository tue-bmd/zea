"""
Script to convert the EchoNet-LVH database to zea format.

Each video is cropped so that the scan cone is centered
without padding, such that it can be converted to polar domain.

For more information about the dataset, resort to the following links:

- The original dataset can be found at `this link <https://stanfordaimi.azurewebsites.net/datasets/5b7fcc28-579c-4285-8b72-e4238eac7bd1>`_.
"""

import csv
import json
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import keras
from keras import ops
from tqdm import tqdm

from zea import log
from zea.backend import jit
from zea.data import generate_zea_dataset
from zea.data.convert.echonet import H5Processor
from zea.data.convert.utils import load_avi
from zea.display import cartesian_to_polar_matrix
from zea.func.tensor import translate, vmap
from zea.tools.fit_scan_cone import (
    _load_first_frame,
    crop_and_center_cone,
    fit_and_crop_around_scan_cone,
)


def load_splits(csv_path: str | Path):
    """
    Load splits from MeasurementsList.csv and return avi filenames

    Args:
        csv_path: Path to the MeasurementsList.csv file
    Returns:
        Dictionary with keys 'train', 'val', 'test', 'rejected' and values as lists of avi filenames
    """
    splits = {"train": [], "val": [], "test": [], "rejected": []}
    # Read CSV using built-in csv module
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        # Group by HashedFileName
        file_split_map = {}
        for row in reader:
            filename = row["HashedFileName"]
            split = row["split"]
            file_split_map.setdefault(filename, split)
        # Now, for each unique filename, add to the correct split
        for filename, split in file_split_map.items():
            splits[split].append(filename + ".avi")
    return splits


def find_avi_file(source_dir: Path, hashed_filename: str, batch=None):
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

    dirs = [source_dir / batch] if batch else source_dir.glob("Batch*")
    for batch_dir in dirs:
        avi_path = batch_dir / f"{hashed_filename}.avi"
        if avi_path.exists():
            return avi_path
    return None


def _find_avi_files(src: Path, splits: dict, batch):
    # Collect and de-extension all filenames across splits
    base_filenames = [
        avi_filename[:-4] if avi_filename.endswith(".avi") else avi_filename
        for split_files in splits.values()
        for avi_filename in split_files
    ]

    # Look up the AVI files in parallel (I/O-bound filesystem checks)
    files_to_process = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            lambda name: (name, find_avi_file(src, name, batch=batch)),
            base_filenames,
        )
        for base_filename, avi_file in tqdm(
            results, total=len(base_filenames), desc="Finding AVI files"
        ):
            if avi_file:
                files_to_process.append(avi_file)
            else:
                log.warning(
                    f"Warning: Could not find AVI file for {base_filename} in batch "
                    f"{batch if batch else 'any'}"
                )
    return files_to_process


def precompute_cone_parameters(
    source_path: Path, measurements_csv: str | Path, cone_params_csv: Path, batch, max_files, force
):
    """
    Precompute and save cone parameters for all AVI files.

    This function loads the first frame from each AVI file, applies fit_scan_cone
    to determine cropping parameters, and saves these parameters to a CSV file
    for later use during the actual data conversion.

    Args:
        source_path: Source directory containing EchoNet-LVH data
        measurements_csv: Path to the MeasurementsList.csv file
        cone_params_csv: Path to the output CSV file
        batch: Specific batch to process (e.g., "Batch2") or None for all
        max_files: Maximum number of files to process (or None for all)
        force: Whether to recompute parameters if they already exist
    Returns:
        Path to the CSV file containing cone parameters
    """

    # Check if parameters already exist
    if cone_params_csv.exists() and not force:
        log.warning(f"Parameters already exist at {cone_params_csv}. Use --force to recompute.")
        return cone_params_csv

    # Get list of files to process
    splits = load_splits(measurements_csv)
    files_to_process = _find_avi_files(source_path, splits, batch)

    # Limit files if max_files is specified
    if max_files is not None:
        files_to_process = files_to_process[:max_files]
        log.info(f"Limited to processing {max_files} files due to max_files parameter")

    log.info(f"Computing cone parameters for {len(files_to_process)} files")

    # Dictionary to store parameters for each file
    all_cone_params = {}

    # CSV field names - only the essential parameters needed for cropping
    fieldnames = [
        "avi_filename",
        "crop_left",
        "crop_right",
        "crop_top",
        "crop_bottom",
        "apex_x",
        "new_width",
        "new_height",
        "opening_angle",
        "status",
    ]

    # Open CSV file for writing
    with open(cone_params_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each file
        for avi_file, avi_filename in tqdm(files_to_process, desc="Computing cone parameters"):
            try:
                # Load only the first frame of video using OpenCV directly
                first_frame = _load_first_frame(avi_file)

                # Detect cone parameters
                _, full_cone_params = fit_and_crop_around_scan_cone(first_frame, return_params=True)

                if (
                    full_cone_params["crop_left"] < 0
                    or full_cone_params["crop_right"] > first_frame.shape[1]
                ):
                    raise ValueError(
                        "Computed crop exceeds frame dimensions, meaning that either cone detection"
                        "failed, due to e.g. DICOM artifacts present in the frame, or the full scan"
                        "cone is not visible in the frame."
                    )

                # Extract only the essential parameters
                essential_params = {
                    "avi_filename": avi_filename,
                    "crop_left": full_cone_params["crop_left"],
                    "crop_right": full_cone_params["crop_right"],
                    "crop_top": full_cone_params["crop_top"],
                    "crop_bottom": full_cone_params["crop_bottom"],
                    "apex_x": full_cone_params["apex_x"],
                    "new_width": full_cone_params["new_width"],
                    "new_height": full_cone_params["new_height"],
                    "opening_angle": full_cone_params["opening_angle"],
                    "status": "success",
                }

                # Save to output CSV
                writer.writerow(essential_params)

                # Store in dictionary
                all_cone_params[avi_filename] = essential_params

            except Exception as e:
                log.error(f"Error processing {avi_file}: {str(e)}")

                # Write failure record
                failure_record = {
                    "avi_filename": avi_filename,
                    "status": f"error: {str(e)}",
                }

                # Fill missing fields with None
                for field in fieldnames:
                    if field not in failure_record:
                        failure_record[field] = None

                writer.writerow(failure_record)

    # Also save as JSON for easier programmatic access
    cone_params_json = cone_params_csv.with_suffix(".json")
    with open(cone_params_json, "w", encoding="utf-8") as jsonfile:
        json.dump(all_cone_params, jsonfile)

    log.info(f"Cone parameters saved to {cone_params_csv} and {cone_params_json}")
    return cone_params_csv


def overwrite_splits(csv_path, rejection_path=None):
    """
    Overwrite splits in a MeasurementsList.csv based on manual_rejections.txt
    or another txt file specifying which hashes to reject.

    Args:
        csv_path: Path to the MeasurementsList.csv to update in place
        rejection_path: Path to the rejection txt file. If None, defaults to ./manual_rejections.txt
    Returns:
        None
    """
    csv_path = Path(csv_path)
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

    # Write to a temp dir on the same filesystem so the final replace is atomic.
    with tempfile.TemporaryDirectory(dir=csv_path.parent) as tmp_dir:
        temp_path = Path(tmp_dir) / "MeasurementsList_temp.csv"
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
        temp_path.replace(csv_path)
    log.info(f"Applied {rejection_counter} rejections to {csv_path}")


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


def crop_sequence_with_params(sequence, cone_params):
    """
    Apply cropping to a sequence of frames using predetermined parameters.

    Args:
        sequence: Input sequence as numpy array of shape (frames, height, width)
        cone_params: Dictionary containing cropping parameters

    Returns:
        Cropped and padded sequence
    """
    crop_sequence = vmap(lambda frame: crop_and_center_cone(frame, cone_params, backend=ops))
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

    def get_split(self, avi_file: Path, sequence):
        """
        Get the split (train/val/test) for a given AVI file.

        Args:
            avi_file: Path to the AVI file
            sequence: Video sequence (unused)

        Returns:
            String indicating the split ('train', 'val', or 'test')
        """
        # Extract base filename without extension
        filename = avi_file.name

        for split, files in self.splits.items():
            if filename in files:
                return split
        raise UserWarning("Unknown split for file: " + filename)

    def __call__(self, avi_file: Path):
        """Takes a single avi_file and generates a zea dataset

        Args:
            avi_file: Path to avi_file to be processed

        Returns:
            zea dataset
        """
        avi_file = avi_file.with_suffix(".avi")
        sequence_np = load_avi(avi_file)
        sequence_processed = ops.convert_to_numpy(sequence_np)
        sequence_processed = translate(sequence_processed, self.range_from, self._process_range)
        # Get pre-computed cone parameters for this file
        cone_params = self.cone_parameters.get(avi_file.name)
        if cone_params is not None:
            # Apply pre-computed cropping parameters
            sequence_processed = crop_sequence_with_params(sequence_processed, cone_params)
        else:
            raise UserWarning(f"No cone parameters for {avi_file.name}")

        split = self.get_split(avi_file, sequence_processed)
        out_h5 = self.path_out_h5 / split / avi_file.with_suffix(".hdf5")

        angle = cone_params["opening_angle"] / 2  # angular field spans (-angle, +angle)
        polar_im_set = self.cart2pol_batched(sequence_processed, angle)
        sequence_processed = translate(sequence_processed, self._process_range, self.range_from)
        sequence_processed_uint8 = ops.cast(ops.floor(sequence_processed + 0.5), "uint8")
        del sequence_processed

        polar_im_set = translate(polar_im_set, self._process_range, (0, 255))
        polar_im_set_uint8 = ops.cast(ops.floor(polar_im_set + 0.5), "uint8")
        del polar_im_set

        if ops.all(sequence_processed_uint8 == 0):
            raise ValueError(f"Processed sequence is all zeros for file {avi_file}")

        if ops.all(polar_im_set_uint8 == 0):
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


def transform_measurements_csv(csv_path, cone_params_csv=None):
    """Update a measurements CSV file in place with coordinates transformed using cone parameters.

    Args:
        csv_path: Path to the CSV file to transform in place
        cone_params_csv: Path to CSV file with cone parameters
    """
    try:
        # Read the CSV file
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
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

        # Save back to the CSV file
        if transformed_rows:
            # Use keys from first row as fieldnames
            out_fieldnames = list(transformed_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=out_fieldnames)
                writer.writeheader()
                writer.writerows(transformed_rows)
        else:
            # Write header only if no rows
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
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
        log.info(f"Converted measurements saved to {csv_path}")

    except Exception as e:
        log.error(f"Error processing CSV file: {str(e)}")
        raise


def unzip(src: Path, dst: Path) -> Path:
    assert src.exists(), f"Source path {src} does not exist."

    log.info(f"Unzipping {src} to {dst}...")
    with zipfile.ZipFile(src, "r") as zip_ref:
        zip_ref.extractall(dst)
    log.info("Unzipping completed.")
    return dst


def convert_echonetlvh(
    src: Path,
    dst: Path,
    no_rejection,
    rejection_path,
    batch,
    convert_measurements,
    convert_images,
    max_files,
    force,
):
    """
    Conversion script for the EchoNet-LVH dataset.
    Unzips, overwrites splits if needed, precomputes cone parameters,
    and converts images and/or measurements to zea format and saves dataset.
    Is called with argparse arguments through zea/zea/data/convert/__main__.py
    """

    if keras.backend.backend() != "jax":
        log.warning("We recommend using jax for speed in the EchoNet-LVH conversion.")

    # Check if unzip is needed
    if src.suffix == ".zip":
        tmp_dir = dst / "unzipped_original_files"
        tmp_dir.mkdir()
        src = unzip(src, tmp_dir, "echonetlvh")

    # Check the required files exist
    for folder in ["Batch1", "Batch2", "Batch3", "Batch4"]:
        assert (src / folder).exists(), f"Missing {folder} folder in {src}."
    assert (src / "MeasurementsList.csv").exists(), f"Missing MeasurementsList.csv in {src}."
    log.info(f"Found Batch1, Batch2, Batch3, Batch4 and MeasurementsList.csv in {src}.")

    # Copy MeasurementsList.csv to dst
    measurements_csv = dst / "MeasurementsList.csv"
    shutil.copy(src / "MeasurementsList.csv", measurements_csv)

    if not no_rejection:
        overwrite_splits(measurements_csv, rejection_path)

    # Precompute cone parameters if needed
    cone_params_csv = dst / "cone_parameters.csv"
    precompute_cone_parameters(measurements_csv, cone_params_csv, batch, max_files, force)

    # If no specific conversion is requested, convert both
    if not (convert_measurements or convert_images):
        convert_measurements = True
        convert_images = True

    # Convert images if requested
    if convert_images:
        splits = load_splits(measurements_csv)

        # Load precomputed cone parameters
        cone_parameters = load_cone_parameters(cone_params_csv)
        log.info(f"Loaded cone parameters for {len(cone_parameters)} files")

        files_to_process = _find_avi_files(src, splits, batch)

        # List files that have already been processed (set for O(1) membership)
        files_done = {
            filename.removesuffix(".hdf5")
            for _, _, filenames in os.walk(dst)
            for filename in filenames
            if filename.endswith(".hdf5")
        }

        # Filter out already processed files
        files_to_process = [f for f in files_to_process if f.stem not in files_done]

        # Limit files if max_files is specified
        if max_files is not None:
            files_to_process = files_to_process[:max_files]
            log.info(f"Limited to processing {max_files} files due to max_files parameter")

        log.info(f"Files left to process: {len(files_to_process)}")

        # Initialize processor with splits and cone parameters
        processor = LVHProcessor(path_out_h5=dst, splits=splits, cone_params=cone_parameters)

        log.info("Starting the conversion process.")

        for file in tqdm(files_to_process):
            try:
                processor(file)
            except Exception as e:
                log.error(f"Error processing {file}: {str(e)}")

        log.info("All image conversion tasks are completed.")

    # Convert measurements if requested
    if convert_measurements:
        transform_measurements_csv(measurements_csv, cone_params_csv)

    log.info("All tasks are completed.")
