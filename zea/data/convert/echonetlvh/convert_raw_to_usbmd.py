"""
Script to convert the EchoNet-LVH database to zea format.

Each video is cropped so that the scan cone is centered
without padding, such that it can be converted to polar domain.

This cropping requires first computing scan cone parameters
using `data/convert/echonetlvh/precompute_crop.py`, which
are then passed to this script.
"""

import os
import csv

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from zea import log
from zea.data import generate_zea_dataset
from zea.data.convert.echonet import H5Processor
from zea.display import cartesian_to_polar_matrix
from zea.tensor_ops import translate
from zea.data.convert.utils import load_avi
from zea.data.convert.echonetlvh.precompute_crop import precompute_cone_parameters
from zea.data.convert.utils import unzip


def overwrite_splits(source_dir):
    """
    Apply manual rejection labels to MeasurementsList.csv using manual_rejections.txt.
    
    Reads manual_rejections.txt located next to this module to obtain a list of HashedFileName values to mark as rejected. Opens MeasurementsList.csv in source_dir, sets the `split` field to "rejected" for any row whose HashedFileName appears in the rejection list, and atomically replaces the original CSV with the updated version. If manual_rejections.txt or MeasurementsList.csv is missing, the function logs a warning and returns without modifying files. The function asserts that exactly 278 rows were marked as rejected and logs the number of rejections written.
    
    Parameters:
        source_dir (str | Path): Directory containing MeasurementsList.csv to be updated.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rejection_path = os.path.join(current_dir, "manual_rejections.txt")
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
            assert rejection_counter == 278, (
                f"Expected 278 rejections, but applied only {rejection_counter}."
            )
    except FileNotFoundError:
        log.warning(f"{csv_path} not found, skipping rejections.")
        return
    temp_path.replace(csv_path)
    log.info(f"Overwritten {rejection_counter}/278 rejections to MeasurementsList.csv")
    return


def load_splits(source_dir):
    """
    Load dataset split mapping from MeasurementsList.csv in source_dir.
    
    Reads MeasurementsList.csv and returns a dictionary with keys "train", "val", "test", and "rejected", each mapping to a list of corresponding AVI filenames (each filename suffixed with ".avi").
    
    Returns:
        dict: Mapping from split name ("train", "val", "test", "rejected") to a list of `.avi` filenames.
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
    """Find AVI file in the specified batch directory or any batch if not specified."""
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
    Crop a single image frame according to precomputed cone parameters and apply zero padding to preserve apex centering.
    
    Parameters:
        frame (numpy.ndarray): 2D image array to crop.
        cone_params (dict): Mapping with keys `crop_left`, `crop_right`, `crop_top`, `crop_bottom`, and `apex_x` used to compute cropping and padding.
    
    Returns:
        numpy.ndarray: The cropped (and if needed, zero-padded) frame with apex horizontally centered within the output.
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
        top_pad = np.zeros((top_padding, cropped.shape[1]), dtype=cropped.dtype)
        cropped = np.concatenate([top_pad, cropped], axis=0)
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
            left_pad = np.zeros((cropped_height, left_padding), dtype=cropped.dtype)
            cropped = np.concatenate([left_pad, cropped], axis=1)

        if right_padding > 0:
            right_pad = np.zeros((cropped_height, right_padding), dtype=cropped.dtype)
            cropped = np.concatenate([cropped, right_pad], axis=1)

    return cropped


def crop_sequence_with_params(sequence, cone_params):
    """
    Crop each frame in a sequence using provided cone parameters and return the stacked result.
    
    Parameters:
        sequence (numpy.ndarray): Array of frames with shape (frames, height, width).
        cone_params (dict): Cropping and padding parameters (as produced by load_cone_parameters) used to crop each frame.
    
    Returns:
        numpy.ndarray: Cropped (and if needed, padded) sequence stacked along the first axis with shape (frames, final_height, final_width).
    """
    cropped_frames = [crop_frame_with_params(frame, cone_params) for frame in sequence]
    return np.stack(cropped_frames, axis=0)


class LVHProcessor(H5Processor):
    """Modified H5Processor for EchoNet-LVH dataset."""

    def __init__(self, *args, cone_params=None, **kwargs):
        """
        Initialize the LVHProcessor and store optional precomputed cone parameters.
        
        Parameters:
            cone_params (dict or None): Mapping from AVI filename to its cone parameter dictionary.
                If None, an empty mapping is used and no precomputed parameters will be applied.
        """
        super().__init__(*args, **kwargs)
        # Store the pre-computed cone parameters
        self.cone_parameters = cone_params or {}

    def get_split(self, avi_file: str, sequence):
        """
        Determine which dataset split the given AVI file belongs to.
        
        Parameters:
            avi_file (str): Path or filename of the AVI file to check.
        
        Returns:
            str: The split name — 'train', 'val', or 'test'.
        
        Raises:
            UserWarning: If the file is not found in any split.
        """
        # Extract base filename without extension
        filename = Path(avi_file).stem + ".avi"

        for split, files in self.splits.items():
            if filename in files:
                return split
        raise UserWarning("Unknown split for file: " + filename)

    def __call__(self, avi_file):
        """
        Process a single AVI file into a zea-format dataset.
        
        Loads the AVI, applies amplitude translation and optional precomputed cone cropping, converts each frame to polar coordinates, optionally saves a .npz with both polar and original sequences, and builds the zea dataset payload with scaled uint8 images.
        
        Parameters:
            avi_file (str or Path): Path to the AVI file to process.
        
        Returns:
            The created zea dataset representation for the processed AVI file (contains HDF5 output path, `image` and `image_sc` arrays, and related metadata).
        """
        avi_filename = Path(avi_file).stem + ".avi"
        sequence = np.array(load_avi(avi_file))

        sequence = translate(sequence, self.range_from, self._process_range)

        # Get pre-computed cone parameters for this file
        cone_params = self.cone_parameters.get(avi_filename)
        if cone_params is not None:
            # Apply pre-computed cropping parameters
            sequence = crop_sequence_with_params(sequence, cone_params)
        else:
            log.warning(f"No cone parameters for {avi_filename}, using original sequence")
        sequence = np.array(sequence)
        split = self.get_split(avi_file, sequence)
        out_h5 = self.path_out_h5 / split / (Path(avi_file).stem + ".hdf5")

        polar_im_set = []
        for im in sequence.astype(np.float32):
            polar_im_set.append(cartesian_to_polar_matrix(im))
        polar_im_set = np.stack(polar_im_set, axis=0)

        if self._to_numpy:
            out_npz = self.path_out / split / (Path(avi_file).stem + ".npz")
            out_npz.mkdir(parents=True, exist_ok=True)
            np.savez(out_npz, image=np.array(polar_im_set), image_sc=np.array(sequence))

        zea_dataset = {
            "path": out_h5,
            # store as uint8 for memory efficiency
            "image_sc": translate(np.array(sequence), self._process_range, (0, 255)).astype(
                np.uint8
            ),
            "probe_name": "generic",
            "description": "EchoNet-LVH dataset converted to zea format",
            "image": translate(np.array(polar_im_set), self._process_range, (0, 255)).astype(
                np.uint8
            ),
            "cast_to_float": False,
        }
        return generate_zea_dataset(**zea_dataset)


def transform_measurement_coordinates_with_cone_params(row, cone_params):
    """
    Transform measurement coordinates (X1, X2, Y1, Y2) using precomputed scan-cone parameters.
    
    Parameters:
        row (dict): Measurement row containing 'HashedFileName' and coordinate fields 'X1', 'X2', 'Y1', 'Y2'.
        cone_params (dict): Cone parameters produced by fit_scan_cone; required keys include
            'crop_left', 'crop_top', 'apex_x', 'crop_right', 'new_width', and 'new_height'.
    
    Returns:
        dict or None: The input row with transformed coordinate values converted to strings, or
        `None` if `cone_params` is `None`.
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
    """
    Transform measurement coordinates in a CSV using precomputed cone parameters and write the transformed rows to a new CSV.
    
    If a cone parameters CSV is provided and exists, its parameters are applied per-file when transforming measurement coordinates. Rows that cannot be transformed are skipped; if no rows are transformed the output CSV will contain only the original header. A summary of processed, converted, and skipped rows is logged.
    
    Parameters:
        source_csv (str or Path): Path to the input measurements CSV.
        output_csv (str or Path): Path where the converted CSV will be written.
        cone_params_csv (str or Path, optional): Path to a CSV containing cone parameters; if omitted or not found, measurements will not be transformed.
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


def convert_echonetlvh(args):
    # Check if unzip is needed
    """
    Coordinate conversion of the EchoNet-LVH dataset into zea-compatible outputs (HDF5/NPZ) and optional measurement CSV transformation.
    
    This function:
    - Ensures the source archive is available and applies manual rejections when enabled.
    - Ensures precomputed cone parameters exist (generating them if missing).
    - Converts image sequences to zea-format HDF5 files (and optional NPZ dumps) using precomputed cone parameters and dataset splits when `convert_images` is enabled.
    - Converts and writes a transformed MeasurementsList.csv using cone parameters when `convert_measurements` is enabled.
    
    Parameters:
        args: An object (typically parsed CLI/namespace) with required attributes:
            src (str or Path): Source path or archive root for EchoNet-LVH input.
            dst (str or Path): Destination directory for HDF5 outputs and converted CSV.
            dst_npz (str or Path): Destination directory for optional NPZ outputs.
            no_rejection (bool): If True, skip applying manual rejection updates to splits.
            convert_images (bool): If True, perform image sequence conversion.
            convert_measurements (bool): If True, perform measurement CSV conversion.
            batch (str or None): Optional batch subdirectory to restrict AVI discovery.
            max_files (int or None): Optional max number of files to process.
            no_hyperthreading (bool): If True, process files sequentially instead of using a process pool.
    
    Side effects:
        Creates/overwrites files under `dst` (HDF5, MeasurementsList.csv) and optionally `dst_npz` (NPZ).
        May call external helpers that read/write additional intermediate files (e.g., cone parameters).
    """
    src = unzip(args.src, "echonetlvh")

    # Overwrite the splits if manual rejections are provided
    if not args.no_rejection:
        overwrite_splits(args.src)

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
        processor = LVHProcessor(
            path_out_h5=args.dst,
            path_out=args.dst_npz,
            splits=splits,
            cone_params=cone_parameters,
        )

        log.info("Starting the conversion process.")

        if not args.no_hyperthreading:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(processor, file): file for file in files_to_process}
                for future in tqdm(as_completed(futures), total=len(files_to_process)):
                    try:
                        future.result()
                    except Exception as e:
                        log.error(f"Error processing file: {str(e)}")
        else:
            log.info("Converting without hyperthreading")
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