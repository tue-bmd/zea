"""
Script to precompute cone parameters for the EchoNet-LVH dataset.
This script should be run separately before the main conversion process.
"""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from zea import log
from zea.tools.fit_scan_cone import fit_and_crop_around_scan_cone


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


def load_first_frame(avi_file):
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
                first_frame = load_first_frame(avi_file)

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
