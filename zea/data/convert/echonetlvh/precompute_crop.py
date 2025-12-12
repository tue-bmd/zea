"""
Script to precompute cone parameters for the EchoNet-LVH dataset.
This script should be run separately before the main conversion process.
"""

import csv
import json
from pathlib import Path

from tqdm import tqdm

from zea import log
from zea.tools.fit_scan_cone import fit_and_crop_around_scan_cone


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


def precompute_cone_parameters(args):
    """
    Precompute and save cone parameters for all AVI files.

    This function loads the first frame from each AVI file, applies fit_scan_cone
    to determine cropping parameters, and saves these parameters to a CSV file
    for later use during the actual data conversion.

    Args:
        args: Argument parser namespace with the following attributes:
            src: Source directory containing EchoNet-LVH data
            dst: Destination directory to save cone parameters
            batch: Specific batch to process (e.g., "Batch2") or None for all
            max_files: Maximum number of files to process (or None for all)
            force: Whether to recompute parameters if they already exist
    Returns:
        Path to the CSV file containing cone parameters
    """

    source_path = Path(args.src)
    output_path = Path(args.dst)
    output_path.mkdir(parents=True, exist_ok=True)

    # Output file for cone parameters
    cone_params_csv = output_path / "cone_parameters.csv"
    cone_params_json = output_path / "cone_parameters.json"

    # Check if parameters already exist
    if cone_params_csv.exists() and not args.force:
        log.warning(f"Parameters already exist at {cone_params_csv}. Use --force to recompute.")
        return cone_params_csv

    # Get list of files to process
    splits = load_splits(source_path)

    files_to_process = []
    for split_files in splits.values():
        for avi_filename in split_files:
            # Strip .avi if present
            base_filename = avi_filename[:-4] if avi_filename.endswith(".avi") else avi_filename
            avi_file = find_avi_file(args.src, base_filename, batch=args.batch)
            if avi_file:
                files_to_process.append((avi_file, avi_filename))
            else:
                log.warning(
                    f"Could not find AVI file for {base_filename} in batch "
                    f"{args.batch if args.batch else 'any'}"
                )

    # Limit files if max_files is specified
    if args.max_files is not None:
        files_to_process = files_to_process[: args.max_files]
        log.info(f"Limited to processing {args.max_files} files due to max_files parameter")

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
    with open(cone_params_json, "w", encoding="utf-8") as jsonfile:
        json.dump(all_cone_params, jsonfile)

    log.info(f"Cone parameters saved to {cone_params_csv} and {cone_params_json}")
    return cone_params_csv
