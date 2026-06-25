"""
Script to convert the EchoNet-LVH database to zea format.

Each video is cropped so that the scan cone is centered
without padding, such that it can be converted to polar domain.

For more information about the dataset, resort to the following links:

- The original dataset can be found at `this link <https://stanfordaimi.azurewebsites.net/datasets/5b7fcc28-579c-4285-8b72-e4238eac7bd1>`_.
"""

import csv
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path

import keras
import numpy as np
from keras import ops
from tqdm import tqdm

from zea import File, log
from zea.backend import jit
from zea.data.convert.utils import load_avi, unzip
from zea.display import cartesian_to_polar_matrix, polar_to_cartesian_matrix
from zea.func.tensor import vmap
from zea.tools.fit_scan_cone import (
    _load_first_frame,
    crop_and_center_cone,
    detect_cone_parameters,
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


def load_shapes(csv_path: str | Path):
    """
    Load shapes from MeasurementsList.csv and return avi filenames

    Args:
        csv_path: Path to the MeasurementsList.csv file

    Returns: dictionary with the filename as key and the shape as value
    """
    shapes = {}
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row["HashedFileName"]
            height = int(row["Height"])
            width = int(row["Width"])
            shape = (height, width)
            if filename not in shapes:
                shapes[filename] = shape
            else:
                assert shapes[filename] == shape, (
                    f"MeasurementsList.csv has multiple entries for {filename}, "
                    "and the shapes are different"
                )
    return shapes


def find_avi_file(source_dir: Path, hashed_filename: str):
    """
    Find AVI file in the source EchoNet-LVH dataset.

    Args:
        source_dir: Source directory containing BatchX subdirectories
        hashed_filename: Hashed filename (with or without .avi extension)

    Returns:
        Path to the AVI file if found, else None
    """
    # If filename already has .avi extension, strip it
    if hashed_filename.endswith(".avi"):
        hashed_filename = hashed_filename[:-4]

    for batch_dir in source_dir.glob("Batch*"):
        avi_path = batch_dir / f"{hashed_filename}.avi"
        if avi_path.exists():
            return avi_path
    raise FileNotFoundError(f"Could not find AVI file for {hashed_filename}")


def _find_avi_files(src: Path, splits: dict):
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
            lambda name: find_avi_file(src, name),
            base_filenames,
        )
        for avi_file in tqdm(results, total=len(base_filenames), desc="Finding AVI files"):
            files_to_process.append(avi_file)
    return files_to_process


def _compute_cone_params_for_file(avi_file, fieldnames):
    """Compute cone parameters for a single AVI file.

    Pure worker function with no shared state, safe to run in a thread pool.
    Returns a row dict (with ``status`` either ``"success"`` or ``"error: ..."``)
    matching ``fieldnames``.
    """
    try:
        # Load only the first frame of video using OpenCV directly
        first_frame = _load_first_frame(avi_file)

        # Detect cone parameters
        full_cone_params = detect_cone_parameters(first_frame, image_range=(0, 255))

        if (
            full_cone_params["crop_left"] < 0
            or full_cone_params["crop_right"] > first_frame.shape[1]
        ):
            raise ValueError(
                "Computed crop exceeds frame dimensions, meaning that either cone "
                "detection failed, due to e.g. DICOM artifacts present in the frame, "
                "or the full scan cone is not visible in the frame."
            )

        # Extract only the essential parameters
        return {
            "avi_filename": avi_file.name,
            "crop_left": full_cone_params["crop_left"],
            "crop_right": full_cone_params["crop_right"],
            "crop_top": full_cone_params["crop_top"],
            "crop_bottom": full_cone_params["crop_bottom"],
            "apex_x": full_cone_params["apex_x"],
            "apex_y": full_cone_params["apex_y"],
            "circle_radius": full_cone_params["circle_radius"],
            "left_slope": full_cone_params["left_slope"],
            "right_slope": full_cone_params["right_slope"],
            "new_width": full_cone_params["new_width"],
            "new_height": full_cone_params["new_height"],
            "opening_angle": full_cone_params["opening_angle"],
            "status": "success",
        }

    except Exception as e:
        log.error(f"Processing {avi_file} failed: {str(e)}")

        # Build failure record, filling missing fields with None
        failure_record = {
            "avi_filename": avi_file.name,
            "status": f"error: {str(e)}",
        }
        for field in fieldnames:
            failure_record.setdefault(field, None)
        return failure_record


def precompute_cone_parameters(
    source_path: Path,
    measurements_csv: str | Path,
    cone_params_csv: Path,
    max_files,
    max_workers: int = 8,
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
        max_files: Maximum number of files to process (or None for all)
        max_workers: Number of worker threads used to process files in parallel

    Returns:
        Path to the CSV file containing cone parameters
    """

    # Get list of files to process
    splits = load_splits(measurements_csv)
    files_to_process = _find_avi_files(source_path, splits)

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
        "apex_y",
        "circle_radius",
        "left_slope",
        "right_slope",
        "new_width",
        "new_height",
        "opening_angle",
        "status",
    ]

    # Open CSV file for writing. Files are processed in parallel worker threads
    # (OpenCV / NumPy release the GIL), but the csv writer and the shared dict are
    # only ever touched from this main thread.
    with open(cone_params_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_cone_params_for_file, avi_file, fieldnames): avi_file
                for avi_file in files_to_process
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Computing cone parameters",
            ):
                result = future.result()

                # Save to output CSV
                writer.writerow(result)

                # Store successful results in dictionary
                if result["status"] == "success":
                    all_cone_params[result["avi_filename"]] = result

    # Also save as JSON for easier programmatic access
    cone_params_json = cone_params_csv.with_suffix(".json")
    with open(cone_params_json, "w", encoding="utf-8") as jsonfile:
        json.dump(all_cone_params, jsonfile)

    log.info(f"Cone parameters saved to {cone_params_csv} and {cone_params_json}")
    return cone_params_csv


def overwrite_splits(csv_path: Path, rejection_path=None):
    """
    Overwrite splits in a MeasurementsList.csv based on manual_rejections.txt
    or another txt file specifying which hashes to reject.

    Args:
        csv_path: Path to the MeasurementsList.csv to update in place
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

    if not Path(rejection_path).exists():
        log.warning(f"{rejection_path} not found, skipping rejections.")
        return

    with open(rejection_path) as f:
        rejected_hashes = [line.strip() for line in f]

    # Write to a temp dir on the same filesystem so the final replace is atomic.
    with tempfile.TemporaryDirectory(dir=csv_path.parent) as tmp_dir:
        temp_path = Path(tmp_dir) / "MeasurementsList_temp.csv"
        rejection_counter = 0
        with (
            csv_path.open("r", newline="", encoding="utf-8") as infile,
            temp_path.open("w", encoding="utf-8", newline="") as outfile,
        ):
            reader = csv.DictReader(infile)
            assert reader.fieldnames is not None, "CSV file has no header row"
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

    Only loads the rows with status "success".

    Args:
        csv_path: Path to the CSV file containing cone parameters

    Returns:
        Dictionary mapping avi_filename to cone parameters
    """
    cone_params = {}

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["status"] != "success":
                continue

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


class LVHProcessor:
    """Processor for EchoNet-LVH dataset."""

    def __init__(
        self,
        path_out_h5: str | Path,
        splits: dict,
        cone_params: dict,
        polar_shape=(600, 600),
        frame_bucket: int = 128,
    ):
        self.path_out_h5 = Path(path_out_h5)
        self.splits = splits
        # Flatten to a filename -> split lookup so get_split is O(1) instead of a
        # linear scan over every split list on each of the (many) files.
        self.split_by_filename = {
            filename: split for split, files in (splits or {}).items() for filename in files
        }
        self.cone_parameters = cone_params or {}
        # Clip lengths vary per file, which is the leading (batch) dim of every GPU
        # op here. Padding it up to a multiple of `frame_bucket` collapses the many
        # distinct shapes into a handful, so XLA compiles each kernel only a few
        # times instead of once per unique frame count.
        self.frame_bucket = frame_bucket

        self.cart2pol_batched = jit(
            vmap(
                lambda matrix, tip_x, tip_y, r_max, theta_min, theta_max: cartesian_to_polar_matrix(
                    matrix,
                    tip=(tip_x, tip_y),
                    r_max=r_max,
                    theta_range=(theta_min, theta_max),
                    polar_shape=polar_shape,  # fixed
                ),
                in_axes=(0, None, None, None, None, None),
            )
        )

    def get_split(self, avi_file: Path):
        """Get the split (train/val/test) for a given AVI file."""
        assert self.splits is not None, "splits not loaded; call load_splits() first"
        split = self.split_by_filename.get(avi_file.name)
        if split is None:
            raise UserWarning("Unknown split for file: " + avi_file.name)
        return split

    @staticmethod
    def scan_convert(image_polar, cone_params, cartesian_shape, order=1):
        """
        Scan convert the 'image_polar' to cartesian coordinates to exactly
        match the cropped original (i.e. the 'image' in the file.), using the cone parameters.
        """
        # Match crop_and_center_cone, which crops with int()-truncated boundaries.
        crop_left = int(cone_params["crop_left"])
        crop_right = int(cone_params["crop_right"])
        crop_top = int(cone_params["crop_top"])
        apex_x_in_crop = cone_params["apex_x"] - crop_left
        cropped_width = crop_right - crop_left
        left_padding = max(0, int(cropped_width / 2 - apex_x_in_crop))
        tip_x = apex_x_in_crop + left_padding
        tip_y = cone_params["apex_y"] - crop_top
        tip = (tip_x, tip_y)

        r_max = cone_params["circle_radius"]

        theta_max = -math.atan(cone_params["right_slope"])
        theta_min = -math.atan(cone_params["left_slope"])
        theta_range = (theta_min, theta_max)

        return polar_to_cartesian_matrix(
            image_polar, cartesian_shape, tip, r_max, theta_range, order=order
        )

    def load(self, avi_file: Path):
        """Stage 1 (I/O + host preprocessing, thread-safe): read+decode the AVI, fetch
        cone params, frame-pad, and build the cropped cartesian view (image_sc).

        Runs no GPU/JAX work, so it is safe to call from worker threads. Returns a
        payload dict consumed by :meth:`compute`, or raises on missing params or an
        all-zero cropped sequence.
        """
        avi_file = avi_file.with_suffix(".avi")

        # Get pre-computed cone parameters for this file
        cone_params = self.cone_parameters.get(avi_file.name)
        if cone_params is None:
            raise UserWarning(f"No cone parameters for {avi_file.name}")

        sequence_np = load_avi(avi_file)
        out_h5 = self.path_out_h5 / self.get_split(avi_file) / (avi_file.stem + ".hdf5")

        # Pad the frame (leading/batch) dimension up to a multiple of frame_bucket
        n_frames = sequence_np.shape[0]
        padded_len = math.ceil(n_frames / self.frame_bucket) * self.frame_bucket
        if padded_len != n_frames:
            sequence_np = np.pad(sequence_np, [[0, padded_len - n_frames], [0, 0], [0, 0]])

        image_sc_np = crop_and_center_cone(sequence_np[:n_frames], cone_params)
        if not image_sc_np.any():
            raise ValueError(f"Processed sequence is all zeros for file {avi_file}")

        return {
            "avi_file": avi_file,
            "cone_params": cone_params,
            "sequence_np": sequence_np,
            "image_sc_np": image_sc_np,
            "n_frames": n_frames,
            "out_h5": out_h5,
        }

    def compute(self, payload: dict):
        """Stage 2 (GPU, main thread only): polar conversion.

        Takes a payload from :meth:`load` and returns the host-side arrays and
        metadata for :meth:`save`. Keep this on the main thread: there is a single
        device and concurrent tracing is not safe.
        """
        avi_file = payload["avi_file"]
        cone_params = payload["cone_params"]
        n_frames = payload["n_frames"]
        image_sc_np = payload["image_sc_np"]
        # Already padded to a multiple of frame_bucket on the host in :meth:`load`, so
        # every device op below sees only a handful of clip-length shapes.
        sequence_processed = ops.cast(ops.convert_to_tensor(payload["sequence_np"]), "float32")

        # Polar conversion runs on the uncropped frame using apex coordinates in
        # original-image space; theta_min/theta_max come from the fitted slopes
        # (polar +theta lands on the left after the 90° rotation in
        # cartesian_to_polar_matrix, so right_slope → theta_min).
        polar_im_set = self.cart2pol_batched(
            sequence_processed,
            ops.convert_to_tensor(cone_params["apex_x"]),
            ops.convert_to_tensor(cone_params["apex_y"]),
            ops.convert_to_tensor(cone_params["circle_radius"]),
            ops.convert_to_tensor(-math.atan(cone_params["right_slope"])),
            ops.convert_to_tensor(-math.atan(cone_params["left_slope"])),
        )

        polar_im_set = ops.cast(ops.floor(polar_im_set + 0.5), "uint8")

        # Drop the padding frames on-device, but do NOT materialise here: the
        # device->host copy is the only blocking step, so we defer it to the saver
        # thread (see :meth:`save`). That lets the main loop dispatch the next
        # file's GPU work immediately instead of parking on the transfer.
        polar_im_set = polar_im_set[:n_frames]

        # TODO: would be cool if we could store all the information of
        # 'MeasurementsList.csv' and 'cone_parameters.csv' in the metadata
        metadata = {
            "annotations": {"anatomy": "heart", "view": "PLAX"},
            "subject": {"id": avi_file.name, "type": "human"},
        }
        return payload["out_h5"], image_sc_np, polar_im_set, metadata

    @staticmethod
    def save(out_h5: Path, image_sc_np, polar, metadata: dict):
        """Stage 3 (I/O, thread-safe): write the zea HDF5 file.

        ``polar`` may still be an unmaterialised device array from :meth:`compute`;
        the blocking device->host copy happens here, off the main thread, so it
        overlaps with the next file's GPU compute. No tracing is involved, so
        materialising it from a worker thread is safe.
        """
        polar_np = np.asarray(polar)
        if not polar_np.any():
            raise ValueError(f"Polar sequence is all zeros for file {out_h5}")
        File.create(
            out_h5,
            data={
                "image": {"values": image_sc_np},
                "image_polar": {"values": polar_np, "unit": "pixels"},
            },
            metadata=metadata,
            description="EchoNet-LVH dataset converted to zea format",
            warn_missing_optional_fields=False,
        )

    def __call__(self, avi_file: Path):
        """Takes a single avi_file and generates a zea dataset.

        Sequential convenience wrapper around :meth:`load`, :meth:`compute` and
        :meth:`save`; the parallel pipeline drives those stages directly.

        Args:
            avi_file: Path to avi_file to be processed
        """
        self.save(*self.compute(self.load(avi_file)))

    def run(
        self,
        files,
        load_workers: int = 8,
        save_workers: int = 2,
        max_pending_saves: int | None = None,
    ):
        """Run over ``files`` as an overlapped load -> compute -> save
        pipeline so the GPU is not stalled on disk I/O.

        Loads (decode) run on a thread pool, GPU compute stays on the main thread,
        and writes go to a small saver pool. A single bad file is logged and
        skipped rather than aborting the whole (multi-hour) run.

        The in-flight save queue is bounded (``max_pending_saves``): each pending
        save pins a decoded volume in memory, so without a cap a transient write
        slowdown would let memory grow until the process is OOM-killed. h5py
        serialises its writes through a global lock, so a couple of save workers is
        enough to overlap the write with GPU compute; more just adds memory pressure.
        """
        if max_pending_saves is None:
            max_pending_saves = max(2 * save_workers, 4)

        prefetch = int(load_workers * 2)

        failures = 0

        def drain_saves(save_futures, block):
            """Reap finished save futures, logging (not raising) per-file errors so
            one failed write does not tear down the run. With ``block`` wait for at
            least one to finish; otherwise only reap those already done."""
            nonlocal failures
            if not save_futures:
                return
            if block:
                done, _ = wait(save_futures, return_when=FIRST_COMPLETED)
            else:
                done = [f for f in save_futures if f.done()]
            for future in done:
                out_h5 = save_futures.pop(future)
                try:
                    future.result()
                except Exception as e:
                    failures += 1
                    log.error(f"Saving {out_h5} failed: {e}")

        with (
            ThreadPoolExecutor(max_workers=load_workers) as loaders,
            ThreadPoolExecutor(max_workers=save_workers) as savers,
        ):
            save_futures = {}

            for file, load_future in tqdm(
                _bounded_map(loaders, self.load, files, prefetch), total=len(files)
            ):
                try:
                    payload = load_future.result()  # surfaces load errors
                    result = self.compute(payload)  # GPU, main thread
                except Exception as e:
                    failures += 1
                    log.error(f"Processing {file} failed: {e}")
                    continue

                # Backpressure: block until the save queue has room before
                # submitting another decoded volume.
                while len(save_futures) >= max_pending_saves:
                    drain_saves(save_futures, block=True)

                # result is (out_h5, image_sc_np, polar_np, metadata)
                save_futures[savers.submit(self.save, *result)] = result[0]
                drain_saves(save_futures, block=False)

            # Drain remaining writes
            while save_futures:
                drain_saves(save_futures, block=True)

        if failures:
            log.warning(f"Conversion completed with {failures} file(s) skipped due to errors.")


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

    # Update Width and Height to reflect the cropped image dimensions
    final_width = cone_params["new_width"]
    final_height = cone_params["new_height"]
    new_row["Width"] = int(final_width)
    new_row["Height"] = int(final_height)

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
    # Read the CSV file
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        fieldnames = reader.fieldnames
        assert fieldnames is not None, "CSV file has no header row"

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
            transformed_row = transform_measurement_coordinates_with_cone_params(row, cone_params)
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


def _fix_faulty_entry(measurements_csv, src):
    """Some entries in the MeasurementsList.csv are faulty, so we fix them here."""
    with open(measurements_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, "MeasurementsList.csv has no header row"
        fieldnames = reader.fieldnames
        rows = list(reader)

    for bad_hash in ["0XBD41EBF599F7EE4F", "0X2061669A27571EA3", "0XA26FCACCC289023E"]:
        try:
            avi_path = find_avi_file(src, bad_hash)
            h, w = _load_first_frame(avi_path).shape
        except FileNotFoundError:
            log.warning(
                f"Trying to fix faulty entry for {bad_hash}, but file not found. "
                "You can ignore this when running on a subset of the dataset."
            )
            continue
        for row in rows:
            if row["HashedFileName"] == bad_hash:
                fps = row["Width"]

                # If they ever update the csv, this will trigger since the FPS is never 500,
                # and the width is always atleast 800.
                if float(fps) > 500:
                    log.warning("Seems like the faulty entries were already fixed.")
                    break

                n_frames = row["FPS"]
                row["Width"] = w
                row["Height"] = h
                row["FPS"] = fps
                row["Frames"] = n_frames

    # Write back
    with open(measurements_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bounded_map(executor, fn, items, max_in_flight):
    """Lazily submit ``fn(item)`` to ``executor`` keeping at most ``max_in_flight``
    futures pending, yielding ``(item, future)`` pairs in input order.

    Order is preserved (so shape-sorting still avoids JIT retracing) while the
    look-ahead bound keeps only a handful of decoded videos in memory at once.
    """
    items = iter(items)
    pending = deque()

    def _submit_next():
        try:
            item = next(items)
        except StopIteration:
            return
        pending.append((item, executor.submit(fn, item)))

    for _ in range(max_in_flight):
        _submit_next()

    while pending:
        item, future = pending.popleft()
        _submit_next()  # refill the window before handing this one back
        yield item, future


def convert_echonetlvh(
    src: Path,
    dst: Path,
    no_rejection,
    rejection_path,
    convert_measurements,
    convert_images,
    max_files,
    force,
    max_workers: int = 8,
):
    """
    Conversion script for the EchoNet-LVH dataset.
    Unzips, overwrites splits if needed, precomputes cone parameters,
    and converts images and/or measurements to zea format and saves dataset.
    Is called with argparse arguments through zea/zea/data/convert/__main__.py
    """

    assert src.exists(), f"Source path {src} does not exist."
    assert dst.exists(), f"Destination path {dst} does not exist."
    assert src.is_dir() or src.suffix == ".zip", (
        f"Source path {src} is not a directory or `.zip` file"
    )
    assert dst.is_dir(), f"Destination path {dst} is not a directory."

    if keras.backend.backend() != "jax":
        log.warning("We recommend using jax for speed in the EchoNet-LVH conversion.")

    # Check if unzip is needed
    if src.suffix == ".zip":
        tmp_dir = dst / "unzipped_original_files"
        tmp_dir.mkdir(exist_ok=True)
        src = unzip(src, tmp_dir)

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

    # There are some mistakes in the csv file, so fix them here
    _fix_faulty_entry(measurements_csv, src)

    # Precompute cone parameters if needed
    cone_params_csv = dst / "cone_parameters.csv"
    if cone_params_csv.exists() and not force:
        log.warning(f"Parameters already exist at {cone_params_csv}. Use --force to recompute.")
    else:
        precompute_cone_parameters(src, measurements_csv, cone_params_csv, max_files, max_workers)

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

        files_to_process = _find_avi_files(src, splits)

        # List files that have already been processed (set for O(1) membership)
        files_done = {
            filename.removesuffix(".hdf5")
            for _, _, filenames in os.walk(dst)
            for filename in filenames
            if filename.endswith(".hdf5")
        }

        # Filter out already processed files
        files_to_process = [f for f in files_to_process if f.stem not in files_done]

        # Filter out files without cone parameters
        files_to_process = [f for f in files_to_process if f.name in cone_parameters]

        # Limit files if max_files is specified
        if max_files is not None:
            files_to_process = files_to_process[:max_files]
            log.info(f"Limited to processing {max_files} files due to max_files parameter")

        log.info(f"Files left to process: {len(files_to_process)}")

        # Initialize processor with splits and cone parameters
        processor = LVHProcessor(path_out_h5=dst, splits=splits, cone_params=cone_parameters)

        # Sort files by (h, w) to avoid retracing
        shapes = load_shapes(measurements_csv)
        files_to_process = sorted(files_to_process, key=lambda x: shapes[x.stem])

        log.info("Starting the conversion process.")

        processor.run(files_to_process, load_workers=max_workers)

        log.info("All image conversion tasks are completed.")

    # Convert measurements if requested
    if convert_measurements:
        transform_measurements_csv(measurements_csv, cone_params_csv)

    log.info("All tasks are completed.")
