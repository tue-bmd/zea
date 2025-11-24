"""Functionality to convert the camus dataset to the zea format.
Requires SimpleITK to be installed: pip install SimpleITK.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor
from venv import logger

import numpy as np
import scipy
from skimage.transform import resize
from tqdm import tqdm

from zea.data.data_format import generate_zea_dataset
from zea.internal.utils import find_first_nonzero_index
from zea.tensor_ops import translate
from zea.data.convert.utils import unzip


def transform_sc_image_to_polar(image_sc, output_size=None, fit_outline=True):
    """
    Convert a scan-converted 2D ultrasound (cone) image into a square (polar-like) representation.
    
    Parameters:
        image_sc (numpy.ndarray): Input 2D image array (height, width) with background assumed to be zero.
        output_size (tuple, optional): Desired output size as (height, width). Defaults to the input shape.
        fit_outline (bool, optional): If True, fit smooth polynomial outlines to image borders before remapping to reduce edge artifacts; if False, use the raw detected outline.
    
    Returns:
        numpy.ndarray: Squared 2D image resized to `output_size`.
    """
    assert len(image_sc.shape) == 2, "function only allows for 2D data"

    # Default output size is the input size
    if output_size is None:
        output_size = image_sc.shape

    # Initialize an empty target array for polar_image
    polar_image = np.zeros_like(image_sc)

    # Flip along the x axis (such that curve of image_sc is pointing up)
    flipped_image = np.flip(image_sc, axis=0)

    # Find index of first non zero element along y axis (for every vertical line)
    non_zeros_flipped = find_first_nonzero_index(flipped_image, 0)

    # Remove any black vertical lines (columns) that do not contain image data
    remove_vertical_lines = np.where(non_zeros_flipped == -1)[0]
    polar_image = np.delete(polar_image, remove_vertical_lines, axis=1)
    non_zeros_flipped = np.delete(non_zeros_flipped, remove_vertical_lines)

    if fit_outline:
        model_fitted_bottom = np.poly1d(
            np.polyfit(range(len(non_zeros_flipped)), non_zeros_flipped, 4)
        )
        non_zeros_flipped = model_fitted_bottom(range(len(non_zeros_flipped)))
        non_zeros_flipped = non_zeros_flipped.round().astype(np.int64)
        non_zeros_flipped = np.clip(non_zeros_flipped, 0, None)

    non_zeros = polar_image.shape[0] - non_zeros_flipped

    # Find the middle of the width of the image
    width = polar_image.shape[1]
    width_middle = round(width / 2)

    # For every vertical line in the image
    for x_i in range(width):
        # Move the flipped first non-zero element to the bottom of the image
        polar_image[non_zeros_flipped[x_i] :, x_i] = image_sc[: non_zeros[x_i], x_i]

    # Find indices of first and last non-zero element along x axis (for every horizontal line)
    non_zeros_left = find_first_nonzero_index(polar_image, 1)
    non_zeros_right = width - find_first_nonzero_index(np.flip(polar_image, 1), 1, width_middle)

    # Remove any black horizontal lines (rows) that do not contain image data
    remove_horizontal_lines = np.max(np.where(non_zeros_left == -1)) + 1
    polar_image = polar_image[remove_horizontal_lines:, :]
    non_zeros_left = non_zeros_left[remove_horizontal_lines:]
    non_zeros_right = non_zeros_right[remove_horizontal_lines:]

    if fit_outline:
        model_fitted_left = np.poly1d(np.polyfit(range(len(non_zeros_left)), non_zeros_left, 2))
        non_zeros_left = model_fitted_left(range(len(non_zeros_left)))
        non_zeros_left = non_zeros_left.round().astype(np.int64)

        model_fitted_right = np.poly1d(np.polyfit(range(len(non_zeros_right)), non_zeros_right, 2))
        non_zeros_right = model_fitted_right(range(len(non_zeros_right)))
        non_zeros_right = non_zeros_right.round().astype(np.int64)

    # For every horizontal line in the image
    for y_i in range(polar_image.shape[0]):
        small_array = polar_image[y_i, non_zeros_left[y_i] : non_zeros_right[y_i]]

        if len(small_array) <= 1:
            # If the array is too small for interpolation, set it to the middle value.
            polar_image[y_i, :] = polar_image[y_i, width_middle]
        else:
            # Perform linear interpolation to stretch the line to the desired width.
            array_interp = scipy.interpolate.interp1d(np.arange(small_array.size), small_array)
            polar_image[y_i, :] = array_interp(np.linspace(0, small_array.size - 1, width))

    # Resize image to output_size
    return resize(polar_image, output_size, preserve_range=True)


def sitk_load(filepath: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load an image file using SimpleITK and return its pixel array along with extracted metadata.
    
    Parameters:
        filepath (str | Path): Path to the image file to load.
    
    Returns:
        im_array (np.ndarray): NumPy array produced by SimpleITK's GetArrayFromImage. For 3D images the shape is (Z, H, W); for 2D images the shape is (H, W). Pixel dtype is preserved.
        metadata (dict): Dictionary with keys:
            - "origin": image origin tuple,
            - "ElementSpacing": spacing tuple,
            - "direction": direction cosines tuple,
            - "NDims": number of image dimensions,
            - "metadata": mapping of SimpleITK metadata keys to their values.
    """
    # Load image and save info
    import SimpleITK as sitk

    image = sitk.ReadImage(str(filepath))

    all_metadata = {}
    for k in image.GetMetaDataKeys():
        all_metadata[k] = image.GetMetaData(k)

    metadata = {
        "origin": image.GetOrigin(),
        "ElementSpacing": image.GetSpacing(),
        "direction": image.GetDirection(),
        "NDims": image.GetDimension(),
        "metadata": all_metadata,
    }

    # Extract numpy array from the SimpleITK image object
    im_array = sitk.GetArrayFromImage(image)

    return im_array, metadata


def process_camus(source_path, output_path, output_path_npz=None, overwrite=False):
    """
    Convert a CAMUS image or sequence to ZE A format and write output files.
    
    This loads the CAMUS source, converts each frame to a polar (squared) representation, rescales both original and polar images to the range [-60, 0] dB, and writes a ZE A HDF5 dataset. Optionally saves a compressed NumPy (.npz) with the polar and scan-converted images.
    
    Parameters:
        source_path (str | os.PathLike): Path to the CAMUS image file or sequence.
        output_path (str | os.PathLike): Destination path for the ZE A HDF5 output.
        output_path_npz (str | os.PathLike, optional): If provided, path to write a compressed .npz containing `image` (polar) and `image_sc` (scan-converted).
        overwrite (bool, optional): If True, overwrite an existing output_path. If False and output_path exists, the function returns without writing. Defaults to False.
    """

    # Check if output file already exists and remove
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            logging.warning("Output file already exists. Skipping conversion.")
            return

    # Open the file
    image_seq, _ = sitk_load(source_path)

    # Convert to polar coordinates
    image_seq_polar = []
    for image in image_seq:
        image_seq_polar.append(transform_sc_image_to_polar(image))
    image_seq_polar = np.stack(image_seq_polar, axis=0)

    # Change range to [-60, 0] dB
    image_seq = translate(image_seq, (0, 255), (-60, 0))
    image_seq_polar = translate(image_seq_polar, (0, 255), (-60, 0))

    if output_path_npz is not None:
        # Save as numpy file
        np.savez_compressed(
            output_path_npz,
            image=image_seq_polar,
            image_sc=image_seq,
        )

    generate_zea_dataset(
        path=output_path,
        image=image_seq_polar,
        image_sc=image_seq,
        probe_name="generic",
        description="camus dataset converted to zea format",
    )


splits = {"train": [1, 401], "val": [401, 451], "test": [451, 501]}


def get_split(patient_id: int) -> str:
    """
    Determine which dataset split a patient ID belongs to.
    
    Returns:
        The split name: "train", "val", or "test".
    
    Raises:
        ValueError: If the patient_id does not fall into any defined split range.
    """
    if splits["train"][0] <= patient_id < splits["train"][1]:
        return "train"
    elif splits["val"][0] <= patient_id < splits["val"][1]:
        return "val"
    elif splits["test"][0] <= patient_id < splits["test"][1]:
        return "test"
    else:
        raise ValueError(f"Did not find split for patient: {patient_id}")


def _process_task(task):
    """
    Unpack a task tuple and invoke process_camus in a worker process.
    
    Creates parent directories for the target outputs, calls process_camus with the unpacked paths, and logs then re-raises any exception raised by processing.
    
    Parameters:
        task (tuple): (source_file_str, output_file_str, output_file_npz_str_or_None)
            - source_file_str: filesystem path to the source CAMUS file as a string.
            - output_file_str: filesystem path for the ZEA output file as a string.
            - output_file_npz_str_or_None: filesystem path for optional NPZ output as a string, or None.
    """
    source_file_str, output_file_str, output_file_npz_str = task
    source_file = Path(source_file_str)
    output_file = Path(output_file_str)

    # Ensure destination directories exist (safe to call from multiple processes)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file_npz = Path(output_file_npz_str) if output_file_npz_str else None
    if output_file_npz is not None:
        output_file_npz.parent.mkdir(parents=True, exist_ok=True)

    # Call the real processing function (must be importable in the worker)
    # If process_camus lives in another module, import it there instead.
    try:
        process_camus(source_file, output_file, output_file_npz, overwrite=False)
    except Exception:
        # Log and re-raise so the main process can handle it
        logger.exception("Error processing %s", source_file)
        raise


def convert_camus(args):
    """
    Orchestrates conversion of the CAMUS dataset into ZE A HDF5 files (and optional compressed NPZ files) across dataset splits.
    
    Processes files found under the CAMUS source folder (after unzipping if needed), assigns each patient to a train/val/test split, creates matching output paths, and executes per-file conversion tasks either serially or in parallel. Ensures output directories do not pre-exist, optionally produces .npz copies when dst_npz is provided, and logs progress and failures.
    
    Parameters:
        args: An object with the following attributes:
            src (str | Path): Path to the CAMUS archive or extracted folder.
            dst (str | Path): Root destination folder for ZE A HDF5 outputs; split subfolders will be created.
            dst_npz (str | Path | None): Optional root destination for compressed NPZ outputs; if provided, NPZ files are produced.
            no_hyperthreading (bool, optional): If True, run tasks serially instead of using a process pool.
    """
    to_numpy = args.dst_npz is not None

    camus_source_folder = Path(args.src)
    camus_output_folder = Path(args.dst)
    camus_output_folder_npz = Path(args.dst_npz) if to_numpy else None

    # Look for either CAMUS_public.zip or folders database_nifti, database_split
    camus_source_folder = unzip(camus_source_folder, "camus")

    # check if output folders already exist
    for split in splits:
        assert not (camus_output_folder / split).exists(), (
            f"Output folder {camus_output_folder / split} exists. Exiting program."
        )

    # clone folder structure of source to output using pathlib
    files = list(camus_source_folder.glob("**/*_half_sequence.nii.gz"))
    tasks = []
    for source_file in files:
        # check if source file in camus database (ignore other files)
        if "database_nifti" not in source_file.parts:
            continue

        patient = source_file.stem.split("_")[0]
        patient_id = int(patient.removeprefix("patient"))
        split = get_split(patient_id)

        output_file = camus_output_folder / split / source_file.relative_to(camus_source_folder)
        # Replace .nii.gz with .hdf5
        output_file = output_file.with_suffix("").with_suffix(".hdf5")
        # make sure folder exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if to_numpy:
            output_file_npz = (
                camus_output_folder_npz / split / source_file.relative_to(camus_source_folder)
            )
            output_file_npz = output_file_npz.with_suffix("").with_suffix(".npz")
            output_file_npz.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((str(source_file), str(output_file), str(output_file_npz)))
        else:
            tasks.append((str(source_file), str(output_file), None))
    if not tasks:
        logger.info("No files found to process.")
        return

    if getattr(args, "no_hyperthreading", False):
        logger.info("no_hyperthreading is True — running tasks serially (no ProcessPoolExecutor)")
        for t in tqdm(tasks, desc="Processing files (serial)"):
            try:
                _process_task(t)
            except Exception as e:
                logger.exception("Task processing failed: %s", e)
        logger.info("Processing finished for %d files (serial)", len(tasks))
        return

    # Submit tasks to the process pool and track progress
    with ProcessPoolExecutor(max_workers=64) as exe:
        for _ in tqdm(exe.map(_process_task, tasks), total=len(tasks), desc="Processing files"):
            pass
    logger.info("Processing finished for %d files", len(tasks))