"""Functionality to convert the camus dataset to the zea format.
Requires SimpleITK to be installed: pip install SimpleITK.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy
from skimage.transform import resize
from tqdm import tqdm

# from zea.display import transform_sc_image_to_polar
from zea import log
from zea.data.data_format import generate_zea_dataset
from zea.utils import find_first_nonzero_index, translate


def transform_sc_image_to_polar(image_sc, output_size=None, fit_outline=True):
    """
    Transform a scan converted input image (cone) into square
        using radial stretching and downsampling. Note that it assumes the background to be zero!
        Please verify if your results make sense, especially if the image contains black parts
        at the edges. This function is not perfect by any means, but it works for most cases.

    Args:
        image (numpy.ndarray): Input image as a 2D numpy array (height, width).
        output_size (tuple, optional): Output size of the image as a tuple.
            Defaults to image_sc.shape.
        fit_outline (bool, optional): Whether to fit a polynomial the outline of the image.
            Defaults to True. If this is set to False, and the ultrasound image contains
            some black parts at the edges, weird artifacts can occur, because the jagged outline
            is stretched to the desired width.

    Returns:
        numpy.ndarray: Squared image as a 2D numpy array (height, width).
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
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
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


def convert_camus(source_path, output_path, overwrite=False):
    """Converts the camus database to the zea format.

    Args:
        source_path (str, pathlike): The path to the original camus file.
        output_path (str, pathlike): The path to the output file.
        overwrite (bool, optional): Set to True to overwrite existing file.
            Defaults to False.
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

    generate_zea_dataset(
        path=output_path,
        image=image_seq_polar,
        image_sc=image_seq,
        probe_name="generic",
        description="camus dataset converted to zea format",
    )


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        # path to CAMUS_public/database_nifti
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


splits = {"train": [1, 401], "val": [401, 451], "test": [451, 501]}


def get_split(patient_id: int) -> str:
    """Determine the dataset split for a given patient ID."""
    if splits["train"][0] <= patient_id < splits["train"][1]:
        return "train"
    elif splits["val"][0] <= patient_id < splits["val"][1]:
        return "val"
    elif splits["test"][0] <= patient_id < splits["test"][1]:
        return "test"
    else:
        raise ValueError(f"Did not find split for patient: {patient_id}")


if __name__ == "__main__":
    if importlib.util.find_spec("SimpleITK") is None:
        log.error("SimpleITK not installed. Please install SimpleITK: `pip install SimpleITK`")
        sys.exit()
    import SimpleITK as sitk

    args = get_args()

    camus_source_folder = Path(args.source)
    camus_output_folder = Path(args.output)

    # check if output folders already exist
    for split in splits:
        assert not (camus_output_folder / split).exists(), (
            f"Output folder {camus_output_folder / split} exists. Exiting program."
        )

    # clone folder structure of source to output using pathlib
    # and run convert_camus() for every hdf5 found in there
    files = list(camus_source_folder.glob("**/*_half_sequence.nii.gz"))
    for source_file in tqdm(files):
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
        convert_camus(source_file, output_file, overwrite=False)
