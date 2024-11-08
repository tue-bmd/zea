"""Functionality to convert the camus dataset to the USBMD format.
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
from tqdm import tqdm

from usbmd.data.data_format import generate_usbmd_dataset
from usbmd.display import transform_sc_image_to_polar
from usbmd.utils import log, translate


def sitk_load(filepath: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    # pylint: disable=possibly-used-before-assignment
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
    """Converts the camus database to the USBMD format.

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

    generate_usbmd_dataset(
        path=output_path,
        image=image_seq_polar,
        image_sc=image_seq,
        probe_name="generic",
        description="camus dataset converted to USBMD format",
    )


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/CAMUS_public/database_nifti",
    )
    parser.add_argument(
        "--output", type=str, default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/CAMUS"
    )
    args = parser.parse_args()
    return args


splits = {"train": [1, 401], "val": [401, 451], "test": [451, 501]}


def get_split(x):
    if splits["train"][0] <= x < splits["train"][1]:
        return "train"
    elif splits["val"][0] <= x < splits["val"][1]:
        return "val"
    elif splits["test"][0] <= x < splits["test"][1]:
        return "test"
    else:
        raise ValueError(f"Did not find split for patient: {x}")


if __name__ == "__main__":
    if importlib.util.find_spec("SimpleITK") is None:
        log.error(
            "SimpleITK not installed. Please install SimpleITK: `pip install SimpleITK`"
        )
        sys.exit()
    import SimpleITK as sitk

    args = get_args()

    camus_source_folder = Path(args.source)
    camus_output_folder = Path(args.output)

    # check if output folders already exist
    for split in splits:
        assert not (
            camus_output_folder / split
        ).exists(), (
            f"Output folder {camus_output_folder / split} exists. Exiting program."
        )

    # clone folder structure of source to output using pathlib
    # and run convert_camus() for every hdf5 found in there
    files = list(camus_source_folder.glob("**/*_half_sequence.nii.gz"))
    for source_file in tqdm(files):
        # check if source file in camus database (ignore other files)
        if not "database_nifti" in source_file.parts:
            continue

        patient = source_file.stem.split("_")[0]
        patient_id = int(patient.removeprefix("patient"))
        split = get_split(patient_id)

        output_file = (
            camus_output_folder / split / source_file.relative_to(camus_source_folder)
        )

        # Replace .nii.gz with .hdf5
        output_file = output_file.with_suffix("").with_suffix(".hdf5")

        # make sure folder exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        convert_camus(source_file, output_file, overwrite=False)
