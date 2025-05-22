"""
The function convert_image_dataset convert an existing dataset of images
or sequences of images to USBMD format.
"""

import os
import re
from itertools import groupby
from pathlib import Path

import numpy as np

from usbmd.data.data_format import generate_usbmd_dataset
from usbmd.io_lib import _SUPPORTED_IMG_TYPES, load_image


def _img_dir_to_h5_dir(
    existing_dataset_root,
    new_dataset_root,
    current_dir,
    files,
    dataset_name,
    group_pattern=re.compile(r"(.*)\..*"),
    sort_pattern=None,
):
    """Internal function to convert a directory of images to hdf5 format.
    This function is intended to be used as a subroutine by ``convert_image_to_dataset``,
    see below for details.

    Args:
        existing_dataset_root (str): Path to the root directory of your image dataset.
        new_dataset_root (str): Path to the directory which will be the root of your new dataset.
        current_dir (str): Path to directory containing ``files``.
        files (list[str]): List of file paths in ``current_dir``.
        dataset_name (str, optional): Dataset name for hdf5 description attribute.
        group_pattern (re.Pattern, optional): Regex pattern to group images into the same hdf5 file.
        sort_pattern (re.Pattern, optional): Regex pattern to extract index for sorting frames in a
            group of images.

    Returns:
        None
    """
    # Make a new directory in new_dataset_root to match the directory tree in existing_dataset_root
    relative_dir = os.path.relpath(current_dir, existing_dataset_root)
    new_dir_path = Path(new_dataset_root) / Path(relative_dir)
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    # Select only image files
    img_files = list(
        filter(
            lambda path: any(
                path.lower().endswith(ext) for ext in _SUPPORTED_IMG_TYPES
            ),
            files,
        )
    )

    # Group and sort the frames, if patterns are present
    get_group = lambda path: group_pattern.match(path)[1]
    get_rank = lambda path: (
        int(sort_pattern.match(path)[1]) if sort_pattern is not None else 0
    )
    grouped_sorted_files = [
        (parent, sorted(child, key=get_rank))
        for parent, child in groupby(img_files, key=get_group)
    ]

    # Convert each group of images to a stacked np array and save as hdf5
    for group_id, imgs_in_group in grouped_sorted_files:
        frames = np.vstack(
            [
                load_image(Path(current_dir) / img_file)[None, ...]
                for img_file in imgs_in_group
            ]
        )

        new_h5_file_path = new_dir_path / f"{group_id}.hdf5"
        generate_usbmd_dataset(
            path=new_h5_file_path,
            image=frames,
            probe_name="generic",
            description=f"{dataset_name or 'image'} dataset converted to USBMD format",
        )


def convert_image_dataset(
    existing_dataset_root,
    new_dataset_root,
    dataset_name=None,
    group_pattern=re.compile(r"(.*)\..*"),
    sort_pattern=None,
):
    r"""Converts an existing dataset of images or sequences of images to USBMD format.

    Maps an image dataset to a hdf5 dataset containing those images, preserving directory structure.
    Can also be used to map a video dataset to hdf5 if the videos are stored as sequences on images.

    Args:
        existing_dataset_root (str): Path to the root directory of your image dataset.
        new_dataset_root (str): Path to the directory which will be the root of your new dataset.
        dataset_name (str, optional): Dataset name for hdf5 description attribute.
        group_pattern (re.Pattern, optional): Regex pattern to group images into the same hdf5 file.
        sort_pattern (re.Pattern, optional): Regex pattern to extract index for sorting frames
            in a group of images.

    Returns:
        None

    Note:
        If you have a video dataset, stored as sequences of images, you may want to group
        the files such that images from the same video clip are stored in order in the same
        hdf5 file, with shape [n_frames, height, width]. This is what the group_pattern and
        sort_pattern regexes are for. Any images in the current_dir whose paths match
        group_pattern will be grouped into a single hdf5 file. If the file paths have some
        index,  e.g. frame_{i}.png, then you can match that index with sort_pattern, and
        the frames will be sorted numerically according to that matched substring.

    Example:
        .. code-block:: python

            convert_image_dataset(
                "/mnt/z/Ultrasound-BMd/data/oisin/camus_test",
                "/mnt/z/Ultrasound-BMd/data/oisin/camus_test_h5",
                group_pattern=re.compile(r"(patient\d+)_\d+\.png"),
                sort_pattern=re.compile(r"patient\d+_(\d+)\.png"),
            )
    """
    assert os.path.exists(
        existing_dataset_root
    ), f"The directory '{existing_dataset_root}' does not exist."

    for current_dir, _, files in os.walk(existing_dataset_root):
        print(f"Mapping {current_dir}")
        _img_dir_to_h5_dir(
            existing_dataset_root,
            new_dataset_root,
            current_dir,
            files,
            dataset_name,
            group_pattern=group_pattern,
            sort_pattern=sort_pattern,
        )
