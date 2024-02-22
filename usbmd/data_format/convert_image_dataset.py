"""
The function convert_image_dataset convert an existing dataset of images
or sequences of images to USBMD format.
"""

import os
import re
from itertools import groupby

from PIL import Image
import numpy as np

from usbmd.data_format.usbmd_data_format import generate_usbmd_dataset

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def img_to_np(path):
    """
    Reads an image at {path} in grayscale as a 1d np array and adds an n_frames dimension.

    Params:
        path (str): the path for the image you would like to read.

    Returns:
        np.array of shape [1, img_height, img_width]
    """
    image = np.array(Image.open(path).convert("L"))  # Read grayscale = 1 channel
    image = image[None, ...]  # add n_frames dimension
    return image


def img_dir_to_h5_dir(
    existing_dataset_root,
    new_dataset_root,
    current_dir,
    files,
    dataset_name,
    group_pattern=re.compile(r"(.*)\..*"),
    sort_pattern=None,
):
    """
    (This function is intended to be used as a subroutine by convert_image_to_dataset,
    see below for details.)

    Params:
        existing_dataset_root (str): path to the root directory of your image dataset
        new_dataset_root (str): path to the directory which will be the root of your new dataset
        current_dir (str): path to directory containing 'files'
        files (list[str]): list of file paths in 'current_dir'
        dataset_name (optional str): dataset name for hdf5 desciption attribute
        group_pattern (optional re.Pattern): regex pattern to group images into the same hdf5 file
        sort_pattern (optional re.Pattern): regex pattern to extract index for sorting frames in a
                                            group of images

    Returns:
        None
    """
    # Make a new directory in new_dataset_root to match the directory tree in existing_dataset_root
    relative_dir = os.path.relpath(current_dir, existing_dataset_root)
    new_dir_path = f"{new_dataset_root}/{relative_dir}/"
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    # Select only image files
    img_files = list(
        filter(
            lambda path: any(path.lower().endswith(ext) for ext in IMG_EXTENSIONS),
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
            [img_to_np(f"{current_dir}/{img_file}") for img_file in imgs_in_group]
        )

        new_h5_file_path = f"{new_dir_path}/{group_id}.hdf5"
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
    # pylint: disable=anomalous-backslash-in-string
    """
    Maps an image dataset to a hdf5 dataset containing those images, preserving directory structure.
    Can also be used to map a video dataset to hdf5 if the videos are stored as sequences on images.

    Params:
        existing_dataset_root (str): path to the root directory of your image dataset
        new_dataset_root (str): path to the directory which will be the root of your new dataset
        dataset_name (optional str): dataset name for hdf5 desciption attribute
        group_pattern (optional re.Pattern): regex pattern to group images into the same hdf5 file
        sort_pattern (optional re.Pattern): regex pattern to extract index for sorting frames
                                            in a group of images

    Returns:
        None

    Note on group_pattern and sort_pattern:
      * If you have a video dataset, stored as sequences of images, you may want to group
        the files such that images from the same video clip are stored in order in the same
        hdf5 file, with shape [n_frames, height, width]. This is what the group_pattern and
        sort_pattern regexes are for. Any images in the current_dir whose paths match
        group_pattern will be grouped into a single hdf5 file. If the file paths have some
        index,  e.g. frame_{i}.png, then you can match that index with sort_pattern, and
        they frames will be sorted numerically according to that matched substring.

    Example usage:
        ```
        convert_image_dataset(
            "/mnt/z/Ultrasound-BMd/data/oisin/camus_test",
            "/mnt/z/Ultrasound-BMd/data/oisin/camus_test_h5",
            group_pattern=re.compile(r"(patient\d+)_\d+\.png"),
            sort_pattern=re.compile(r"patient\d+_(\d+)\.png"),
        )
        ```
    """
    # pylint: enable=anomalous-backslash-in-string
    assert os.path.exists(
        existing_dataset_root
    ), f"The directory '{existing_dataset_root}' does not exist."

    for current_dir, _, files in os.walk(existing_dataset_root):
        print(f"Mapping {current_dir}")
        img_dir_to_h5_dir(
            existing_dataset_root,
            new_dataset_root,
            current_dir,
            files,
            dataset_name,
            group_pattern=group_pattern,
            sort_pattern=sort_pattern,
        )
