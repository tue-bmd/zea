"""
Functions to write and validate datasets in the zea format.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from zea.data.file import File
from zea.internal.checks import _DATA_TYPES


@dataclass
class DatasetElement:
    """Class to store a dataset element with a name, data, description and unit. Used to
    supply and load additional dataset elements in the zea format."""

    # The name of the dataset. This will be the key in the group.
    dataset_name: str
    # The data to store in the dataset.
    data: np.ndarray
    description: str
    unit: str
    # The group name to store the dataset under. This can be a nested group, e.g.
    # "lens/profiles"
    group_name: str = ""


def validate_input_data(raw_data, aligned_data, envelope_data, beamformed_data, image, image_sc):
    """
    Validates input data for a zea dataset

    Args:
        raw_data (np.ndarray): The raw data of the ultrasound measurement of
            shape (n_frames, n_tx, n_ax, n_el, n_ch).
        aligned_data (np.ndarray): The aligned data of the ultrasound measurement of
            shape (n_frames, n_tx, n_ax, n_el, n_ch).
        envelope_data (np.ndarray): The envelope data of the ultrasound measurement of
            shape (n_frames, grid_size_z, grid_size_x). Must be an ndarray.
        beamformed_data (np.ndarray): The beamformed data of the ultrasound measurement of
            shape (n_frames, grid_size_z, grid_size_x). Must be an ndarray.
        image (np.ndarray): The image data of shape (n_frames, grid_size_z, grid_size_x).
            Must be an ndarray.
        image_sc (np.ndarray): The scan converted images of shape
            (n_frames, output_size_z, output_size_x). Must be an ndarray.
    """
    assert (
        raw_data is not None
        or aligned_data is not None
        or envelope_data is not None
        or beamformed_data is not None
        or image is not None
        or image_sc is not None
    ), f"At least one of the data types {_DATA_TYPES} must be specified."

    # specific checks for each data type are done in validate_file


def load_description(path):
    """Loads the description of a zea dataset.

    Args:
        path (str): The path to the zea dataset.

    Returns:
        str: The description of the dataset, or an empty string if not found.
    """
    path = Path(path)

    with File(path, "r") as file:
        description = file.attrs.get("description", "")

    return description


def load_additional_elements(path):
    """Loads additional dataset elements from a zea dataset.

    Args:
        path (str): The path to the zea dataset.

    Returns:
        list: A list of DatasetElement objects.
    """
    path = Path(path)

    with File(path, "r") as file:
        if "non_standard_elements" not in file:
            return []

        additional_elements = _load_additional_elements_from_group(file, "non_standard_elements")

    return additional_elements


def _load_additional_elements_from_group(file, path):
    """Recursively loads additional dataset elements from a group."""
    elements = []
    for name, item in file[path].items():
        if isinstance(item, h5py.Dataset):
            elements.append(_load_dataset_element_from_group(file, f"{path}/{name}"))
        elif isinstance(item, h5py.Group):
            elements.extend(_load_additional_elements_from_group(file, f"{path}/{name}"))
    return elements


def _load_dataset_element_from_group(file, path):
    """Loads a specific dataset element from a group.

    Args:
        file (h5py.File): The HDF5 file object.
        path (str): The full path to the dataset element.
            e.g., "non_standard_elements/lens/lens_profile"

    Returns:
        DatasetElement: The loaded dataset element.
    """

    dataset = file[path]
    description = dataset.attrs.get("description", "")
    unit = dataset.attrs.get("unit", "")
    data = dataset[()]

    path_parts = path.split("/")

    return DatasetElement(
        dataset_name=path_parts[-1],
        data=data,
        description=description,
        unit=unit,
        group_name="/".join(path_parts[1:-1]),
    )
