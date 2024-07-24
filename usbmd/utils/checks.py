"""Check functions for data types and shapes.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 30 2023
"""

import importlib.util
import logging
from pathlib import Path

import h5py
import numpy as np

from usbmd.registry import checks_registry

_DATA_TYPES = [
    "raw_data",
    "aligned_data",
    "beamformed_data",
    "envelope_data",
    "image",
    "image_sc",
]

_BACKENDS = [None, "torch", "tensorflow", "keras", "jax", "numpy"]

_ML_LIB_AVAILABLE = False
for lib in _BACKENDS:
    if importlib.util.find_spec(str(lib)):
        if lib == "torch":
            _ML_LIB_AVAILABLE = True
        if lib == "tensorflow":
            _ML_LIB_AVAILABLE = True

_MOD_TYPES = [None, "rf", "iq"]

_REQUIRED_SCAN_KEYS = [
    "n_ax",
    "n_el",
    "n_tx",
    "n_ch",
    "probe_geometry",
    "sampling_frequency",
    "center_frequency",
    "t0_delays",
    "n_frames",
]

_IMAGE_DATA_TYPES = ["image", "image_sc", "envelope_data", "beamformed_data"]

_NON_IMAGE_DATA_TYPES = ["raw_data", "aligned_data"]


def get_check(data_type):
    """Get check function for data type.

    Args:
        data_type (str): data type to get check function for

    Raises:
        ValueError: if data type is not valid

    Returns:
        function: check function for data type
    """
    if data_type not in _DATA_TYPES:
        raise ValueError(
            f"Data type {data_type} not valid. Must be one of {_DATA_TYPES}"
        )
    return checks_registry[data_type]


@checks_registry("raw_data")
def _check_raw_data(data=None, shape=None, with_batch_dim=False):
    """Check raw data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): raw data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 5 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """
    assert (
        data is not None or shape is not None
    ), "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if not with_batch_dim:
        assert len(shape) == 4, (
            "raw data must be 4D, with expected shape [n_tx, n_ax, n_el, n_ch], "
            f"got {shape}"
        )
    else:
        assert len(shape) == 5, (
            "raw data must be 5D, with expected shape [n_fr, n_tx, n_ax, n_el, n_ch], "
            f"got {shape}"
        )
    assert shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {shape[-1]} channels"
    )


@checks_registry("aligned_data")
def _check_aligned_data(data=None, shape=None, with_batch_dim=False):
    """Check aligned data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): aligned data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 5 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """
    assert (
        data is not None or shape is not None
    ), "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if not with_batch_dim:
        assert len(shape) == 4, (
            "aligned data must be 4D, with expected shape [n_tx, n_ax, n_el, n_ch], "
            f"got {shape}"
        )
    else:
        assert len(shape) == 5, (
            "aligned data must be 5D, with expected shape [n_fr, n_tx, n_ax, n_el, n_ch], "
            f"got {shape}"
        )
    assert shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {shape[-1]} channels"
    )


@checks_registry("beamformed_data")
def _check_beamformed_data(data=None, shape=None, with_batch_dim=False):
    """Check beamformed data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): beamformed data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """
    assert (
        data is not None or shape is not None
    ), "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if not with_batch_dim:
        assert len(shape) == 3, (
            "beamformed data must be 3D, with expected shape [Ny, Nx, n_ch], "
            f"got {shape}"
        )
    else:
        assert len(shape) == 4, (
            "beamformed data must be 4D, with expected shape [n_fr, Ny, Nx, n_ch], "
            f"got {shape}"
        )
    assert shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {shape[-1]} channels"
    )


@checks_registry("envelope_data")
def _check_envelope_data(data=None, shape=None, with_batch_dim=False):
    """Check envelope data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): envelope data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
    """
    assert (
        data is not None or shape is not None
    ), "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if not with_batch_dim:
        assert len(shape) == 2, (
            "envelope data must be 2D, with expected shape [Ny, Nx], " f"got {shape}"
        )
    else:
        assert len(shape) == 3, (
            "envelope data must be 3D, with expected shape [n_fr, Ny, Nx], "
            f"got {shape}"
        )


@checks_registry("image")
def _check_image(data=None, shape=None, with_batch_dim=False):
    """Check image data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): image data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape.
    """
    assert (
        data is not None or shape is not None
    ), "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if not with_batch_dim:
        assert len(shape) == 2, (
            "image data must be 2D, with expected shape [Ny, Nx], " f"got {shape}"
        )
    else:
        assert len(shape) == 3, (
            "image data must be 3D, with expected shape [n_fr, Ny, Nx], " f"got {shape}"
        )


@checks_registry("image_sc")
def _check_image_sc(data=None, shape=None, with_batch_dim=False):
    """Check image data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): scan-converted data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape.
    """
    assert (
        data is not None or shape is not None
    ), "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if not with_batch_dim:
        assert len(shape) == 2, (
            "image data must be 2D, with expected shape [Ny, Nx], " f"got {shape}"
        )
    else:
        assert len(shape) == 3, (
            "image data must be 3D, with expected shape [n_fr, Ny, Nx], " f"got {shape}"
        )


def validate_dataset(path: str = None, dataset: h5py.File = None):
    """Reads the hdf5 dataset at the given path and validates its structure.

    Provide either the path or the dataset, but not both.

    Args:
        path (str, pathlike): The path to the hdf5 dataset.
        dataset (h5py.File): The hdf5 dataset.

    """
    assert (path is not None) ^ (
        dataset is not None
    ), "Provide either the path or the dataset, but not both."

    if path is not None:
        path = Path(path)
        with h5py.File(path, "r") as _dataset:
            event_structure, num_events = _validate_hdf5_dataset(_dataset)
    else:
        event_structure, num_events = _validate_hdf5_dataset(dataset)

    return {
        "status": "success",
        "event_structure": event_structure,
        "num_events": num_events,
    }


def check_key(dataset, key):
    """Check if the key is present in the dataset."""
    assert key in dataset.keys(), f"The dataset does not contain the key `{key}`."


def _validate_hdf5_dataset(dataset):
    all_keys = list(dataset.keys())

    event_structure = False
    if "data" not in all_keys:
        event_structure = True

    if event_structure:
        num_events = len(all_keys)
        for event_no in range(num_events):
            check_key(dataset, f"event_{event_no}")
            _validate_structure(dataset[f"event_{event_no}"])
    else:
        num_events = 0
        _validate_structure(dataset)

    return event_structure, num_events


def _validate_structure(dataset):
    # Validate the root group
    check_key(dataset, "data")

    # Check if there is only image data
    not_only_image_data = (
        len([i for i in _NON_IMAGE_DATA_TYPES if i in dataset["data"].keys()]) > 0
    )

    # Only check scan group if there is non-image data
    if not_only_image_data:
        check_key(dataset, "scan")

        for key in _REQUIRED_SCAN_KEYS:
            check_key(dataset["scan"], key)

    # validate the data group
    for key in dataset["data"].keys():
        assert key in _DATA_TYPES, "The data group contains an unexpected key."

        # Validate data shape
        data_shape = dataset["data"][key].shape
        if key == "raw_data":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert (
                data_shape[0] == dataset["scan"]["n_frames"][()]
            ), "n_frames does not match the first dimension of raw_data."
            assert (
                data_shape[1] == dataset["scan"]["n_tx"][()]
            ), "n_tx does not match the second dimension of raw_data."
            assert (
                data_shape[2] == dataset["scan"]["n_ax"][()]
            ), "n_ax does not match the third dimension of raw_data."
            assert (
                data_shape[3] == dataset["scan"]["n_el"][()]
            ), "n_el does not match the fourth dimension of raw_data."
        elif key == "aligned_data":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert (
                data_shape[0] == dataset["scan"]["n_frames"][()]
            ), "n_frames does not match the first dimension of aligned_data."
        elif key == "beamformed_data":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert (
                data_shape[0] == dataset["scan"]["n_frames"][()]
            ), "n_frames does not match the first dimension of beamformed_data."
        elif key == "envelope_data":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert (
                data_shape[0] == dataset["scan"]["n_frames"][()]
            ), "n_frames does not match the first dimension of envelope_data."
        elif key == "image":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert (
                data_shape[0] == dataset["scan"]["n_frames"][()]
            ), "n_frames does not match the first dimension of image."
        elif key == "image_sc":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert (
                data_shape[0] == dataset["scan"]["n_frames"][()]
            ), "n_frames does not match the first dimension of image_sc."

    if not_only_image_data:
        _assert_scan_keys_present(dataset)

    _assert_unit_and_description_present(dataset)


def _assert_scan_keys_present(dataset):
    """Ensure that all required keys are present.

    Args:
        dataset (h5py.File): The dataset instance to check.

    Raises:
        AssertionError: If a required key is missing or does not have the right shape.
    """
    for required_key in _REQUIRED_SCAN_KEYS:
        assert (
            required_key in dataset["scan"].keys()
        ), f"The scan group does not contain the required key {required_key}."

    # Ensure that all keys have the correct shape
    for key in dataset["scan"].keys():
        if isinstance(dataset["scan"][key], h5py.Group):
            shape_dataset = None
        else:
            shape_dataset = dataset["scan"][key].shape

        if key == "probe_geometry":
            correct_shape = (dataset["scan"]["n_el"][()], 3)

        elif key == "t0_delays":
            correct_shape = (
                dataset["scan"]["n_tx"][()],
                dataset["scan"]["n_el"][()],
            )
        elif key == "tx_apodizations":
            correct_shape = (
                dataset["scan"]["n_tx"][()],
                dataset["scan"]["n_el"][()],
            )

        elif key == "focus_distances":
            correct_shape = (dataset["scan"]["n_tx"][()],)

        elif key == "polar_angles":
            correct_shape = (dataset["scan"]["n_tx"][()],)

        elif key == "azimuth_angles":
            correct_shape = (dataset["scan"]["n_tx"][()],)

        elif key == "initial_times":
            correct_shape = (dataset["scan"]["n_tx"][()],)

        elif key == "time_to_next_transmit":
            correct_shape = (
                dataset["scan"]["n_frames"][()],
                dataset["scan"]["n_tx"][()],
            )

        elif key in (
            "sampling_frequency",
            "center_frequency",
            "n_frames",
            "n_tx",
            "n_el",
            "n_ax",
            "n_ch",
            "sound_speed",
            "bandwidth_percent",
        ):
            correct_shape = 1
            shape_dataset = dataset["scan"][key].size

        else:
            correct_shape = None
            logging.warning(f"No validation has been defined for {key}.")

        if correct_shape is not None:
            assert shape_dataset == correct_shape, (
                f"`{key}` does not have the correct shape. "
                f"Expected shape: {correct_shape}, got shape: {shape_dataset}"
            )


def _assert_unit_and_description_present(hdf5_file, _prefix=""):
    """Checks that all datasets have a unit and description attribute.

    Args:
        hdf5_file (h5py.File): The hdf5 file to check.

    Raises:
        AssertionError: If a dataset does not have a unit or description
            attribute.
    """
    for key in hdf5_file.keys():
        if isinstance(hdf5_file[key], h5py.Group):
            _assert_unit_and_description_present(
                hdf5_file[key], _prefix=_prefix + key + "/"
            )
        else:
            assert (
                "unit" in hdf5_file[key].attrs.keys()
            ), f"The dataset {_prefix}/{key} does not have a unit attribute."
            assert (
                "description" in hdf5_file[key].attrs.keys()
            ), f"The dataset {_prefix}/{key} does not have a description attribute."


def _assert_uint8_images(images: np.ndarray):
    """
    Asserts that the input images have the correct properties.

    Args:
        images (np.ndarray): The input images.

    Raises:
        AssertionError: If the dtype of images is not uint8.
        AssertionError: If the shape of images is not (n_frames, height, width, channels)
            or (n_frames, height, width) for grayscale images.
        AssertionError: If images have anything other than 1 (grayscale),
            3 (rgb) or 4 (rgba) channels.
    """
    assert (
        images.dtype == np.uint8
    ), f"dtype of images should be uint8, got {images.dtype}"

    assert images.ndim in (3, 4), (
        "images must have shape (n_frames, height, width, channels),"
        f" or (n_frames, height, width) for grayscale images. Got {images.shape}"
    )

    if images.ndim == 4:
        assert images.shape[-1] in (1, 3, 4), (
            "Grayscale images must have 1 channel, "
            "RGB images must have 3 channels, and RGBA images must have 4 channels. "
            f"Got shape: {images.shape}, channels: {images.shape[-1]}"
        )
