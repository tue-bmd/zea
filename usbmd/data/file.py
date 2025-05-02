"""USBMD H5 file functionality.

- **Author(s)**     : Tristan Stevens, Wessel van Nierop
"""

import inspect
from pathlib import Path
from typing import List

import h5py
import numpy as np

from usbmd.probes import get_probe
from usbmd.scan import Scan, cast_scan_parameters
from usbmd.utils import log
from usbmd.utils.checks import (
    _DATA_TYPES,
    _NON_IMAGE_DATA_TYPES,
    _REQUIRED_SCAN_KEYS,
    get_check,
)


def get_shape_hdf5_file(filepath, key):
    """Retrieve the shape of some key in a hdf5 file."""
    with File(filepath, mode="r") as f:
        return f.shape(key)


def assert_key(file, key):
    """Asserts key is in file."""
    if key not in file.keys():
        raise KeyError(f"{key} not found in file")


class File(h5py.File):
    """h5py.File in usbmd format."""

    def __init__(self, *args, **kwargs):
        if "locking" not in kwargs and "mode" in kwargs and kwargs["mode"] == "r":
            # If the file is opened in read mode, disable locking
            kwargs["locking"] = False
        super().__init__(*args, **kwargs)

    def print(self):
        """Print the contents of the file."""
        print_hdf5_attrs(self)

    @property
    def path(self):
        """Return the path of the file."""
        return Path(self.filename)

    @property
    def name(self):
        """Return the name of the file."""
        return self.path.name

    @property
    def stem(self):
        """Return the stem of the file."""
        return self.path.stem

    @property
    def event_keys(self):
        """Return all events in the file."""
        return [key for key in self.keys() if "event" in key]

    @property
    def has_events(self):
        """Check if the file has events."""
        return any("event" in key for key in self.keys())
        # return self.attrs.get("event_structure", False)

    @property
    def num_frames(self):
        """Return number of frames in a file."""

        if "scan" in self.file:
            return int(self.file["scan"]["n_frames"][()])
        else:
            return sum(
                int(event["scan"]["n_frames"][()]) for event in self.file.values()
            )

    def get_event_shapes(self, key):
        """Get the shapes of a key for all events."""
        for event_key in self.event_keys:
            yield self[event_key][key].shape

    def events_have_same_shape(self, key):
        """Check if all events have the same shape for a given key."""
        if not self.has_events:
            return True

        shapes = list(self.get_event_shapes(key))
        return len(np.unique(shapes)) == 1

    def assert_key(self, key):
        """Asserts key is in file."""
        return assert_key(self, key)

    def _simple_index(self, key):
        return not self.has_events or "event" in key

    def shape(self, key):
        """Return shape of some key, or all events."""

        self.assert_key(key)

        if self._simple_index(key):
            return list(self[key].shape)
        else:
            raise NotImplementedError

    @staticmethod
    def _prepare_indices(indices):
        """Prepare the indices for loading data from hdf5 files.
        Options:
            - str("all")
            - int -> single frame
            - list of ints -> indexes first axis (frames)
            - list of list, ranges or slices -> indexes multiple axes
        """
        # TODO: assert the typing of indices and test the options
        if isinstance(indices, str):
            if indices == "all":
                return slice(None)
            else:
                raise ValueError(f"Invalid value for indices: {indices}. ")

        if isinstance(indices, int):
            return indices

        # Convert ranges to lists
        processed_indices = [
            list(idx) if isinstance(idx, range) else idx for idx in indices
        ]

        # Check if items are list-like and cast to tuple (needed for hdf5)
        if any(isinstance(idx, (list, tuple, slice)) for idx in processed_indices):
            processed_indices = tuple(processed_indices)

        return processed_indices

    def load_scan(self, event=None):
        """Alias for get_scan_parameters."""
        return self.get_scan_parameters(event)

    @staticmethod
    def check_data(data, key):
        """Check the data for a given key. For example, will check if the shape matches
        the data type (such as raw_data, ...)"""
        if key in _DATA_TYPES:
            get_check(key)(data, with_batch_dim=None)

    @staticmethod
    def format_key(key):
        """Format the key to match the data type."""
        # TODO: support events
        if "data/" not in key:
            return "data/" + key
        else:
            return key

    def to_iterator(self, key):
        """Convert the data to an iterator over all frames."""
        for frame_idx in range(self.num_frames):
            yield self.load_data(key, frame_idx)

    def load_data(self, dtype, indices: str | int | List[int] = "all"):
        """Load data from the file.

        Args:
            dtype (str): The type of data to load. Options are 'raw_data', 'aligned_data',
                'beamformed_data', 'envelope_data', 'image' and 'image_sc'.
            indices (str, int, list, optional): The indices to load. Defaults to "all" in
                which case all frames are loaded. If an int is provided, it will be used
                as a single index. If a list is provided, it will be used as a list of
                indices.
        """
        key = self.format_key(dtype)
        self.assert_key(key)
        indices = self._prepare_indices(indices)

        if self._simple_index(key):
            data = self[key][indices]
            self.check_data(data, key)
        elif self.events_have_same_shape(key):
            raise NotImplementedError
        else:
            raise NotImplementedError

        return data

    @property
    def probe_name(self):
        """Reads the probe name from the data file and returns it."""
        assert "probe" in self.attrs, (
            "Probe name not found in file attributes. "
            "Make sure you are using a USBMD dataset. "
            f"Found attributes: {list(self.attrs)}"
        )
        probe_name = self.attrs["probe"]
        return probe_name

    @property
    def description(self):
        """Reads the description from the data file and returns it."""
        assert "description" in self.attrs, (
            "Description not found in file attributes. "
            "Make sure you are using a USBMD dataset. "
            f"Found attributes: {list(self.attrs)}"
        )
        description = self.attrs["description"]
        return description

    def get_parameters(self, event=None):
        """Returns a dictionary of parameters to initialize a scan
        object that comes with the dataset (stored inside datafile).

        If there are no scan parameters in the hdf5 file, returns
        an empty dictionary.

        Args:
            file (h5py or mat): File container.
            event (int, optional): Event number. When specified an event structure
                is expected as follows:
                    - event_0/scan
                    - event_1/scan
                    - ...
                Defaults to None. In that case no event structure is expected.

        Returns:
            dict: The scan parameters.
        """
        scan_parameters = {}
        if "scan" in self:
            scan_parameters = recursively_load_dict_contents_from_group(self, "scan")
        elif "event" in list(self.keys())[0]:
            if event is None:
                raise ValueError(
                    log.error(
                        "Please specify an event number to read scan parameters "
                        "from a file with an event structure."
                    )
                )

            assert f"event_{event}/scan" in self, (
                f"Could not find scan parameters for event {event} in file. "
                f"Found number of events: {len(self.keys())}."
            )

            scan_parameters = recursively_load_dict_contents_from_group(
                self, f"event_{event}/scan"
            )
        else:
            log.warning("Could not find scan parameters in file.")

        scan_parameters = cast_scan_parameters(scan_parameters)
        return scan_parameters

    def get_scan_parameters(self, event=None):
        """Returns a dictionary of default parameters to initialize a scan
        object that works with the dataset.

        Returns:
            dict: The default parameters (the keys are identical to the
                __init__ parameters of the Scan class).
        """
        file_scan_parameters = self.get_parameters(event)

        sig = inspect.signature(Scan.__init__)
        scan_parameters = {
            key: file_scan_parameters[key]
            for key in sig.parameters
            if key in file_scan_parameters
        }
        if len(scan_parameters) == 0:
            log.info(f"Could not find proper scan parameters in {self}.")
        return scan_parameters

    def scan(self, event=None):
        """Returns a Scan object initialized with the parameters from the file.

        Args:
            event (int, optional): Event number. When specified an event structure
                is expected as follows:
                    - event_0/scan
                    - event_1/scan
                    - ...
                Defaults to None. In that case no event structure is expected.

        Returns:
            Scan: The scan object.
        """
        return Scan.safe_initialize(**self.get_scan_parameters(event))

    def get_probe_parameters(self, event=None):
        """Returns a dictionary of probe parameters to initialize a probe
        object that comes with the dataset (stored inside datafile).

        Returns:
            dict: The probe parameters.
        """
        file_scan_parameters = self.get_parameters(event)

        sig = inspect.signature(get_probe("generic").__init__)
        probe_parameters = {
            key: file_scan_parameters[key]
            for key in sig.parameters
            if key in file_scan_parameters
        }
        return probe_parameters

    def probe(self, event=None):
        """Returns a Probe object initialized with the parameters from the file.

        Args:
            event (int, optional): Event number. When specified an event structure
                is expected as follows:
                    - event_0/scan
                    - event_1/scan
                    - ...
                Defaults to None. In that case no event structure is expected.

        Returns:
            Probe: The probe object.
        """
        probe_parameters = self.get_probe_parameters(event)
        if self.probe_name == "generic":
            return get_probe(self.probe_name, **probe_parameters)
        else:
            probe = get_probe(self.probe_name)

            probe_geometry = probe_parameters.get("probe_geometry", None)
            if not np.allclose(probe_geometry, probe.probe_geometry):
                probe.probe_geometry = probe_geometry
                log.warning(
                    "The probe geometry in the data file does not "
                    "match the probe geometry of the probe. The probe "
                    "geometry has been updated to match the data file."
                )

    def recursively_load_dict_contents_from_group(
        self, path: str, squeeze: bool = False
    ) -> dict:
        """Load dict from contents of group

        Values inside the group are converted to numpy arrays
        or primitive types (int, float, str). Single element
        arrays are converted to the corresponding primitive type (if squeeze=True)

        Args:
            path (str): path to group
            squeeze (bool, optional): squeeze arrays with single element.
                Defaults to False.
        Returns:
            dict: dictionary with contents of group
        """
        return recursively_load_dict_contents_from_group(self, path, squeeze)


def load_usbmd_file(
    path, frames=None, transmits=None, data_type="raw_data", scan: Scan = None
):
    """Loads a hdf5 file in the USBMD format and returns the data together with
    a scan object containing the parameters of the acquisition and a probe
    object containing the parameters of the probe.

    # TODO: add support for event

    Args:
        path (str, pathlike): The path to the hdf5 file.
        frames (tuple, list, optional): The frames to load. Defaults to None in
            which case all frames are loaded.
        transmits (tuple, list, optional): The transmits to load. Defaults to
            None in which case all transmits are used.
        data_type (str, optional): The type of data to load. Defaults to
            'raw_data'. Other options are 'aligned_data', 'beamformed_data',
            'envelope_data', 'image' and 'image_sc'.
        scan (utils.config.Config, optional): A Scan object to override the scan parameters
            in the data file. Defaults to None.

    Returns:
        (np.ndarray): The raw data of shape (n_frames, n_tx, n_ax, n_el, n_ch).
        (Scan): A scan object containing the parameters of the acquisition.
        (Probe): A probe object containing the parameters of the probe.
    """

    if frames is not None:
        # Assert that all frames are integers
        assert all(
            isinstance(frame, int) for frame in frames
        ), "All frames must be integers."
    else:
        frames = "all"

    # Define the additional keyword parameters from the scan object
    if scan is None:
        scan = {}

    if transmits is not None:
        # Assert that all frames are integers
        assert all(
            isinstance(tx, int) for tx in transmits
        ), "All transmits must be integers."

    with File(path, mode="r") as file:
        # Load the probe object from the file
        probe = file.probe()

        # Load the desired frames from the file
        data = file.load_data(data_type, indices=frames)

        scan["selected_transmits"] = transmits
        scan = Scan.merge(file.scan(), scan)

        return data, scan, probe


def recursively_load_dict_contents_from_group(
    h5file: h5py._hl.files.File, path: str, squeeze: bool = False
) -> dict:
    """Load dict from contents of group

    Values inside the group are converted to numpy arrays
    or primitive types (int, float, str). Single element
    arrays are converted to the corresponding primitive type (if squeeze=True)

    Args:
        h5file (h5py._hl.files.File): h5py file object
        path (str): path to group
        squeeze (bool, optional): squeeze arrays with single element.
            Defaults to False.
    Returns:
        dict: dictionary with contents of group
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
            # all ones in shape
            if squeeze:
                if ans[key].shape == () or all(i == 1 for i in ans[key].shape):
                    # check for strings
                    if isinstance(ans[key], str):
                        ans[key] = str(ans[key])
                    # check for integers
                    elif int(ans[key]) == float(ans[key]):
                        ans[key] = int(ans[key])
                    else:
                        ans[key] = float(ans[key])
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + "/" + key + "/"
            )
    return ans


def print_hdf5_attrs(hdf5_obj, prefix=""):
    """Recursively prints all keys, attributes, and shapes in an HDF5 file.

    Args:
        hdf5_obj (h5py.File, h5py.Group, h5py.Dataset): HDF5 object to print.
        prefix (str, optional): Prefix to print before each line. This
            parameter is used in internal recursion and should not be supplied
            by the user.
    """
    assert isinstance(
        hdf5_obj, (h5py.File, h5py.Group, h5py.Dataset)
    ), "ERROR: hdf5_obj must be a File, Group, or Dataset object"

    if isinstance(hdf5_obj, h5py.File):
        name = "root" if hdf5_obj.name == "/" else hdf5_obj.name
        print(prefix + name + "/")
        prefix += "    "
    elif isinstance(hdf5_obj, h5py.Dataset):
        shape_str = str(hdf5_obj.shape).replace(",)", ")")
        print(prefix + "├── " + hdf5_obj.name + " (shape=" + shape_str + ")")
        prefix += "│   "

    # Print all attributes
    for key, val in hdf5_obj.attrs.items():
        print(prefix + "├── " + key + ": " + str(val))

    # Recursively print all keys, attributes, and shapes in groups
    if isinstance(hdf5_obj, h5py.Group):
        for i, key in enumerate(hdf5_obj.keys()):
            is_last = i == len(hdf5_obj.keys()) - 1
            if is_last:
                marker = "└── "
                new_prefix = prefix + "    "
            else:
                marker = "├── "
                new_prefix = prefix + "│   "
            print(prefix + marker + key + "/")
            print_hdf5_attrs(hdf5_obj[key], new_prefix)


def validate_dataset(path: str = None, dataset: File = None):
    """Reads the hdf5 dataset at the given path and validates its structure.

    Provide either the path or the dataset, but not both.

    Args:
        path (str, pathlike): The path to the hdf5 dataset.
        dataset (File): The hdf5 dataset.

    """
    assert (path is not None) ^ (
        dataset is not None
    ), "Provide either the path or the dataset, but not both."

    if path is not None:
        path = Path(path)
        with File(path, "r") as _dataset:
            event_structure, num_events = _validate_hdf5_dataset(_dataset)
    else:
        event_structure, num_events = _validate_hdf5_dataset(dataset)

    return {
        "status": "success",
        "event_structure": event_structure,
        "num_events": num_events,
    }


def _validate_hdf5_dataset(dataset: File):
    all_keys = list(dataset.keys())

    if dataset.has_events:
        num_events = len(all_keys)
        for event_no in range(num_events):
            assert_key(dataset, f"event_{event_no}")
            _validate_structure(dataset[f"event_{event_no}"])
    else:
        num_events = 0
        _validate_structure(dataset)

    return dataset.has_events, num_events


def _validate_structure(dataset: File):
    # Validate the root group
    assert_key(dataset, "data")

    # Check if there is only image data
    not_only_image_data = (
        len([i for i in _NON_IMAGE_DATA_TYPES if i in dataset["data"].keys()]) > 0
    )

    # Only check scan group if there is non-image data
    if not_only_image_data:
        assert_key(dataset, "scan")

        for key in _REQUIRED_SCAN_KEYS:
            assert_key(dataset["scan"], key)

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
        elif key == "tgc_gain_curve":
            correct_shape = (dataset["scan"]["n_ax"][()],)
        elif key == "tx_waveform_indices":
            correct_shape = (dataset["scan"]["n_tx"][()],)
        elif key in ("waveforms_one_way", "waveforms_two_way"):
            correct_shape = None

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
            "element_width",
            "lens_correction",
        ):
            correct_shape = ()
            shape_dataset = dataset["scan"][key].shape

        else:
            correct_shape = None
            log.warning(f"No validation has been defined for {log.orange(key)}.")

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
