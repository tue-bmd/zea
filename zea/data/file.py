"""zea H5 file functionality."""

import enum
from pathlib import Path
from typing import List

import h5py
import numpy as np

from zea import log
from zea.data.preset_utils import HF_PREFIX, _hf_resolve_path
from zea.internal.checks import (
    _DATA_TYPES,
    _NON_IMAGE_DATA_TYPES,
    _REQUIRED_SCAN_KEYS,
    get_check,
)
from zea.probes import Probe
from zea.scan import Scan
from zea.utils import reduce_to_signature


def assert_key(file: h5py.File, key: str):
    """Asserts key is in a h5py.File."""
    if key not in file.keys():
        raise KeyError(f"{key} not found in file")


class File(h5py.File):
    """h5py.File in zea format."""

    def __init__(self, name, *args, **kwargs):
        """Initialize the file.

        Args:
            name (str, Path, HFPath): The path to the file.
                Can be a string or a Path object. Additionally can be a string with
                the prefix 'hf://', in which case it will be resolved to a
                huggingface path.
            *args: Additional arguments to pass to h5py.File.
            **kwargs: Additional keyword arguments to pass to h5py.File.
        """

        if str(name).startswith(HF_PREFIX):
            name = _hf_resolve_path(str(name))

        if "locking" not in kwargs and "mode" in kwargs and kwargs["mode"] == "r":
            # If the file is opened in read mode, disable locking
            kwargs["locking"] = False

        super().__init__(name, *args, **kwargs)

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
    def n_frames(self):
        """Return number of frames in a file."""

        if "scan" in self.file:
            return int(self.file["scan"]["n_frames"][()])
        else:
            return sum(int(event["scan"]["n_frames"][()]) for event in self.file.values())

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

    def _simple_index(self, key):
        return not self.has_events or "event" in key

    def shape(self, key) -> tuple:
        """Return shape of some key, or all events."""
        key = self.format_key(key)

        if self._simple_index(key):
            return self[key].shape
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

        Returns:
            indices (tuple): A tuple of indices / slices to use for indexing.
        """
        _value_error_msg = (
            f"Invalid value for indices: {indices}. "
            "Indices can be a 'all', int or a List[int, tuple, list, slice, range]."
        )

        # Check all options that only index the first axis
        if isinstance(indices, str):
            if indices == "all":
                return slice(None)
            else:
                raise ValueError(_value_error_msg)

        if isinstance(indices, range):
            return list(indices)

        if isinstance(indices, (int, slice, np.integer)):
            return indices

        # At this point, indices should be a list or tuple
        assert isinstance(indices, (list, tuple, np.ndarray)), _value_error_msg

        assert all(
            isinstance(idx, (list, tuple, int, slice, range, np.ndarray, np.integer))
            for idx in indices
        ), _value_error_msg

        # Convert ranges to lists
        processed_indices = [list(idx) if isinstance(idx, range) else idx for idx in indices]

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

    def format_key(self, key):
        """Format the key to match the data type."""
        # TODO: support events

        if isinstance(key, enum.Enum):
            key = key.value

        assert isinstance(key, str), f"Key must be a string, got {type(key)}. "

        # Return the key if it is in the file
        if key in self.keys():
            return key

        # Add 'data/' prefix if not present
        if "data/" not in key:
            key = "data/" + key

        assert key in self.keys(), (
            f"Key {key} not found in file. Available keys: {list(self['data'].keys())}"
        )

        return key

    def to_iterator(self, key):
        """Convert the data to an iterator over all frames."""
        for frame_idx in range(self.n_frames):
            yield self.load_data(key, frame_idx)

    @staticmethod
    def key_to_data_type(key):
        """Convert the key to a data type."""
        data_type = key.split("/")[-1]
        return data_type

    def load_transmits(self, key, selected_transmits):
        """Load raw_data or aligned_data for a given list of transmits.
        Args:
            data_type (str): The type of data to load. Options are 'raw_data' and 'aligned_data'.
            selected_transmits (list, np.ndarray): The transmits to load.
        """
        key = self.format_key(key)
        data_type = self.key_to_data_type(key)
        assert data_type in ["raw_data", "aligned_data"], (
            f"Cannot load transmits for {data_type}. Only raw_data and aligned_data are supported."
        )
        indices = [slice(None), np.array(selected_transmits)]
        return self.load_data(key, indices)

    def load_data(self, data_type, indices: str | int | List[int] = "all"):
        """Load data from the file.

        Args:
            data_type (str): The type of data to load. Options are 'raw_data', 'aligned_data',
                'beamformed_data', 'envelope_data', 'image' and 'image_sc'.
            indices (str, int, list, optional): The indices to load. Defaults to "all" in
                which case all frames are loaded. If an int is provided, it will be used
                as a single index. If a list is provided, it will be used as a list of
                indices.
        """
        key = self.format_key(data_type)
        indices = self._prepare_indices(indices)

        if self._simple_index(key):
            data = self[key]
            try:
                data = data[indices]
            except (OSError, IndexError) as exc:
                raise ValueError(
                    f"Invalid indices {indices} for key {key}. {key} has shape {data.shape}."
                ) from exc
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
            "Make sure you are using a zea file. "
            f"Found attributes: {list(self.attrs)}"
        )
        probe_name = self.attrs["probe"]
        return probe_name

    @property
    def description(self):
        """Reads the description from the data file and returns it."""
        assert "description" in self.attrs, (
            "Description not found in file attributes. "
            "Make sure you are using a zea file. "
            f"Found attributes: {list(self.attrs)}"
        )
        description = self.attrs["description"]
        return description

    def get_parameters(self, event=None):
        """Returns a dictionary of parameters to initialize a scan
        object that comes with the file (stored inside datafile).

        If there are no scan parameters in the hdf5 file, returns
        an empty dictionary.

        Args:
            event (int, optional): Event number. When specified, an event structure
                is expected as follows::

                    event_0 / scan
                    event_1 / scan
                    ...

                Defaults to None. In that case no event structure is expected.

        Returns:
            dict: The scan parameters.
        """
        scan_parameters = {}
        if "scan" in self:
            scan_parameters = self.recursively_load_dict_contents_from_group("scan")
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

            scan_parameters = self.recursively_load_dict_contents_from_group(f"event_{event}/scan")
        else:
            log.warning("Could not find scan parameters in file.")

        return scan_parameters

    def get_scan_parameters(self, event=None) -> dict:
        """Returns a dictionary of scan parameters stored in the file."""
        return self.get_parameters(event)

    def scan(self, event=None, safe=True, **kwargs) -> Scan:
        """Returns a Scan object initialized with the parameters from the file.

        Args:
            event (int, optional): Event number. When specified, an event structure
                is expected as follows::

                    event_0 / scan
                    event_1 / scan
                    ...

                Defaults to None. In that case no event structure is expected.
            safe (bool, optional): If True, will only use parameters that are
                defined in the Scan class. If False, will use all parameters
                from the file. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the Scan object.
                These will override the parameters from the file if they are
                present in the file.

        Returns:
            Scan: The scan object.
        """
        return Scan.merge(self.get_scan_parameters(event), kwargs, safe=safe)

    def get_probe_parameters(self, event=None) -> dict:
        """Returns a dictionary of probe parameters to initialize a probe
        object that comes with the file (stored inside datafile).

        Returns:
            dict: The probe parameters.
        """
        file_scan_parameters = self.get_parameters(event)

        probe_parameters = reduce_to_signature(Probe.__init__, file_scan_parameters)
        return probe_parameters

    def probe(self, event=None) -> Probe:
        """Returns a Probe object initialized with the parameters from the file.

        Args:
            event (int, optional): Event number. When specified, an event structure
                is expected as follows::

                    event_0 / scan
                    event_1 / scan
                    ...

                Defaults to None. In that case, no event structure is expected.

        Returns:
            Probe: The probe object.
        """
        probe_parameters_file = self.get_probe_parameters(event)
        return Probe.from_parameters(self.probe_name, probe_parameters_file)

    def recursively_load_dict_contents_from_group(self, path: str) -> dict:
        """Load dict from contents of group

        Values inside the group are converted to numpy arrays
        or primitive types (int, float, str).

        Args:
            path (str): path to group
        Returns:
            dict: dictionary with contents of group
        """
        ans = {}
        for key, item in self[path].items():
            if isinstance(item, h5py.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(path + "/" + key + "/")
        return ans

    @classmethod
    def get_shape(cls, path: str, key: str) -> tuple:
        """Get the shape of a key in a file.

        Args:
            path (str): The path to the file.
            key (str): The key to get the shape of.

        Returns:
            tuple: The shape of the key.
        """
        with cls(path, mode="r") as file:
            return file.shape(key)

    def validate(self):
        """Validate the file structure.

        Returns:
            dict: A dictionary with the validation results.
        """
        return validate_file(file=self)

    def __repr__(self):
        return (
            f"<zea.data.file.File at 0x{id(self):x} "
            f'("{Path(self.filename).name}" mode={self.mode})>'
        )

    def __str__(self):
        return f"zea HDF5 File: '{self.path.name}' (mode={self.mode})"

    def copy_key(self, key: str, dst: "File"):
        """Copy a specific key to another file.

        Will always copy the attributes and the scan data if it exists. Will warn if the key is
        not in this file or if the key already exists in the destination file.

        Args:
            key (str): The key to copy.
            dst (File): The destination file to copy the key to.
        """
        key = self.format_key(key)

        # Copy the key if it does not already exist in the destination file
        if key in dst:
            log.warning(f"Skipping key '{key}' because it already exists in dst file {dst.path}.")
        elif key in self:
            self.copy(key, dst, name=key)
        else:
            log.warning(f"Key '{key}' not found in src file {self.path}. Skipping copy.")

        # Copy attributes from src to dst
        for attr_key, attr_value in self.attrs.items():
            dst[key].attrs[attr_key] = attr_value

        # Copy scan data if requested
        if "scan" in self and "scan" not in dst:
            # Copy the scan data if it exists
            self.copy("scan", dst)

    def summary(self):
        """Print the contents of the file."""
        _print_hdf5_attrs(self)


def load_file(
    path,
    data_type="raw_data",
    indices: str | int | List[int] = "all",
    scan_kwargs: dict = None,
):
    """Loads a zea data files (h5py file).

    Returns the data together with a scan object containing the parameters
    of the acquisition and a probe object containing the parameters of the probe.

    Additionally, it can load a specific subset of frames / transmits.

    # TODO: add support for event

    Args:
        path (str, pathlike): The path to the hdf5 file.
        data_type (str, optional): The type of data to load. Defaults to
            'raw_data'. Other options are 'aligned_data', 'beamformed_data',
            'envelope_data', 'image' and 'image_sc'.
        indices (str, int, list, optional): The indices to load. Defaults to "all" in
            which case all frames are loaded. If an int is provided, it will be used
            as a single index. If a list is provided, it will be used as a list of
            indices.
        scan_kwargs (Config, dict, optional): Additional keyword arguments
            to pass to the Scan object. These will override the parameters from the file
            if they are present in the file. Defaults to None.

    Returns:
        (np.ndarray): The raw data of shape (n_frames, n_tx, n_ax, n_el, n_ch).
        (Scan): A scan object containing the parameters of the acquisition.
        (Probe): A probe object containing the parameters of the probe.
    """
    # Define the additional keyword parameters from the scan object
    if scan_kwargs is None:
        scan_kwargs = {}

    with File(path, mode="r") as file:
        # Load the probe object from the file
        probe = file.probe()

        # Load the desired frames from the file
        data = file.load_data(data_type, indices=indices)

        # extract transmits from indices
        # we only have to do this when the data has a n_tx dimension
        # in that case we also have update scan parameters to match
        # the number of selected transmits
        if data_type in ["raw_data", "aligned_data"]:
            indices = File._prepare_indices(indices)
            if isinstance(indices, tuple) and len(indices) > 1:
                scan_kwargs["selected_transmits"] = indices[1]

        scan = file.scan(**scan_kwargs)

        return data, scan, probe


def _print_hdf5_attrs(hdf5_obj, prefix=""):
    """Recursively prints all keys, attributes, and shapes in an HDF5 file.

    Args:
        hdf5_obj (h5py.File, h5py.Group, h5py.Dataset): HDF5 object to print.
        prefix (str, optional): Prefix to print before each line. This
            parameter is used in internal recursion and should not be supplied
            by the user.
    """
    assert isinstance(hdf5_obj, (h5py.File, h5py.Group, h5py.Dataset)), (
        "ERROR: hdf5_obj must be a File, Group, or Dataset object"
    )

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
            _print_hdf5_attrs(hdf5_obj[key], new_prefix)


def validate_file(path: str = None, file: File = None):
    """Reads the hdf5 file at the given path and validates its structure.

    Provide either the path or the file, but not both.

    Args:
        path (str, pathlike): The path to the hdf5 file.
        file (File): The hdf5 file.

    """
    assert (path is not None) ^ (file is not None), (
        "Provide either the path or the file, but not both."
    )

    if path is not None:
        path = Path(path)
        with File(path, "r") as _file:
            event_structure, num_events = _validate_hdf5_file(_file)
    else:
        event_structure, num_events = _validate_hdf5_file(file)

    return {
        "status": "success",
        "event_structure": event_structure,
        "num_events": num_events,
    }


def _validate_hdf5_file(file: File):
    all_keys = list(file.keys())

    if file.has_events:
        num_events = len(all_keys)
        for event_no in range(num_events):
            assert_key(file, f"event_{event_no}")
            _validate_structure(file[f"event_{event_no}"])
    else:
        num_events = 0
        _validate_structure(file)

    return file.has_events, num_events


def _validate_structure(file: File):
    # Validate the root group
    assert_key(file, "data")

    # Assert file["data"] is a group
    assert isinstance(file["data"], h5py.Group), (
        "The data group is not a group. Please check the file structure. "
        "Maybe this is not a zea file?"
    )

    # Check if there is only image data
    not_only_image_data = len([i for i in _NON_IMAGE_DATA_TYPES if i in file["data"].keys()]) > 0

    # Only check scan group if there is non-image data
    if not_only_image_data:
        assert_key(file, "scan")

        for key in _REQUIRED_SCAN_KEYS:
            assert_key(file["scan"], key)

    # validate the data group
    for key in file["data"].keys():
        assert key in _DATA_TYPES, "The data group contains an unexpected key."

        # Validate data shape
        data_shape = file["data"][key].shape
        if key == "raw_data":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert data_shape[0] == file["scan"]["n_frames"][()], (
                "n_frames does not match the first dimension of raw_data."
            )
            assert data_shape[1] == file["scan"]["n_tx"][()], (
                "n_tx does not match the second dimension of raw_data."
            )
            assert data_shape[2] == file["scan"]["n_ax"][()], (
                "n_ax does not match the third dimension of raw_data."
            )
            assert data_shape[3] == file["scan"]["n_el"][()], (
                "n_el does not match the fourth dimension of raw_data."
            )
        elif key == "aligned_data":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert data_shape[0] == file["scan"]["n_frames"][()], (
                "n_frames does not match the first dimension of aligned_data."
            )
        elif key == "beamformed_data":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert data_shape[0] == file["scan"]["n_frames"][()], (
                "n_frames does not match the first dimension of beamformed_data."
            )
        elif key == "envelope_data":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert data_shape[0] == file["scan"]["n_frames"][()], (
                "n_frames does not match the first dimension of envelope_data."
            )
        elif key == "image":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert data_shape[0] == file["scan"]["n_frames"][()], (
                "n_frames does not match the first dimension of image."
            )
        elif key == "image_sc":
            get_check(key)(shape=data_shape, with_batch_dim=True)
            assert data_shape[0] == file["scan"]["n_frames"][()], (
                "n_frames does not match the first dimension of image_sc."
            )

    if not_only_image_data:
        _assert_scan_keys_present(file)

    _assert_unit_and_description_present(file)


def _assert_scan_keys_present(file: File):
    """Ensure that all required keys are present.

    Args:
        file (h5py.File): The file instance to check.

    Raises:
        AssertionError: If a required key is missing or does not have the right shape.
    """
    for required_key in _REQUIRED_SCAN_KEYS:
        assert required_key in file["scan"].keys(), (
            f"The scan group does not contain the required key {required_key}."
        )

    # Ensure that all keys have the correct shape
    for key in file["scan"].keys():
        if isinstance(file["scan"][key], h5py.Group):
            shape_file = None
        else:
            shape_file = file["scan"][key].shape

        if key == "probe_geometry":
            correct_shape = (file["scan"]["n_el"][()], 3)

        elif key == "t0_delays":
            correct_shape = (
                file["scan"]["n_tx"][()],
                file["scan"]["n_el"][()],
            )
        elif key == "tx_apodizations":
            correct_shape = (
                file["scan"]["n_tx"][()],
                file["scan"]["n_el"][()],
            )

        elif key == "focus_distances":
            correct_shape = (file["scan"]["n_tx"][()],)

        elif key == "polar_angles":
            correct_shape = (file["scan"]["n_tx"][()],)

        elif key == "azimuth_angles":
            correct_shape = (file["scan"]["n_tx"][()],)

        elif key == "initial_times":
            correct_shape = (file["scan"]["n_tx"][()],)

        elif key == "time_to_next_transmit":
            correct_shape = (
                file["scan"]["n_frames"][()],
                file["scan"]["n_tx"][()],
            )
        elif key == "tgc_gain_curve":
            correct_shape = (file["scan"]["n_ax"][()],)
        elif key == "tx_waveform_indices":
            correct_shape = (file["scan"]["n_tx"][()],)
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
            shape_file = file["scan"][key].shape

        else:
            correct_shape = None
            log.warning(f"No validation has been defined for {log.orange(key)}.")

        if correct_shape is not None:
            assert shape_file == correct_shape, (
                f"`{key}` does not have the correct shape. "
                f"Expected shape: {correct_shape}, got shape: {shape_file}"
            )


def _assert_unit_and_description_present(hdf5_file, _prefix=""):
    """Checks that all keys have a unit and description attribute.

    Args:
        hdf5_file (h5py.File): The hdf5 file to check.

    Raises:
        AssertionError: If a file does not have a unit or description attribute.
    """
    for key in hdf5_file.keys():
        if isinstance(hdf5_file[key], h5py.Group):
            _assert_unit_and_description_present(hdf5_file[key], _prefix=_prefix + key + "/")
        else:
            assert "unit" in hdf5_file[key].attrs.keys(), (
                f"The file {_prefix}/{key} does not have a unit attribute."
            )
            assert "description" in hdf5_file[key].attrs.keys(), (
                f"The file {_prefix}/{key} does not have a description attribute."
            )
