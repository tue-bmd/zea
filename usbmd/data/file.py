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
from usbmd.utils.checks import _DATA_TYPES, get_check


def get_shape_hdf5_file(filepath, key):
    """Retrieve the shape of some key in a hdf5 file."""
    with File(filepath, mode="r") as f:
        return f.shape(key)


class File(h5py.File):
    """h5py.File in usbmd format."""

    def __init__(self, *args, **kwargs):
        if not "locking" in kwargs:
            kwargs["locking"] = False
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(self, dataset_folder, file_path, user, **kwargs):
        """Create a File object from a config file.

        Args:
            dataset_folder (Path): Path to the dataset folder.
            file_path (Path): Path to the file.
            dtype (str): Data type of the file.
            **kwargs: Additional arguments for h5py.File.

        Returns:
            File: The File object.
        """

        data_root = user.data_root
        path = data_root / dataset_folder / file_path

        return self(path)

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
                int(self.file[event]["scan"]["n_frames"][()])
                for event in self.file.keys()
            )

    def get_event_shapes(self, key):
        for event_key in self.event_keys:
            yield self[event_key][key].shape

    def events_have_same_shape(self, key):
        """Check if all events have the same shape for a given key."""
        if not self.has_events:
            return True

        shapes = list(self.get_event_shapes(key))
        return len(np.unique(shapes)) == 1

    def _check_key(self, key):
        """Check if key is in file."""
        if key not in self.keys():
            raise KeyError(f"{key} not found in file")

    def _simple_index(self, key):
        return not self.has_events or "event" in key

    def shape(self, key):
        """Return shape of some key, or all events."""

        self._check_key(key)

        if self._simple_index(key):
            return list(self[key].shape)
        else:
            raise NotImplementedError

    @staticmethod
    def _prepare_indices(indices):
        if indices == "all":
            return slice(None)

        if isinstance(indices, int):
            return indices

        processed_indices = tuple(
            list(idx) if isinstance(idx, range) else idx for idx in indices
        )
        return processed_indices

    def load_scan(self, event=None):
        """Alias for get_scan_parameters."""
        return self.get_scan_parameters(event)

    @staticmethod
    def check_data(data, key):
        if key in _DATA_TYPES:
            get_check(key)(data, with_batch_dim=None)

    def load_data(self, dtype, indices: str | int | List[int] = "all"):
        key = "data/" + dtype
        self._check_key(key)
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
        scan_parameters = self.get_scan_parameters(event)
        return Scan(**scan_parameters)

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
