"""Read functionality.

- **Author(s)**     : Tristan Stevens
- **Date**          : -
"""

import sys
from pathlib import Path

import h5py
import numpy as np

from usbmd.utils import log


class ReadH5:
    """Read H5 files object class"""

    def __init__(self, file_path):
        """Open a .h5 file for reading.

        Args:
            file_path :  path to the .h5 HDF5 file

        """
        self.file_path = Path(file_path)
        self.file = None

    def open(self):
        """Open the .hdf5 HDF5 file for reading."""
        try:
            self.file = h5py.File(self.file_path, "r")
        except Exception as e:
            if "Unable to open file" in str(e):
                log.error(
                    f"Unable to open file {self.file_path}. It may be locked by another process."
                )
                sys.exit(1)
            elif "No such file or directory" in str(e):
                log.error(f"File {self.file_path} not found.")
                sys.exit(1)
            else:
                raise e
        return self.file

    def close(self):
        """Close the .hdf5 HDF5 file after reading."""
        self.file.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_extension(self):
        """Get file extension

        Returns:
            str: extension.

        """
        return self.file_path.suffix

    def __getitem__(self, i, keys=None):
        if keys is None:
            return self._get(i=i, group=self.file)
        else:
            return self._get_from_keys(i=i, keys=keys, group=self.file)

    @staticmethod
    def _get_from_keys(i, keys, group):
        alist = []
        for key in keys:
            alist.append(group[key][i])
        return alist

    def get_all(self):
        """Get all data (for all indices) in file"""
        return self._get(i=None, group=self.file)

    def _get(self, i=None, group=None):
        alist = []
        for key in group.keys():
            sub_group = group.get(key)
            if isinstance(sub_group, h5py.Group):
                output = self._get(i, sub_group)
            elif isinstance(sub_group, h5py.Dataset):
                if i is None:
                    # get all using ':'
                    output = sub_group[:]
                else:
                    output = sub_group[i]
            else:
                raise ValueError(f"{type(group)}")
            alist.append(output)
        return alist

    def keys(self):
        """Return all keys in the hdf5 object.

        Returns:
            keys (list with strings): keys.

        """
        return self.file.keys()

    def summary(self):
        """Summary of the hdf5 object"""
        print_hdf5_attrs(self.file)

    def __len__(self):
        key = self.get_largest_group_name()
        if key is None:
            return 0
        return len(self.file[key])

    @property
    def shape(self):
        """Return shape of largest group in dataset"""
        key = self.get_largest_group_name()
        return self.file[key].shape

    def get_largest_group_name(self):
        """Returns key which contains a value with most number of elements.

        Usefull when the key is different in each data file, but you would
        like to retrieve the main data and not the metadata.

        Returns:
            key_name (str): key name.

        """
        group_info = []

        def visit_func(name, node):
            if isinstance(node, h5py.Dataset):
                n_elements = np.prod(np.array(node.shape, dtype=np.float64))
                group_info.append((name, n_elements))

        self.file.visititems(visit_func)
        if not group_info:
            log.warning("hdf5 file does not contain any datasets")
            return None
        idx = np.argmax([gi[1] for gi in group_info])
        key_name, _ = group_info[idx]
        return key_name


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
                h5file, path + key + "/"
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
