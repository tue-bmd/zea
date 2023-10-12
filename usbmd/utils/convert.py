"""This module contains functionality for converting between hdf5, mat and
dictionary.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 3th, 2022
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

from usbmd.utils.io import filename_from_window_dialog


def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != "float64":
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)


def save_dict_to_file(filename, dic):
    """Save dict to .mat or .hdf5"""

    filetype = Path(filename).suffix
    assert filetype in [".mat", ".hdf5"]

    if filetype == ".hdf5":
        with h5py.File(filename, "w") as h5file:
            recursively_save_dict_contents_to_group(h5file, "/", dic)
    elif filetype == ".mat":
        sio.savemat(filename, dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """Save dict contents to group"""
    for key, item in dic.items():
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        else:
            h5file[path + key] = item


def load_dict_from_file(filename, squeeze=True):
    """dict from file"""
    filetype = Path(filename).suffix
    assert filetype in [".mat", ".hdf5"]

    v_7_3 = False
    if filetype == ".mat":
        try:
            return load_mat(filename)
        except:
            v_7_3 = True

    if (filetype == ".hdf5") or (v_7_3 is True):
        with h5py.File(filename, "r") as h5file:
            return recursively_load_dict_contents_from_group(h5file, "/", squeeze)


def recursively_load_dict_contents_from_group(h5file, path, squeeze=True):
    """Load dict from contents of group"""
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            if squeeze:
                ans[key] = np.squeeze(item[()])
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans


def strip_matfile_keys(dic):
    """Strip some unecessary .mat keys"""
    new_dic = {}
    for key, val in dic.items():
        if key not in ["__globals__", "__header__", "__version__"]:
            new_dic[key] = np.squeeze(val)
    return new_dic


def get_args():
    """Command line argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="hdf5 file or mat file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    data = {}
    if args.file is None:
        file = filename_from_window_dialog(
            "Choose .mat or .hdf5 file",
            filetypes=(("mat or hdf5", "*.mat *.hdf5 *.h5"),),
        )
    else:
        file = Path(args.file)

    dic = load_dict_from_file(file)
    if file.suffix == ".mat":
        dic = strip_matfile_keys(dic)
        save_dict_to_file(file.with_suffix(".hdf5"), dic)

    elif file.suffix in [".h5", ".hdf5"]:
        save_dict_to_file(file.with_suffix(".mat"), dic)

    print(f"Succesfully converted {file}")
