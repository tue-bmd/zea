"""
Convert utilities. Converting between h5, mat and dictionary.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from usbmd.utils.utils import filename_from_window_dialog

parser = argparse.ArgumentParser()
parser.add_argument('--file', default=None, help='h5 file or mat file')
args = parser.parse_args()


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
        if ndarray.dtype != 'float64':
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
    """Save dict to .mat or .h5"""

    filetype = Path(filename).suffix
    assert filetype in ['.mat', '.h5']

    if filetype == '.h5':
        with h5py.File(filename, 'w') as h5file:
            recursively_save_dict_contents_to_group(h5file, '/', dic)
    elif filetype == '.mat':
        sio.savemat(filename, dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """Save dict contents to group"""
    for key, item in dic.items():
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            h5file[path + key] = item

def load_dict_from_file(filename, squeeze=True):
    """dict from file"""
    filetype = Path(filename).suffix
    assert filetype in ['.mat', '.h5']

    v_7_3 = False
    if filetype == '.mat':
        try:
            return load_mat(filename)
        except:
            v_7_3 = True

    if (filetype == '.h5') or (v_7_3 is True):
        with h5py.File(filename, 'r') as h5file:
            return recursively_load_dict_contents_from_group(h5file, '/', squeeze)

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
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

if __name__ == '__main__':
    data = {}
    if args.file is None:
        file = filename_from_window_dialog(
            'Choose .mat or .h5 file',
            filetypes=(
                ('mat or h5', '*.mat *.hdf5 *.h5'),
            ),
        )
    else:
        file = Path(args.file)

    dic = load_dict_from_file(file)
    if file.suffix == '.mat':
        save_dict_to_file(dic, file.with_suffix('.h5'))

    elif (file.suffix == '.h5') or (file.suffix == '.hdf5'):
        save_dict_to_file(dic, file.with_suffix('.mat'))

    # if (file.suffix == '.h5') or (file.suffix == '.hdf5'):
    #     with h5py.File(file) as fd:
    #         for i in fd.keys():
    #             data[i] = fd[i][...]
    #     sio.savemat(file.with_suffix('.mat'), data)
    # elif (file.suffix == '.mat'):
    #     try:
    #         data = sio.loadmat(file)
    #     except:
    #         with h5py.File(file, 'r') as fd:
    #             data = fd[...]
    #     with h5py.File(file.with_suffix('.h5'), 'w') as fd:
    #         for i in data.keys():
    #             if i not in ['__globals__',  '__header__', '__version__']:
    #                 fd[i] = np.squeeze(data[i])
    # else:
    #     raise ValueError('filename must ends with .h5, .hdf5 or .mat')

    print(f'Succesfully converted {file}')
