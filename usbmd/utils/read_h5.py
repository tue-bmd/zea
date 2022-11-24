"""Read functionality
Author(s): Tristan Stevens
"""
from pathlib import Path

import h5py
import numpy as np


class ReadH5:
    """Read H5 files object class"""
    def __init__(self, file_path):
        """Open a .h5 file for reading.

        Args:
            file_path :  path to the .h5 HDF5 file

        """
        self.file_path = Path(file_path)
        self.h5f = h5py.File(self.file_path, 'r')

    def get_extension(self):
        """Get file extension

        Returns:
            str: extension.

        """
        return self.file_path.suffix

    def __getitem__(self, i, keys=None):
        if keys is None:
            return self._get(i=i, group=self.h5f)
        else:
            return self._get_from_keys(i=i, keys=keys, group=self.h5f)

    @staticmethod
    def _get_from_keys(i, keys, group):
        alist = []
        for key in keys:
            alist.append(group[key][i])
        return alist

    def get_all(self):
        """Get all data (for all indices) in h5f"""
        return self._get(i=None, group=self.h5f)

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
                raise ValueError(f'{type(group)}')
            alist.append(output)
        return alist

    def keys(self):
        """Return all keys in the hdf5 object.

        Returns:
            keys (list with strings): keys.

        """
        return self.h5f.keys()

    def summary(self):
        """Summary of the hdf5 object"""
        self.h5f.visititems(print)
        # self.h5f.visititems(self._visit_func)

    @staticmethod
    def _visit_func(_, node):
        print(f'{node.name}: ')

    @staticmethod
    def frame_as_first(frames):
        """permute the dataset to have the frame indices as the first dimension

        Args:
            frames (ndarray): array with frame indices as last dimension

        Returns:
            frames (ndarray): array of shape num_frames x ....
        """

        # always start with frame dim:
        last_dim = len(np.shape(frames)) - 1
        order = (last_dim,) + tuple(np.arange(0, last_dim))
        frames = np.array(frames).transpose(order)
        return frames

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

        self.h5f.visititems(visit_func)
        idx = np.argmax([gi[1] for gi in group_info])
        key_name, _ = group_info[idx]
        return key_name

    def __len__(self):
        key = self.get_largest_group_name()
        return len(self.h5f[key])

    @property
    def shape(self):
        """Return shape of largest group in dataset"""
        key = self.get_largest_group_name()
        return self.h5f[key].shape

    def close(self):
        """Close the .hdf5 HDF5 file for reading.

        Returns:
            void
        """
        self.h5f.close()
