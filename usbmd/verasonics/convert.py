"""Convert verasonics files to usbmd format
Author(s): Tristan Stevens
"""
from pathlib import Path

import numpy as np

from usbmd.utils.convert import get_args, save_dict_to_file
from usbmd.utils.read_h5 import ReadH5
from usbmd.utils.utils import filename_from_window_dialog


def save_to_usbmd_format(filename):
    """Save mat / hdf5 file to usbmd format (hdf5).

    Args:
        filename (str): path to .mat / .hdf5 file
    """
    filename = Path(filename)
    file = ReadH5(filename)

    key = list(file.keys())[-1]

    data = file[:]
    data = data[0][1]
    N_frames, N_el, _ = data.shape

    # hardcoded for now
    N_ax = 8192
    N_planes = 101

    data = np.reshape(data, (N_frames, N_el, N_planes, N_ax))
    data = np.transpose(data, (0, 2, 3, 1))

    dic = {
        'data/' + key: data,
    }
    save_dict_to_file(filename.with_suffix('.hdf5'), dic)

if __name__ == '__main__':
    args = get_args()
    data = {}
    if args.file is None:
        file = filename_from_window_dialog(
            'Choose .mat or .hdf5 file',
            filetypes=(
                ('mat or hdf5', '*.mat *.hdf5 *.h5'),
            ),
        )
    else:
        file = Path(args.file)

    save_to_usbmd_format(file)
