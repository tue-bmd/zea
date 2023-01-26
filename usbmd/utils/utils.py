"""Utility functions
Author(s): Tristan Stevens
"""
import os
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image


def filename_from_window_dialog(window_name=None, filetypes=None, initialdir=None):
    """ Get filename through dialog window
    Args:
        window_name: string with name of window
        filetypes: tuple of tuples containing (name, filetypes)
            example:
                (('mat or hdf5 or whatever you want', '*.mat *.hdf5 *'), (ckpt, *.ckpt))
        initialdir: path to directory where window will start
    Returns:
        filename: string containing path to selected file
    """
    if filetypes is None:
        filetypes = (('all files', '*.*'),)

    root = Tk()
    # open in foreground
    root.wm_attributes('-topmost', 1)
    # we don't want a full GUI, so keep the root window from appearing
    root.withdraw()
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename(
        parent=root,
        title=window_name,
        filetypes=filetypes,
        initialdir=initialdir,
    )
    # check whether a file was selected
    if filename:
        return Path(filename)
    else:
        raise Exception('No file selected.')

def translate(array, range_from, range_to):
    """ Map values in array from one range to other.

    Args:
        array (ndarray): input array.
        range_from (Tuple): lower and upper bound of original array.
        range_to (Tuple): lower and upper bound to which array should be mapped.

    Returns:
        (ndarray): translated array
    """
    left_min, left_max = range_from
    right_min, right_max = range_to
    if left_min == left_max:
        return np.ones_like(array) * right_max

    # Convert the left range into a 0-1 range (float)
    value_scaled = (array - left_min) / (left_max - left_min)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * (right_max - right_min))

def search_file_tree(directory, filetypes=None, write=True):
    """Lists all files in directory and sub-directories.

    If file_paths.txt is detected in the directory, that file is read and used.

    Args:
        directory (str): path to directory.
        filetypes (Tuple of strings, optional): filetypes.
            Defaults to image types (.png etc.).
        write (bool, optional): Whether to write to file. Useful has searching
            the tree takes quite a while. Defaults to True.

    Returns:
        file_paths (List): List with str to all file paths.

    """
    directory = Path(directory)
    if (directory / 'file_paths.txt').is_file():
        print('Using pregenerated txt file in the following directory for reading file paths: ')
        print(directory)
        with open(directory / 'file_paths.txt', encoding='utf-8') as file:
            file_paths = file.read().splitlines()
        return file_paths

    # set default file type
    if filetypes is None:
        filetypes = ('jpg', 'jpeg', 'JPEG', 'png', 'PNG')

    file_paths = []

    # Traverse file tree to index all dicom files
    for dirpath, _, filenames in tqdm.tqdm(os.walk(directory), desc='Searching file tree'):
        for file in filenames:
            # Append to file_paths if it is a filetype file
            if file.endswith(filetypes):
                file_paths.append(str(Path(dirpath)/Path(file)))

    print(f'\nFound {len(file_paths)} image files in .\\{Path(directory)}\n')
    assert len(file_paths) > 0, 'ERROR: No image files were found'

    if write:
        with open(directory / 'file_paths.txt', 'w', encoding='utf-8') as file:
            file_paths = [file + '\n' for file in file_paths]
            file.writelines(file_paths)

    return file_paths

def find_key(dictionary, contains, case_sensitive=False):
    """Find key in dictionary that contains partly the string `contains`

    Args:
        dictionary (dict): Dictionary to find key in.
        contains (str): String which the key should .
        case_sensitive (bool, optional): Whether the search is case sensitive.
            Defaults to False.

    Returns:
        str: the key of the dictionary that contains the query string.

    """
    if case_sensitive:
        key = [k for k in dictionary.keys() if contains in k]
    else:
        key = [k for k in dictionary.keys() if contains in k.lower()]
    return key[0]

def plt_window_has_been_closed(fig):
    """Checks whether matplotlib plot window is closed"""
    return not plt.fignum_exists(fig.number)

def print_clear_line():
    """Clears line. Helpful when printing in a loop on the same line."""
    line_up = '\033[1A'
    line_clear = '\x1b[2K'
    print(line_up, end=line_clear)

def to_image(image, value_range: tuple=None, pillow: bool=True):
    """Convert numpy array to uint8 image format.

    Args:
        image (ndarray): input array image
        value_range (tuple, optional): assumed range of input data.
            Defaults to None.
        pillow (bool, optional): whether to convert the image
            array to pillow object. Defaults to True.

    Returns:
        image: output image array uint8 [0, 255]
            (pillow if set to True)
    """
    if value_range:
        image = translate(
            np.clip(image, *value_range), value_range, (0, 255)
        )

    image = image.astype(np.uint8)
    if pillow:
        image = Image.fromarray(image)
    return image

def strtobool(val: str):
    """Convert a string representation of truth to True or False.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f'invalid truth value {val}')

def save_to_gif(images, filename, fps=20):
    """ Saves a sequence of images to .gif file.
    Args:
        images: list of images (numpy arrays).
        filename: string containing filename to which data should be written.
        fps: frames per second of rendered format.
    """
    duration = 1 / (fps) * 1000 # milliseconds per frame

    # convert grayscale images to RGB
    if len(images[0].shape) == 2:
        images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images]

    pillow_img, *pillow_imgs = [
        Image.fromarray(img) for img in images
    ]

    pillow_img.save(
        fp=filename, format='GIF', append_images=pillow_imgs, save_all=True,
        loop=0, duration=duration, interlace=False, optimize=False,
    )
    return print(f'Succesfully saved GIF to -> {filename}')
