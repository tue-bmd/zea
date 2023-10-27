"""Utility functions

- **Author(s)**     : Tristan Stevens
- **Date**          : October 25th, 2022
"""
import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def translate(array, range_from, range_to):
    """Map values in array from one range to other.

    Args:
        array (ndarray): input array.
        range_from (Tuple): lower and upper bound of original array.
        range_to (Tuple): lower and upper bound to which array should be mapped.

    Returns:
        (ndarray): translated array
    """
    left_min, left_max = range_from
    right_min, right_max = range_to
    assert left_min <= left_max, "boundaries are set incorrectly"
    assert right_min < right_max, "boundaries are set incorrectly"
    if left_min == left_max:
        return np.ones_like(array) * right_max

    # Convert the left range into a 0-1 range (float)
    value_scaled = (array - left_min) / (left_max - left_min)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * (right_max - right_min))


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
    line_up = "\033[1A"
    line_clear = "\x1b[2K"
    print(line_up, end=line_clear)


def to_image(image, value_range: tuple = None, pillow: bool = True):
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
        image = translate(np.clip(image, *value_range), value_range, (0, 255))

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
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


def save_to_gif(images, filename, fps=20):
    """Saves a sequence of images to .gif file.
    Args:
        images: list of images (numpy arrays).
        filename: string containing filename to which data should be written.
        fps: frames per second of rendered format.
    """
    duration = 1 / (fps) * 1000  # milliseconds per frame

    # convert grayscale images to RGB
    if len(images[0].shape) == 2:
        images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images]

    pillow_img, *pillow_imgs = [Image.fromarray(img) for img in images]

    pillow_img.save(
        fp=filename,
        format="GIF",
        append_images=pillow_imgs,
        save_all=True,
        loop=0,
        duration=duration,
        interlace=False,
        optimize=False,
    )
    return print(f"Succesfully saved GIF to -> {filename}")


def update_dictionary(dict1: dict, dict2: dict, keep_none: bool = False) -> dict:
    """Updates dict1 with values dict2

    Args:
        dict1 (dict): base dictionary
        dict2 (dict): update dictionary
        keep_none (bool, optional): whether to keep keys
            with None values in dict2. Defaults to False.

    Returns:
        dict: updated dictionary
    """
    if not keep_none:
        dict2 = {k: v for k, v in dict2.items() if v is not None}
    # dict merging python > 3.9: default_scan_params | config_scan_params
    dict_out = {**dict1, **dict2}
    return dict_out


def get_date_string(string: str = None):
    """Generate a date string for current time, according to format specified by `string`."""
    now = datetime.datetime.now()
    if string is None:
        string = "%Y_%m_%d_%H%M%S"

    date_str = now.strftime(string)
    return date_str


def find_first_nonzero_index(arr, axis, invalid_val=-1):
    """
    Find the index of the first non-zero element along a specified axis in a NumPy array.

    Args:
        arr (numpy.ndarray): The input array to search for the first non-zero element.
        axis (int): The axis along which to perform the search.
        invalid_val (int, optional): The value to assign to elements where no
            non-zero values are found along the axis.

    Returns:
        numpy.ndarray: An array of indices where the first non-zero element
            occurs along the specified axis. Elements with no non-zero values along
            the axis are replaced with the 'invalid_val'.

    """
    nonzero_mask = arr != 0
    return np.where(
        nonzero_mask.any(axis=axis), nonzero_mask.argmax(axis=axis), invalid_val
    )
