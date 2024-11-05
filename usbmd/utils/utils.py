"""Utility functions

- **Author(s)**     : Tristan Stevens
- **Date**          : October 25th, 2022
"""

import datetime
import functools
import hashlib
import inspect
import platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from usbmd.utils import log
from usbmd.utils.checks import _assert_uint8_images


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

    Raises:
        TypeError: if not all keys are strings.
        KeyError: if no key is found containing the query string.
    """
    # Assert that all keys are strings
    if not all(isinstance(k, str) for k in dictionary.keys()):
        raise TypeError("All keys must be strings.")

    if case_sensitive:
        key = [k for k in dictionary.keys() if contains in k]
    else:
        key = [k for k in dictionary.keys() if contains in k.lower()]

    if len(key) == 0:
        raise KeyError(f"Key containing '{contains}' not found in dictionary.")

    return key[0]


def print_clear_line():
    """Clears line. Helpful when printing in a loop on the same line."""
    line_up = "\033[1A"
    line_clear = "\x1b[2K"
    print(line_up, end=line_clear)


def strtobool(val: str):
    """Convert a string representation of truth to True or False.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    assert isinstance(val, str), f"Input value must be a string, not {type(val)}"
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


def grayscale_to_rgb(image):
    """Converts a grayscale image to an RGB image.

    Args:
        image (ndarray): Grayscale image. Must have shape (height, width).

    Returns:
        ndarray: RGB image.
    """
    assert image.ndim == 2, "Input image must be grayscale."
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def preprocess_for_saving(images):
    """Preprocesses images for saving to GIF or MP4.

    Args:
        images (ndarray, list[ndarray]): Images. Must have shape (n_frames, height, width, channels)
            or (n_frames, height, width).
    """
    images = np.array(images)
    _assert_uint8_images(images)

    # Remove channel axis if it is 1 (grayscale image)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    # convert grayscale images to RGB
    if images.ndim == 3:
        images = [grayscale_to_rgb(image) for image in images]
        images = np.array(images)

    return images


def save_to_gif(images, filename, fps=20):
    """Saves a sequence of images to .gif file.
    Args:
        images: list of images (numpy arrays). Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename: string containing filename to which data should be written.
        fps: frames per second of rendered format.
    """
    assert isinstance(
        filename, (str, Path)
    ), f"Filename must be a string or Path object, not {type(filename)}"
    images = preprocess_for_saving(images)

    if fps > 50:
        log.warning(f"Cannot set fps ({fps}) > 50. Setting it automatically to 50.")
        fps = 50

    duration = 1 / (fps) * 1000  # milliseconds per frame

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
    return log.success(f"Succesfully saved GIF to -> {log.yellow(filename)}")


def save_to_mp4(images, filename, fps=20):
    """Saves a sequence of images to .mp4 file.
    Args:
        images: list of images (numpy arrays). Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename: string containing filename to which data should be written.
        fps: frames per second of rendered format.
    """
    assert isinstance(
        filename, (str, Path)
    ), f"Filename must be a string or Path object, not {type(filename)}"
    images = preprocess_for_saving(images)

    filename = str(filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _, height, width, _ = images.shape
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for image in images:
        video_writer.write(image)

    video_writer.release()
    return log.success(f"Successfully saved MP4 to -> {filename}")


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
    """Generate a date string for current time, according to format specified by
    `string`. Refer to the documentation of the datetime module for more information
    on the formatting options.

    If no string is specified, the default format is used: "%Y_%m_%d_%H%M%S".
    """
    if string is not None and not isinstance(string, str):
        raise TypeError("Input must be a string.")

    # Get the current time
    now = datetime.datetime.now()

    # If no string is specified, use the default format
    if string is None:
        string = "%Y_%m_%d_%H%M%S"

    # Generate the date string
    date_str = now.strftime(string)

    return date_str


def date_string_to_readable(date_string: str, include_time: bool = False):
    """Converts a date string to a more readable format.

    Args:
        date_string (str): The input date string.
        include_time (bool, optional): Whether to include the time in the output.
            Defaults to False.

    Returns:
        str: The date string in a more readable format.
    """
    date = datetime.datetime.strptime(date_string, "%Y_%m_%d_%H%M%S")
    if include_time:
        return date.strftime("%B %d, %Y %I:%M %p")
    else:
        return date.strftime("%B %d, %Y")


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


def first_not_none_item(arr):
    """
    Finds and returns the first non-None item in the given array.

    Args:
        arr (list): The input array.

    Returns:
        The first non-None item found in the array, or None if no such item exists.
    """
    non_none_items = [item for item in arr if item is not None]
    return non_none_items[0] if non_none_items else None


def deprecated(replacement=None):
    """Decorator to mark a function, method, or attribute as deprecated.

    Args:
        replacement (str, optional): The name of the replacement function, method, or attribute.

    Returns:
        callable: The decorated function, method, or property.

    Raises:
        DeprecationWarning: A warning is issued when the deprecated item is called or accessed.

    Example:
        >>> class MyClass:
        ...     @deprecated(replacement="new_method")
        ...     def old_method(self):
        ...         print("This is the old method.")
        ...
        ...     @deprecated(replacement="new_attribute")
        ...     def __init__(self):
        ...         self._old_attribute = "Old value"
        ...
        ...     @deprecated(replacement="new_property")
        ...     @property
        ...     def old_property(self):
        ...         return self._old_attribute
        ...

        >>> # Using the deprecated method
        >>> obj = MyClass()
        >>> obj.old_method()
        This is the old method.

        >>> # Accessing the deprecated attribute
        >>> print(obj.old_property)
        Old value

        >>> # Setting value to the deprecated attribute
        >>> obj.old_property = "New value"
        __main__:28: DeprecationWarning: Access to deprecated attribute old_property.
        __main__:29: DeprecationWarning: Use new_property instead.
        __main__:32: DeprecationWarning: Setting value to deprecated attribute old_property.
        __main__:33: DeprecationWarning: Use new_property instead.
    """

    def decorator(item):
        if callable(item):
            # If it's a function or method
            @functools.wraps(item)
            def wrapper(*args, **kwargs):
                if replacement:
                    log.deprecated(
                        f"Call to deprecated {item.__name__}."
                        f" Use {replacement} instead."
                    )
                else:
                    log.deprecated(f"Call to deprecated {item.__name__}.")
                return item(*args, **kwargs)

            return wrapper
        elif isinstance(item, property):
            # If it's a property of a class
            def getter(self):
                if replacement:
                    log.deprecated(
                        f"Access to deprecated attribute {item.fget.__name__}, "
                        f"use {replacement} instead."
                    )
                else:
                    log.deprecated(
                        f"Access to deprecated attribute {item.fget.__name__}."
                    )
                return item.fget(self)

            def setter(self, value):
                if replacement:
                    log.deprecated(
                        f"Setting value to deprecated attribute {item.fget.__name__}, "
                        f"use {replacement} instead."
                    )
                else:
                    log.deprecated(
                        f"Setting value to deprecated attribute {item.fget.__name__}."
                    )
                item.fset(self, value)

            def deleter(self):
                if replacement:
                    log.deprecated(
                        f"Deleting deprecated attribute {item.fget.__name__}, "
                        f"use {replacement} instead."
                    )
                else:
                    log.deprecated(
                        f"Deleting deprecated attribute {item.fget.__name__}."
                    )
                item.fdel(self)

            return property(getter, setter, deleter)

        else:
            raise TypeError(
                "Decorator can only be applied to functions, methods, or properties."
            )

    return decorator


def calculate_file_hash(file_path, omit_line_str=None):
    """Calculates the hash of a file.
    Args:
        file_path (str): Path to file.
        omit_line_str (str): If this string is found in a line, the line will
            be omitted when calculating the hash. This is useful for example
            when the file contains the hash itself.
    Returns:
        str: The hash of the file.
    """
    hash_object = hashlib.sha256()
    with open(file_path, "rb") as f:
        for line in f:
            if omit_line_str is not None and omit_line_str in str(line):
                continue
            hash_object.update(line)
    return hash_object.hexdigest()


def check_architecture():
    """Checks the architecture of the system."""
    return platform.uname()[-1]


def get_function_args(func):
    """Get the names of the arguments of a function."""
    sig = inspect.signature(func)
    return tuple(sig.parameters)


def keep_trying(fn, args=None, required_set=None):
    """Keep trying to run a function until it succeeds.
    Args:
        fn (function): function to run
        args (dict, optional): arguments to pass to function
        required_set (set, optional): set of required outputs
            if output is not in required_set, function will be rerun
    """
    while True:
        try:
            out = fn(**args) if args is not None else fn()
            if required_set is not None:
                assert out is not None
                assert out in required_set, f"Output {out} not in {required_set}"
            return out
        except Exception as e:
            print(e)


def safe_initialize_class(cls, **kwargs):
    """Safely initialize a class by removing any invalid arguments."""
    # Retrieve the argument names of the Scan class
    sig = inspect.signature(cls.__init__)

    # Filter out the arguments that are not part of the Scan class
    reduced_params = {key: kwargs[key] for key in sig.parameters if key in kwargs}

    return cls(**reduced_params)
