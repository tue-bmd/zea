"""General utility functions."""

import collections.abc
import datetime
import functools
import hashlib
import inspect
import platform
import time
from functools import wraps
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import yaml
from keras import ops
from PIL import Image

from zea import log


def _assert_uint8_images(images: np.ndarray):
    """
    Asserts that the input images have the correct properties.

    Args:
        images (np.ndarray): The input images.

    Raises:
        AssertionError: If the dtype of images is not uint8.
        AssertionError: If the shape of images is not (n_frames, height, width, channels)
            or (n_frames, height, width) for grayscale images.
        AssertionError: If images have anything other than 1 (grayscale),
            3 (rgb) or 4 (rgba) channels.
    """
    assert images.dtype == np.uint8, f"dtype of images should be uint8, got {images.dtype}"

    assert images.ndim in (3, 4), (
        "images must have shape (n_frames, height, width, channels),"
        f" or (n_frames, height, width) for grayscale images. Got {images.shape}"
    )

    if images.ndim == 4:
        assert images.shape[-1] in (1, 3, 4), (
            "Grayscale images must have 1 channel, "
            "RGB images must have 3 channels, and RGBA images must have 4 channels. "
            f"Got shape: {images.shape}, channels: {images.shape[-1]}"
        )


def translate(array, range_from=None, range_to=(0, 255)):
    """Map values in array from one range to other.

    Args:
        array (ndarray): input array.
        range_from (Tuple, optional): lower and upper bound of original array.
            Defaults to min and max of array.
        range_to (Tuple, optional): lower and upper bound to which array should be mapped.
            Defaults to (0, 255).

    Returns:
        (ndarray): translated array
    """
    if range_from is None:
        left_min, left_max = ops.min(array), ops.max(array)
    else:
        left_min, left_max = range_from
    right_min, right_max = range_to

    # Convert the left range into a 0-1 range (float)
    value_scaled = (array - left_min) / (left_max - left_min)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * (right_max - right_min))


def map_negative_indices(indices: list, length: int):
    """Maps negative indices for array indexing to positive indices.
    Example:
        >>> from zea.utils import map_negative_indices
        >>> map_negative_indices([-1, -2], 5)
        [4, 3]
    """
    return [i if i >= 0 else length + i for i in indices]


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
    # Stack the grayscale image into 3 channels (RGB)
    return np.stack([image] * 3, axis=-1)


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


def save_to_gif(images, filename, fps=20, shared_color_palette=False):
    """Saves a sequence of images to a GIF file.

    Args:
        images (list or np.ndarray): List or array of images. Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename (str or Path): Filename to which data should be written.
        fps (int): Frames per second of rendered format.
        shared_color_palette (bool, optional): If True, creates a global
            color palette across all frames, ensuring consistent colors
            throughout the GIF. Defaults to False, which is default behavior
            of PIL.Image.save. Note: True can cause slow saving for longer sequences.

    """
    images = preprocess_for_saving(images)

    if fps > 50:
        log.warning(f"Cannot set fps ({fps}) > 50. Setting it automatically to 50.")
        fps = 50

    duration = 1 / (fps) * 1000  # milliseconds per frame

    pillow_imgs = [Image.fromarray(img) for img in images]

    if shared_color_palette:
        # Apply the same palette to all frames without dithering for consistent color mapping
        # Convert all images to RGB and combine their colors for palette generation
        all_colors = np.vstack([np.array(img.convert("RGB")).reshape(-1, 3) for img in pillow_imgs])
        combined_image = Image.fromarray(all_colors.reshape(-1, 1, 3))

        # Generate palette from all frames
        global_palette = combined_image.quantize(
            colors=256,
            method=Image.MEDIANCUT,
            kmeans=1,
        )

        # Apply the same palette to all frames without dithering
        pillow_imgs = [
            img.convert("RGB").quantize(
                palette=global_palette,
                dither=Image.NONE,
            )
            for img in pillow_imgs
        ]

    pillow_img, *pillow_imgs = pillow_imgs

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
    log.success(f"Succesfully saved GIF to -> {log.yellow(filename)}")


def save_to_mp4(images, filename, fps=20):
    """Saves a sequence of images to an MP4 file.

    Args:
        images (list or np.ndarray): List or array of images. Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename (str or Path): Filename to which data should be written.
        fps (int): Frames per second of rendered format.

    Returns:
        str: Success message.

    """
    images = preprocess_for_saving(images)

    filename = str(filename)

    parent_dir = Path(filename).parent
    if not parent_dir.exists():
        raise FileNotFoundError(f"Directory '{parent_dir}' does not exist.")

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to save MP4 files. "
            "Please install it with 'pip install opencv-python' or "
            "'pip install opencv-python-headless'."
        ) from exc

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
    return np.where(nonzero_mask.any(axis=axis), nonzero_mask.argmax(axis=axis), invalid_val)


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
        >>> from zea.utils import deprecated
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

        >>> # Using the deprecated method
        >>> obj = MyClass()
        >>> obj.old_method()
        This is the old method.
        >>> # Accessing the deprecated attribute
        >>> print(obj.old_property)
        Old value
        >>> # Setting value to the deprecated attribute
        >>> obj.old_property = "New value"
    """

    def decorator(item):
        if callable(item):
            # If it's a function or method
            @functools.wraps(item)
            def wrapper(*args, **kwargs):
                if replacement:
                    log.deprecated(
                        f"Call to deprecated {item.__name__}. Use {replacement} instead."
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
                    log.deprecated(f"Access to deprecated attribute {item.fget.__name__}.")
                return item.fget(self)

            def setter(self, value):
                if replacement:
                    log.deprecated(
                        f"Setting value to deprecated attribute {item.fget.__name__}, "
                        f"use {replacement} instead."
                    )
                else:
                    log.deprecated(f"Setting value to deprecated attribute {item.fget.__name__}.")
                item.fset(self, value)

            def deleter(self):
                if replacement:
                    log.deprecated(
                        f"Deleting deprecated attribute {item.fget.__name__}, "
                        f"use {replacement} instead."
                    )
                else:
                    log.deprecated(f"Deleting deprecated attribute {item.fget.__name__}.")
                item.fdel(self)

            return property(getter, setter, deleter)

        else:
            raise TypeError("Decorator can only be applied to functions, methods, or properties.")

    return decorator


def calculate_file_hash(file_path, omit_line_str=None):
    """Calculates the hash of a file.

    Args:
        file_path (str): Path to file.
        omit_line_str (str, optional): If this string is found in a line, the line will
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


def fn_requires_argument(fn, arg_name):
    """Returns True if the function requires the argument 'arg_name'."""
    params = get_function_args(fn)
    return arg_name in params


def find_methods_with_return_type(cls, return_type_hint):
    """
    Find all methods in a class that have the specified return type hint.

    Args:
        cls: The class to inspect.
        return_type_hint: The return type hint to match (as a string).

    Returns:
        A list of method names that match the return type hint.
    """
    matching_methods = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        annotations = getattr(member, "__annotations__", {})
        if annotations.get("return") == return_type_hint:
            matching_methods.append(name)
    return matching_methods


def keep_trying(fn, args=None, required_set=None):
    """Keep trying to run a function until it succeeds.

    Args:
        fn (callable): Function to run.
        args (dict, optional): Arguments to pass to function.
        required_set (set, optional): Set of required outputs.
            If output is not in required_set, function will be rerun.

    Returns:
        Any: The output of the function if successful.

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


def reduce_to_signature(func, kwargs):
    """Reduce the kwargs to the signature of the function."""
    # Retrieve the argument names of the function
    sig = inspect.signature(func)

    # Filter out the arguments that are not part of the function
    reduced_params = {key: kwargs[key] for key in sig.parameters if key in kwargs}

    return reduced_params


def safe_initialize_class(cls, **kwargs):
    """Safely initialize a class by removing any invalid arguments."""

    # Filter out the arguments that are not part of the Scan class
    reduced_params = reduce_to_signature(cls.__init__, kwargs)

    return cls(**reduced_params)


def deep_compare(obj1, obj2):
    """Recursively compare two objects for equality."""
    # Only recurse into dicts
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_compare(obj1[k], obj2[k]) for k in obj1)

    # If not dict, but both are iterable (excluding strings/bytes), compare items
    if (
        isinstance(obj1, collections.abc.Iterable)
        and isinstance(obj2, collections.abc.Iterable)
        and not isinstance(obj1, (str, bytes))
        and not isinstance(obj2, (str, bytes))
    ):
        return all(deep_compare(a, b) for a, b in zip(obj1, obj2))

    # Fallback to direct comparison (also handles int, float, str, etc.)
    return obj1 == obj2


class FunctionTimer:
    """
    A decorator class for timing the execution of functions.

    Example:
        >>> from zea.utils import FunctionTimer
        >>> timer = FunctionTimer()
        >>> my_function = lambda: sum(range(10))
        >>> my_function = timer(my_function)
        >>> _ = my_function()
        >>> print(timer.get_stats("my_function"))
    """

    def __init__(self):
        self.timings = {}
        self.last_append = 0

    def __call__(self, func, name=None):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # Store the timing result
            _name = name if name is not None else func.__name__
            if _name not in self.timings:
                self.timings[_name] = []
            self.timings[_name].append(elapsed_time)

            return result

        return wrapper

    def get_stats(self, func_name, drop_first: bool | int = False):
        """Calculate statistics for the given function."""
        if func_name not in self.timings:
            raise ValueError(f"No timings recorded for function '{func_name}'.")

        if isinstance(drop_first, bool):
            idx = 1 if drop_first else 0
        elif isinstance(drop_first, int):
            idx = drop_first
        else:
            raise ValueError("drop_first must be a boolean or an integer.")

        times = self.timings[func_name][idx:]
        return {
            "mean": mean(times),
            "median": median(times),
            "std_dev": stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "count": len(times),
        }

    def export_to_yaml(self, filename):
        """Export the timing data to a YAML file."""
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(self.timings, f, default_flow_style=False)
        print(f"Timing data exported to '{filename}'.")

    def append_to_yaml(self, filename, func_name):
        """Append the timing data to a YAML file."""
        cropped_timings = self.timings[func_name][self.last_append :]

        with open(filename, "a", encoding="utf-8") as f:
            yaml.dump(cropped_timings, f, default_flow_style=False)

        self.last_append = len(self.timings[func_name])

    def print(self, drop_first: bool | int = False):
        """Print timing statistics for all recorded functions using formatted output."""

        # Print title
        print(log.bold("Function Timing Statistics"))
        header = (
            f"{log.cyan('Function'):<30} {log.green('Mean'):<22} "
            f"{log.green('Median'):<22} {log.green('Std Dev'):<22} "
            f"{log.yellow('Min'):<22} {log.yellow('Max'):<22} {log.magenta('Count'):<18}"
        )
        length = len(log.remove_color_escape_codes(header))
        print("=" * length)

        # Print header
        print(header)
        print("-" * length)

        # Print data rows
        for func_name in self.timings.keys():
            stats = self.get_stats(func_name, drop_first=drop_first)
            row = (
                f"{log.cyan(func_name):<30} "
                f"{log.green(log.number_to_str(stats['mean'], 6)):<22} "
                f"{log.green(log.number_to_str(stats['median'], 6)):<22} "
                f"{log.green(log.number_to_str(stats['std_dev'], 6)):<22} "
                f"{log.yellow(log.number_to_str(stats['min'], 6)):<22} "
                f"{log.yellow(log.number_to_str(stats['max'], 6)):<22} "
                f"{log.magenta(str(stats['count'])):<18}"
            )
            print(row)
