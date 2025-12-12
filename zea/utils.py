"""General utility functions."""

import collections.abc
import datetime
import time
from functools import wraps
from statistics import mean, median, stdev

import keras
import yaml

from zea import log


def canonicalize_axis(axis, num_dims) -> int:
    """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
    if not -num_dims <= axis < num_dims:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {num_dims}")
    if axis < 0:
        axis = axis + num_dims
    return axis


def map_negative_indices(indices: list, num_dims: int):
    """Maps negative indices for array indexing to positive indices.
    Example:
        >>> from zea.utils import map_negative_indices
        >>> map_negative_indices([-1, -2], 5)
        [4, 3]
    """
    return [canonicalize_axis(idx, num_dims) for idx in indices]


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


def block_until_ready(func):
    """Decorator that ensures asynchronous (gpu) operations complete before returning."""
    if keras.backend.backend() == "jax":
        import jax

        def _block(value):
            if hasattr(value, "__array__"):
                return jax.block_until_ready(value)
            else:
                return value
    else:

        def _block(value):
            if hasattr(value, "__array__"):
                # convert to numpy but return as original type
                _ = keras.ops.convert_to_numpy(value)
            return value

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Handle different return types
        if isinstance(result, (list, tuple)):
            # For multiple outputs, block each one
            blocked_results = [_block(r) for r in result]
            return type(result)(blocked_results)
        elif isinstance(result, dict):
            # For dict outputs, block array values
            return {k: _block(v) for k, v in result.items()}
        else:
            # Single output
            return _block(result)

    return wrapper


class FunctionTimer:
    """
    A decorator class for timing the execution of functions.

    Example:
        .. doctest::

            >>> from zea.utils import FunctionTimer
            >>> timer = FunctionTimer()
            >>> my_function = lambda: sum(range(10))
            >>> my_function = timer(my_function, name="my_function")
            >>> _ = my_function()
            >>> print(timer.get_stats("my_function"))  # doctest: +ELLIPSIS
            {'mean': ..., 'median': ..., 'std_dev': ..., 'min': ..., 'max': ..., 'count': ...}
    """

    def __init__(self):
        self.timings = {}
        self.last_append = 0
        self.decorated_functions = {}  # Track decorated functions

    def __call__(self, func, name=None):
        _name = name if name is not None else func.__name__

        # Create a unique identifier for this function
        func_id = id(func)

        # Check if this exact function has already been decorated
        if func_id in self.decorated_functions:
            existing_name = self.decorated_functions[func_id]
            raise ValueError(
                f"Function '{func.__name__}' (id: {func_id}) has already been "
                f"decorated with timer name '{existing_name}'. "
                f"Cannot decorate the same function instance multiple times."
            )

        # Handle name conflicts by appending a suffix
        original_name = _name
        counter = 1
        while _name in self.timings:
            _name = f"{original_name}_{counter}"
            counter += 1

        # Initialize timing storage for this function
        self.timings[_name] = []

        # Track this decorated function
        self.decorated_functions[func_id] = _name

        func = block_until_ready(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # Store the timing result
            self.timings[_name].append(elapsed_time)

            return result

        return wrapper

    def _parse_drop_first(self, drop_first: bool | int):
        if isinstance(drop_first, bool):
            idx = 1 if drop_first else 0
        elif isinstance(drop_first, int):
            idx = drop_first
        else:
            raise ValueError("drop_first must be a boolean or an integer.")
        return idx

    def get_stats(self, func_name, drop_first: bool | int = False):
        """Calculate statistics for the given function."""
        if func_name not in self.timings:
            raise ValueError(f"No timings recorded for function '{func_name}'.")

        idx = self._parse_drop_first(drop_first)
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

    def print(self, drop_first: bool | int = False, total_time: bool = False):
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

        if total_time:
            idx = self._parse_drop_first(drop_first)
            total = sum(mean(times[idx:]) for times in self.timings.values())
            print("-" * length)
            print(f"{log.bold('Mean Total Time:')} {log.bold(log.number_to_str(total, 6))} seconds")
