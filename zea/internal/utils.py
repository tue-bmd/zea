"""Utility functions used internally.

These are not exposed to the public API.
"""

import contextlib
import functools
import hashlib
import inspect
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

from zea import log


def find_key(dictionary, contains, case_sensitive=False):
    """Find key in dictionary that contains partly the string `contains`

    Args:
        dictionary (dict): Dictionary to find key in.
        contains (str): String which the key should contain.
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
        key = [k for k in dictionary.keys() if contains.lower() in k.lower()]

    if len(key) == 0:
        raise KeyError(f"Key containing '{contains}' not found in dictionary.")

    return key[0]


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


def deprecated_alias(value, old_value, old, new, caller):
    """Map the deprecated keyword argument ``old`` onto its replacement ``new``.

    Args:
        value: The value passed as ``new``.
        old_value: The value passed as ``old``.
        old (str): The deprecated argument name.
        new (str): The name of its replacement.
        caller (str): The name of the function, for the messages.

    Returns:
        ``old_value`` (with a deprecation warning) when given, else ``value``.

    Raises:
        TypeError: If both ``old`` and ``new`` are given.
    """
    if old_value is None:
        return value
    if value is not None:
        raise TypeError(f"{caller}() got both {new!r} and its deprecated alias {old!r}.")
    log.deprecated(f"{caller}({old}=...) is deprecated, use {new}=... instead.")
    return old_value


def deprecated_distance_to_apex(distance_to_apex, grid_apex, caller):
    """Map the deprecated ``distance_to_apex`` argument onto ``grid_apex``.

    Returns:
        ``(0, 0, -distance_to_apex)`` (with a deprecation warning) when given, else
        ``grid_apex``.
    """
    if distance_to_apex is None:
        return grid_apex
    log.deprecated(f"{caller}(distance_to_apex=d) is deprecated, use grid_apex=(0, 0, -d).")
    return (0.0, 0.0, -distance_to_apex)


def deprecated(replacement=None):
    """Decorator to mark a function, method, or attribute as deprecated.

    Args:
        replacement (str, optional): The name of the replacement function, method, or attribute.

    Returns:
        callable: The decorated function, method, or property.

    Raises:
        DeprecationWarning: A warning is issued when the deprecated item is called or accessed.

    Example:
        >>> from zea.internal.utils import deprecated
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

            # Preserve the original signature so introspection-based tooling (e.g.
            # keras.Operation's auto get_config, which uses inspect.getfullargspec)
            # still sees the real parameters instead of this wrapper's *args/**kwargs.
            wrapper.__signature__ = inspect.signature(item)  # ty: ignore[unresolved-attribute]
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

                if item.fset is None:
                    raise AttributeError(f"{item.fget.__name__} is read-only")
                item.fset(self, value)

            def deleter(self):
                if replacement:
                    log.deprecated(
                        f"Deleting deprecated attribute {item.fget.__name__}, "
                        f"use {replacement} instead."
                    )
                else:
                    log.deprecated(f"Deleting deprecated attribute {item.fget.__name__}.")

                if item.fdel is None:
                    raise AttributeError(f"{item.fget.__name__} cannot be deleted")
                item.fdel(self)

            return property(getter, setter, deleter)

        else:
            raise TypeError("Decorator can only be applied to functions, methods, or properties.")

    return decorator


@contextlib.contextmanager
def atomic_write(path, suffix=None):
    """Write to a temporary file and move it into place only once it is complete.

    Readers of `path` never observe a half-written file, and a failed write leaves any
    existing `path` untouched instead of replacing it with a truncated one. The
    temporary file is created next to the destination so that it lives on the same
    filesystem, which is what makes the final move atomic.

    Args:
        path (str, Path): Destination path. Its parent directory must already exist.
        suffix (str, optional): Suffix of the temporary file. Writers that pick their
            format from the extension (h5py, PIL, ...) need this to match the
            destination. Defaults to the suffix of `path`.

    Yields:
        Path: Temporary path to write to. It exists (empty) when the block starts.

    Example:
        .. code-block:: python

            with atomic_write(path) as tmp:
                tmp.write_bytes(data)

    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
        suffix=path.suffix if suffix is None else suffix,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    except BaseException:
        # Includes KeyboardInterrupt/SystemExit: clean up the partial temp file.
        tmp_path.unlink(missing_ok=True)
        raise


def load_json_cache(path, max_age: float):
    """The JSON stored at `path` by :func:`store_json_cache`, or ``None`` if it is missing,
    unreadable or older than `max_age` seconds."""
    import json
    import time

    path = Path(path)
    try:
        if time.time() - path.stat().st_mtime >= max_age:
            return None
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def store_json_cache(path, data) -> None:
    """Store `data` as JSON at `path`, readable only by this user, for
    :func:`load_json_cache`. A failure is logged, not raised: the cache is optional."""
    import json

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # atomic_write goes through mkstemp, so the file is private to this user (0600).
        with atomic_write(path, suffix=".part") as tmp:
            tmp.write_text(json.dumps(data))
    except OSError as exc:
        log.debug(f"Could not write the cache file {path}: {exc}")


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
            if omit_line_str is not None and omit_line_str.encode() in line:
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
            log.warning(f"Function {fn.__name__} failed with error: {e}. Retrying...")


def reduce_to_signature(func, kwargs):
    """Reduce the kwargs to the signature of the function."""
    # Retrieve the argument names of the function
    sig = inspect.signature(func)

    # Filter out the arguments that are not part of the function
    reduced_params = {key: kwargs[key] for key in sig.parameters if key in kwargs}

    return reduced_params
