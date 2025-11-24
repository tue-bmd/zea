"""Utility functions used internally.

These are not exposed to the public API.
"""

import functools
import hashlib
import inspect
import platform

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


def find_methods_with_return_type(cls, return_type_hint: str):
    """
    Find all methods in a class that have the specified return type hint.

    Args:
        cls: The class to inspect.
        return_type_hint (str): The return type hint to match.

    Returns:
        A list of method names that match the return type hint.
    """
    matching_methods = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        annotations = getattr(member, "__annotations__", {})
        return_annotation = annotations.get("return")
        if return_annotation is None:
            continue

        # Convert annotation to string for comparison
        if hasattr(return_annotation, "__name__"):
            # For types like bool, int, str, custom classes
            annotation_str = return_annotation.__name__
        else:
            # For string annotations or other types, convert to string
            annotation_str = str(return_annotation)

        if annotation_str == return_type_hint:
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
            log.warning(f"Function {fn.__name__} failed with error: {e}. Retrying...")


def reduce_to_signature(func, kwargs):
    """Reduce the kwargs to the signature of the function."""
    # Retrieve the argument names of the function
    sig = inspect.signature(func)

    # Filter out the arguments that are not part of the function
    reduced_params = {key: kwargs[key] for key in sig.parameters if key in kwargs}

    return reduced_params
