"""Caching utilities for function outputs.

>[!NOTE]
> It can be useful to inherit custom classes from `usbmd.core.Object`, as
> these classes will be serialized properly, just like regular python objects. Otherwise
> custom classes will not be recognized as equal if they have the same attributes by the
> caching mechanism.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 11th, 2024
"""

import hashlib
import inspect
import os
from pathlib import Path

import joblib

from usbmd.utils import log

_CACHE_DIR = Path.home() / ".usbmd_cache"

# Set backend based on USBMD_CACHE_DIR flag, if applicable.
if "USBMD_CACHE_DIR" in os.environ:
    _cache_dir = os.environ["USBMD_CACHE_DIR"]
    if _cache_dir:
        _CACHE_DIR = _cache_dir

_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def serialize_elements(key_elements):
    """Serialize elements to generate a cache key.

    In general uses the string representation of the elements unless
    the element has a `serialized` attribute, in which case it uses that.
    For instance this is useful for custom classes that inherit from `usbmd.core.Object`.

    Args:
        key_elements (list): List of elements to serialize. Can be nested lists
            or tuples. In this case the elements are serialized recursively.
    Returns:
        list[str]: List of serialized elements. In cases where the elements were
            lists of tuples those are combined into a single string.

    """
    serialized_elements = []
    for element in key_elements:
        if isinstance(element, (list, tuple)):
            element = serialize_elements(element)
            serialized_elements.append("_".join(element))
        elif hasattr(element, "serialized"):
            serialized_elements.append(str(element.serialized))
        else:
            serialized_elements.append(str(element))

    return serialized_elements


def generate_cache_key(func, args, kwargs, arg_names):
    """Generate a unique cache key based on function name and specified parameters."""
    key_elements = [func.__name__]  # function name
    try:
        key_elements.append(inspect.getsource(func))  # source code
    except OSError:
        log.warning(
            f"Could not get source code for function {func.__name__}. Proceeding without it."
        )
    if not arg_names:
        key_elements.extend(args)
        key_elements.extend(f"{k}={v}" for k, v in kwargs.items())
    else:
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        for name in arg_names:
            if name in bound_args.arguments:
                key_elements.append(f"{name}={bound_args.arguments[name]}")

    key = "_".join(serialize_elements(key_elements))
    return f"{func.__name__}_" + hashlib.md5(key.encode()).hexdigest()


def cache_output(*arg_names, verbose=False):
    """Decorator to cache function outputs using joblib."""
    assert all(isinstance(arg_name, str) for arg_name in arg_names), (
        "All argument names must be strings, "
        "please use cache_output with strings as arguments or leave it empty "
        "to cache all arguments."
    )

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(func, args, kwargs, arg_names)
            cache_file = _CACHE_DIR / f"{cache_key}.pkl"
            if cache_file.exists():
                if verbose:
                    log.info(f"Loading cached result for {func.__name__}.")
                return joblib.load(cache_file)
            result = func(*args, **kwargs)
            joblib.dump(result, cache_file)
            return result

        return wrapper

    return decorator


def clear_cache(func_name=None):
    """Clear cache files.

    If func_name is specified, only clear cache files related to that function.
    Otherwise, clear the entire cache directory.

    Provides a summary of how much was cleared and logs the information.
    """
    total_cleared = 0

    if func_name:
        pattern = f"{func_name}_*.pkl"
    else:
        pattern = "*.pkl"

    for cache_file in _CACHE_DIR.glob(pattern):
        file_size = cache_file.stat().st_size
        cache_file.unlink()
        total_cleared += file_size

    if total_cleared > 0:
        if func_name:
            log.info(
                f"Cleared {total_cleared / (1024 * 1024):.2f} MB "
                f"from cache for function '{func_name}'."
            )
        else:
            log.info(
                f"Cleared {log.yellow(f'{total_cleared / (1024 * 1024):.2f}')} "
                "MB from cache."
            )
    else:
        log.info("No cache files to clear.")


def cache_summary():
    """Print a summary of the cache, grouping by function name and summing the sizes."""
    summary = {}
    for cache_file in _CACHE_DIR.glob("*.pkl"):
        # Assuming cache files are named as '{func_name}_{hash}.pkl'
        func_name = "_".join(cache_file.stem.split("_")[:-1])
        file_size = cache_file.stat().st_size
        summary[func_name] = summary.get(func_name, 0) + file_size

    if not summary:
        log.info(f"usbmd cache at {_CACHE_DIR} is empty.")
        return

    log.info(f"usbmd cache summary at {_CACHE_DIR}:")
    for func_name, total_size in summary.items():
        log.info(
            f"Function '{func_name}' has a total cache size of "
            f"{total_size / (1024 * 1024):.2f} MB"
        )


def set_cache_dir(cache_dir):
    """Set the cache directory to a custom location.

    Args:
        cache_dir (str | Path): Path to the new cache directory
    """
    global _CACHE_DIR  # pylint: disable=global-statement
    previous_cache_dir = _CACHE_DIR

    # Convert to Path and resolve
    new_cache_dir = Path(cache_dir).resolve()

    # Set environment variable
    os.environ["USBMD_CACHE_DIR"] = str(new_cache_dir)

    # Update module-level cache dir
    _CACHE_DIR = new_cache_dir
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Changed cache directory from {previous_cache_dir} to {_CACHE_DIR}.")
