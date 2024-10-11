"""Caching utilities for function outputs.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 11th, 2024
"""

import hashlib
import inspect
from pathlib import Path

import joblib

CACHE_DIR = Path.home() / ".usbmd_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def generate_cache_key(func, args, kwargs, arg_names):
    """Generate a unique cache key based on function name and specified parameters."""
    key_elements = [func.__name__]
    if not arg_names:
        key_elements.extend(args)
        key_elements.extend(f"{k}={v}" for k, v in kwargs.items())
    else:
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        for name in arg_names:
            if name in bound_args.arguments:
                key_elements.append(f"{name}={bound_args.arguments[name]}")
    key = "_".join(map(str, key_elements))
    return hashlib.md5(key.encode()).hexdigest()


def cache_output(*arg_names):
    """Decorator to cache function outputs using joblib."""
    assert all(isinstance(arg_name, str) for arg_name in arg_names), (
        "All argument names must be strings, "
        "please use cache_output with strings as arguments or leave it empty "
        "to cache all arguments."
    )

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(func, args, kwargs, arg_names)
            cache_file = CACHE_DIR / f"{cache_key}.pkl"
            if cache_file.exists():
                return joblib.load(cache_file)
            result = func(*args, **kwargs)
            joblib.dump(result, cache_file)
            return result

        return wrapper

    return decorator
