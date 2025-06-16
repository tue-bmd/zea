"""Caching utilities for function outputs.

>[!TIP]
> Caching works best for functions that take long, but output small results. If loading of a large
> cached tensor for instance take longer than the function itself,
> it is better to not cache the result.

>[!NOTE]
> It can be useful to inherit custom classes from `zea.core.Object`, as
> these classes will be serialized properly, just like regular python objects. Otherwise
> custom classes will not be recognized as equal if they have the same attributes by the
> caching mechanism.

>[!NOTE]
> For large experiments, it can be recommended to either set a custom cache directory
> or disable the cache completely. This can be done by setting the environment variable
> `ZEA_CACHE_DIR` to a custom directory or `ZEA_DISABLE_CACHE` to `1` or `true`.
> Otherwise, the cache will be stored in `~/.zea_cache` by default, which can pile up over time.

"""

import ast
import atexit
import hashlib
import inspect
import os
import pickle
import tempfile
import textwrap
from pathlib import Path

import joblib
import keras

from zea import log

_DEFAULT_ZEA_CACHE_DIR = Path.home() / ".cache" / "zea"


def _disable_cache():
    """Disable caching by creating a temporary directory and setting the environment variable."""
    os.environ["ZEA_DISABLE_CACHE"] = "1"
    _tmp_dir = tempfile.TemporaryDirectory(prefix="zea_cache_")
    atexit.register(lambda: _tmp_dir.cleanup())
    return Path(_tmp_dir.name)


def is_cache_disabled():
    """Check if caching is disabled via environment variable."""
    val = os.environ.get("ZEA_DISABLE_CACHE", "0").strip().lower()
    return val in ("1", "true", "yes")


def _make_cache_dir(path: Path):
    """Try to create the cache directory.
    If it fails, disable the cache and return a temporary directory instead."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        log.warning(
            f"Could not create cache directory {ZEA_CACHE_DIR}: {e} \n"
            + "Disabling cache globally. Set ZEA_CACHE_DIR to a different directory "
            + "to enable caching again."
        )
        return _disable_cache()


ZEA_CACHE_DIR = Path(os.environ.get("ZEA_CACHE_DIR", _DEFAULT_ZEA_CACHE_DIR)).resolve()

# Even if we cannot create the cache directory, we still want to use a temporary directory
# to avoid errors in the rest of the code (particularly huggingface)
if is_cache_disabled():
    ZEA_CACHE_DIR = _disable_cache()
else:
    ZEA_CACHE_DIR = _make_cache_dir(ZEA_CACHE_DIR)


_CACHE_DIR = ZEA_CACHE_DIR / "cached_funcs"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def serialize_elements(key_elements: list):
    """Serialize elements of a list to generate a cache key.

    In general uses the string representation of the elements unless
    the element has a `serialized` attribute, in which case it uses that.
    For instance this is useful for custom classes that inherit from `zea.core.Object`.

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
            # If element is a list or tuple, serialize its elements recursively
            element = serialize_elements(element)
            serialized_elements.append("_".join(element))
        elif hasattr(element, "serialized"):
            # Use the serialized attribute if it exists (e.g. for zea.core.Object)
            serialized_elements.append(str(element.serialized))
        elif isinstance(element, str):
            # If element is a string, use it as is
            serialized_elements.append(element)
        elif isinstance(element, keras.random.SeedGenerator):
            # If element is a SeedGenerator, use the state
            element = keras.ops.convert_to_numpy(element.state.value)
            element = pickle.dumps(element)
            element = hashlib.md5(element).hexdigest()
            serialized_elements.append(element)
        else:
            # Otherwise, serialize the element using pickle and hash it
            element = pickle.dumps(element)
            element = hashlib.md5(element).hexdigest()
            serialized_elements.append(element)

    return serialized_elements


def get_function_source(func):
    """Recursively get the source code of a function and its nested functions."""
    try:
        source = inspect.getsource(func)
    except OSError:
        return None  # Do not cache if source code cannot be retrieved

    # Parse the source code into an AST
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    called_functions = set()

    class FunctionCallVisitor(ast.NodeVisitor):
        """AST visitor to collect function calls."""

        def visit_Call(self, node):
            """Visit a Call node and add the function name to the set."""
            if isinstance(node.func, ast.Name):
                called_functions.add(node.func.id)
            self.generic_visit(node)

    FunctionCallVisitor().visit(tree)

    # Sorting the called functions to ensure consistent cache keys
    for called_func_name in sorted(called_functions):
        try:
            called_func = func.__globals__.get(called_func_name)
            if inspect.isfunction(called_func) and called_func.__module__ != "zea.internal.cache":
                nested_source = get_function_source(called_func)
                if nested_source is None:
                    # If any nested function's source cannot be retrieved, do not cache
                    return None
                source += nested_source
        except (NameError, TypeError):
            continue

    return source


def generate_cache_key(func, args, kwargs, arg_names):
    """Generate a unique cache key based on function name and specified parameters."""
    key_elements = [func.__qualname__]  # qualified function name
    source = get_function_source(func)
    if source is None:
        log.warning(
            f"Could not get source code for function {func.__qualname__}. Not caching the result."
        )
        return None  # Do not cache if source code cannot be retrieved
    key_elements.append(source)  # source code
    if not arg_names:
        key_elements.extend(args)
        key_elements.extend(v for _, v in sorted(kwargs.items()))
    else:
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        for name in arg_names:
            if name in bound_args.arguments:
                key_elements.append(bound_args.arguments[name])

    key = "_".join(serialize_elements(key_elements))
    return f"{func.__qualname__}_" + hashlib.md5(key.encode()).hexdigest()


def cache_output(*arg_names, verbose=False):
    """Decorator to cache function outputs using joblib."""
    assert all(isinstance(arg_name, str) for arg_name in arg_names), (
        "All argument names must be strings, "
        "please use cache_output with strings as arguments or leave it empty "
        "to cache all arguments."
    )

    def decorator(func):
        def wrapper(*args, **kwargs):
            if is_cache_disabled():
                if verbose:
                    log.info(f"Caching is globally disabled for {func.__qualname__}.")
                return func(*args, **kwargs)
            try:
                cache_key = generate_cache_key(func, args, kwargs, arg_names)
            except Exception as e:
                log.warning(
                    f"Could not cache result for {func.__qualname__}: {e}. "
                    "Running the function without caching. "
                    "Often happens for a function wrapped with jax.jit or tf.function."
                )
                return func(*args, **kwargs)
            if cache_key is None:
                return func(*args, **kwargs)  # Run function without caching
            cache_file = _CACHE_DIR / f"{cache_key}.pkl"
            if cache_file.exists():
                if verbose:
                    log.info(f"Loading cached result for {func.__qualname__}.")
                return joblib.load(cache_file)
            elif verbose:
                log.info(f"Running {func.__qualname__} and caching the result to {cache_file}.")
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
            log.info(f"Cleared {log.yellow(f'{total_cleared / (1024 * 1024):.2f}')} MB from cache.")
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
        log.info(f"zea cache at {_CACHE_DIR} is empty.")
        return

    log.info(f"zea cache summary at {_CACHE_DIR}:")
    for func_name, total_size in summary.items():
        log.info(
            f"Function '{func_name}' has a total cache size of {total_size / (1024 * 1024):.2f} MB"
        )
