"""Check that all Python files in the project can be compiled, and that no import errors occur, for
example due to missing dependencies in the pyproject.toml file."""

import builtins
import contextlib
import glob
import importlib
import os
import traceback
from pathlib import Path

import pytest

from .helpers import run_in_subprocess


@contextlib.contextmanager
def no_ml_lib_import(backends: list = None, allow_keras_backend=True):
    """Context manager to check if any backend in backends gets imported inside of it.
    Will raise an ImportError if any of the backends are imported."""

    if backends is None:
        backends = ["jax", "tensorflow", "torch"]

    if allow_keras_backend:
        curr_backend = os.environ.get("KERAS_BACKEND", None)
        assert curr_backend is not None, "KERAS_BACKEND environment variable is not set."

        # remove curr_backend from backends
        backends = [backend for backend in backends if backend != curr_backend]

    # Save the original built-in import function
    original_import_func = builtins.__import__

    # Define a custom import function
    def import_fail_on_ml_libs(name, *args, **kwargs):
        """Raise an error if a disallowed backend is imported."""
        if name.lower() in backends:
            raise ImportError(
                f"Disallowed backend import detected: '{name}'. "
                f"The following backends are not allowed: {backends}. "
                f"Current KERAS_BACKEND is set to '{curr_backend}'."
            )
        return original_import_func(name, *args, **kwargs)

    # Override the built-in import function
    builtins.__import__ = import_fail_on_ml_libs

    try:
        yield
    finally:
        # Restore the original import function after exiting the context
        builtins.__import__ = original_import_func


@pytest.mark.parametrize("directory", [Path(__file__).parent.parent])
def test_check_imports_errors(directory, verbose=False):
    """Check all Python files in a directory for import errors."""
    python_files = glob.glob(f"{directory}/**/*.py", recursive=True)

    for python_file in python_files:
        if verbose:
            print(python_file)

    success = True
    for python_file in python_files:
        try:
            # Attempt to compile the Python file (checks for import errors)
            with open(python_file, "rb") as file:
                compile(file.read(), python_file, "exec")
        except SyntaxError as e:
            print(f"Syntax error in {python_file}:\n{e}")
            success = False
        except ImportError as e:
            print(f"Import error in {python_file}:\n{e}")
            success = False
        except Exception as e:
            print(f"Error in {python_file}:\n{e}")
            traceback.print_exc()
            success = False

    assert success, "Import errors found in one or more Python files."


@run_in_subprocess
def test_package_only_imports_keras_backend():
    """Test that the package does not import heavy ML libraries like torch, tensorflow,
    or jax running in a fresh environment.

    NOTE: Only torch and tensorflow because keras with numpy backend will also import jax.
    See: /usr/local/lib/python3.10/dist-packages/keras/src/backend/numpy/image.py"""

    with no_ml_lib_import():
        importlib.import_module("zea")
