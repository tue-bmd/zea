"""Check that all Python files in the project can be compiled, and that no import errors occur, for
example due to missing dependencies in the pyproject.toml file."""

import builtins
import contextlib
import glob
import importlib
import sys
import traceback
from pathlib import Path

import pytest


@contextlib.contextmanager
def _no_ml_lib_import(backends=["jax", "tensorflow", "torch"]):
    """Context manager to override and restore the built-in import function."""

    # Check if any of the disallowed backends are already imported
    for backend in backends:
        if backend in sys.modules:
            raise ImportError(f"{backend} is not allowed to be imported at this point.")

    # Save the original built-in import function
    original_import_func = builtins.__import__

    # Define a custom import function
    def import_fail_on_ml_libs(name, *args, **kwargs):
        """Raise an error if a disallowed backend is imported."""
        if name.lower() in backends:
            raise ImportError(f"{name} is not allowed to be imported in this program.")
        return original_import_func(name, *args, **kwargs)

    # Override the built-in import function
    builtins.__import__ = import_fail_on_ml_libs

    try:
        yield
    finally:
        # Restore the original import function after exiting the context
        builtins.__import__ = original_import_func


@pytest.fixture()
def no_ml_lib_import():
    with _no_ml_lib_import():
        yield


@pytest.fixture
def no_torch_tensorflow():
    """
    NOTE: This function exists because keras with numpy backend will also import jax.
    See: /usr/local/lib/python3.10/dist-packages/keras/src/backend/numpy/image.py
    """
    with _no_ml_lib_import(["torch", "tensorflow"]):
        yield


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


def usbmd_import():
    with _no_ml_lib_import(["torch", "tensorflow"]):
        try:
            importlib.import_module("usbmd")
        except Exception as e:
            # Print the error message to stdout so that it can be captured by the parent process
            print(e)
            sys.exit(1)


def test_package_does_not_import_heavy_ml_libraries():
    """Test that the package does not import heavy ML libraries like torch, tensorflow,
    or jax running in a fresh environment."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")

    process = ctx.Process(target=usbmd_import)
    process.start()
    process.join()
    assert process.exitcode == 0, "Process failed with exit code {}".format(
        process.exitcode
    )
