""" Check that all Python files in the project can be compiled, and that no import errors occur, for
example due to missing dependencies in the pyproject.toml file. """

import builtins
import glob
import importlib
import traceback
from pathlib import Path

import pytest


@pytest.fixture
def _no_ml_lib_import():
    """Fixture to override and restore the built-in import function."""
    # Override the built-in import function
    original_import_func = builtins.__import__

    # Define a custom import function
    def import_fail_on_ml_libs(name, *args, **kwargs):
        """Custom import function that raises an error if torch, tensorflow, or jax is imported."""
        if name.lower() in ["jax", "tensorflow", "torch"]:
            raise ImportError(f"{name} is not allowed to be imported in this program.")
        return original_import_func(name, *args, **kwargs)

    builtins.__import__ = import_fail_on_ml_libs
    yield
    # Restore the original import function after the test
    builtins.__import__ = original_import_func


@pytest.fixture
def _no_torch_tensorflow():
    """
    Fixture to override and restore the built-in import function.
    NOTE: This function exists because keras with numpy backend will also import jax.
    See: /usr/local/lib/python3.10/dist-packages/keras/src/backend/numpy/image.py
    """
    # Override the built-in import function
    original_import_func = builtins.__import__

    # Define a custom import function
    def import_fail_on_ml_libs(name, *args, **kwargs):
        """Custom import function that raises an error if torch, tensorflow, or jax is imported."""
        if name.lower() in ["tensorflow", "torch"]:
            raise ImportError(f"{name} is not allowed to be imported in this program.")
        return original_import_func(name, *args, **kwargs)

    builtins.__import__ = import_fail_on_ml_libs
    yield
    # Restore the original import function after the test
    builtins.__import__ = original_import_func


@pytest.mark.parametrize("directory", [Path(__file__).parent.parent])
def test_check_imports_errors(directory):
    """Check all Python files in a directory for import errors."""
    python_files = glob.glob(f"{directory}/**/*.py", recursive=True)

    for python_file in python_files:
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


def test_package_does_not_import_heavy_ml_libraries(_no_torch_tensorflow):
    """Test that the package does not import heavy ML libraries like torch, tensorflow, or jax."""
    importlib.import_module("usbmd")
