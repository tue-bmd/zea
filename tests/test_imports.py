""" Check that all Python files in the project can be compiled, and that no import errors occur, for
example due to missing dependencies in the pyproject.toml file. """

import glob
import importlib
import sys
import traceback
from pathlib import Path

import pytest


def _assert_ml_libs_not_imported():
    """Check if 'torch', 'tensorflow', and 'jax' were imported"""
    torch_imported = "torch" in sys.modules
    tf_imported = "tensorflow" in sys.modules
    jax_imported = "jax" in sys.modules
    if torch_imported or tf_imported or jax_imported:
        raise AssertionError(
            "Torch, TensorFlow, and/or JAX was imported! Please ensure that no ML library "
            "is imported in the device module. "
            f"Imported modules: torch={torch_imported}, tensorflow={tf_imported}, "
            f"jax={jax_imported}"
        )


def _clear_ml_libs():
    """Clear ML libraries from sys.modules"""
    sys.modules.pop("torch", None)
    sys.modules.pop("tensorflow", None)
    sys.modules.pop("jax", None)


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


def test_package_does_not_import_heavy_ml_libraries():
    """Test that the package does not import heavy ML libraries like torch, tensorflow, or jax."""
    _clear_ml_libs()
    importlib.import_module("usbmd")
    _assert_ml_libs_not_imported()
