"""This file creates a :class:`zea.Operation` for all unary :mod:`keras.ops`
and :mod:`keras.ops.image` functions.

They can be used in zea pipelines like any other :class:`zea.Operation`, for example:

.. doctest::

    >>> from zea.ops.keras_ops import Squeeze
    >>> op = Squeeze(axis=1)
"""

import inspect
import re
import shutil
import sys
import tempfile
from pathlib import Path

import keras


def _filter_funcs_by_first_arg(funcs, arg_name):
    """Filter a list of (name, func) tuples to those whose first argument matches arg_name."""
    filtered = []
    for name, func in funcs:
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if params and params[0] == arg_name:
                filtered.append((name, func))
        except (ValueError, TypeError):
            # Skip functions that can't be inspected
            continue
    return filtered


def _functions_from_namespace(namespace):
    """Get all functions from a given namespace."""
    return [(name, obj) for name, obj in inspect.getmembers(namespace) if inspect.isfunction(obj)]


def _unary_functions_from_namespace(namespace, arg_name="x"):
    """Get all unary functions from a given namespace."""
    funcs = _functions_from_namespace(namespace)
    return _filter_funcs_by_first_arg(funcs, arg_name)


def _snake_to_pascal(name):
    """Convert a snake_case name to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _generate_operation_class_code(name, namespace):
    """Generate Python code for a zea.Operation class for a given keras.ops function."""
    class_name = _snake_to_pascal(name)
    module_path = f"{namespace.__name__}.{name}"
    doc = f"Operation wrapping {module_path}."

    return f'''
@ops_registry("{module_path}")
class {class_name}(Lambda):
    """{doc}"""

    def __init__(self, **kwargs):
        try:
            super().__init__(func={module_path}, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("{class_name}", "{module_path}") from e
'''


def _generate_ops_file():
    """Generate a .py file with all operation class definitions."""

    # File header with version info
    content = f'''"""Auto-generated :class:`zea.Operation` for all unary :mod:`keras.ops`
and :mod:`keras.ops.image` functions.

They can be used in zea pipelines like any other :class:`zea.Operation`, for example:

.. doctest::

    >>> from zea.ops.keras_ops import Squeeze

    >>> op = Squeeze(axis=1)

This file is generated automatically. Do not edit manually.
Generated with Keras {keras.__version__}
"""

import keras

from zea.internal.registry import ops_registry
from zea.ops.base import Lambda

class MissingKerasOps(ValueError):
    def __init__(self, class_name: str, func: str):
        super().__init__(
            f"Failed to create {{class_name}} with {{func}}. " +
            "This may be due to an incompatible version of `keras`. " +
            "Please try to upgrade `keras` to the latest version by running " +
            "`pip install --upgrade keras`."
        )

'''

    for name, _ in _unary_functions_from_namespace(keras.ops, "x"):
        content += _generate_operation_class_code(name, keras.ops)

    for name, _ in _unary_functions_from_namespace(keras.ops.image, "images"):
        content += _generate_operation_class_code(name, keras.ops.image)

    # Write to a temporary file first, then move to final location
    target_path = Path(__file__).parent.parent / "ops/keras_ops.py"
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(content)
        temp_path = Path(tmp_file.name)

    # Atomic move to avoid partial writes
    shutil.move(temp_path, target_path)

    print("Done generating `ops/keras_ops.py`.")


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of ints, ignoring pre-release suffixes."""
    parts = []
    for segment in version_str.split("."):
        # Extract only the leading integer from each segment (e.g. '0rc1' -> 0)
        m = re.match(r"(\d+)", segment)
        if m:
            parts.append(int(m.group(1)))
    return tuple(parts)


def _get_generated_keras_version(target_path: Path) -> tuple[int, ...] | None:
    """Extract the Keras version from the header of an existing generated file.

    Returns the version as a tuple of ints, or ``None`` if the file does not
    exist or the version cannot be parsed.
    """
    if not target_path.exists():
        return None
    header_pattern = re.compile(r"Generated with Keras\s+(\S+)")
    try:
        with target_path.open(encoding="utf-8") as f:
            for line in f:
                m = header_pattern.search(line)
                if m:
                    return _parse_version(m.group(1))
    except Exception:
        pass
    return None


def _check_version_and_generate(target_path: Path) -> None:
    """Check Keras version and generate ops file if not downgrading.

    If the installed Keras version is older than the version used to generate
    the existing file, prints a warning and exits with code 1 to prevent
    downgrading the file.
    """
    current_version = _parse_version(keras.__version__)
    generated_version = _get_generated_keras_version(target_path)

    if generated_version is not None and current_version < generated_version:
        print(
            f"WARNING: Your installed Keras version ({keras.__version__}) is older than "
            f"the version used to generate `keras_ops.py` "
            f"({'.'.join(str(x) for x in generated_version)}). "
            "Regenerating would downgrade the file and remove operations that are "
            "available in newer Keras releases.\n"
            "Please upgrade Keras to avoid this:\n"
            "    pip install --upgrade keras"
        )
        sys.exit(1)

    _generate_ops_file()
