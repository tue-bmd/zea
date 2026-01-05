"""This file creates a :class:`zea.Operation` for all unary :mod:`keras.ops`
and :mod:`keras.ops.image` functions.

They can be used in zea pipelines like any other :class:`zea.Operation`, for example:

.. doctest::

    >>> from zea.ops.keras_ops import Squeeze
    >>> op = Squeeze(axis=1)
"""

import inspect
import shutil
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


if __name__ == "__main__":
    _generate_ops_file()
