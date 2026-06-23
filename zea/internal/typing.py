"""Shared type aliases for ``zea``.

These aliases keep annotations across the codebase consistent and *informative*
without coupling the type hints to a single ML backend. ``zea`` runs on top of
Keras with interchangeable backends (TensorFlow, PyTorch, JAX, NumPy), so a
"tensor" is deliberately backend-agnostic.

Import the aliases from here rather than redefining them per module::

    from zea.internal.typing import Tensor, NDArray, PathOrStr
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

if TYPE_CHECKING:
    # Imported for typing only so that ``import zea`` does not eagerly pull in
    # keras (and therefore a backend); see ``zea/__init__.py``.
    import keras

__all__ = [
    "Tensor",
    "Scalar",
    "ArrayLike",
    "DTypeLike",
    "NDArray",
    "PathOrStr",
]

# A real-valued scalar accepted by the spec/parameter system. Values are stored
# internally as ``np.float32`` after casting, but the public API also accepts
# plain Python ``int``/``float`` and other NumPy scalar types.
Scalar = Union[float, int, np.floating, np.integer]

# A backend-agnostic tensor: either a (symbolic or eager) Keras/backend tensor
# or a concrete NumPy array. Most ``zea`` ops accept and return values of this
# kind regardless of the configured Keras backend.
Tensor = Union["keras.KerasTensor", np.ndarray]

# Anything accepted where a filesystem path is expected.
PathOrStr = Union[str, "PathLike[str]"]
