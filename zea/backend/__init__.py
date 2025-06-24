"""Backend subpackage for ``zea``.

This subpackage provides backend-specific utilities for the ``zea`` library. Most backend logic is handled by Keras 3, but a few features require custom wrappers to ensure compatibility and performance across JAX, TensorFlow, and PyTorch.

.. note::
    Most backend-specific logic is handled by Keras 3, so this subpackage is intentionally minimal. Only features not natively supported by Keras (such as JIT and autograd) are implemented here.

Key Features
------------

- **JIT Compilation** (:func:`zea.backend.jit`):
  Provides a unified interface for just-in-time (JIT) compilation of functions, dispatching to the appropriate backend (JAX or TensorFlow) as needed. This enables accelerated execution of computationally intensive routines.

- **Automatic Differentiation** (:class:`zea.backend.AutoGrad`):
  Offers a backend-agnostic wrapper for automatic differentiation, allowing gradient computation regardless of the underlying ML library.

- **Backend Submodules:**

  - :mod:`zea.backend.jax` -- JAX-specific utilities and device management.
  - :mod:`zea.backend.torch` -- PyTorch-specific utilities and device management.
  - :mod:`zea.backend.tensorflow` -- TensorFlow-specific utilities, and device management, as well as data loading utilities.

- **Data Loading** (:func:`zea.backend.tensorflow.make_dataloader`):
  This function is implemented using TensorFlow's efficient data pipeline utilities. It provides a convenient way to load and preprocess data for machine learning workflows, leveraging TensorFlow's ``tf.data.Dataset`` API.

"""

import keras

from zea import log


def _import_tf():
    try:
        import tensorflow as tf

        return tf
    except ImportError:
        return None


def _import_jax():
    try:
        import jax

        return jax
    except ImportError:
        return None


def _import_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


tf_mod = _import_tf()
jax_mod = _import_jax()


def tf_function(func=None, jit_compile=False, **kwargs):
    """Applies default tf.function to the given function. Only in TensorFlow backend."""
    return jit(func, jax=False, jit_compile=jit_compile, **kwargs)


def jit(func=None, jax=True, tensorflow=True, **kwargs):
    """
    Applies JIT compilation to the given function based on the current Keras backend.
    Can be used as a decorator or as a function.

    Args:
        func (callable): The function to be JIT compiled.
        jax (bool): Whether to enable JIT compilation in the JAX backend.
        tensorflow (bool): Whether to enable JIT compilation in the TensorFlow backend.
        **kwargs: Keyword arguments to be passed to the JIT compiler.

    Returns:
        callable: The JIT-compiled function.
    """
    if func is None:

        def decorator(func):
            return _jit_compile(func, jax=jax, tensorflow=tensorflow, **kwargs)

        return decorator
    else:
        return _jit_compile(func, jax=jax, tensorflow=tensorflow, **kwargs)


def _jit_compile(func, jax=True, tensorflow=True, **kwargs):
    backend = keras.backend.backend()

    if backend == "tensorflow" and tensorflow:
        if tf_mod is None:
            raise ImportError("TensorFlow is not installed. Please install it to use this backend.")
        jit_compile = kwargs.pop("jit_compile", True)
        return tf_mod.function(func, jit_compile=jit_compile, **kwargs)
    elif backend == "jax" and jax:
        if jax_mod is None:
            raise ImportError("JAX is not installed. Please install it to use this backend.")
        return jax_mod.jit(func, **kwargs)
    elif backend == "tensorflow" and not tensorflow:
        return func
    elif backend == "jax" and not jax:
        return func
    else:
        log.warning(
            f"Unsupported backend: {backend}. Supported backends are 'tensorflow' and 'jax'."
        )
        log.warning("Falling back to non-compiled mode.")
        return func
