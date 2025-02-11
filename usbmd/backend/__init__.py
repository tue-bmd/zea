"""Backend module for USBMD.
The existance of this (maybe empty) file is essential for the package to work."""

import keras


def tf_function(func=None, jit_compile=False, **kwargs):
    """Applies default tf.function to the given function. Only in TensorFlow backend."""
    return jit(func, jax=False, jit_compile=jit_compile, **kwargs)


def jit(func=None, jax=True, tensorflow=True, **kwargs):
    """
    Applies JIT compilation to the given function based on the current Keras backend.
    Can be used as a decorator or as a function.
    Args:
        func (callable): The function to be JIT compiled.
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

    # Jit with TensorFlow
    if backend == "tensorflow" and tensorflow:
        try:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel

            jit_compile = kwargs.pop("jit_compile", True)
            return tf.function(func, jit_compile=jit_compile, **kwargs)
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. Please install it to use this backend."
            ) from exc
    # Jit with JAX
    elif backend == "jax" and jax:
        try:
            import jax  # pylint: disable=import-outside-toplevel

            return jax.jit(func, **kwargs)
        except ImportError as exc:
            raise ImportError(
                "JAX is not installed. Please install it to use this backend."
            ) from exc
    # No JIT compilation, because disabled
    elif backend == "tensorflow" and not tensorflow:
        return func
    elif backend == "jax" and not jax:
        return func
    else:
        print(
            f"Unsupported backend: {backend}. Supported backends are 'tensorflow' and 'jax'."
        )
        print("Falling back to non-compiled mode.")
        return func
