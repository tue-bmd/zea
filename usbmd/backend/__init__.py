"""Backend module for USBMD.
The existance of this (maybe empty) file is essential for the package to work."""

import keras


def jit(func):
    """
    Applies JIT compilation to the given function based on the current Keras backend.

    Args:
        func (callable): The function to be JIT compiled.

    Returns:
        callable: The JIT-compiled function.

    Raises:
        ValueError: If the backend is unsupported or if the necessary libraries are not installed.
    """
    backend = keras.backend.backend()

    if backend == "tensorflow":
        try:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel

            return tf.function(func, jit_compile=True)
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. Please install it to use this backend."
            ) from exc
    elif backend == "jax":
        try:
            import jax  # pylint: disable=import-outside-toplevel

            return jax.jit(func)
        except ImportError as exc:
            raise ImportError(
                "JAX is not installed. Please install it to use this backend."
            ) from exc
    else:
        print(
            f"Unsupported backend: {backend}. Supported backends are 'tensorflow' and 'jax'."
        )
        print("Falling back to non-compiled mode.")
        return func
