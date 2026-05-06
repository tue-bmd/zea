"""Backend utilities for ``zea``.

.. note::
    Most tensor operations are handled by Keras 3. This module only wraps the
    features that Keras does not expose directly: JIT compilation, automatic
    differentiation, and device placement.

Public API
----------

:func:`jit`
    Unified JIT compilation for JAX (``jax.jit``) and TensorFlow
    (``tf.function``).  A no-op for the ``torch`` backend.

:class:`device`
    Context manager that pins all Keras ops to a specific device.
    Re-exported as :func:`zea.device`.

:func:`func_on_device`
    Run a callable with its tensor arguments moved to a target device.
    For ``torch`` this also calls ``.to(device)`` on every input tensor.

:class:`AutoGrad`
    Backend-agnostic automatic differentiation wrapper.
"""

from contextlib import nullcontext

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


def _get_backend():
    try:
        backend_result = keras.backend.backend()
        if isinstance(backend_result, str):
            return backend_result
        else:
            # to handle mocked backends during testing
            return None
    except Exception:
        return None


tf_mod = _import_tf()
jax_mod = _import_jax()
backend = _get_backend()


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
            f"JIT compilation not currently supported for backend {backend}. "
            "Supported backends are 'tensorflow' and 'jax'."
        )
        log.warning("Initialize zea.Pipeline with jit_options=None to suppress this warning.")
        log.warning("Falling back to non-compiled mode.")
        return func


class device:
    """Context manager to run operations on a specific device, regardless of backend.

    Normalises device strings across JAX, TensorFlow, and PyTorch so that
    ``'gpu:0'``, ``'cuda:0'`` and ``'cpu'`` all work with every backend, then
    delegates to :func:`keras.device` which handles the per-backend dispatch.

    For the ``torch`` backend, :func:`keras.device` sets Keras's internal
    device-tracking state so that tensors created by Keras ops land on the
    correct device.  Existing input tensors are **not** moved automatically —
    use ``pipeline(device=..., **inputs)`` or
    :func:`zea.backend.func_on_device` when you also need to relocate
    pre-existing tensors.

    Args:
        device (str): Device string, e.g. ``'cuda:0'``, ``'gpu:0'``, or
            ``'cpu'``.

    Example:
        .. code-block:: python

            # All backends: tensors created by Keras ops are placed on gpu:0
            with zea.device("gpu:0"):
                output = pipeline(data=data)

            # Per-call device with automatic input-tensor movement (all backends)
            output = pipeline(device="gpu:0", data=data)
    """

    def __init__(self, device: str):
        if device is None:
            self._context = nullcontext()
        else:
            normalized = self._normalize(device)
            self._context = keras.device(normalized)

    @staticmethod
    def _normalize(device: str) -> str:
        """Normalize device string before passing to ``keras.device``.

        Converts ``cuda:N`` → ``gpu:N`` so the string is backend-agnostic;
        ``keras.device`` itself then converts ``gpu:N`` → ``cuda:N`` when
        running under the ``torch`` backend.
        """
        device = device.lower()
        if device.startswith("auto:"):
            raise ValueError(
                f"``zea.device`` does not accept 'auto:N' device strings (got {device!r}). "
                "Use zea.init_device('auto:N') first to resolve a concrete device, "
                "then pass the returned string (e.g. 'gpu:0') to ``zea.device``."
            )
        # Normalise to gpu:N; keras.device handles gpu → cuda for the torch backend.
        return device.replace("cuda", "gpu")

    def __enter__(self):
        self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context.__exit__(exc_type, exc_val, exc_tb)


# Private alias so func_on_device can reference the class without clashing
# with its own `device` parameter.
_DeviceContext = device


def func_on_device(func, device, *args, **kwargs):
    """Run ``func`` with all tensor arguments placed on ``device``.

    For the ``torch`` backend, every tensor argument is explicitly moved with
    ``.to(device)`` before the call.  For JAX and TensorFlow the function is
    executed inside an :class:`zea.backend.device` context, which routes newly created
    tensors to the requested device.

    Args:
        func (callable): Function to call.
        device (str): Target device, e.g. ``'cpu'``, ``'gpu:0'``, ``'cuda:1'``.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        Output of ``func(*args, **kwargs)``.
    """
    if device is None:
        return func(*args, **kwargs)

    if keras.backend.backend() == "torch":
        import torch

        _device = torch.device(device.lower().replace("gpu", "cuda"))

        def _move(x):
            if isinstance(x, torch.Tensor):
                return x.to(_device)
            if isinstance(x, (list, tuple)):
                return type(x)(_move(i) for i in x)
            if isinstance(x, dict):
                return {k: _move(v) for k, v in x.items()}
            return x

        args = _move(args)
        kwargs = _move(kwargs)

    with _DeviceContext(device):
        return func(*args, **kwargs)
