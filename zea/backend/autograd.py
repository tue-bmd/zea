"""Autograd wrapper for different backends."""

import functools

import keras

from . import _import_jax, _import_tf, _import_torch

tf = _import_tf()
jax = _import_jax()
torch = _import_torch()


class AutoGrad:
    """Wrapper class for autograd using different backends."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.function = None

        if verbose:
            print(f"Using backend: {self.backend}")

    @property
    def backend(self):
        """Get Keras backend. Machine learning library of choice."""
        return keras.backend.backend()

    @backend.setter
    def backend(self, backend):
        """Set Keras backend. Machine learning library of choice."""
        raise ValueError("Cannot change backend currently. Needs reimport of keras.")
        # keras.config.set_backend(backend)

    def set_function(self, function):
        """Set the function to calculate the gradients of."""
        self.function = function

    def gradient(self, variable, **kwargs):
        """Returns the gradients of the function w.r.t. variable.

        Args:
            variable (Tensor): Input tensor.
            **kwargs: Keyword arguments to pass to self.function.

        Returns:
            gradients (Tensor): Gradients of the function at variable.
                ∇f(x)
        """
        variable = keras.ops.convert_to_tensor(variable)
        if self.function is None:
            raise ValueError("Function not set. Use `set_function` to set a custom function.")
        assert self.backend in [
            "torch",
            "tensorflow",
            "jax",
        ], f"Unsupported backend: {self.backend}"

        if self.backend == "torch":
            if torch is None:
                raise ImportError("PyTorch is not installed.")
            variable = variable.detach().requires_grad_(True)
            out = self.function(variable, **kwargs)
            gradients = torch.autograd.grad(out, variable)[0]
            return gradients
        elif self.backend == "tensorflow":
            if tf is None:
                raise ImportError("TensorFlow is not installed.")
            with tf.GradientTape() as tape:
                tape.watch(variable)
                out = self.function(variable, **kwargs)
            gradients = tape.gradient(out, variable)
            return gradients
        elif self.backend == "jax":
            if jax is None:
                raise ImportError("JAX is not installed.")
            func = functools.partial(self.function, **kwargs)
            return jax.grad(func)(variable)

    def gradient_and_value(self, variable, has_aux: bool = False, **kwargs):
        """Returns both the gradients w.r.t. variable and outputs of the function.

        Note that self.function should return a tuple of (out, aux) if has_aux=True.
        with aux being a tuple of auxiliary variables.
        If has_aux=False, self.function should return out only.

        Args:
            variable (Tensor): Input tensor.
            has_aux (bool): Whether the function returns auxiliary variables.
            **kwargs: Keyword arguments to pass to self.function.

        Returns:
            gradients (Tensor): Gradients of the function at variable.
                ∇f(x)
            out (Tuple or Tensor): Outputs of the function at variable.
                if has_aux: out = (f(x), aux)
                else: out = f(x)
        """
        variable = keras.ops.convert_to_tensor(variable)
        if self.function is None:
            raise ValueError("Function not set. Use `set_function` to set a custom function.")
        assert self.backend in [
            "torch",
            "tensorflow",
            "jax",
        ], f"Unsupported backend: {self.backend}"

        aux = None
        if self.backend == "torch":
            if torch is None:
                raise ImportError("PyTorch is not installed.")
            variable = variable.detach().requires_grad_(True)
            if has_aux:
                out, aux = self.function(variable, **kwargs)
            else:
                out = self.function(variable, **kwargs)
            gradients = torch.autograd.grad(out, variable)[0]
        elif self.backend == "tensorflow":
            if tf is None:
                raise ImportError("TensorFlow is not installed.")
            with tf.GradientTape() as tape:
                tape.watch(variable)
                if has_aux:
                    out, aux = self.function(variable, **kwargs)
                else:
                    out = self.function(variable, **kwargs)
            gradients = tape.gradient(out, variable)
        elif self.backend == "jax":
            if jax is None:
                raise ImportError("JAX is not installed.")
            out, gradients = jax.value_and_grad(self.function, argnums=0, has_aux=has_aux)(
                variable, **kwargs
            )
            if has_aux:
                out, aux = out
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        if has_aux:
            return gradients, (out, aux)
        return gradients, out

    def get_gradient_jit_fn(self):
        """Returns a jitted function for calculating the gradients."""
        if self.backend == "jax":
            jax_mod = _import_jax()
            if jax_mod is None:
                raise ImportError("JAX is not installed.")
            return jax_mod.jit(self.gradient)
        elif self.backend == "tensorflow":
            tf = _import_tf()
            if tf is None:
                raise ImportError("TensorFlow is not installed.")
            return tf.function(self.gradient, jit_compile=True)
        elif self.backend == "torch":
            torch = _import_torch()
            if torch is None:
                raise ImportError("PyTorch is not installed.")
            return torch.compile(self.gradient)

    def get_gradient_and_value_jit_fn(self, has_aux: bool = False, disable_jit=False):
        """Returns a jitted function for calculating the gradients and function outputs."""
        func = lambda x, **kwargs: self.gradient_and_value(x, has_aux=has_aux, **kwargs)
        if disable_jit:
            return func
        if self.backend == "jax":
            if jax is None:
                raise ImportError("JAX is not installed.")
            return jax.jit(func)
        elif self.backend == "tensorflow":
            if tf is None:
                raise ImportError("TensorFlow is not installed.")
            return tf.function(
                func,
                jit_compile=True,
            )
        elif self.backend == "torch":
            # return torch.compile(func)
            raise NotImplementedError("Jitting not supported for torch backend.")
        else:
            raise UserWarning("You haven't set a jittable keras backend!")
