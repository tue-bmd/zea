"""Allow different backends for autograd.

Author(s): Tristan Stevens, Oisín Nolan
Date: 22/01/2024
"""

import functools

import jax
import keras
import numpy as np
import tensorflow as tf
import torch


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
            raise ValueError(
                "Function not set. Use `set_function` to set a custom function."
            )
        assert self.backend in [
            "torch",
            "tensorflow",
            "jax",
        ], f"Unsupported backend: {self.backend}"

        if self.backend == "torch":
            variable = variable.detach().requires_grad_(True)
            out = self.function(variable, **kwargs)
            gradients = torch.autograd.grad(out, variable)[0]
            return gradients
        elif self.backend == "tensorflow":
            with tf.GradientTape() as tape:
                tape.watch(variable)
                out = self.function(variable, **kwargs)
            gradients = tape.gradient(out, variable)
            return gradients
        elif self.backend == "jax":
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
            raise ValueError(
                "Function not set. Use `set_function` to set a custom function."
            )
        assert self.backend in [
            "torch",
            "tensorflow",
            "jax",
        ], f"Unsupported backend: {self.backend}"

        if self.backend == "torch":
            # We can use: https://pytorch.org/docs/stable/generated/torch.func.grad_and_value.html
            variable = variable.detach().requires_grad_(True)
            if has_aux:
                out, aux = self.function(variable, **kwargs)
            else:
                out = self.function(variable, **kwargs)
            gradients = torch.autograd.grad(out, variable)[0]
        elif self.backend == "tensorflow":
            with tf.GradientTape() as tape:
                tape.watch(variable)
                if has_aux:
                    out, aux = self.function(variable, **kwargs)
                else:
                    out = self.function(variable, **kwargs)
            gradients = tape.gradient(out, variable)
        elif self.backend == "jax":
            out, gradients = jax.value_and_grad(
                self.function, argnums=0, has_aux=has_aux
            )(variable, **kwargs)
            if has_aux:
                out, aux = out

        if has_aux:
            return gradients, (out, aux)
        return gradients, out

    def get_gradient_jit_fn(self):
        """Returns a jitted function for calculating the gradients."""
        if self.backend == "jax":
            return jax.jit(self.gradient)
        elif self.backend == "tensorflow":
            return tf.function(self.gradient, jit_compile=True)
        elif self.backend == "torch":
            return torch.compile(self.gradient)

    def get_gradient_and_value_jit_fn(self, has_aux: bool = False, disable_jit=False):
        """Returns a jitted function for calculating the gradients and function outputs."""
        func = lambda x, **kwargs: self.gradient_and_value(x, has_aux=has_aux, **kwargs)
        if disable_jit:
            return func
        if self.backend == "jax":
            return jax.jit(func)

        elif self.backend == "tensorflow":
            return tf.function(
                func,
                jit_compile=True,
            )
        elif self.backend == "torch":
            raise NotImplementedError("Jitting not supported for torch backend.")
            # return torch.compile(func)
        else:
            raise UserWarning("You haven't set a jittable keras backend!")


if __name__ == "__main__":
    # when running this make sure to uncomment backend line above imports
    # and set the backend to the desired backend
    keras.utils.set_random_seed(42)

    def _custom_function(x):
        """A testing function for AutogradWrapper."""
        return keras.ops.sum(x**2)

    def _custom_function_with_aux(x):
        y = x**2
        test_var = y + 1
        return keras.ops.sum(y), (y, test_var)

    wrapper = AutoGrad()
    wrapper.set_function(_custom_function)

    # Example input
    x_input = np.random.rand(5)

    # Calculate autograd using the specified backend and custom function
    result = wrapper.gradient(x_input)

    print(f"Using backend: {wrapper.backend}")
    print(f"Autograd result: {result}")
    print(f"Expected result: {2 * x_input}")

    wrapper = AutoGrad()
    wrapper.set_function(_custom_function_with_aux)

    # Example input
    x_input = np.random.rand(5)

    # Calculate autograd using the specified backend and custom function
    has_aux = True
    result, out = wrapper.gradient_and_value(x_input, has_aux=has_aux)

    if has_aux:
        out, aux = out

    print(f"Using backend: {wrapper.backend}")
    print(f"Autograd result: {result}")
    print(f"Expected result: {2 * x_input}")
    print(f"out: {out}")

    if has_aux:
        print(f"Len of aux: {len(aux)}")
        for i, _aux in enumerate(aux):
            print(f"aux[{i}]: {_aux}")

    # testing jitted output
    jit_fn = wrapper.get_gradient_and_value_jit_fn(has_aux=has_aux)
    result, out = jit_fn(x_input)
    print(f"Jitted autograd result: {result}")

    # benchmark speed of jit_fn versus non-jitted function
    import timeit

    num_runs = 1000

    def _run_without_jitting():
        for _ in range(num_runs):
            wrapper.gradient_and_value(x_input, has_aux=has_aux)

    without_jitting_time = timeit.timeit(_run_without_jitting, number=1)

    def _run_with_jitting():
        for _ in range(num_runs):
            jit_fn(x_input)

    with_jitting_time = timeit.timeit(_run_with_jitting, number=1)

    print(
        f"Time taken for {num_runs} runs without jitting: {without_jitting_time:.4f} seconds"
    )
    print(
        f"Time taken for {num_runs} runs with jitting: {with_jitting_time:.4f} seconds"
    )

    del jit_fn
