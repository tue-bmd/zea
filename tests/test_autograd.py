"""Tests for AutoGrad."""

import time

import keras
import numpy as np
import pytest

from zea.backend.autograd import AutoGrad

from . import backend_equality_check, DEFAULT_TEST_SEED

GT_BACKEND = "jax"  # ground truth backend for equality check
OTHER_BACKENDS = ["torch", "tensorflow"]  # reference backends for equality check


@pytest.fixture
def x_input():
    """Generate random input tensor for testing."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    return rng.standard_normal(5)


@pytest.fixture
def wrapper():
    """Create an instance of AutoGrad wrapper."""
    return AutoGrad()


@backend_equality_check(
    backends=OTHER_BACKENDS,
    gt_backend=GT_BACKEND,
)  # no numpy which has no autograd
def test_gradient_simple(wrapper, x_input):
    """Test the gradient of a simple function."""

    def f(x):
        return keras.ops.sum(x**2)

    wrapper.set_function(f)
    grad = wrapper.gradient(x_input)
    np.testing.assert_allclose(grad, 2 * x_input, rtol=1e-5)
    return grad


@backend_equality_check(
    backends=OTHER_BACKENDS,
    gt_backend=GT_BACKEND,
)  # no numpy, which has no autograd
def test_gradient_and_value_with_aux(wrapper, x_input):
    """Test the gradient and value of a function with auxiliary outputs."""

    def f(x):
        y = x**2
        test_var = y + 1
        return keras.ops.sum(y), (y, test_var)

    wrapper.set_function(f)
    grad, (out, aux) = wrapper.gradient_and_value(x_input, has_aux=True)

    grad = keras.ops.convert_to_numpy(grad)
    x_input = keras.ops.convert_to_numpy(x_input)
    out = keras.ops.convert_to_numpy(out)
    aux = tuple(keras.ops.convert_to_numpy(a) for a in aux)

    np.testing.assert_allclose(grad, 2 * x_input, rtol=1e-5)
    np.testing.assert_allclose(out, np.sum(x_input**2), rtol=1e-5)
    assert len(aux) == 2
    np.testing.assert_allclose(aux[0], x_input**2, rtol=1e-5)
    np.testing.assert_allclose(aux[1], x_input**2 + 1, rtol=1e-5)
    return grad


def test_gradient_function_not_set(wrapper, x_input):
    """Test that an error is raised when the function is not set."""
    with pytest.raises(ValueError):
        wrapper.gradient(x_input)


def test_gradient_and_value_function_not_set(wrapper, x_input):
    """Test that an error is raised when the function is not set."""
    with pytest.raises(ValueError):
        wrapper.gradient_and_value(x_input)


@pytest.mark.performance
def test_gradient_and_value_jit_timing(wrapper, x_input):
    """Performance test for jitted vs non-jitted gradient_and_value."""
    has_aux = True

    def f(x):
        y = x**2
        test_var = y + 1
        return keras.ops.sum(y), (y, test_var)

    wrapper.set_function(f)
    jit_fn = wrapper.get_gradient_and_value_jit_fn(has_aux=has_aux)

    num_runs = 1000

    # Warm up JIT
    jit_fn(x_input)

    start = time.time()
    for _ in range(num_runs):
        wrapper.gradient_and_value(x_input, has_aux=has_aux)
    non_jit_time = time.time() - start

    start = time.time()
    for _ in range(num_runs):
        jit_fn(x_input)
    jit_time = time.time() - start

    print(f"Non-jitted: {non_jit_time:.4f}s, Jitted: {jit_time:.4f}s")
