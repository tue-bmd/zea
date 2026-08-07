"""Tests for AutoGrad."""

import time

import keras
import numpy as np
import pytest

from zea.backend.autograd import AutoGrad

from . import DEFAULT_TEST_SEED, backend_equality_check, run_in_backend

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


def test_backend_property_matches_keras(wrapper):
    """The backend property reflects the active keras backend and is read-only."""
    assert wrapper.backend == keras.backend.backend()

    with pytest.raises(ValueError, match="Cannot change backend"):
        wrapper.backend = "jax"


def test_verbose_reports_backend(capsys):
    """``verbose=True`` prints the backend that will be used."""
    AutoGrad(verbose=True)
    assert keras.backend.backend() in capsys.readouterr().out


@backend_equality_check(
    backends=["tensorflow", "torch"],
    gt_backend=GT_BACKEND,
)  # no numpy, which has no autograd
def test_get_gradient_jit_fn(wrapper, x_input):
    """The jitted gradient function returns the same gradients as the eager one."""

    def f(x):
        return keras.ops.sum(x**3)

    wrapper.set_function(f)
    grad = keras.ops.convert_to_numpy(wrapper.get_gradient_jit_fn()(x_input))

    np.testing.assert_allclose(grad, 3 * np.asarray(x_input) ** 2, rtol=1e-4)
    return grad


def test_get_gradient_and_value_jit_fn_disable_jit(wrapper, x_input):
    """``disable_jit=True`` returns a plain (non-compiled) callable on every backend."""

    def f(x):
        return keras.ops.sum(x**2)

    wrapper.set_function(f)
    grad, out = wrapper.get_gradient_and_value_jit_fn(disable_jit=True)(x_input)

    np.testing.assert_allclose(keras.ops.convert_to_numpy(grad), 2 * np.asarray(x_input), rtol=1e-5)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(out), np.sum(np.asarray(x_input) ** 2), rtol=1e-5
    )


@pytest.mark.torch
@run_in_backend("torch")
def test_get_gradient_and_value_jit_fn_torch():
    """``torch.compile`` backs the jitted gradient_and_value on the torch backend."""
    from zea.backend.autograd import AutoGrad

    def cube_sum(x):
        return keras.ops.sum(x**3)

    wrapper = AutoGrad()
    wrapper.set_function(cube_sum)
    x = keras.ops.convert_to_tensor(np.array([1.0, 2.0, 3.0], dtype="float32"))

    grad, out = wrapper.get_gradient_and_value_jit_fn()(x)

    np.testing.assert_allclose(keras.ops.convert_to_numpy(grad), [3.0, 12.0, 27.0], rtol=1e-5)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(out), 36.0, rtol=1e-5)
