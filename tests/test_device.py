"""GPU usage testing"""

from itertools import product

import numpy as np
import pytest

from zea.backend import func_on_device
from zea.internal.device import init_device

from . import backend_equality_check, DEFAULT_TEST_SEED

devices = ["cpu", "gpu:0", "cuda:0", "auto:-1", "auto:1"]
backends = ["tensorflow", "torch", "jax", "auto", "numpy"]


@pytest.mark.parametrize("device, backend", list(product(devices, backends)))
def test_init_device(device, backend):
    """Test device initialization with combinations of device and backend"""
    init_device(device=device, backend=backend, verbose=False)


@pytest.mark.parametrize("backend", backends)
def test_default_init_device(backend):
    """Test gpu usage setting script with defaults"""
    init_device(backend=backend, verbose=False)


@backend_equality_check()
def test_func_on_device():
    """Test func_on_device with all backends."""

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    x = rng.standard_normal((3, 3))
    y = rng.standard_normal((3, 3))

    def f(x, y):
        return x + y

    return func_on_device(f, "cpu", x, y)
