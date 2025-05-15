"""GPU usage testing"""

from itertools import product

import pytest

from usbmd.internal.device import init_device

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
