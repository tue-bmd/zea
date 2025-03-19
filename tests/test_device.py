"""GPU usage testing"""

import pytest

from usbmd.utils.device import init_device


@pytest.mark.parametrize(
    "device, backend",
    [
        ("cpu", "tensorflow"),
        ("cpu", "torch"),
        ("gpu:0", "tensorflow"),
        ("gpu:0", "torch"),
        ("cuda:0", "tensorflow"),
        ("cuda:0", "torch"),
        ("auto:-1", "tensorflow"),
        ("auto:-1", "torch"),
    ],
)
def test_init_device(device, backend):
    """Test gpu usage setting script"""
    init_device(device=device, backend=backend)


@pytest.mark.parametrize(
    "backend",
    [
        "tensorflow",
        "torch",
        "jax",
        "auto",
        "numpy",
        None,
    ],
)
def test_default_init_device(backend):
    """Test gpu usage setting script with defaults"""
    init_device(backend=backend)
