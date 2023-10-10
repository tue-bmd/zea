"""GPU usage testing
"""
import pytest

from usbmd.pytorch_ultrasound.utils.gpu_config import get_device as get_device_pytorch
from usbmd.tensorflow_ultrasound.utils.gpu_config import (
    set_gpu_usage as set_gpu_usage_tf,
)


@pytest.mark.parametrize(
    "device, ml_lib",
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
def test_gpu_usage(device, ml_lib):
    """Test gpu usage setting script"""
    if ml_lib == "tensorflow":
        set_gpu_usage_tf(device=device)
    elif ml_lib == "torch":
        get_device_pytorch(device=device)


@pytest.mark.parametrize(
    "ml_lib",
    [
        "tensorflow",
        "torch",
    ],
)
def test_default_gpu_usage(ml_lib):
    """Test gpu usage setting script with defaults"""
    if ml_lib == "tensorflow":
        set_gpu_usage_tf()
    elif ml_lib == "torch":
        get_device_pytorch()
