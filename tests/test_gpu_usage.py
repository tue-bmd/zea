"""GPU usage testing
"""
import pytest

from usbmd.pytorch_ultrasound.utils.gpu_config import \
    set_gpu_usage as set_gpu_usage_pytorch
from usbmd.tensorflow_ultrasound.utils.gpu_config import \
    set_gpu_usage as set_gpu_usage_tf


@pytest.mark.parametrize('device', ['cpu', 'gpu:0', 'auto:-1', 'cuda:0'])
def test_gpu_usage(device):
    """Test gpu usage setting script"""
    # this only works when this tests runs first, since gpu usage should be done first
    # set_gpu_usage()
    set_gpu_usage_tf(device=device)

    set_gpu_usage_pytorch(device=device)
