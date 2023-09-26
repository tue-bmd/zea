"""GPU usage testing
"""
import pytest

from usbmd.pytorch_ultrasound.utils.gpu_config import \
    set_gpu_usage as set_gpu_usage_pytorch
from usbmd.tensorflow_ultrasound.utils.gpu_config import \
    set_gpu_usage as set_gpu_usage_tf


@pytest.mark.parametrize('device, ml_lib', [
    ('cpu', 'tensorflow'),
    ('cpu', 'torch'),
    ('gpu:0', 'tensorflow'),
    ('gpu:0', 'torch'),
    ('cuda:0', 'tensorflow'),
    ('cuda:0', 'torch'),
    ('auto:-1', 'tensorflow'),
    ('auto:-1', 'torch'),
])
def test_gpu_usage(device, ml_lib):
    """Test gpu usage setting script"""
    if ml_lib == 'tensorflow':
        set_gpu_usage_tf(device=device)
    elif ml_lib == 'torch':
        set_gpu_usage_pytorch(device=device)
