"""GPU usage testing
"""
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage


def test_gpu_usage():
    """Test gpu usage setting script"""
    set_gpu_usage()
    set_gpu_usage(device='cpu')
