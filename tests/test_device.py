""" Tests for the device module. """

from tests.test_imports import _no_ml_lib_import  # pylint: disable=unused-import
from usbmd.utils.device import init_device


def test_init_device_without_ml_libs(_no_ml_lib_import):
    """
    Test that the init_device function does not import any ML libraries if backend is None.
    This is important because, for example, Jax should not be imported before
    CUDA_VISIBLE_DEVICES is set.
    """
    # Run the init_device function within the mock_import fixture context
    init_device(None, "auto:1")
