"""Tests for the device module."""

from tests.test_imports import _no_ml_lib_import
from usbmd.utils.device import init_device


def test_init_device_without_ml_libs():
    """
    Test that the init_device function does not import any ML libraries if backend is None.
    This is important because, for example, Jax should not be imported before
    CUDA_VISIBLE_DEVICES is set.
    """
    with _no_ml_lib_import(allow_keras_backend=False):
        init_device(backend=None)
