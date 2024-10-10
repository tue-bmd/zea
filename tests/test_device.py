""" Tests for the device module. """

import builtins

from tests.test_imports import import_fail_on_ml_libs, original_import
from usbmd.utils.device import init_device


def test_init_device_without_ml_libs():
    """
    Test that the init_device function does not import any ML libraries if ml_library is None.
    This is important because for example Jax should not be imported before
    CUDA_VISIBLE_DEVICES is set.
    """

    # Override the built-in import function
    builtins.__import__ = import_fail_on_ml_libs

    init_device(None, "auto:1")

    # Restore the original import function
    builtins.__import__ = original_import
