from tests.test_imports import _assert_ml_libs_not_imported, _clear_ml_libs
from usbmd.utils.device import init_device


def test_init_device_without_ml_libs():
    """Test that the init_device function does not import any ML libraries if ml_library is None.
    This is important because for example Jax should not be imported before CUDA_VISIBLE_DEVICES is set.
    """
    _clear_ml_libs()
    init_device("tensorflow", "auto:1")
    _assert_ml_libs_not_imported()


if __name__ == "__main__":
    test_init_device_without_ml_libs()
