"""Loads the environment variable USBMD_BACKEND can import the corresponding backend."""

import importlib.util

from usbmd.utils.checks import _BACKENDS

_ML_LIB_SET = False
for lib in _BACKENDS:
    if importlib.util.find_spec(str(lib)):
        if lib == "torch":
            import usbmd.backend.pytorch  # pylint: disable=import-outside-toplevel, unused-import

            _ML_LIB_SET = True
        if lib == "tensorflow":
            import usbmd.backend.tensorflow  # pylint: disable=import-outside-toplevel, unused-import

            _ML_LIB_SET = True
