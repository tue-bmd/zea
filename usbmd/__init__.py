"""
The official documentation for the Python package `usbmd` - a convenient ultrasound toolbox.

.. include:: ./README.md
"""

__version__ = "2.1.0"

import inspect
import os


def imported_from_main():
    """Check if the module was imported from __main__.py
    or usbmd was called from the command line.

    In this case, we don't need to print the warning message, as
    we will set the backend in __main__.py just after this __init__.py.
    """
    for frame in inspect.stack():
        print(frame.filename)
        filename = frame.filename
        if filename.endswith("__main__.py") or filename.endswith("bin/usbmd"):
            return True
    return False


# set to numpy as default to prevent unecessary imports (of tensorflow),
# but preferred action is to set it manually before importing usbmd
# for instance at top of script:
# os.environ["KERAS_BACKEND"] = "jax"
# or in your terminal:
# export KERAS_BACKEND=jax

# this module was imported from __main__.py we don't need this warning
# as this will set the backend in __main__.py just after this __init__.py
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "numpy"
    if not imported_from_main():
        print(
            "`KERAS_BACKEND` not set. usbmd will set it to `numpy` by default. "
            "It is recommended to set it manually using `os.environ['KERAS_BACKEND']` "
            "at top of your script before importing usbmd or any other library."
        )

# Main (isort: split)
from .config import Config, load_config_from_yaml
from .datapaths import set_data_paths
from .interface import Interface
from .setup_usbmd import set_backend, setup, setup_config
from .utils.device import init_device
