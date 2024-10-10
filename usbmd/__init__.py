"""
The official documentation for the Python package `usbmd` - a convenient ultrasound toolbox.

.. include:: ./README.md
"""

__version__ = "2.1.0"

import os

# set to numpy as default to prevent unecessary imports,
# but will be overwritten by config.ml_library later if using setup()
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "numpy"
    print(
        "`KERA_BACKEND` not set. usbmd will set it to `numpy` by default. "
        "It is recommended to set it manually using `os.environ['KERAS_BACKEND']` "
        "at top of your script before importing usbmd or any other library."
    )

# Main (isort: split)
from .config import Config, load_config_from_yaml
from .datapaths import set_data_paths
from .interface import Interface
from .setup_usbmd import setup, setup_config
from .utils import log
from .utils.device import init_device
