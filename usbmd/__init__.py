"""
The official documentation for the Python package `usbmd` - a convenient ultrasound toolbox.

.. include:: ./README.md
"""

__version__ = "2.0.0"

# Main (isort: split)
from .config import Config, load_config_from_yaml
from .setup_usbmd import setup, setup_config
from .utils import log
from .utils.device import init_device
from .datapaths import set_data_paths
