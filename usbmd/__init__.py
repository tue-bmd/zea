"""
The official documentation for the Python package `usbmd` - a convenient ultrasound toolbox.

.. include:: ./README.md
"""

# import usbmd.backend.aliases

from .__version__ import __version__

# Main (isort: split)
from .config import Config, load_config_from_yaml
from .setup_usbmd import setup, setup_config
