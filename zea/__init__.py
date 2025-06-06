"""zea: A Toolbox for Cognitive Ultrasound Imaging."""

__version__ = "2.4.0"

import inspect
import os

from . import log


def _imported_from_main():
    """Check if the module was imported from __main__.py
    or zea was called from the command line.

    In this case, we don't need to print the warning message, as
    we will set the backend in __main__.py just after this __init__.py.
    """
    for frame in inspect.stack():
        filename = frame.filename
        if filename.endswith("__main__.py") or filename.endswith("bin/zea"):
            return True
    return False


# set to numpy as default to prevent unecessary imports (of tensorflow),
# but preferred action is to set it manually before importing zea
# for instance at top of script:
# os.environ["KERAS_BACKEND"] = "jax"
# or in your terminal:
# export KERAS_BACKEND=jax

# this module was imported from __main__.py we don't need this warning
# as this will set the backend in __main__.py just after this __init__.py
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "numpy"
    if not _imported_from_main():
        print(
            "`KERAS_BACKEND` not set. zea will set it to `numpy` by default. "
            "It is recommended to set it manually using `os.environ['KERAS_BACKEND']` "
            "at top of your script before importing zea or any other library."
        )

# Main (isort: split)
from .config import Config
from .data.datasets import Dataset, Folder
from .data.file import File, load_zea_file
from .datapaths import set_data_paths
from .interface import Interface
from .internal.device import init_device
from .internal.setup_zea import set_backend, setup, setup_config
from .ops import Pipeline
from .probes import Probe
from .scan import Scan
