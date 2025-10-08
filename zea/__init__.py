"""``zea``: *A Toolbox for Cognitive Ultrasound Imaging.*"""

import importlib.util
import os

from . import log

# dynamically add __version__ attribute (see pyproject.toml)
# __version__ = __import__("importlib.metadata").metadata.version(__package__)
__version__ = "0.0.6"


def _bootstrap_backend():
    """Setup function to initialize the zea package."""

    def _check_backend_installed():
        """Assert that at least one ML backend (torch, tensorflow, jax) is installed.
        If not, raise an AssertionError with a helpful install message.
        """

        ml_backends = ["torch", "tensorflow", "jax"]
        for backend in ml_backends:
            if importlib.util.find_spec(backend) is not None:
                return

        backend_env = os.environ.get("KERAS_BACKEND", "numpy")
        install_guide_urls = {
            "torch": "https://pytorch.org/get-started/locally/",
            "tensorflow": "https://www.tensorflow.org/install",
            "jax": "https://docs.jax.dev/en/latest/installation.html",
        }
        guide_url = install_guide_urls.get(backend_env, "https://keras.io/getting_started/")
        raise ImportError(
            "No ML backend (torch, tensorflow, jax) installed in current environment. "
            f"Please install at least one ML backend before importing {__package__} or "
            f"any other library. Current KERAS_BACKEND is set to '{backend_env}', "
            f"please install it first, see: {guide_url}. One simple alternative is to "
            f"install with default backend: `pip install {__package__}[jax]`."
        )

    _check_backend_installed()

    from keras.backend import backend as keras_backend

    log.info(f"Using backend {keras_backend()!r}")


# call and clean up namespace
_bootstrap_backend()
del _bootstrap_backend

from . import (
    agent,
    beamform,
    data,
    display,
    io_lib,
    keras_ops,
    metrics,
    models,
    simulator,
    tensor_ops,
    utils,
    visualize,
)
from .config import Config
from .data.datasets import Dataset, Folder
from .data.file import File, load_file
from .datapaths import set_data_paths
from .interface import Interface
from .internal.device import init_device
from .internal.setup_zea import setup, setup_config
from .ops import Pipeline
from .probes import Probe
from .scan import Scan
