"""``zea``: *A Toolbox for Cognitive Ultrasound Imaging.*"""

import importlib.util
import os

from . import log

# dynamically add __version__ attribute (see pyproject.toml)
# __version__ = __import__("importlib.metadata").metadata.version(__package__)
__version__ = "0.0.8"


def _bootstrap_backend():
    """Setup function to initialize the zea package."""

    def _check_backend_installed():
        """Verify that the required ML backend is installed.

        Raises ImportError if:
        1. No ML backend (torch, tensorflow, jax) is installed
        2. KERAS_BACKEND points to a backend that is not installed
        """
        ML_BACKENDS = ["torch", "tensorflow", "jax"]
        INSTALL_URLS = {
            "torch": "https://pytorch.org/get-started/locally/",
            "tensorflow": "https://www.tensorflow.org/install",
            "jax": "https://docs.jax.dev/en/latest/installation.html",
        }
        KERAS_DEFAULT_BACKEND = "tensorflow"
        DOCS_URL = "https://zea.readthedocs.io/en/latest/installation.html"

        # Determine which backend Keras will try to use
        backend_env = os.environ.get("KERAS_BACKEND")
        effective_backend = backend_env or KERAS_DEFAULT_BACKEND

        # Find all installed ML backends
        installed_backends = [
            backend for backend in ML_BACKENDS if importlib.util.find_spec(backend) is not None
        ]

        # Error if no backends are installed
        if not installed_backends:
            if backend_env:
                backend_status = f"KERAS_BACKEND is set to '{backend_env}'"
            else:
                backend_status = f"KERAS_BACKEND is not set (defaults to '{KERAS_DEFAULT_BACKEND}')"
            install_url = INSTALL_URLS.get(effective_backend, "https://keras.io/getting_started/")
            raise ImportError(
                f"No ML backend (torch, tensorflow, jax) installed in current "
                f"environment. Please install at least one ML backend before importing "
                f"{__package__}. {backend_status}, please install it first, see: "
                f"{install_url}. One simple alternative is to install with default "
                f"backend: `pip install {__package__}[jax]`. For more information, "
                f"see: {DOCS_URL}"
            )

        # Error if the effective backend is not installed
        # (skip numpy which doesn't need installation)
        if effective_backend not in ["numpy"] and effective_backend not in installed_backends:
            if backend_env:
                backend_status = f"KERAS_BACKEND environment variable is set to '{backend_env}'"
            else:
                backend_status = (
                    f"KERAS_BACKEND is not set, which defaults to '{KERAS_DEFAULT_BACKEND}'"
                )
            install_url = INSTALL_URLS.get(effective_backend, "https://keras.io/getting_started/")
            raise ImportError(
                f"{backend_status}, but this backend is not installed. "
                f"Installed backends: {', '.join(installed_backends)}. "
                f"Please either install '{effective_backend}' (see: {install_url}) "
                f"or set KERAS_BACKEND to one of the installed backends "
                f"(e.g., export KERAS_BACKEND={installed_backends[0]}). "
                f"For more information, see: {DOCS_URL}"
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
