"""usbmd: A Python package for ultrasound image reconstructing using deep learning."""

import importlib.util
import os

from . import log


def setup():

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
        guide_url = install_guide_urls.get(
            backend_env, "https://keras.io/getting_started/"
        )
        raise AssertionError(
            "No ML backend (torch, tensorflow, jax) installed in current environment. "
            f"Please install at least one ML backend before importing {__package__} or any other library. "
            f"Current KERAS_BACKEND is set to '{backend_env}', please install it first, see: {guide_url}. "
            f"One simple alternative is to install with default backend: `pip install {__package__}[jax]`."
        )

    _check_backend_installed()

    import keras  # pylint: disable=import-outside-toplevel

    log.info(f"Using backend {keras.backend.backend()!r}")

    # dynamically add __version__ attribute (see pyproject.toml)
    globals()["__version__"] = __import__("importlib.metadata").metadata.version(
        __package__
    )


# call and clean up namespace
setup()
del setup

# Main (isort: split)
from .config import Config
from .data.datasets import Dataset, Folder
from .data.file import File, load_usbmd_file
from .datapaths import set_data_paths
from .interface import Interface
from .internal.device import init_device
from .internal.setup_usbmd import set_backend, setup, setup_config
from .ops import Pipeline
from .probes import Probe
from .scan import Scan
