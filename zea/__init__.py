"""``zea``: *A Toolbox for Cognitive Ultrasound Imaging.*"""

import importlib
import importlib.util
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from . import log

if TYPE_CHECKING:
    # Static-only imports so IDEs and type checkers can resolve the public API
    # without actually importing keras (or any backend) at runtime.
    from . import (
        agent,
        beamform,
        data,
        display,
        func,
        io_lib,
        metrics,
        models,
        ops,
        simulator,
        utils,
        visualize,
    )
    from .backend import device
    from .config import Config
    from .data.dataloader import Dataloader
    from .data.datasets import Dataset
    from .data.file import File, load_file
    from .datapaths import set_data_paths
    from .internal.device import init_device
    from .internal.setup_zea import setup, setup_config
    from .ops import Pipeline
    from .probes import Probe
    from .scan import Parameters

try:
    # dynamically add __version__ attribute (see pyproject.toml)
    __version__ = version("zea")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source)
    __version__ = "dev"


def _bootstrap_backend():
    """Setup function to initialize the zea package."""

    # No printing when using --help flag
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        return

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

    # Read from the env var rather than calling ``keras.backend.backend()``
    # so that importing ``zea`` does not import ``keras``.
    log.info(f"Using backend {os.environ.get('KERAS_BACKEND', 'tensorflow')!r}")


# Skip backend bootstrap when building on ReadTheDocs
if os.environ.get("READTHEDOCS") != "True":
    _bootstrap_backend()

del _bootstrap_backend

# Public API is loaded lazily so that ``import zea`` does not pull in
# ``keras`` (or any ML backend) transitively. In particular this lets
# ``zea.init_device(...)`` be called *before* keras is imported, which is the
# whole point of ``init_device``: it sets ``CUDA_VISIBLE_DEVICES`` and related
# env vars that must be in place before the backend initialises.
_LAZY_SUBMODULES = (
    "agent",
    "beamform",
    "data",
    "display",
    "func",
    "io_lib",
    "metrics",
    "models",
    "ops",
    "simulator",
    "utils",
    "visualize",
)

_LAZY_ATTRS = {
    "device": ("zea.backend", "device"),
    "Config": ("zea.config", "Config"),
    "Dataloader": ("zea.data.dataloader", "Dataloader"),
    "Dataset": ("zea.data.datasets", "Dataset"),
    "File": ("zea.data.file", "File"),
    "load_file": ("zea.data.file", "load_file"),
    "set_data_paths": ("zea.datapaths", "set_data_paths"),
    "init_device": ("zea.internal.device", "init_device"),
    "setup": ("zea.internal.setup_zea", "setup"),
    "setup_config": ("zea.internal.setup_zea", "setup_config"),
    "Pipeline": ("zea.ops", "Pipeline"),
    "Probe": ("zea.probes", "Probe"),
    "Parameters": ("zea.scan", "Parameters"),
    # Deprecated alias for Parameters (emits a DeprecationWarning when used).
    "Scan": ("zea.scan", "Scan"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(_LAZY_SUBMODULES))
