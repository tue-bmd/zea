"""``zea``: *A Toolbox for Cognitive Ultrasound Imaging.*"""

import importlib
import importlib.util
import json
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
        simulator_time_domain,
        utils,
        visualize,
    )
    from .backend import device
    from .config import Config
    from .data.dataloader import Dataloader
    from .data.datasets import EXISTS, Dataset
    from .data.file import File, load_file
    from .datapaths import set_data_paths
    from .internal.device import init_device
    from .internal.setup_zea import setup, setup_config
    from .ops import Pipeline
    from .parameters import Parameters
    from .probes import Probe

try:
    # dynamically add __version__ attribute (see pyproject.toml)
    __version__ = version("zea")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source)
    __version__ = "dev"


def _bootstrap_backend():
    """Resolve, validate and announce the Keras backend.

    Runs before keras is imported, so ``KERAS_BACKEND`` is in place by the time it is.
    The backend is read from the environment and from ``keras.json`` directly rather
    than through keras itself, which would defeat the lazy imports below.
    """

    # What Keras falls back to when no backend is configured anywhere. Only used to
    # explain in error messages what would have happened without zea.
    KERAS_DEFAULT_BACKEND = "tensorflow"
    # Ordered by preference: when the user did not pick a backend, the first of these
    # that is installed is used. zea is built on top of Keras, so we mirror the Keras
    # default of preferring tensorflow rather than imposing an order of our own.
    ML_BACKENDS = ["tensorflow", "jax", "torch"]
    INSTALL_URLS = {
        "torch": "https://pytorch.org/get-started/locally/",
        "tensorflow": "https://www.tensorflow.org/install",
        "jax": "https://docs.jax.dev/en/latest/installation.html",
    }
    DOCS_URL = "https://zea.readthedocs.io/en/latest/installation.html"

    def _backend_from_keras_config():
        """Read the backend from ``keras.json``, without importing keras.

        Keras documents this file as an equal alternative to ``KERAS_BACKEND``
        (https://keras.io/getting_started/), so a backend set there is a deliberate
        choice that zea must not silently override. Returns None when the file is
        absent, unreadable or does not name a backend.
        """
        keras_home = os.environ.get("KERAS_HOME") or os.path.join("~", ".keras")
        config_path = os.path.expanduser(os.path.join(keras_home, "keras.json"))
        try:
            with open(config_path, encoding="utf-8") as file:
                return json.load(file).get("backend")
        except (OSError, ValueError):
            return None

    def _resolve_backend():
        """Pick the backend to use and verify that it is usable.

        Returns a ``(backend, installed_backends, origin)`` tuple, where ``origin`` is
        ``"env"``, ``"keras.json"`` or ``"auto"``, depending on where the backend came
        from.

        Raises ImportError if:
        1. No ML backend (torch, tensorflow, jax) is installed
        2. The chosen backend is not installed
        """
        backend_env = os.environ.get("KERAS_BACKEND")

        # Find all installed ML backends
        installed_backends = [
            backend for backend in ML_BACKENDS if importlib.util.find_spec(backend) is not None
        ]

        # Keras writes keras.json itself on first import, so the file exists even for
        # users who never touched it. Only trust it when it names a backend that is
        # actually usable here; a stale entry falls through to automatic selection
        # instead of erroring, whereas KERAS_BACKEND, being a deliberate act, does not.
        config_backend = None
        if backend_env is None:
            config_backend = _backend_from_keras_config()
            if config_backend not in installed_backends and config_backend != "numpy":
                config_backend = None

        if backend_env:
            origin = "env"
            source = f"KERAS_BACKEND is set to '{backend_env}'"
        elif config_backend:
            origin = "keras.json"
            source = f"keras.json selects backend '{config_backend}'"
        else:
            origin = "auto"
            source = f"KERAS_BACKEND is not set (Keras defaults to '{KERAS_DEFAULT_BACKEND}')"

        # If the user did not pick a backend, use one that is actually installed
        backend = (
            backend_env
            or config_backend
            or (installed_backends[0] if installed_backends else KERAS_DEFAULT_BACKEND)
        )

        # Keras' numpy backend is not standalone: it imports jax internally (see
        # keras/src/backend/numpy/nn.py), so jax is required no matter which other
        # backends happen to be installed.
        if backend == "numpy" and "jax" not in installed_backends:
            if installed_backends:
                backend_status = f"Installed backends: {', '.join(installed_backends)}."
            else:
                backend_status = (
                    "No ML backend (torch, tensorflow, jax) installed in current environment."
                )
            raise ImportError(
                f"{backend_status} {source}, but Keras' numpy "
                f"backend is not standalone: jax must be installed as well. Install it with "
                f"`pip install {__package__}[jax]` (GPU) or `pip install 'jax[cpu]'` "
                f"(CPU-only). For more information, see: {DOCS_URL}"
            )

        # Error if no backends are installed
        if not installed_backends:
            install_url = INSTALL_URLS.get(backend, "https://keras.io/getting_started/")
            raise ImportError(
                f"No ML backend (torch, tensorflow, jax) installed in current "
                f"environment. Please install at least one ML backend before importing "
                f"{__package__}. {source}, please install it first, see: "
                f"{install_url}. One simple alternative is to install with default "
                f"backend: `pip install {__package__}[jax]`. For more information, "
                f"see: {DOCS_URL}"
            )

        # Error if the chosen backend is not installed. Only reachable for an explicit
        # choice: automatic selection always lands on an installed backend.
        # (skip numpy which doesn't need installation)
        if backend != "numpy" and backend not in installed_backends:
            install_url = INSTALL_URLS.get(backend, "https://keras.io/getting_started/")
            raise ImportError(
                f"{source}, but this backend is not installed. "
                f"Installed backends: {', '.join(installed_backends)}. "
                f"Please either install '{backend}' (see: {install_url}) "
                f"or set KERAS_BACKEND to one of the installed backends "
                f"(e.g., export KERAS_BACKEND={installed_backends[0]}). "
                f"For more information, see: {DOCS_URL}"
            )

        return backend, installed_backends, origin

    backend, installed_backends, origin = _resolve_backend()

    # Export the resolution so that keras, and anything reading KERAS_BACKEND
    # (e.g. ``zea.init_device``), agrees with what was decided here.
    os.environ["KERAS_BACKEND"] = backend

    # Keras resolves its backend once, at import time, so if something imported it
    # before zea the line above came too late. Reading it back is free here: the
    # module is already loaded, so this does not import keras.
    if "keras" in sys.modules:
        try:
            active_backend = sys.modules["keras"].backend.backend()
        except AttributeError:  # keras is still initialising
            active_backend = backend
        if active_backend != backend:
            log.warning(
                f"keras was imported before zea and is using the {active_backend!r} backend, "
                f"not {backend!r}. Import zea, or set KERAS_BACKEND, before importing keras."
            )

    # No printing when using --help flag
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        return

    # Speak up only when the backend was not pinned and more than one was available:
    # neither automatic selection nor keras.json travels with a script, so the same
    # code can pick a different backend elsewhere. A single installed backend leaves
    # nothing to decide, and KERAS_BACKEND is already reproducible.
    if origin != "env" and len(installed_backends) > 1:
        alternatives = ", ".join(b for b in installed_backends if b != backend)
        picked_by = " (from keras.json)" if origin == "keras.json" else ""
        log.warning(
            f"Using backend {backend!r}{picked_by}, but {alternatives} also installed. "
            f"Set KERAS_BACKEND to pin it."
        )
    else:
        log.info(f"Using backend {backend!r}")


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
    "simulator_time_domain",
    "utils",
    "visualize",
)

_LAZY_ATTRS = {
    "device": ("zea.backend", "device"),
    "Config": ("zea.config", "Config"),
    "Dataloader": ("zea.data.dataloader", "Dataloader"),
    "Dataset": ("zea.data.datasets", "Dataset"),
    "EXISTS": ("zea.data.datasets", "EXISTS"),
    "File": ("zea.data.file", "File"),
    "load_file": ("zea.data.file", "load_file"),
    "set_data_paths": ("zea.datapaths", "set_data_paths"),
    "init_device": ("zea.internal.device", "init_device"),
    "setup": ("zea.internal.setup_zea", "setup"),
    "setup_config": ("zea.internal.setup_zea", "setup_config"),
    "Pipeline": ("zea.ops", "Pipeline"),
    "Probe": ("zea.probes", "Probe"),
    "Parameters": ("zea.parameters", "Parameters"),
    # Deprecated alias for Parameters (emits a DeprecationWarning when used).
    "Scan": ("zea.parameters", "Scan"),
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
