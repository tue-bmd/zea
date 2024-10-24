import copy
import importlib
import sys

import keras


def reload_usbmd():
    """Reloads usbmd. This is useful when changing the backend.
    Taken from `keras.config.set_backend`"""

    # Clear module cache.
    loaded_modules = [key for key in sys.modules.keys() if key.startswith("usbmd")]
    for key in loaded_modules:
        del sys.modules[key]
    # Reimport usbmd with the new backend (set via KERAS_BACKEND).
    import usbmd  # pylint: disable=import-outside-toplevel

    # Finally: refresh all imported Keras submodules.
    globs = copy.copy(globals())
    for key, value in globs.items():
        if value.__class__ == usbmd.__class__:
            if str(value).startswith("<module 'usbmd."):  # pylint: disable=no-member
                module_name = str(value)
                module_name = module_name[module_name.find("'") + 1 :]
                module_name = module_name[: module_name.find("'")]
                globals()[key] = importlib.import_module(module_name)


def set_backend(backend: str):
    """Set compute backend

    Note: Make sure to reimport any module you are using that uses keras
    directly (has import keras or derivative at the top of the file).
    This can be done with the importlib module.

    """
    # set keras backend
    if keras.config.backend() != backend:
        keras.config.set_backend(backend)
        reload_usbmd()
