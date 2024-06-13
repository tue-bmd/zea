"""Loads the environment variable USBMD_BACKEND can import the corresponding backend."""

import os

from usbmd.utils.checks import _BACKENDS

USBMD_BACKEND = os.environ.get("USBMD_BACKEND")
if USBMD_BACKEND is not None:
    USBMD_BACKEND = USBMD_BACKEND.lower()

assert USBMD_BACKEND in _BACKENDS, ValueError(f"Unsupported backend: {USBMD_BACKEND}.")


def import_backend():
    """Useful to load the registries"""
    if USBMD_BACKEND == "torch":
        import usbmd.backend.pytorch  # pylint: disable=import-outside-toplevel, unused-import

        return True
    elif USBMD_BACKEND == "tensorflow":
        import usbmd.backend.tensorflow  # pylint: disable=import-outside-toplevel, unused-import

        return True
    return False
