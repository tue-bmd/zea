"""XLA flag defaults, applied from :mod:`zea`'s bootstrap before any backend loads.

XLA's GPU fusion autotuner is on by default, and it causes jax jit compilation times to
explode on large grids. The flag to disable it is part of an experimental feature that
may disappear in future versions (and invalid XLA flags crash the compiler), so this
first checks whether jaxlib supports it.
"""

import importlib.util
import mmap
import os
from pathlib import Path

from zea import log

FUSION_AUTOTUNER_FLAG = "xla_gpu_experimental_enable_fusion_autotuner"


def flag_supported(flag):
    """Check if ``flag`` shows up in jaxlib's compiled libraries."""
    spec = importlib.util.find_spec("jaxlib")
    if spec is None or not spec.submodule_search_locations:
        return False
    for root in spec.submodule_search_locations:
        for lib in sorted(Path(root).rglob("*.so")):
            try:
                if lib.stat().st_size < 5 * 1024**2:
                    continue
                with open(lib, "rb") as handle:
                    with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                        if mapped.find(flag.encode()) != -1:
                            return True
            except OSError:
                continue
    return False


def disable_fusion_autotuner(backend=None):
    """Turn the autotuner off via ``XLA_FLAGS``, unless the user asked for it.

    Args:
        backend (str, optional): Keras backend in use. Defaults to ``KERAS_BACKEND``.

    Returns:
        bool: whether the flag was added.
    """
    if backend is None:
        backend = os.environ.get("KERAS_BACKEND", "tensorflow")
    if backend.lower() not in ("jax", "numpy"):  # numpy falls back to jax for some operations
        return False

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if FUSION_AUTOTUNER_FLAG in xla_flags:
        if f"--{FUSION_AUTOTUNER_FLAG}=false" not in xla_flags:
            log.warning(
                "The XLA fusion autotuner is enabled. This may cause excessive jit "
                "compilation times when beamforming large grids."
            )
        return False

    if not flag_supported(FUSION_AUTOTUNER_FLAG):
        return False

    os.environ["XLA_FLAGS"] = f"{xla_flags} --{FUSION_AUTOTUNER_FLAG}=false".strip()
    return True
