"""__init__ for tests"""

import os

from .backend_utils import get_test_backend

# Set default backend for tests
TEST_BACKEND = get_test_backend()
os.environ["KERAS_BACKEND"] = TEST_BACKEND
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["ZEA_FIND_H5_SHAPES_PARALLEL"] = "0"

# Persist JAX compilations across runs, including small per-op ones.
if os.environ.get("ZEA_DISABLE_CACHE", "").lower() not in ("1", "true"):
    _zea_cache_dir = os.environ.get("ZEA_CACHE_DIR", os.path.join("~", ".cache", "zea"))
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.join(os.path.expanduser(_zea_cache_dir), "jax_test_compilation_cache"),
    )
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")


# Initializing the backend workers for `backend_equality_check` and `run_in_backend`.
# Note that these workers only have CPU access!
from .helpers import BackendEqualityCheck

backend_workers = BackendEqualityCheck()
backend_equality_check = backend_workers.backend_equality_check
run_in_backend = backend_workers.run_in_backend

# Parameters for dummy dataset
DUMMY_DATASET_N_FRAMES = 4
DUMMY_DATASET_GRID_SIZE_Z = 256
DUMMY_DATASET_GRID_SIZE_X = 256

DEFAULT_TEST_SEED = 42

# Populated during notebook test runs: {notebook_name: (folder, duration_seconds)}
_notebook_timings: dict[str, tuple[str, float]] = {}
