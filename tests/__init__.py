"""__init__ for tests"""

import os

# Set default backend for tests
DEFAULT_TEST_BACKEND = "tensorflow"
os.environ["KERAS_BACKEND"] = DEFAULT_TEST_BACKEND
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ZEA_FIND_H5_SHAPES_PARALLEL"] = "0"
os.environ["JAX_PLATFORMS"] = "cpu"

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
