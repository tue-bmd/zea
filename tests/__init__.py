"""__init__ for tests"""

import os

# Set default backend for tests
DEFAULT_TEST_BACKEND = "tensorflow"
os.environ["KERAS_BACKEND"] = DEFAULT_TEST_BACKEND
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Initializing the backend workers for `backend_equality_check` and `run_in_backend`.
# Note that these workers only have CPU access!
from .helpers import BackendEqualityCheck

backend_workers = BackendEqualityCheck()
backend_equality_check = backend_workers.backend_equality_check
run_in_backend = backend_workers.run_in_backend

# Parameters for dummy dataset
DUMMY_DATASET_N_FRAMES = 4
DUMMY_DATASET_N_Z = 256
DUMMY_DATASET_N_X = 256
