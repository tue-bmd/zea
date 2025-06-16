"""__init__ for tests"""

import os
import sys

# Set default backend for tests
DEFAULT_TEST_BACKEND = "tensorflow"
os.environ["KERAS_BACKEND"] = DEFAULT_TEST_BACKEND
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# import sys
# def _should_use_gpu():
#     """Only use GPU if --gpu is passed and CUDA devices are available

#     In vscode you can add the following to your settings.json to enable GPU tests:
#     ```json
#     {
#         // ...existing settings...
#         "python.testing.pytestArgs": [
#             "tests",
#             "--gpu"
#         ]
#     }
#     ```
#     """
#     # pytest stores options in sys.argv, check for --gpu
#     use_gpu = "--gpu" in sys.argv
#     cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
#     return use_gpu and cuda_visible


# from zea.internal.device import init_device

# if _should_use_gpu():
#     init_device("auto:1")
# else:
#     init_device("cpu")

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
