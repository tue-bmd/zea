"""__init__ for tests"""

import os

# Running tests on cpu for now...
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

# Set default backend for tests
DEFAULT_TEST_BACKEND = "tensorflow"
os.environ["KERAS_BACKEND"] = DEFAULT_TEST_BACKEND
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
