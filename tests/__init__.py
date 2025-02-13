"""__init__ for tests"""

import os

DEFAULT_TEST_BACKEND = "tensorflow"

os.environ["KERAS_BACKEND"] = DEFAULT_TEST_BACKEND
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from usbmd import init_device

init_device(backend=None)
