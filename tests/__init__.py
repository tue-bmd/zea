"""__init__ for tests"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from usbmd import init_device

init_device(backend=None)
