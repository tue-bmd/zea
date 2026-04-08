"""Tensorflow Ultrasound Beamforming Library.

Initialize modules for registries.
"""

import sys
from pathlib import PosixPath

# Convert PosixPath objects to strings in sys.path
# this is necessary due to weird TF bug when importing
sys.path = [str(p) if isinstance(p, PosixPath) else p for p in sys.path]

import tensorflow as tf  # noqa: E402
