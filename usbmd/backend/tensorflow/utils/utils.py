""" Utility functions for USBMD tensorflow modules."""

import numpy as np
import tensorflow as tf


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def tf_snapshot(obj) -> dict:
    """Returns a snapshot of the object parameters as a dictionary of tensors.

    Returns:
        dict: The scan parameters as a dictionary of tensors.
    """
    EXCEPTIONS = ["angles", "_angles"]

    snapshot = DotDict()
    for key in dir(obj):
        if key[0] != "_" and key not in EXCEPTIONS:
            value = getattr(obj, key)
            if isinstance(value, (np.ndarray, int, float, list)):

                # if data is of double precision, convert to float32
                if isinstance(value, np.ndarray) and value.dtype == np.float64:
                    dtype = tf.float32
                else:
                    dtype = None

                snapshot[key] = tf.convert_to_tensor(value, dtype=dtype)
    return snapshot
