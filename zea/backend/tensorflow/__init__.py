"""Tensorflow Ultrasound Beamforming Library.

Initialize modules for registries.
"""

import sys
from pathlib import PosixPath

import numpy as np

# Convert PosixPath objects to strings in sys.path
# this is necessary due to weird TF bug when importing
sys.path = [str(p) if isinstance(p, PosixPath) else p for p in sys.path]

import tensorflow as tf  # noqa: E402

from .dataloader import make_dataloader  # noqa: E402


def on_device_tf(func, inputs, device, return_numpy=False, **kwargs):
    """Applies a Tensorflow function to inputs on a specified device.

    Args:
        func (function): Function to apply to the input data.
        inputs (ndarray): Input array.
        device (str): Device string, e.g. ``'cuda'``, ``'gpu'``, or ``'cpu'``.
        return_numpy (bool, optional): Whether to convert output
            data back to numpy. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the ``func``.

    Returns:
        tf.Tensor or ndarray: The output data.

    Raises:
        AssertionError: If ``func`` is not a function from the tensorflow library.

    Note:
        This function converts the ``inputs`` array to a tf.Tensor and moves
        it to the specified ``device``. It then applies the ``func`` function to the inputs
        and returns the output data. If the output is a dictionary, it extracts the first value
        from the dictionary. If ``return_numpy`` is True, it converts the output data back to a
        numpy array before returning.

    Example:
        .. code-block:: python

            import tensorflow as tf


            def square(x):
                return x**2


            inputs = [1, 2, 3, 4, 5]
            device = "cuda"
            output = on_device_tf(square, inputs, device)
    """
    device = device.replace("cuda", "gpu")

    with tf.device(device):
        outputs = func(inputs, **kwargs)

    if return_numpy:
        if not isinstance(outputs, np.ndarray):
            outputs = outputs.numpy()
    return outputs
