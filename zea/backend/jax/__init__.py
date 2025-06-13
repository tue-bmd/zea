"""Jax utilities for zea."""

import jax
import numpy as np


def on_device_jax(func, inputs, device, return_numpy=False, **kwargs):
    """Applies a JAX function to inputs on a specified device.

    Args:
        func (callable): The function to apply.
        inputs (ndarray): Input array.
        device (str): Device string, e.g. ``'cuda'``, ``'gpu'``, or ``'cpu'``.
        return_numpy (bool, optional): Whether to convert output
            data back to numpy. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the ``func``.

    Returns:
        jax.numpy.DeviceArray or ndarray: The output data.

    Raises:
        AssertionError: If ``func`` is not a function from the JAX library.

    Note:
        This function converts the ``inputs`` array to a JAX array and moves
        it to the specified ``device``. It then applies the ``func`` function to the inputs
        and returns the output data. If the output is a dictionary, it extracts the first value
        from the dictionary. If ``return_numpy`` is True, it converts the output data back to a
        numpy array before returning.

    Example:
        .. code-block:: python

            import jax.numpy as jnp


            def square(x):
                return x**2


            inputs = [1, 2, 3, 4, 5]
            device = "gpu"
            output = on_device_jax(square, inputs, device)
    """
    device = device.split(":")
    if len(device) == 2:
        device_type, device_number = device
        device_number = int(device_number)
    else:
        # if no device number is specified, use the first device
        device_type = device[0]
        device_number = 0

    if device_number > len(jax.devices(device_type)):
        raise ValueError(
            f"Device {device} is not available from JAX devices: {jax.devices(device_type)}"
        )

    jax_device = jax.devices(device_type)[device_number]

    with jax.default_device(jax_device):
        outputs = func(inputs, **kwargs)

    if isinstance(outputs, dict):
        outputs = list(outputs.values())[0]

    if return_numpy:
        outputs = np.array(outputs)

    return outputs
