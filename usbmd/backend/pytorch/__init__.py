"""Pytorch Ultrasound Beamforming Library.

Initialize modules for registries.
"""

import numpy as np
import torch

# pylint: disable=unused-import
from usbmd.backend.pytorch.layers import beamformers


def on_device_torch(func, inputs, device, return_numpy=False, **kwargs):
    """Compute func on device.

    Args:
        func (function): function to apply to the input data
        inputs (ndarray): input array
        device (str): cuda / gpu / cpu
        return_numpy (bool, optional): Whether to convert output
            data back to numpy. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the `func`.

    Returns:
        torch.Tensor or ndarray: The output data.

    Raises:
        AssertionError: If `func` is not a function from the torch library.

    Note:
        This function converts the `inputs` array to a torch.Tensor and moves it to
            the specified `device`.
        It then applies the `func` function to the inputs and returns the output data.
        If the output is a dictionary, it extracts the first value from the dictionary.
        If `return_numpy` is True, it converts the output data back to a numpy array
            before returning.

    Example:
        >>> import torch
        >>> def square(x):
        ...     return x ** 2
        >>> inputs = [1, 2, 3, 4, 5]
        >>> device = 'cuda'
        >>> output = on_device_torch(square, inputs, device)
        >>> print(output)
        tensor([ 1,  4,  9, 16, 25], device='cuda:0')
    """
    device = device.replace("gpu", "cuda")

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs)

    inputs = inputs.to(device)

    # check that function is a function from torch library
    # assert "torch" in str(type(func)), f"func: {func} should be a torch function"
    if hasattr(func, "to"):
        func = func.to(device)

    outputs = func(inputs, **kwargs)

    if isinstance(outputs, dict):
        # depends a bit how flexible we want to be...
        # but for now quick and dirty solution
        key = list(outputs.keys())[0]
        outputs = outputs[key]

    if return_numpy:
        if not isinstance(outputs, np.ndarray):
            outputs = outputs.cpu().numpy()

    return outputs
