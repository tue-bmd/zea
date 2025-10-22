"""Pytorch Ultrasound Beamforming Library.

Initialize modules for registries.
"""

import torch


def func_on_device(func, device, *args, **kwargs):
    """Moves all tensor arguments of a function to a specified device before calling it.

    Args:
        func (callable): Function to be called.
        device (str or torch.device): Device to move tensors to.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.
    Returns:
        The output of the function.
    """
    if device is None:
        return func(*args, **kwargs)

    if isinstance(device, str):
        device = torch.device(device)

    def move_to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return type(x)(move_to_device(i) for i in x)
        elif isinstance(x, dict):
            return {k: move_to_device(v) for k, v in x.items()}
        else:
            return x

    args = move_to_device(args)
    kwargs = move_to_device(kwargs)

    return func(*args, **kwargs)
