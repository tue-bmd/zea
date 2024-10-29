"""GPU configuration utilities for Jax.

- **Author(s)**     : Tristan Stevens
- **Date**          : 25/10/2024
"""

import jax

from usbmd.utils import log
from usbmd.utils.gpu_utils import get_device as _get_device
from usbmd.utils.gpu_utils import selected_gpu_ids_to_device


def get_device(device="auto:1", verbose=True, hide_others=True):
    """Returns a device for Jax using the device argument and
    by searching for available GPUs and their available memory.
    If CUDA is unavailable or CPU is selected, use CPU.

    Note that currently only one GPU is supported for Jax.

    Args:
        device (str/int/list): GPU device(s) to select. Defaults to 'auto:1'.
            - If 'cpu', use CPU.
            - If 'gpu', select GPU based on available memory.
                Throw an error if no GPU is available.
            - If None, try to select GPU based on available memory.
                Fall back to CPU if no GPU is available.
            - If an integer or a list of integers, use the corresponding
                GPU(s). If the list contains None values (e.g. [0, None, 2]), a
                GPU will be selected based on available memory.
            - If formatted as 'cuda:xx' or 'gpu:xx', where xx is an integer,
                use the corresponding GPU(s).
            - If formatted as 'auto:xx', where xx is an integer, automatically
                select xx GPUs based on available memory. If xx is -1, use all available GPUs.
        verbose (bool): prints output if True.
        hide_others (bool): if True, hide other GPUs from the system by setting
            the CUDA_VISIBLE_DEVICES environment variable.

    Returns:
        device (str): Available device that torch can use for its computations.
    """
    try:
        jax.devices("gpu")
    except RuntimeError:
        log.warning("Cuda not available, fallback to CPU.")
        return "cpu"

    selected_gpu_ids = _get_device(device, verbose=verbose, hide_others=hide_others)

    return selected_gpu_ids_to_device(selected_gpu_ids)


if __name__ == "__main__":
    # Initialize GPU
    # Example on how to use gpu config functions
    device = get_device("auto:1")
    x = jax.random.normal(jax.random.PRNGKey(0), (10,))

    # or use on_device_jax from usbmd.backend.jax
    device = device.split(":")
    device_type, device_number = device
    device_number = int(device_number)

    jax_device = jax.devices(device_type)[device_number]
    jax.device_put(x, jax_device)
    print(x)
