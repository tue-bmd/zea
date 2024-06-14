"""GPU configuration utilities for Pytorch.

- **Author(s)**     : Ben Luijten, Tristan Stevens, Wessel van Nierop
- **Date**          : 24 May 2023
"""

import torch

from usbmd.utils import log
from usbmd.utils.gpu_utils import get_gpu_memory, select_gpus


def get_device(device="auto:1", verbose=True, hide_others=True):
    """Returns a device for Pytorch using the device argument and
    by searching for available GPUs and their available memory.
    If CUDA is unavailable or CPU is selected, use CPU.

    Note that currently only one GPU is supported for Pytorch.

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
    if device.lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        log.warning("Cuda not available, fallback to CPU.")
        return "cpu"

    if verbose:
        header = "GPU settings"
        print("-" * 2 + header.center(50 - 4, "-") + "-" * 2)

    memory = get_gpu_memory(verbose=verbose)
    if memory is None:  # nvidia-smi not working, fallback to CPU
        return "cpu"

    gpu_ids = list(range(torch.cuda.device_count()))

    selected_gpu_ids = select_gpus(
        available_gpu_ids=gpu_ids,
        memory_free=memory,
        device=device,
        verbose=verbose,
        hide_others=hide_others,
    )
    if selected_gpu_ids is None:
        return "cpu"
    if len(selected_gpu_ids) > 1:
        log.warning(
            (
                "Config specified multiple GPU's while our Pytorch"
                "implementation does not currently support this."
                f"Fallback to single GPU: {selected_gpu_ids[0]}"
            )
        )
    if verbose:
        print("-" * 50)

    return f"cuda:{selected_gpu_ids[0]}"


if __name__ == "__main__":
    # Initialize GPU
    # Example on how to use gpu config functions
    device = get_device("auto:1")
    x = torch.randn(10).to(device)
    print(x)
