"""Device utilities"""

import os
from typing import Union

from usbmd.utils.gpu_utils import get_device, hide_gpus, selected_gpu_ids_to_device


def set_memory_growth_tf():
    """Attempts to allocate only as much GPU memory as needed for the runtime allocations"""
    try:
        import tensorflow as tf
    except:
        return

    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in tf.config.get_visible_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def backend_key(backend):
    if backend == "torch":
        return "cuda"
    if backend == "tensorflow":
        return "gpu"
    if backend == "jax":
        return "gpu"
    return "gpu"


def backend_cuda_available(backend):
    if backend == "torch":
        try:
            import torch
        except:
            return False
        return torch.cuda.is_available()
    if backend == "tensorflow":
        try:
            import tensorflow as tf
        except:
            return False
        return tf.test.is_gpu_available()
    if backend == "jax":
        try:
            import jax
        except:
            return False
        try:
            return bool(jax.devices("gpu"))
        except:
            return False
    return False


def init_device(
    device: Union[str, int, list] = "auto:1",
    backend: Union[str, None] = "auto",
    hide_devices: Union[int, list] = None,
    allow_preallocate: bool = True,
    verbose: bool = True,
):
    """Selects a GPU or CPU device based on the config.

    Args:
        backend (str): String indicating which backend to use. Can be
            'torch', 'tensorflow', 'jax', 'numpy', `None` or "auto".
                - When "auto", the function will select the backend based on the
                `KERAS_BACKEND` environment variable.
                - For numpy this function will return 'cpu'.
        device (str/int/list): device(s) to select.
            Examples: 'cuda:1', 'gpu:2', 'auto:-1', 'cpu', 0, or [0,1,2,3].

            For more details see: `usbmd.utils.gpu_utils.get_device`
        hide_devices (int/list): device(s) to hide from the system.
            Examples: 0, or [0,1,2,3]. Can be useful when some GPUs have too
            little tensor cores to be useful for training, or when some GPUs
            are reserved for other tasks. Defaults to None, in which case no
            GPUs are hidden and all are available for use.
        allow_preallocate (bool, optional): allow preallocation of memory.
            Used for jax and tensorflow.
        verbose (bool, optional): print device selection. Defaults to True.
    Returns:
        device (str/int/list): selected device(s).
    """
    if hide_devices is not None:
        hide_gpus(hide_devices)

    # Get backend from environment variable
    if backend == "auto":
        backend = os.environ.get("KERAS_BACKEND")

    if backend in ["jax", "tensorflow", "torch"]:
        selected_gpu_ids = get_device(device, verbose=verbose)
        device = selected_gpu_ids_to_device(selected_gpu_ids, key=backend_key(backend))
    elif backend == "numpy" or backend == "cpu":
        device = "cpu"
    else:
        raise ValueError(f"Unknown backend ({backend}) in config.")

    # Early exit if device is CPU
    if device == "cpu":
        return device

    # Set if jax and tensorflow should preallocate memory
    if not allow_preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        set_memory_growth_tf()

    # Check if the selected backend is installed with CUDA support
    # -> Run this last because it will mess up the hiding of GPUs!
    if not backend_cuda_available(backend):
        device = "cpu"

    return device
