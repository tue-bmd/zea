"""GPU configuration utilities for Tensorflow.

- **Author(s)**     : Tristan Stevens, Ben Luijten
- **Date**          : 24 May 2023
"""

import os

# set TF logging level here, any out of {"0", "1", "2", "3"}
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from usbmd.utils import log
from usbmd.utils.gpu_utils import get_device as _get_device

# Get a list of all tf devices
GPUS = tf.config.experimental.list_physical_devices("GPU")


def set_memory_growth(gpu_ids=None):
    """Set memory growth for all visible GPUs.
    Args:
        gpu_ids: list of GPU ids to set memory growth for.
            If None, set memory growth for all visible GPUs.
    """

    # Select GPUs
    if gpu_ids is None:
        gpu_ids = list(range(len(GPUS)))

    tf.config.experimental.set_visible_devices(
        [GPUS[gpu] for gpu in gpu_ids],
        "GPU",
    )

    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in tf.config.experimental.get_visible_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        log.warning(
            "Please use `get_device()` before using any TensorFlow functionality."
        )
        print(e)


def get_device(device="auto:1", verbose=True, hide_others=True):
    """Sets the GPU usage for Tensorflow by searching for available GPUs and
    selecting one or more GPUs based on the device argument.
    If CUDA is unavailable, fallback to CPU.

    Hides other GPUs from the system by default by setting the
    CUDA_VISIBLE_DEVICES environment variable. Use the hide_others argument to
    disable this behavior.

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
        gpu_ids: list of selected GPU ids. If no GPU is selected, returns an
            empty list. If a CPU is selected, returns None.
    """
    if not tf.test.is_built_with_cuda() or len(GPUS) == 0:
        log.warning("Cuda not available, fallback to CPU.")
        return None

    selected_gpu_ids = _get_device(device, verbose=verbose, hide_others=hide_others)

    set_memory_growth(gpu_ids=selected_gpu_ids)

    if len(selected_gpu_ids) > 1:
        log.warning(
            (
                "Specified multiple GPU's but this function will just return "
                f"one GPU: {selected_gpu_ids[0]}"
            )
        )

    return f"gpu:{selected_gpu_ids[0]}"


if __name__ == "__main__":
    # Initialize GPU
    # Example on how to use gpu config functions
    gpu_ids = get_device()
