""" Device utilities """

from typing import Union

from usbmd.utils.gpu_utils import hide_gpus, selected_gpu_ids_to_device


def init_device(
    backend: Union[str, None] = None,
    device: Union[str, int, list] = "auto:1",
    hide_devices: Union[int, list] = None,
    verbose: bool = True,
):
    """Selects a GPU or CPU device based on the config.
    For PyTorch, this will return the device.
    For TensorFlow, this will hide all other GPUs from the system
    by setting the CUDA_VISIBLE_DEVICES.

    Args:
        backend (str): String indicating which ml library to use. Can be
            'torch', 'tensorflow', 'jax', 'numpy' or `None`.
                - When `None` or 'jax', the function will select GPU(s) without specific features
                for the ml library and thus will not import any ml library.
                - Selecting Tensorflow will set memory growth, check if CUDA is available
                and format the device string for Tensorflow.
                - Selecting Torch will also check if CUDA is available.
                - For numpy this function will return 'cpu'.
        device (str/int/list): device(s) to select.
            Examples: 'cuda:1', 'gpu:2', 'auto:-1', 'cpu', 0, or [0,1,2,3].

            For more details see: `usbmd.utils.gpu_utils.get_device`
        hide_devices (int/list): device(s) to hide from the system.
            Examples: 0, or [0,1,2,3]. Can be useful when some GPUs have too
            little tensor cores to be useful for training, or when some GPUs
            are reserved for other tasks. Defaults to None, in which case no
            GPUs are hidden and all are available for use.
        verbose (bool, optional): print device selection. Defaults to True.
    Returns:
        device (str/int/list): selected device(s).
    """
    if hide_devices is not None:
        hide_gpus(hide_devices)

    # Init GPU / CPU according to config
    if backend == "torch":
        # pylint: disable=import-outside-toplevel
        from usbmd.backend.torch.utils.gpu_config import get_device

        device = get_device(device, verbose=verbose)
    elif backend == "tensorflow":
        # pylint: disable=import-outside-toplevel
        from usbmd.backend.tensorflow.utils.gpu_config import get_device

        device = get_device(device, verbose=verbose)
    elif backend is None or backend == "jax":
        # pylint: disable=import-outside-toplevel
        from usbmd.utils.gpu_utils import get_device

        selected_gpu_ids = get_device(device, verbose=verbose)
        device = selected_gpu_ids_to_device(selected_gpu_ids, key="gpu")
    elif backend == "numpy":
        device = "cpu"
    else:
        raise ValueError(f"Unknown backend ({backend}) in config.")

    return device


if __name__ == "__main__":
    init_device("torch", "auto:1", hide_devices=None)
