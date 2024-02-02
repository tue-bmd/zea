""" Device utilities """

from typing import Union

from usbmd.utils.gpu_utils import hide_gpus


def init_device(
    ml_library: str,
    device: Union[str, int, list],
    hide_devices: Union[int, list] = None,
):
    """Selects a GPU or CPU device based on the config.
    For PyTorch, this will return the device.
    For TensorFlow, this will hide all other GPUs from the system
    by setting the CUDA_VISIBLE_DEVICES.

    Args:
        ml_library (str): String indicating which ml library to use.
        device (str/int/list): device(s) to select.
            Examples: 'cuda:1', 'gpu:2', 'auto:-1', 'cpu', 0, or [0,1,2,3].

            for more details see:
                pytorch_ultrasound.utils.gpu_config.get_device and
                tensorflow_ultrasound.utils.gpu_config.set_gpu_usage.
        hide_devices (int/list): device(s) to hide from the system.
            Examples: 0, or [0,1,2,3]. Can be useful when some GPUs have too
            little tensor cores to be useful for training, or when some GPUs
            are reserved for other tasks. Defaults to None, in which case no
            GPUs are hidden and all are available for use.
    Returns:
        device (str/int/list): selected device(s).
    """
    if hide_devices is not None:
        hide_gpus(hide_devices)

    # Init GPU / CPU according to config
    if ml_library == "torch":
        # pylint: disable=import-outside-toplevel
        from usbmd.pytorch_ultrasound.utils.gpu_config import get_device

        device = get_device(device)
    elif ml_library == "tensorflow":
        # pylint: disable=import-outside-toplevel
        from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage

        set_gpu_usage(device)
    elif ml_library == "disable" or ml_library is None:
        device = "cpu"
    else:
        raise ValueError(f"Unknown ml_library ({ml_library}) in config.")

    return device


if __name__ == "__main__":
    init_device("torch", "auto:1", hide_devices=None)
