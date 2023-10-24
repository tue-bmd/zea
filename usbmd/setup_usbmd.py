"""This module contains setup functions for usbmd. In general
these setup funcs group together several functions that are often
used together for ease of use.

setup_config: Setup function for config. Retrieves config file and checks for validity.
setup: General setup function for usbmd. Runs setup_config, sets data paths and
    initializes gpu if available.

Author(s): Tristan Stevens
Date: 25/09/2023
"""
from pathlib import Path
from typing import Union

from usbmd.common import set_data_paths
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.config_validation import check_config
from usbmd.utils.git_info import get_git_summary
from usbmd.utils.io_lib import filename_from_window_dialog


def setup_config(file=None):
    """Setup function for config. Retrieves config file and checks for validity.

    Args:
        file (str, optional): file path to config yaml. Defaults to None.
            if None, argparser is checked. If that is None as well, the window
            ui will pop up for choosing the config file manually.

    Returns:
        config (dict): config object / dict.

    """
    if file is None:
        # if no argument is provided resort to UI window
        filetype = "yaml"
        try:
            file = filename_from_window_dialog(
                f"Choose .{filetype} file",
                filetypes=((filetype, "*." + filetype),),
                initialdir="./configs",
            )
        except Exception as e:
            raise ValueError(
                "Please specify the path to a config file through --config flag "
                "if GUI is not working (usually on headless servers)."
            ) from e

    config = load_config_from_yaml(Path(file))
    print(f"Using config file: {file}")
    config = check_config(config)

    ## git
    cwd = Path.cwd().stem
    if cwd in ("Ultrasound-BMd", "usbmd"):
        config["git"] = get_git_summary()

    return config


def setup(
    config_path: str = None,
    user_config: Union[str, dict] = None,
):
    """General setup function for usbmd. Loads config, sets data paths and
    initializes gpu if available. Will return config object.

    Args:
        config_path (str, optional): file path to config yaml.
            Defaults to None, in which case a window dialog will pop up.
        user_config (str or dict, optional): path that points to yaml file with user info.
            Alternively dictionary with user info. Defaults to None.

    Returns:
        config (dict): config object / dict.
    """

    # Load config
    config = setup_config(config_path)

    # Set data paths
    config.data.user = set_data_paths(user_config, local=config.data.local)

    # Init GPU / CPU according to config
    config.device = init_device(config.ml_library, config.device)

    return config


def init_device(ml_library: str, device: Union[str, int, list]):
    """Selects a GPU or CPU device based on the config.
    For PyTorch, this will return the device.
    For TensorFlow, this will hide all other GPUs from the system
    by setting the CUDA_VISIBLE_DEVICES.

    Args:
        ml_library (str): String indicating which ml library to use.
        device (str/int/list): device(s) to select.

    Returns:
        device (str):
    """

    # Init GPU / CPU according to config
    if ml_library == "torch":
        # pylint: disable=import-outside-toplevel
        from usbmd.pytorch_ultrasound.utils.gpu_config import get_device

        device = get_device(device)
    if ml_library == "tensorflow":
        # pylint: disable=import-outside-toplevel
        from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage

        set_gpu_usage(device)
    elif ml_library == "disable" or ml_library is None:
        pass
    else:
        raise ValueError(f"Unknown ml_library ({ml_library}) in config.")

    return device
