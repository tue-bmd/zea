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
from usbmd.utils.device import init_device
from usbmd.utils.git_info import get_git_summary
from usbmd.utils.io_lib import filename_from_window_dialog


def setup(
    config_path: str = None,
    user_config: Union[str, dict] = None,
    verbose: bool = True,
):
    """General setup function for usbmd. Loads config, sets data paths and
    initializes gpu if available. Will return config object.

    Args:
        config_path (str, optional): file path to config yaml.
            Defaults to None, in which case a window dialog will pop up.
        user_config (str or dict, optional): path that points to yaml file with user info.
            Alternively dictionary with user info. Defaults to None.
        verbose (bool, optional): print config file path and git summary. Defaults to True.
    Returns:
        config (dict): config object / dict.
    """

    # Load config
    config = setup_config(config_path, verbose=verbose)

    # Set data paths
    config.data.user = set_data_paths(user_config, local=config.data.local)

    # Init GPU / CPU according to config
    config.device = init_device(config.ml_library, config.device, config.hide_devices)

    return config


def setup_config(config_path: str = None, verbose: bool = True):
    """Setup function for config. Retrieves config file and checks for validity.

    Args:
        config_path (str, optional): file path to config yaml. Defaults to None.
            if None, argparser is checked. If that is None as well, the window
            ui will pop up for choosing the config file manually.
        verbose (bool, optional): print config file path and git summary. Defaults to True.
    Returns:
        config (dict): config object / dict.

    """
    if config_path is None:
        # if no argument is provided resort to UI window
        filetype = "yaml"
        try:
            config_path = filename_from_window_dialog(
                f"Choose .{filetype} file",
                filetypes=((filetype, "*." + filetype),),
                initialdir="./configs",
            )
        except Exception as e:
            raise ValueError(
                "Please specify the path to a config file through --config flag "
                "if GUI is not working (usually on headless servers)."
            ) from e

    config = load_config_from_yaml(Path(config_path))
    if verbose:
        print(f"Using config file: {config_path}")
    config = check_config(config)

    config["git"] = get_git_summary(verbose=verbose)

    return config
