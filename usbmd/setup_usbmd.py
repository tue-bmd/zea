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

import yaml

from usbmd.config import load_config_from_yaml
from usbmd.config.validation import check_config
from usbmd.datapaths import set_data_paths
from usbmd.utils import log
from usbmd.utils.device import init_device
from usbmd.utils.git_info import get_git_summary
from usbmd.utils.io_lib import filename_from_window_dialog


def setup(
    config_path: str = None,
    user_config: Union[str, dict] = None,
    verbose: bool = True,
    disable_config_check: bool = False,
    loader=yaml.FullLoader,
):
    """General setup function for usbmd. Loads config, sets data paths and
    initializes gpu if available. Will return config object.

    Args:
        config_path (str, optional): file path to config yaml.
            Defaults to None, in which case a window dialog will pop up.
        user_config (str or dict, optional): path that points to yaml file with user info.
            Alternively dictionary with user info. Defaults to None.
        verbose (bool, optional): print config file path and git summary. Defaults to True.
        disable_config_check (bool, optional): whether to check for usbmd config validity.
            Defaults to False. Can be set to True if you are using some other config that
            does not have to adhere to usbmd config standards.
        loader (yaml.Loader, optional): yaml loader. Defaults to yaml.FullLoader.
    Returns:
        config (dict): config object / dict.
    """

    # Load config
    config = setup_config(
        config_path,
        verbose=verbose,
        disable_config_check=disable_config_check,
        loader=loader,
    )

    # Set data paths
    config.data.user = set_data_paths(user_config, local=config.data.local)

    # Init GPU / CPU according to config
    config.device = init_device(config.ml_library, config.device, config.hide_devices)

    return config


def setup_config(
    config_path: str = None,
    verbose: bool = True,
    disable_config_check: bool = False,
    loader=yaml.FullLoader,
):
    """Setup function for config. Retrieves config file and checks for validity.

    Args:
        config_path (str, optional): file path to config yaml. Defaults to None.
            if None, argparser is checked. If that is None as well, the window
            ui will pop up for choosing the config file manually.
        verbose (bool, optional): print config file path and git summary. Defaults to True.
        disable_config_check (bool, optional): whether to check for usbmd config validity.
            Defaults to False. Can be set to True if you are using some other config that
            does not have to adhere to usbmd config standards.
        loader (yaml.Loader, optional): yaml loader. Defaults to yaml.FullLoader.
            for custom objects, you might want to use yaml.UnsafeLoader.
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

    config = load_config_from_yaml(Path(config_path), loader=loader)

    if verbose:
        log.info(f"Using config file: {log.yellow(config_path)}")

    config["git"] = get_git_summary(verbose=verbose)

    if not disable_config_check:
        config = check_config(config)

    return config
