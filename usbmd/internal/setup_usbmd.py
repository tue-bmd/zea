"""
usbmd setup
===========

This module provides setup functions for the usbmd package, grouping together
commonly used initialization routines for convenience.

Overview
--------

The main entry point is the :func:`setup` function, which performs several key initialization steps for you:

- Loads and validates the configuration file (YAML) via :func:`usbmd.internal.setup_usbmd.setup_config`.
- Sets up user data paths using :func:`usbmd.datapaths.set_data_paths`.
- Initializes the device (GPU/CPU) according to the configuration using :func:`usbmd.internal.device.init_device`.
- Optionally creates a new user and prompts for datapath information via :func:`usbmd.datapaths.create_new_user`.

By calling :func:`setup`, you can prepare your usbmd environment in a single step, ensuring that configuration, data paths, and device setup are all handled for you.

.. code-block:: python


    # Basic usage: loads config, sets paths, initializes device
    config = setup_usbmd.setup(config_path="my_config.yaml")

    # With user creation prompt
    config = setup_usbmd.setup(config_path="my_config.yaml", create_user=True)

Function Details
----------------

.. autofunction:: setup
    :noindex:

    Calls:
        - :func:`usbmd.internal.setup_usbmd.setup_config`
        - :func:`usbmd.datapaths.set_data_paths`
        - :func:`usbmd.internal.device.init_device`
        - :func:`usbmd.datapaths.create_new_user` (if ``create_user=True``)

.. autofunction:: setup_config
    :noindex:

    Loads and validates the configuration file for usbmd. Supports interactive
    selection if no path is provided.

"""

import copy
import importlib
import sys
from pathlib import Path
from typing import Union

import keras
import yaml

from usbmd import Config, log
from usbmd.config.validation import check_config
from usbmd.datapaths import create_new_user, set_data_paths
from usbmd.internal.device import init_device
from usbmd.internal.git_info import get_git_summary
from usbmd.internal.viewer import filename_from_window_dialog


def reload_module(name):
    """Reloads module. This is useful when changing the backend.
    Taken from `keras.config.set_backend`"""

    # Clear module cache.
    loaded_modules = [key for key in sys.modules if key.startswith(name)]
    for key in loaded_modules:
        del sys.modules[key]

    __class__ = importlib.import_module(name).__class__

    # Finally: refresh all imported submodules.
    globs = copy.copy(globals())
    for key, value in globs.items():
        if value.__class__ == __class__:
            if str(value).startswith(f"<module '{name}."):
                module_name = str(value)
                module_name = module_name[module_name.find("'") + 1 :]
                module_name = module_name[: module_name.find("'")]
                globals()[key] = importlib.import_module(module_name)


def set_backend(backend: str):
    """Set compute backend

    Note: Make sure to reimport any module you are using that uses keras
    directly (has import keras or derivative at the top of the file).
    This can be done with the importlib module.

    """
    # set keras backend
    if keras.config.backend() != backend:
        keras.config.set_backend(backend)
        reload_module("usbmd")


def setup(
    config_path: str = None,
    user_config: Union[str, dict] = None,
    verbose: bool = True,
    disable_config_check: bool = False,
    loader=yaml.FullLoader,
    create_user: bool = False,
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
        create_user (bool, optional): whether to create a new user. Defaults to False.
            If True, it will prompt the user to enter their datapaths.
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

    # Prompt user to enter datapath information
    if create_user:
        create_new_user(user_config, local=config.data.local)

    # Set data paths
    config.data.user = set_data_paths(user_config, local=config.data.local)

    # Init GPU / CPU according to config
    config.device = init_device(
        device=config.device,
        backend="auto",
        hide_devices=config.hide_devices,
    )

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
                "i.e. `usbmd --config <path-to-config.yaml>` if GUI is not working "
                "(usually on headless servers)."
            ) from e

    config = Config.from_yaml(Path(config_path), loader=loader)

    if verbose:
        log.info(f"Using config file: {log.yellow(config_path)}")

    config["git"] = get_git_summary(verbose=verbose)

    if not disable_config_check:
        config = check_config(config)

    return config
