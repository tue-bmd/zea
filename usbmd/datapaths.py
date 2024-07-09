"""Set custom, user specific data paths in this file.

- **Author(s)**     : Tristan Stevens, Frits de Bruijn
- **Date**          : -
"""

import copy
import getpass
import os
import platform
import socket
import sys
import warnings
from functools import reduce
from pathlib import Path
from typing import Union

import yaml

from usbmd.utils import log, strtobool

DEFAULT_WINDOWS_DATA_ROOT = "Z:/Ultrasound-BMd/data"
DEFAULT_LINUX_DATA_ROOT = "/home/data/ultrasound"
DEFAULT_USERS_CONFIG_PATH = "./users.yaml"

DEFAULT_USER = {
    "windows_hostname": {
        "system": "windows",
        "data_root": DEFAULT_WINDOWS_DATA_ROOT,
    },
    "linux_hostname": {
        "system": "linux",
        "data_root": DEFAULT_LINUX_DATA_ROOT,
    },
}


class UnknownUsernameWarning(UserWarning):
    """
    Custom Warning indicating that the username was not found
    in the user.yaml file
    """


class UnknownHostnameWarning(UserWarning):
    """
    Custom Warning indicating that the hostname was not found
    for this user in the user.yaml file
    """


class UnknownLocalRemoteWarning(UserWarning):
    """
    Custom Warning indicating that the data_root corresponding to
    the local or remote key was not found
    in the user.yaml file
    """


def _fallback_to_default_data_root(config, system):

    default_windows_data_root = DEFAULT_WINDOWS_DATA_ROOT
    default_linux_data_root = DEFAULT_LINUX_DATA_ROOT

    if "default_user" in config:
        default_config = config["default_user"]
        if "windows_hostname" in default_config:
            default_windows_data_root = default_config["windows_hostname"]["data_root"]
        elif "linux_hostname" in default_config:
            default_linux_data_root = default_config["linux_hostname"]["data_root"]

    if system == "windows":
        return default_windows_data_root
    elif system == "linux":
        return default_linux_data_root
    else:
        return "./"


def _verify_user_config_and_get_paths(username, config, system, hostname, local):
    """
    Get the user configuration and verify the paths.

    Args:
        username (str): The username of the user.
        config (dict): The configuration dictionary containing user information.
        system (str): The current operating system.
        hostname (str): The hostname of the system.
        local (bool): Flag indicating whether to use local paths or remote paths.

    Returns:
        dict: A dictionary containing the verified paths.
    """
    # Get config for user and hostname
    config = config[username]
    if hostname in config:
        config = config[hostname]
    elif "hostname" in config:
        config = config["hostname"]
    else:
        warnings.warn(
            f"Unknown hostname {hostname} for user {username}",
            UnknownHostnameWarning,
        )
        return _fallback_to_default_data_root(config, system), "./output"

    # Check if set os system matches with the current system
    if "system" in config:
        assert (
            config["system"] == system
        ), f'Current OS {system} does not match user settings: {config["system"]}'
        config.pop("system")

    # Assert that data_root is set
    assert "data_root" in config, "Please add a data_root key to your user / hostname."

    # Assert config only contains data_root and output
    unknown_keys = [x for x in config.keys() if x not in ["data_root", "output"]]
    assert len(unknown_keys) == 0, f"Unknown keys in user config: {unknown_keys}"

    def _error_msg(key):
        return (
            f"{key} key should be either a string or a dict containing "
            "local and / or remote keys with data_root paths as values."
        )

    paths = {}
    # config will contain the data_root and optionally output paths
    for key, path in config.items():
        assert isinstance(path, (str, dict)), _error_msg(key)

        if isinstance(path, str):
            paths[key] = path
            continue

        assert set(path.keys()) <= set(["local", "remote"]), _error_msg(key)
        if local is True:
            if "local" in path:
                paths[key] = path["local"]
            else:
                warnings.warn(
                    f"Unknown local path for {key} in user config. Falling back to default.",
                    UnknownLocalRemoteWarning,
                )
                paths[key] = _fallback_to_default_data_root(config, system)

        elif local is False:
            if "remote" in path:
                paths[key] = path["remote"]
            else:
                warnings.warn(
                    f"Unknown remote path for {key} in user config. Falling back to default.",
                    UnknownLocalRemoteWarning,
                )
                paths[key] = _fallback_to_default_data_root(config, system)
        else:
            raise ValueError(
                f"Please set local to True or False or have the {key} "
                + "specified as a string (without local / remote key)."
            )

    # Set output path if not set
    if "output" not in paths:
        paths["output"] = Path(paths["data_root"], "output")
        log.warning("No output path set, using data_root/output as output path.")

    return paths["data_root"], paths["output"]


def _verify_paths(data_path):
    """Verify that the paths exist and are directories."""
    for key, path in data_path.items():
        if key not in ["data_root", "output"]:
            continue
        if not Path(path).is_dir():
            log.warning(
                f"{key} path {path} does not exist, please update your "
                f"{log.yellow('users.yaml')} file."
            )


def set_data_paths(user_config: Union[str, dict] = None, local: bool = True) -> dict:
    """Get data paths (absolute paths to location of data).

    Args:
        user_config (str or dict): Path that points to yaml file with user info.
            Defaults to None. In that case `./users.yaml` is taken
            as default file. If not string, could also be a dictionary.
            Should be structured see example below.

            ```yaml
                my_username:
                    my_hostname:
                        system: windows
                        data_root: C:/path_to_my_data_root/
                        output: C:/other_paths/
                    linux_hostname:
                        system: linux
                        data_root: /home/path_to_my_data_root/
                        output: C:/other_paths/
                    # if both my_hostname and linux_hostname are not matching, fallback to hostname:
                    hostname:
                        system: linux
                        data_root: /home/path_to_my_data_root/

                other_username:
                    other_hostname:
                        system: windows
                        data_root:
                            local: C:/path_to_my_local_data_root/
                            remote: Z:/path_to_my_remote_data_root/
                        output:
                            local: C:/path_to_my_local_output/
                            remote: Z:/path_to_my_remote_output/
            ```
            the default_user can also be set in the users config but should
            always be of the form:

            ```yaml
            default_user:
                windows_hostname:
                    system: windows
                    data_root: C:/path_to_my_default_data_root/
                linux_hostname:
                    system: linux
                    data_root: /home/path_to_my_default_data_root/
            ```
            When a username is not set, these paths are used as fallback.

        local (bool): Use local dataset or get from NAS.

    Returns:
        data_path (dict): absolute paths to location of data. Stores the following
            parameters:
                `data_root`, `repo_root`, `output`,
                `system`, `username`, `hostname`

    """
    username = getpass.getuser()
    system = platform.system().lower()
    hostname = socket.gethostname()
    repo_root = Path(__file__)

    assert isinstance(
        user_config, (str, dict, type(None))
    ), "user_config should be a string or dictionary."

    # If user_config is None, use the default users.yaml file
    if isinstance(user_config, type(None)):
        user_config = DEFAULT_USERS_CONFIG_PATH

    # If user_config is a dictionary, use it as the config
    if isinstance(user_config, dict):
        config = copy.deepcopy(user_config)

    # If user_config is a string, load the yaml file
    if isinstance(user_config, str):
        config_path = Path(user_config)

        # If there is no users.yaml file yet, create one.
        if not config_path.is_file():
            default_config = DEFAULT_USER
            with open(config_path, "w", encoding="utf-8") as file:
                yaml.dump(default_config, file, default_flow_style=False)
            try:
                create_new_user(local=local)
            except:
                log.warning(
                    f"Could not create user profile for {username} on {hostname}, using default."
                )

        # Load YAML file with user info
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            # Raise error if config is not a dictionary, for example if its empty.
            # Lets not overwrite the users config file in this case.
            raise ValueError(
                f"""YAML file should contain a dictionary, but found {type(config)}".
                Please check your users.yaml file for corruptions. In case you want to create a
                new users.yaml file, please delete the current one."""
            )

    # Check if username is in the config
    if username in config:
        data_root, output = _verify_user_config_and_get_paths(
            username, config, system, hostname, local
        )
    else:
        warnings.warn(
            (
                f"Unknown user {username} in user file.\nFalling back to default path. "
                f"Please update the `{config_path}` file with "
                "your data-path settings."
            ),
            UnknownUsernameWarning,
        )
        data_root = _fallback_to_default_data_root(config, system)
        output = Path(data_root, "output")

    # Add repo_root to sys.path
    sys.path.insert(1, repo_root)

    data_path = {
        "data_root": Path(data_root),
        "repo_root": repo_root,
        "output": Path(output),
        "system": system,
        "username": username,
        "hostname": hostname,
    }

    _verify_paths(data_path)

    return data_path


## Helper functions for handling user input


def _build_user_profile_string(data_paths, local: bool = None):
    """Builds a string that can be written to users.yaml to create a new user profile."""
    tab = "    "  # 4 spaces required in yaml
    base_string = (
        f"'{data_paths['username']}':\n"
        + f"  {data_paths['hostname']}:\n"
        + f"    system: {data_paths['system']}\n"
    )
    if local is None:
        return base_string + f'{tab}data_root: {data_paths["data_root"]}'
    elif local is False:
        return base_string + (
            f"{tab}data_root:\n" + f'{tab}{tab}remote: {data_paths["data_root"]}'
        )
    elif local is True:
        return base_string + (
            f"{tab}data_root:\n" + f'{tab}{tab}local: {data_paths["data_root"]}'
        )
    else:
        raise ValueError("local should set to a boolean or None.")


def _to_write_user_profile_to_file(user_profile_string, user_config_path=DEFAULT_USERS_CONFIG_PATH):
    with open(user_config_path, "a", encoding="utf-8") as file:
        file.write("\n\n" + user_profile_string + "\n")
    print(
        "\n✅ Your user profile was successfully added to" f" `{user_config_path}`.\n"
    )


def _pretty_print_data_paths(data_paths):
    for key, value in data_paths.items():
        print(f"\t{key}: {log.yellow(value)}")
    print()


def _prompt_user_for_data_root():
    data_root_input = input(
        "\nℹ️ Please enter the path to your data directory, "
        "or press Enter to use the default Windows path "
        f"`{DEFAULT_WINDOWS_DATA_ROOT}`: "
    )
    return DEFAULT_WINDOWS_DATA_ROOT if data_root_input == "" else data_root_input


def _acquire_and_validate_data_root():
    data_root_input = _prompt_user_for_data_root()
    while not os.path.isdir(data_root_input):
        print(
            "\n The path you entered does not point to a directory, please try again."
        )
        data_root_input = _prompt_user_for_data_root()
    return data_root_input


def _warning_type_was_thrown(warning_type, list_of_warnings):
    """Returns True iff list_of_warnings contains a warning of type warning_type"""
    if not list_of_warnings:
        return False
    return reduce(
        lambda acc, w: acc and isinstance(w.message, warning_type),
        list_of_warnings,
        True,
    )


def _to_read_yaml_file(path_str):
    path = Path(path_str)
    if not path.is_file():
        raise ValueError("YAML file path provided does not lead to a file.")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        return config


def _to_write_yaml_file(data, path_str):
    path = Path(path_str)
    if _check_for_comments_yaml_file(path_str):
        log.warning(
            f"YAML file {path_str} contains comments. "
            "These will be removed if you write to the file."
        )
        input("Press Enter to continue or Ctrl+C to cancel.")

    if not path.is_file():
        raise ValueError("YAML file path provided does not lead to a file.")

    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def _try(fn, args):
    try:
        return fn(**args)
    except Exception as e:
        print(f"Encountered an error in {fn.__name__}")
        print(e)


def _check_for_comments_yaml_file(path_str):
    """Returns True iff the YAML file at path_str contains comments."""
    path = Path(path_str)
    if not path.is_file():
        raise ValueError("YAML file path provided does not lead to a file.")

    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # just look for # anywhere
        return any("#" in line for line in lines)


def create_new_user(user_config_path: str = None, local: bool = None):
    """Creates a new user profile in `users.yaml` if one does not already exist.

    Args:
        user_config (str): Path that points to yaml file with user info.
            Defaults to None. In that case `./users.yaml` is taken
        local (bool): Use local dataset or get from remote (NAS).
            Per machine, the data_root can be set to a local or remote path.
            Each user can also have a different data_root for each machine.
            Default is None, which means that the data_root is shared for either
            local or remote (i.e. this parameter is ignored), see doc set_data_paths().
    """
    with warnings.catch_warnings(record=True) as list_of_warnings:
        data_paths = set_data_paths(user_config=user_config_path, local=local)
        if user_config_path is None:
            user_config_path = DEFAULT_USERS_CONFIG_PATH
        assert isinstance(user_config_path, str), "user_config_path should be a string."

        # Display any warnings that were thrown during set_data_paths
        if list_of_warnings:
            for w in list_of_warnings:
                print(f"🚨 {w.message}")
        else:
            log.info("Data paths set successfully.")
            log.info("Here's a summary of your data paths:")
            _pretty_print_data_paths(data_paths)

        # If there was no profile found in users.yaml for the current user,
        # give them the option to create a user profile automatically
        user_warning_was_thrown = _warning_type_was_thrown(
            UnknownUsernameWarning, list_of_warnings
        )
        hostname_warning_was_thrown = _warning_type_was_thrown(
            UnknownHostnameWarning, list_of_warnings
        )
        local_remote_warning_was_thrown = _warning_type_was_thrown(
            UnknownLocalRemoteWarning, list_of_warnings
        )

        if user_warning_was_thrown:
            print("ℹ️ Follow the instructions below to create your user profile.")
            data_root = _acquire_and_validate_data_root()
            data_paths["data_root"] = data_root
            user_profile_string = _build_user_profile_string(data_paths, local=local)
            user_response = input(
                "\n"
                + user_profile_string
                + "\n"
                + "\nℹ️ Would you like to automatically create your user"
                + "profile with the details above? [y]: "
            )
            if user_response == "" or strtobool(user_response):
                _try(
                    _to_write_user_profile_to_file,
                    {
                        "user_profile_string": user_profile_string,
                        "user_config_path": user_config_path,
                    },
                )
        elif hostname_warning_was_thrown:
            print(
                f"ℹ️ The hostname '{data_paths['hostname']}' was "
                f"not found for username '{data_paths['username']}'.\n"
            )
            print(
                "ℹ️ Follow the instructions below to create a new "
                f"entry for hostname: '{data_paths['hostname']}:"
            )
            data_root = _acquire_and_validate_data_root()
            data_paths["data_root"] = data_root
            users_yaml_dict = _try(_to_read_yaml_file, {"path_str": user_config_path})
            users_yaml_dict[data_paths["username"]][data_paths["hostname"]] = {
                "system": data_paths["system"],
                "data_root": data_root,
            }
            user_response = input(
                "\n"
                + yaml.dump(users_yaml_dict[data_paths["username"]])
                + "\nℹ️ Would you like to update your user profile "
                + "with the user info above? [y]: "
            )
            if user_response == "" or strtobool(user_response):
                _try(
                    _to_write_yaml_file,
                    {"data": users_yaml_dict, "path_str": user_config_path},
                )
                log.success("Profile updated successfully.")
        elif local_remote_warning_was_thrown:
            local_remote_str = "local" if local else "remote"
            print(
                f"ℹ️ The data_root for '{data_paths['username']}' was "
                f"not found for location: {local_remote_str}.\n"
            )
            print(
                "ℹ️ Follow the instructions below to create a new entry for "
                f"data_root for location: {local_remote_str}:"
            )
            data_root = _acquire_and_validate_data_root()
            data_paths["data_root"] = data_root
            users_yaml_dict = _try(_to_read_yaml_file, {"path_str": user_config_path})
            ## now update the data_root for the user and hostname in the yaml file
            ## use local or remote subkey depending on the local parameter
            users_yaml_dict[data_paths["username"]][data_paths["hostname"]][
                "data_root"
            ].update({local_remote_str: data_root})
            user_response = input(
                "\n"
                + yaml.dump(users_yaml_dict[data_paths["username"]])
                + "\nℹ️ Would you like to update your user profile "
                + "with the user info above? [y]: "
            )
            if user_response == "" or strtobool(user_response):
                _try(
                    _to_write_yaml_file,
                    {"data": users_yaml_dict, "path_str": user_config_path},
                )
                log.success("Profile updated successfully.")

    return data_paths


if __name__ == "__main__":
    create_new_user("users.yaml", local=None)
