"""Utility functions for handling local data paths.

This module provides utilities for managing local and remote data paths in ``zea`` projects.
It supports user- and machine-specific configuration via a ``users.yaml`` file, allowing
dynamic resolution of data roots for portable and reproducible workflows.

The main entry point is :func:`set_data_paths`, which resolves the ``data_root`` and
``output`` paths for the current user and machine. :func:`format_data_path` then turns
the relative paths used in ``zea`` configs into absolute ones. To set up a ``users.yaml``
interactively, run ``python -m zea.datapaths`` (see :func:`create_new_user`).

See the notebook :doc:`../notebooks/data/zea_local_data` for an extensive example of how
to set up your local data paths.


Example usage
^^^^^^^^^^^^^

.. doctest::

    >>> import yaml
    >>> from zea.datapaths import set_data_paths

    >>> user_config = {"data_root": "/path/to/data", "output": "/path/to/output"}
    >>> with open("users.yaml", "w", encoding="utf-8") as file:
    ...     yaml.dump(user_config, file)

    >>> user = set_data_paths("users.yaml")
    >>> print(user.data_root)
    /path/to/data

.. testcleanup::

    import os

    os.remove("users.yaml")

"""

import copy
import getpass
import importlib.resources
import os
import platform
import socket
import sys
import warnings
from pathlib import Path
from typing import Union

import yaml

from zea import log
from zea.config import Config
from zea.internal.config.users import UserProfileSpec, validate_users_config
from zea.internal.preset_utils import HF_PREFIX
from zea.tools.hf import HFPath
from zea.utils import strtobool

DEFAULT_DATA_ROOT = {
    "windows": "Z:/data",
    "linux": "/mnt/z/data",
    "darwin": "/mnt/z/data",
    None: "/mnt/z/data",  # for other system
}

#: The keys a users.yaml section may set directly; anything else is a user or machine.
_PROFILE_FIELDS = UserProfileSpec.field_names()

DEFAULT_LINUX_DATA_ROOT = DEFAULT_DATA_ROOT["linux"]
DEFAULT_USERS_CONFIG_PATH = "./users.yaml"
DEFAULT_OUTPUT_PATH = "{data_root}/output"


class NoYamlFileError(Warning):
    """Raised when the users.yaml file is not found."""


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


def _create_empty_yaml(path):
    # Create empty file if it does not exist
    with open(path, "a", encoding="utf-8"):
        pass


def _is_interactive():
    """Returns True iff we can prompt the user for input on stdin."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        # stdin can be replaced (notebooks, papermill) or already closed
        return False


def _fallback_to_default_data_root(system):
    if system not in DEFAULT_DATA_ROOT:
        system = None
    return DEFAULT_DATA_ROOT[system]


def _default_output_path(data_root):
    return Path(DEFAULT_OUTPUT_PATH.format(data_root=data_root))


def _config_levels(config, username, hostname):
    """The ``users.yaml`` sections that apply here, from most general to most specific.

    A ``users.yaml`` has up to three levels that can carry paths: the shared, userless
    root, a username section, and a hostname section inside it (or at the root, when the
    user has no section of their own).
    """
    levels = [config if isinstance(config, dict) else {}]
    if isinstance(levels[-1].get(username), dict):
        levels.append(levels[-1][username])
    if isinstance(levels[-1].get(hostname), dict):
        levels.append(levels[-1][hostname])
    return levels


def _resolve_profile(config, username, hostname):
    """Return the paths that apply to this user and machine, and whether the user is known.

    The levels are overlaid from general to specific, so a section that sets only some of
    the keys inherits the rest from the level above -- a machine that only pins ``system``
    still gets the user's ``data_root`` and ``output``.
    """
    levels = _config_levels(config, username, hostname)
    username_found = len(levels) > 1 and levels[1] is config.get(username)

    profile = {}
    for level in levels:
        profile.update({key: value for key, value in level.items() if key in _PROFILE_FIELDS})
    return profile, username_found


def _verify_user_config_and_get_paths(config, system, local):
    """
    Get the user configuration and verify the paths.

    Args:
        config (dict): The configuration dictionary containing user information.
        system (str): The current operating system.
        local (bool): Flag indicating whether to use local paths or remote paths.

    Returns:
        dict: A dictionary containing the verified paths.
    """
    # Check if set os system matches with the current system
    if "system" in config:
        assert config["system"] == system, (
            f"Current OS {system} does not match user settings: {config['system']}"
        )
        config.pop("system")

    paths = {}
    # config will contain the data_root and optionally output paths. Their shape is
    # already guaranteed by the users.yaml schema (see zea.internal.config.users).
    for key, path in config.items():
        if not isinstance(path, dict):
            paths[key] = path
            continue

        if local is True:
            if "local" in path:
                paths[key] = path["local"]
            else:
                warnings.warn(
                    f"Unknown local path for {key} in user config. Falling back to default.",
                    UnknownLocalRemoteWarning,
                )
                paths[key] = _fallback_to_default_data_root(system)

        elif local is False:
            if "remote" in path:
                paths[key] = path["remote"]
            else:
                warnings.warn(
                    f"Unknown remote path for {key} in user config. Falling back to default.",
                    UnknownLocalRemoteWarning,
                )
                paths[key] = _fallback_to_default_data_root(system)
        else:
            raise ValueError(
                f"Please set local to True or False or have the {key} "
                "specified as a string (without local / remote sub keys). "
                f"Current value, 'data_root': {path}."
            )

    # Set output path if not set
    if "output" not in paths:
        paths["output"] = _default_output_path(paths["data_root"])
        log.warning("No output path set, using data_root/output as output path.")

    return paths["data_root"], paths["output"]


def _verify_paths(data_path):
    """Verify that the paths exist and are directories."""
    for key in ["data_root", "output"]:
        path = data_path[key]
        if not Path(path).is_dir():
            log.warning(
                f"{key} path `{path}` does not exist, please update your "
                f"{log.yellow('users.yaml')} file."
            )


def _load_users_yaml(user_config, local, username, hostname):
    config_path = Path(user_config)

    # If there is no users.yaml file yet, create one.
    if not config_path.is_file():
        warnings.warn(
            f"No {user_config} file found, creating a new one. "
            "Consider running `python -m zea.datapaths` to setup your paths. ",
            NoYamlFileError,
        )

        _create_empty_yaml(config_path)

        # Only prompt when there is someone at the keyboard to answer. Non-interactive
        # runs (CI, notebooks, scripts) silently fall back to the defaults below.
        if _is_interactive():
            try:
                create_new_user(str(config_path), local=local)
            except Exception:
                log.warning(
                    f"Could not create user profile for {username} on {hostname}, using default."
                )

    # Load YAML file with user info
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        # Raise error if config is not a dictionary, for example if its empty.
        # Lets not overwrite the users config file in this case.
        raise ValueError(
            f"""YAML file should contain a dictionary, but found {type(config)}".
            Please check your users.yaml file for corruptions. In case you want to create a
            new users.yaml file, please delete the current one."""
        )
    return config


def set_data_paths(
    user_config: Union[str, dict, None] = None, local: bool | None = True, verify: bool = True
) -> Config:
    """Get data paths (absolute paths to location of data).

    Args:
        user_config (str or dict, optional): Path to a YAML file with user info.
            If None, uses ``./users.yaml`` as the default file. Can also be a dictionary
            structured as shown below.
        local (bool, optional): Whether to pick the ``local`` or the ``remote`` path for
            entries that define both (e.g. a local disk and a remote share). Default is True.
            Set to None when every path that applies is a plain string, i.e. shared
            between local and remote.
        verify (bool, optional): Verify that the paths exist and are directories.
            Default is True.

    Example YAML structure::

        data_root: ...
        output: ...

    You can also specify different ``data_root`` for different users and machines::

        my_username:
          my_hostname:
            system: windows
            data_root: ...
            output: ...
          other_hostname:
            system: linux
            data_root:
              local: ...
              remote: ...
          # If both my_hostname and other_hostname are not matching, fallback to:
          system: linux
          data_root: ...

        other_username:
          data_root: ...

    The machine section takes precedence over the user section, which takes precedence
    over the userless and machineless one at the bottom. Precedence is per key, so a
    section only needs to set what it changes: a machine that pins just ``system``
    still inherits ``data_root`` and ``output`` from the levels above it.

    Returns:
        Config: Absolute paths to location of data. Stores the following parameters:
            ``data_root``, ``zea_root``, ``output``, ``system``, ``username``, ``hostname``

    Raises:
        ValueError: If ``user_config`` is not a string, dictionary or None.

    Note:
        When ``user_config`` points to a YAML file that does not exist yet, an empty one
        is created and -- when running interactively -- you are offered to set up a
        profile, see :func:`create_new_user`. When no ``data_root`` can be resolved for
        the current user and machine, a warning is raised and a default path for the
        current operating system is used. When no ``output`` is given, it defaults to
        ``data_root/output``.

    """
    username = getpass.getuser()
    system = platform.system().lower()
    hostname = socket.gethostname()
    zea_root = importlib.resources.files("zea")

    # If user_config is None, use the default users.yaml file
    if isinstance(user_config, type(None)):
        user_config = DEFAULT_USERS_CONFIG_PATH

    # If user_config is a dictionary, use it as the config
    if isinstance(user_config, dict):
        config = copy.deepcopy(user_config)
        source = "user config"
    # If user_config is a string, load the yaml file
    elif isinstance(user_config, str):
        config = _load_users_yaml(user_config, local, username, hostname)
        source = f"`{user_config}`"
    else:
        raise ValueError("user_config should be a string or dictionary.")

    # Check the file against the users.yaml schema before resolving anything, so a
    # typo is reported as such instead of silently falling back to the defaults.
    try:
        config = validate_users_config(config)
    except ValueError as exc:
        raise ValueError(f"Invalid user config in {source}: {exc}") from exc

    # Overlay the shared, user and machine sections that apply here
    config, username_found = _resolve_profile(config, username, hostname)

    # Ensure that the resolved profile contains a `data_root` key
    if "data_root" not in config:
        default_data_root = _fallback_to_default_data_root(system)
        # Distinguish "we don't know this user at all" from "we know this user, but we
        # cannot resolve a data_root for this machine", so that `create_new_user` updates
        # the existing profile in place instead of appending a second, shadowing one.
        if username_found:
            warning_type = UnknownHostnameWarning
            reason = (
                f"Cannot find data_root for hostname={hostname} under username={username} "
                "in user file. Also no default data_root found for this user."
            )
        else:
            warning_type = UnknownUsernameWarning
            reason = (
                f"Cannot find data_root for username={username} "
                f"and hostname={hostname} in user file. Also no default data_root found."
            )
        warnings.warn(
            (
                f"{reason} "
                f"Falling back to default path for {system}: {default_data_root}. "
                f"Please update the `{user_config}` with your data-path settings."
            ),
            warning_type,
        )
        # Only the data_root is missing; a configured output still applies, so resolve
        # it the usual way rather than forcing <default_data_root>/output.
        data_root, output = _verify_user_config_and_get_paths(
            {**config, "data_root": default_data_root}, system, local
        )
    else:
        data_root, output = _verify_user_config_and_get_paths(config, system, local)

    data_path = {
        "data_root": Path(data_root),
        "zea_root": zea_root,
        "output": Path(output),
        "system": system,
        "username": username,
        "hostname": hostname,
    }

    if verify:
        _verify_paths(data_path)

    return Config(data_path)


## Helper functions for handling user input


def _build_user_profile_string(data_paths, local: bool | None = None):
    """Builds a string that can be written to users.yaml to create a new user profile."""
    tab = "    "  # 4 spaces required in yaml
    base_string = (
        f"'{data_paths['username']}':\n"
        + f"  {data_paths['hostname']}:\n"
        + f"    system: {data_paths['system']}\n"
    )
    if local is None:
        return base_string + f"{tab}data_root: {data_paths['data_root']}"
    elif local is False:
        return base_string + (f"{tab}data_root:\n" + f"{tab}{tab}remote: {data_paths['data_root']}")
    elif local is True:
        return base_string + (f"{tab}data_root:\n" + f"{tab}{tab}local: {data_paths['data_root']}")
    else:
        raise ValueError("local should set to a boolean or None.")


def _to_write_user_profile_to_file(user_profile_string, user_config_path=DEFAULT_USERS_CONFIG_PATH):
    with open(user_config_path, "a", encoding="utf-8") as file:
        file.write("\n\n" + user_profile_string + "\n")
    print(f"\n✅ Your user profile was successfully added to `{user_config_path}`.\n")


def _pretty_print_data_paths(data_paths):
    for key, value in data_paths.items():
        print(f"\t{key}: {log.yellow(value)}")
    print()


def _prompt_user_for_data_root():
    data_root_input = input(
        "\nℹ️  Please enter the path to your data directory, "
        "or press Enter to use the default Linux path "
        f"`{DEFAULT_LINUX_DATA_ROOT}`: "
    )
    return DEFAULT_LINUX_DATA_ROOT if data_root_input == "" else data_root_input


def _acquire_and_validate_data_root():
    data_root_input = _prompt_user_for_data_root()
    while not os.path.isdir(data_root_input):
        print("\n The path you entered does not point to a directory, please try again.")
        data_root_input = _prompt_user_for_data_root()
    return data_root_input


def _warning_type_was_thrown(warning_type, list_of_warnings):
    """Returns True iff list_of_warnings contains a warning of type warning_type"""
    if not list_of_warnings:
        return False
    return any(isinstance(w.message, warning_type) for w in list_of_warnings)


def _to_read_yaml_file(path_str):
    path = Path(path_str)
    if not path.is_file():
        raise ValueError("YAML file path provided does not lead to a file.")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        return config


def _to_write_yaml_file(data, path_str):
    path = Path(path_str)
    if not path.is_file():
        raise ValueError("YAML file path provided does not lead to a file.")

    if not isinstance(data, dict):
        # Writing e.g. None here would replace the whole users.yaml with `null`.
        raise ValueError(f"Refusing to write {type(data).__name__} to {path_str}, expected a dict.")

    if _check_for_comments_yaml_file(path_str):
        log.warning(
            f"YAML file {path_str} contains comments. "
            "These will be removed if you write to the file."
        )
        input("Press Enter to continue or Ctrl+C to cancel.")

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


def _read_users_yaml_for_update(user_config_path):
    """Read ``users.yaml`` for an in-place update, or return None if that failed.

    Guards the update flows in :func:`create_new_user`: writing back what we could
    not read would replace the user's file with ``null``.
    """
    users_yaml_dict = _try(_to_read_yaml_file, {"path_str": user_config_path})
    if isinstance(users_yaml_dict, dict):
        return users_yaml_dict
    log.warning(
        f"Could not read `{user_config_path}`, so it will not be updated. "
        "Please check the file for corruptions."
    )
    return None


def _resolve_config_section(config, username, hostname, owns=None):
    """Returns the part of the ``users.yaml`` dict to write this user's paths back to.

    Follows the same levels as :func:`set_data_paths`. With ``owns`` set to a key, the
    section that actually supplies that key is returned instead of the most specific one,
    so that updating an inherited value edits it where it lives. Writing it to a deeper
    section would shadow the rest of the original mapping rather than extend it.
    """
    levels = _config_levels(config, username, hostname)
    if owns is not None:
        for level in reversed(levels):
            if owns in level:
                return level
    return levels[-1]


def create_new_user(user_config_path: "str | Path | None" = None, local: bool | None = None):
    """Creates a new user profile in `users.yaml` if one does not already exist.

    Args:
        user_config_path (str or Path, optional): Path that points to yaml file with user
            info. Defaults to None. In that case ``./users.yaml`` is taken.
        local (bool): Use the local dataset, or the one at the remote location.
            Per machine, the data_root can be set to a local or remote path.
            Each user can also have a different data_root for each machine.
            Default is None, which means that the data_root is shared for either
            local or remote (i.e. this parameter is ignored), see doc set_data_paths().

    Returns:
        Config: The data paths for the (possibly newly created) user profile,
            as returned by :func:`set_data_paths`.
    """
    if user_config_path is None:
        user_config_path = DEFAULT_USERS_CONFIG_PATH
    user_config_path = str(user_config_path)

    # Create empty file if it does not exist
    _create_empty_yaml(user_config_path)

    with warnings.catch_warnings(record=True) as list_of_warnings:
        warnings.simplefilter("always")
        data_paths = set_data_paths(user_config=user_config_path, local=local)

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
        user_warning_was_thrown = _warning_type_was_thrown(UnknownUsernameWarning, list_of_warnings)
        hostname_warning_was_thrown = _warning_type_was_thrown(
            UnknownHostnameWarning, list_of_warnings
        )
        local_remote_warning_was_thrown = _warning_type_was_thrown(
            UnknownLocalRemoteWarning, list_of_warnings
        )
        no_yaml_file_error_was_thrown = _warning_type_was_thrown(NoYamlFileError, list_of_warnings)

        if user_warning_was_thrown or no_yaml_file_error_was_thrown:
            print("ℹ️  Follow the instructions below to create your user profile.")
            data_root = _acquire_and_validate_data_root()
            data_paths["data_root"] = data_root
            user_profile_string = _build_user_profile_string(data_paths, local=local)
            user_response = input(
                "\n"
                + user_profile_string
                + "\n"
                + "\nℹ️  Would you like to automatically create your user"
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
            users_yaml_dict = _read_users_yaml_for_update(user_config_path)
            if users_yaml_dict is None:
                return data_paths
            data_root = _acquire_and_validate_data_root()
            data_paths["data_root"] = data_root
            section = _resolve_config_section(users_yaml_dict, data_paths["username"], None)
            # Keep whatever the machine already had (an `output`, for instance); we are
            # only filling in the data_root it was missing.
            existing = section.get(data_paths["hostname"])
            section[data_paths["hostname"]] = {
                **(existing if isinstance(existing, dict) else {}),
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
            users_yaml_dict = _read_users_yaml_for_update(user_config_path)
            if users_yaml_dict is None:
                return data_paths
            data_root = _acquire_and_validate_data_root()
            data_paths["data_root"] = data_root
            ## now update the data_root for the user and hostname in the yaml file
            ## use local or remote subkey depending on the local parameter. The data_root
            ## can live at the top level as well, so look up where it actually is.
            section = _resolve_config_section(
                users_yaml_dict, data_paths["username"], data_paths["hostname"], owns="data_root"
            )
            existing = section.get("data_root")
            section["data_root"] = {
                **(existing if isinstance(existing, dict) else {}),
                local_remote_str: data_root,
            }
            user_response = input(
                "\n"
                + yaml.dump(users_yaml_dict)
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


def format_data_path(path: "str | Path | HFPath", user: "Config | None" = None) -> "Path | HFPath":
    """Resolve a dataset path against the user's ``data_root``.

    Absolute paths and ``hf://`` paths are returned as-is, relative paths are
    interpreted relative to ``user.data_root``.

    Args:
        path (str, Path or HFPath): Path to the dataset. Can be absolute, relative
            to the user's ``data_root``, or a ``hf://`` path.
        user (Config, optional): User config as returned by :func:`set_data_paths`.
            Only required for relative paths.

    Returns:
        Path or HFPath: The resolved path. An :class:`~zea.tools.hf.HFPath` is
        returned for ``hf://`` paths, a :class:`pathlib.Path` otherwise.

    Raises:
        AssertionError: If ``path`` is relative and no ``user`` is provided.
    """
    if str(path).startswith(HF_PREFIX):
        return HFPath(path)
    if Path(path).is_absolute():
        return Path(path)

    assert user is not None, (
        "The dataset folder is relative, but no user is provided. "
        "Please provide a user to load the dataset relative to "
        "the user's data_root."
    )
    return Path(user.data_root) / path


if __name__ == "__main__":
    create_new_user("users.yaml", local=None)
