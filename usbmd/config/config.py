"""Config utilities.
Load settings from yaml files and access them as objects / dicts.

- **Author(s)**     : Tristan Stevens
- **Date**          : 14-09-2021
"""

import copy
import difflib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path

import yaml


def update_leaves(data, func):
    """
    Recursively update the leaves of a nested structure using a given function.

    Args:
        data: The nested structure (e.g., dict, list, tuple, Config) to process.
        func: A function to apply to the leaves.

    Returns:
        The updated structure with the function applied to its leaves.
    """
    if isinstance(data, (Mapping, Config)):
        # If it's a dictionary, recurse for each value
        return type(data)(
            **{key: update_leaves(value, func) for key, value in data.items()}
        )
    elif isinstance(data, (list, tuple)):
        # If it's a list or tuple, recurse for each element
        updated = [update_leaves(item, func) for item in data]
        return type(data)(updated)  # Preserve the original type (list or tuple)
    else:
        # Base case: Apply the function to the leaf
        return func(data)


class Config:
    """Config class.

    This Config class extends a normal dictionary with easydict such that
    values can be accessed as class attributes.

    Other features:
    - `save_to_yaml` method to save the config to a yaml file.
    - `copy` method to create a deep copy of the config.
    - dictionary-like methods `keys`, `values`, `items`, `pop`, `update`, `get`.
    - Propose similar attribute names if a non-existing attribute is accessed.
    - Freeze the config object to prevent new attributes from being added.
    - Load config object from yaml file.

    We took inspiration from the following sources:
    - [EasyDict](https://pypi.org/project/easydict/)
    - [keras.utils.Config](https://keras.io/api/utils/experiment_management_utils/#config-class)
    But this implementation is superior :)
    """

    __frozen__ = False

    def __init__(self, dictionary=None, **kwargs):
        """
        Initializes a Config object.

        Args:
            dictionary (dict, optional): A dictionary containing key-value pairs
                to initialize the Config object. Defaults to None.
            **kwargs: Additional key-value pairs to initialize the Config object.
                Will override values in the dictionary if they have the same key.
        """
        super().__setattr__("__config__", {})

        # Get all methods of the Config class and store them in a list as protected attributes
        super().__setattr__(
            "__protected__",
            [x[0] for x in inspect.getmembers(Config, predicate=inspect.isroutine)]
            + ["__config__", "__protected__"],
        )

        if dictionary is None:
            dictionary = {}
        if kwargs:
            dictionary.update(**kwargs)
        for k, v in dictionary.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        # Check if attribute is a method of the Config class, this cannot be overridden
        if hasattr(self, "__protected__") and name in self.__protected__:
            raise AttributeError(
                f"Cannot set attribute `{name}`. It is used by the Config class."
            )

        # Check if config is frozen
        if self.__frozen__ and not hasattr(self, name):
            raise TypeError(
                f"Config is a frozen, no new attributes can be added. Tried to add: `{name}`"
            )

        # Ensures lists and tuples of dictionaries are converted to Config objects as well
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        else:
            value = self.__class__(value) if isinstance(value, dict) else value

        self.__config__[name] = value

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __getattr__(self, name):
        if name in self.__config__:
            return self.__config__[name]

        msg = f"Unknown attribute: '{name}'."
        if "difflib" in globals():
            closest_matches = difflib.get_close_matches(
                name, self.__config__.keys(), n=1, cutoff=0.7
            )
            if closest_matches:
                msg += f" Did you mean '{closest_matches[0]}'?"
        raise AttributeError(msg)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def to_json(self):
        """Return the config as a json string."""
        return json.dumps(self.__config__)

    def keys(self):
        """Return the keys of the config."""
        return self.__config__.keys()

    def values(self):
        """Return the values of the config."""
        return self.__config__.values()

    def items(self):
        """Return the items of the config."""
        return self.__config__.items()

    def pop(self, *args):
        """Remove and return the value of the given key."""
        return self.__config__.pop(*args)

    def update(self, override_dict):
        """
        Update the configuration object with values from the given dictionary.

        Args:
            override_dict (dict): A dictionary containing the values to update.
        """
        for name, value in override_dict.items():
            setattr(self, name, value)

    def get(self, keyname, value=None):
        """Get the value of the given key."""
        return self.__config__.get(keyname, value)

    def __delattr__(self, name):
        del self.__config__[name]

    def __delitem__(self, key):
        self.__delattr__(key)

    def __iter__(self):
        keys = sorted(self.__config__.keys())
        yield from keys

    def __contains__(self, item):
        return item in self.__config__

    def __len__(self):
        return len(self.__config__)

    def __repr__(self):
        return f"<Config {self.as_dict()}>"

    def as_dict(self):
        """Convert the config to a dictionary (recursively)."""
        dictionary = {}
        for key, value in self.items():
            if isinstance(value, Config):
                value = value.as_dict()
            dictionary[key] = value
        return dictionary

    def serialize(self):
        """Return a dict of this config object with all Path objects converted to strings."""
        return update_leaves(
            self.as_dict(), lambda x: str(x) if isinstance(x, Path) else x
        )

    def deep_copy(self):
        """Deep copy"""
        return Config(copy.deepcopy(self.as_dict()))

    def save_to_yaml(self, path):
        """Save config contents to yaml"""
        with open(Path(path), "w", encoding="utf-8") as save_file:
            yaml.dump(
                self.serialize(),
                save_file,
                default_flow_style=False,
                sort_keys=False,
            )

    def freeze(self):
        """
        Freeze config object. This means that no new attributes can be added.
        Only existing attributes can be modified.
        """
        super().__setattr__("__frozen__", True)

    def unfreeze(self):
        """Unfreeze config object. This means that new attributes can be added."""
        super().__setattr__("__frozen__", False)

    @staticmethod
    def load_from_yaml(path):
        """Load config object from yaml file"""
        return load_config_from_yaml(path)


def load_config_from_yaml(path, loader=yaml.FullLoader):
    """Load config object from yaml file
    Args:
        path (str): path to yaml file.
        loader (yaml.Loader, optional): yaml loader. Defaults to yaml.FullLoader.
            for custom objects, you might want to use yaml.UnsafeLoader.
    Returns:
        Config: config object.
    """
    with open(Path(path), "r", encoding="utf-8") as file:
        dictionary = yaml.load(file, Loader=loader)
    if dictionary:
        return Config(dictionary)
    else:
        return Config()
