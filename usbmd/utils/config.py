"""Config utilities.
Load settings from yaml files and access them as objects / dicts.

- **Author(s)**     : Tristan Stevens
- **Date**          : 14-09-2021
"""

import copy
import inspect
from pathlib import Path

import yaml


class Config(dict):
    """Config class.

    This Config class extends a normal dictionary with easydict such that
    values can be accessed as class attributes. Furthermore it enables
    saving and loading to a yaml.

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

        # Get all methods of the Config class and store them in a list as protected attributes
        super().__setattr__(
            "__protected__",
            [x[0] for x in inspect.getmembers(Config, predicate=inspect.isroutine)],
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
                f"Cannot set attribute `{name}`. It is a method of the Config class."
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

        # Set attribute and update dictionary
        super().__setattr__(name, value)
        self[name] = value

    def update(self, override_dict):
        """
        Update the configuration object with values from the given dictionary.

        Args:
            override_dict (dict): A dictionary containing the values to update.
        """
        for name, value in override_dict.items():
            setattr(self, name, value)

    def serialize(self):
        """Serialize config object to dictionary"""
        dictionary = {}
        for key, value in self.items():
            if isinstance(value, Config):
                dictionary[key] = value.serialize()
            elif isinstance(value, Path):
                dictionary[key] = str(value)
            else:
                dictionary[key] = value
        return dictionary

    def deep_copy(self):
        """Deep copy"""
        return Config(copy.deepcopy(self.serialize()))

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
        self.__frozen__ = True

    def unfreeze(self):
        """Unfreeze config object. This means that new attributes can be added."""
        self.__frozen__ = False


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
        return {}
