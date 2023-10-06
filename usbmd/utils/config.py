"""Config utilities.
Load settings from yaml files and access them as objects / dicts.

- **Author(s)**     : Tristan Stevens
- **Date**          : 14-09-2021
"""
import copy
from pathlib import Path

import yaml


class Config(dict):
    """Config class.

    This Config class extends a normal dictionary with easydict such that
    values can be accessed as class attributes. Furthermore it enables
    saving and loading to a yaml.

    """
    def __init__(self, dictionary=None, **kwargs):
        if dictionary is None:
            dictionary = {}
        if kwargs:
            dictionary.update(**kwargs)
        for k, v in dictionary.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__:
            if not (k.startswith('__') and k.endswith('__')):
                if k not in ['serialize', 'deep_copy', 'save_to_yaml']:
                    setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        else:
            value = self.__class__(value) if isinstance(value, dict) else value
        super(Config, self).__setattr__(name, value)
        self[name] = value

    def serialize(self):
        """Serialize config object to dictionary"""
        dictionary = {}
        for key, value in self.items():
            if isinstance(value, Config):
                dictionary[key] = value.serialize()
            else:
                dictionary[key] = value
        return dictionary

    def deep_copy(self):
        """Deep copy"""
        return Config(copy.deepcopy(self.serialize()))

    def save_to_yaml(self, path):
        """Save config contents to yaml"""
        with open(Path(path), "w", encoding="utf-8") as save_file:
            yaml.dump(self.serialize(), save_file, default_flow_style=False)


def load_config_from_yaml(path):
    """Load config object from yaml file"""
    with open(Path(path), "r", encoding="utf-8") as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    if dictionary:
        return Config(dictionary)
    else:
        return {}
