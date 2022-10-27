"""
Config utilities. Load settings from yaml files and access them as objects / dicts.
"""
import copy
from pathlib import Path

import yaml

class Config(dict):
    """Config class.

    This Config class extends a normal dictionary with the getattr and
    setattr functionality, and it enables saving to a yml.

    """

    def __init__(self, dictionary):
        super().__init__(dictionary)

        for key, value in self.items():
            assert(key not in ['keys', 'values', 'items']), \
                'The configuration contains the following key {key} which is '\
                'reserved already as a standard attribute of a dict.'

            # Change the key: TODO: now only string keys are supported.
            # new_key = str(key).replace('-', '_')
            #self.update({new_key, self.pop(key)})

            if isinstance(value, (list, tuple)):
                detected_dict = 0
                for idx, val in enumerate(value):
                    if isinstance(val, dict):
                        val = Config(val)
                        self[key][idx] = val
                        #setattr(self, key, val)
                        detected_dict += 1
                if not detected_dict:
                    setattr(self, key, value)

            elif isinstance(value, dict):
                value = Config(value)
                setattr(self, key, value)
            else:
                setattr(self, key, value)


    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def serialize(self):
        """Serialize config object to dictionary"""
        dictionary = {}
        for key, value in self.items():
            if isinstance(value, Config):
                dictionary[key] = value.serialize()
            # elif isinstance(value, list) or isinstance(value,tuple):
            #     for idx,
            else:
                dictionary[key] = value
        return dictionary

    def deep_copy(self):
        """Deep copy"""
        return Config(copy.deepcopy(self.serialize()))

    def save_to_yaml(self, path):
        """Save config contents to yaml"""
        with open(Path(path), 'w', encoding='utf-8') as save_file:
            yaml.dump(self.serialize(), save_file, default_flow_style=False)

def load_config_from_yaml(path):
    """Load config object from yaml file"""
    with open(Path(path), 'r', encoding='utf-8') as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    if dictionary:
        return Config(dictionary)
    else:
        return {}
