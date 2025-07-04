"""Config module for managing configuration settings.

This module provides the :class:`Config` class for managing configuration settings,
with support for loading from YAML files, HuggingFace Hub, and dot notation access.

Features
--------

- Dot notation access to dictionary keys.
- Recursive conversion of nested dictionaries/lists to Config objects.
- Attribute access logging and suggestion of similar attribute names.
- Freezing/unfreezing to prevent/allow new attributes.
- Serialization to YAML/JSON.
- Integration with Hugging Face Hub.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

    from zea import Config

    # Load from YAML
    config = Config.from_yaml("config.yaml")
    # Load from HuggingFace Hub
    config = Config.from_hf("zea/diffusion-echonet-dynamic", "train_config.yaml")

    # Access attributes with dot notation
    print(config.model.name)

    # Update recursively
    config.update_recursive({"model": {"name": "new_model"}})

    # Save to YAML
    config.save_to_yaml("new_config.yaml")

"""

import copy
import difflib
import inspect
import json
from pathlib import Path
from typing import Union

import yaml
from huggingface_hub import hf_hub_download

from zea import log
from zea.internal.config.validation import config_schema
from zea.internal.core import object_to_tensor


class Config(dict):
    """Config class.

    This Config class extends a normal dictionary with dot notation access.

    Features:
        - `Config.from_yaml` method to load a config from a yaml file.
        - `Config.from_hf` method to load a config from a huggingface hub.
        - `save_to_yaml` method to save the config to a yaml file.
        - `copy` method to create a deep copy of the config.
        - Normal dictionary methods such as `keys`, `values`, `items`, `pop`, `update`, `get`.
        - Propose similar attribute names if a non-existing attribute is accessed.
        - Freeze the config object to prevent new attributes from being added.
        - Load config object from yaml file.
        - Logs all accessed attributes such that you can check if all attributes have been accessed.

    We took inspiration from the following sources:
        - `EasyDict <https://pypi.org/project/easydict/>`_
        - `keras.utils.Config <https://keras.io/api/utils/experiment_management_utils/#config-class>`_

    But this implementation is superior :)

    """  # noqa: E501

    __frozen__ = False

    def __init__(self, dictionary=None, __parent__=None, **kwargs):
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
            [x[0] for x in inspect.getmembers(Config, predicate=inspect.isroutine)]
            + ["__protected__", "__accessed__", "__parent__"],
        )
        super().__setattr__("__accessed__", {})
        super().__setattr__("__parent__", __parent__)

        if dictionary is None:
            dictionary = {}
        if kwargs:
            dictionary.update(**kwargs)
        for k, v in dictionary.items():
            setattr(self, k, v)

    def clear(self):
        """Clear the config object."""
        super().clear()
        self._reset_accessed()

    def fromkeys(self, keys, value=None):
        """Returns a config with the specified keys and value"""
        super().fromkeys(keys, value)

    def get(self, key, default=None):
        """Returns the value of the specified key"""
        self._mark_accessed(key)
        return super().get(key, default)

    def items(self):
        """Returns a list containing a tuple for each key value pair"""
        # Use a generator that calls __getitem__ for every key
        return [(key, self[key]) for key in self.keys()]

    def keys(self):
        """Returns a list containing the config's keys"""
        return super().keys()

    def pop(self, key, default=None):
        """Removes the element with the specified key"""
        self._mark_accessed(key)
        return super().pop(key, default)

    def popitem(self):
        """Removes the last inserted key-value pair"""
        key, value = super().popitem()
        self._mark_accessed(key)
        return key, value

    def setdefault(self, key, default=None):
        """Returns the value of the specified key. If the key does not exist: insert the key,
        with the specified value"""
        # Use __getitem__ to get values and __setitem__ to set values
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, dictionary: dict | None = None, **kwargs):
        """Updates the config with the specified key-value pairs"""
        # Use __setitem__ to set values
        if dictionary is None:
            dictionary = {}
        dictionary.update(kwargs)
        for key, value in dictionary.items():
            self[key] = value

    def update_recursive(self, dictionary: dict | None = None, **kwargs):
        """Recursively update the config with the provided dictionary and keyword arguments.

        If a key corresponds to another Config object, the update_recursive
        method is called recursively on that object. This makes it possible
        to update nested Config objects without replacing them.

        If a value is a list and the corresponding config value is also a list,
        each element is updated recursively if it is a Config, otherwise replaced.

        Example:

        .. code-block:: python

            config = Config({"a": 1, "b": {"c": 2, "d": 3}})
            config.update_recursive({"a": 4, "b": {"c": 5}})
            print(config)
            # <Config {'a': 4, 'b': {'c': 5, 'd': 3}}>
            # Notice how "d" is kept and only "c" is updated.

        Args:
            dictionary (dict, optional): Dictionary to update from.
            **kwargs: Additional key-value pairs to update.
        """
        if dictionary is None:
            dictionary = {}
        dictionary.update(kwargs)
        for key, value in dictionary.items():
            if key in self and isinstance(self[key], Config):
                self[key].update_recursive(value)
            elif key in self and isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, Config):
                        self[key][i].update_recursive(v)
                    else:
                        self[key][i] = v
            else:
                self[key] = value

    def values(self):
        """Returns a list of all the values in the config"""
        # Use __getitem__ to get values
        return (self[key] for key in self.keys())

    def __or__(self, other):
        """
        `self | other` operator. Returns a new config object with the contents of both configs.
        """
        return Config(super().__or__(other))

    def __ror__(self, other):
        """
        `other | self` operator. Returns a new config object with the contents of both configs.
        """
        return Config(super().__ror__(other))

    def __ior__(self, other):
        """
        `|=` operator. Updates the config with the contents of the other config.
        """
        self.update(other)
        return self

    def __iter__(self):
        """Returns an iterator that iterates through the keys of the config"""
        # Overwritten to ensure iteration respects our logic
        return iter(self.keys())

    def __contains__(self, key):
        """Returns True if the specified key exists in the config"""
        return super().__contains__(key)

    def __setattr__(self, name, value):
        # Check if attribute is a method of the Config class, this cannot be overridden
        if hasattr(self, "__protected__") and name in self.__protected__:
            raise AttributeError(f"Cannot set attribute `{name}`. It is used by the Config class.")

        # Check if config is frozen
        if self.__frozen__ and not hasattr(self, name):
            raise TypeError(
                f"Config is a frozen, no new attributes can be added. Tried to add: `{name}`"
            )

        # If overriding an existing attribute, mark it as unaccessed
        self._mark_unaccessed(name)

        # Convert tuple to list to allow for item assignment
        if isinstance(value, tuple):
            value = list(value)

        # Ensures lists and tuples of dictionaries are converted to Config objects as well
        if isinstance(value, list):
            value = [
                self.__class__(x, __parent__=self) if isinstance(x, dict) else x for x in value
            ]
        # Ensures dictionaries are converted to Config objects as well
        elif isinstance(value, dict):
            value = self.__class__(value, __parent__=self)

        super().__setitem__(name, value)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def _unknown_attr(self, name):
        msg = f"Unknown attribute: '{name}'."
        if "difflib" in globals():
            closest_matches = difflib.get_close_matches(name, self.keys(), n=1, cutoff=0.7)
            if closest_matches:
                msg += f" Did you mean '{closest_matches[0]}'?"
        return msg

    def _reset_accessed(self):
        """Reset accessed attributes."""
        self._recursive_setattr("__accessed__", {})

    def _mark_accessed(self, name):
        """Mark an attribute as accessed."""
        if name in self:
            self.__accessed__[name] = True

    def _mark_unaccessed(self, name):
        """Mark an attribute as unaccessed."""
        if name in self.__accessed__:
            del self.__accessed__[name]

    def _mark_accessed_recursive(self):
        """Mark an attribute and all its children as accessed."""

        def mark_accessed(config, key, value):
            config._mark_accessed(key)
            return key, value

        self.as_dict(mark_accessed)

    def _dict_items(self):
        """Return the items of the config object. Only used for internal purposes."""
        return super().items()

    def _trace_through_ancestors(self, key_trace=None):
        """Find the root ancestor of the config object."""
        if key_trace is None:
            key_trace = []
        if self.__parent__ is None:
            return self, key_trace
        for key, value in self.__parent__._dict_items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if v == self:
                        return self.__parent__._trace_through_ancestors([key + f"_{i}"] + key_trace)
            if value == self:
                return self.__parent__._trace_through_ancestors([key] + key_trace)
        raise ValueError("Parent not found in ancestors. Report to zea developers.")

    @staticmethod
    def _assert_key_accessed(config, key, value, _assert=True):
        """Assert that a key has been accessed."""
        if key not in config.__accessed__:
            key_trace = config._trace_through_ancestors()[1]
            msg = f"Attribute '{key}'='{value}' has not been accessed."
            if key_trace:
                msg += f" Has ancestors through '{key_trace}'"
            if _assert:
                raise AssertionError(msg)
            log.warning(msg)
        return key, value

    def _assert_all_accessed(self):
        """Assert that all attributes have been accessed."""
        # Temporary remove parent to avoid recursion if not being called from ancestor.
        self._all_unaccessed(_assert=True)

    def _log_all_unaccessed(self):
        """Log all unaccessed attributes."""
        self._all_unaccessed(_assert=False)

    def _all_unaccessed(self, _assert=False):
        """Assert or log all unaccessed attributes."""
        # Temporary remove parent to avoid recursion if not being called from ancestor.
        parent = self.__parent__
        super().__setattr__("__parent__", None)
        self.as_dict(lambda *args: self._assert_key_accessed(*args, _assert=_assert))
        super().__setattr__("__parent__", parent)

    def __getattr__(self, name):
        if name in self:
            self._mark_accessed(name)
            return super().__getitem__(name)

        msg = self._unknown_attr(name)
        raise AttributeError(msg)

    def __getitem__(self, key):
        if key in self:
            self._mark_accessed(key)
            return super().__getitem__(key)

        msg = self._unknown_attr(key)
        raise KeyError(msg)

    def __delattr__(self, name):
        del self[name]

    def __repr__(self):
        return f"<Config {self.as_dict()}>"

    def to_json(self):
        """Return the config as a json string."""
        return json.dumps(self)

    def as_dict(self, func_on_leaves=None):
        """Convert the config to a normal dictionary (recursively).

        Args:
            func_on_leaves (callable, optional): Function to apply to each leaf node.
                The function should take three arguments: the config object, the key, and the value.
                You can change the key and value inside the function. Defaults to None.
        """
        dictionary = {}
        for key, value in self._dict_items():
            if isinstance(value, Config):
                value = value.as_dict(func_on_leaves)
            elif isinstance(value, list):
                value = [v.as_dict(func_on_leaves) if isinstance(v, Config) else v for v in value]
            # a dict does not exist inside a Config object, because it is a Config object itself
            if func_on_leaves:
                key, value = func_on_leaves(self, key, value)
            dictionary[key] = value
        return dictionary

    def serialize(self):
        """Return a dict of this config object with all Path objects converted to strings."""
        return self.as_dict(lambda _, key, value: (key, _path_to_str(value)))

    def copy(self):
        """Deep copy the config object.

        This is useful when you want to modify the config object
        without changing the original. Does not preserve the access history or frozen state!
        """
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
        """Freeze config object.

        This means that no new attributes can be added.
        Only existing attributes can be modified.
        """
        self._recursive_setattr("__frozen__", True)

    def unfreeze(self):
        """Unfreeze config object. This means that new attributes can be added."""
        self._recursive_setattr("__frozen__", False)

    def _recursive_setattr(self, set_key, set_value):
        """Helper function to recursively set an attribute on all nested configs."""
        super().__setattr__(set_key, set_value)
        for _, value in self._dict_items():
            if isinstance(value, Config):
                value._recursive_setattr(set_key, set_value)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, Config):
                        v._recursive_setattr(set_key, set_value)

    @classmethod
    def from_yaml(cls, path, **kwargs):
        """Load config object from yaml file"""
        return _load_config_from_yaml(path, config_class=cls, **kwargs)

    @classmethod
    def from_hf(cls, repo_id, path, **kwargs):
        """Load config object from huggingface hub.

        Example:

        .. code-block:: python

            config = Config.from_hf("zeahub/configs", "config_camus.yaml", repo_type="dataset")

        Args:
            repo_id (str): huggingface hub repo id.
                For example: "zeahub/configs"
            path (str): path to the config file in the repo.
                For example: "train_config.yaml"
            **kwargs: additional arguments to pass to the `hf_hub_download`
                function. For example, use repo_type="dataset" to download
                from a dataset repo, or revision="main" to download from
                a specific branch.

        Returns:
            Config: config object.
        """
        local_path = hf_hub_download(repo_id, path, **kwargs)
        return _load_config_from_yaml(local_path, config_class=cls)

    def to_tensor(self):
        """Convert the attributes in the object to keras tensors"""
        return object_to_tensor(self)


def check_config(config: Union[dict, Config], verbose: bool = False):
    """Check a config given dictionary"""

    def _try_validate_config(config):
        try:
            config = config_schema.validate(config)
            return config
        except Exception as e:
            log.error(f"Config is not valid: {e}")
            raise e

    assert type(config) in [
        dict,
        Config,
    ], f"Config must be a dictionary or Config object, not {type(config)}"
    if isinstance(config, Config):
        config = config.serialize()
        config = _try_validate_config(config)
        config = Config(config)
        config.freeze()  # freeze because schema will add all defaults
    else:
        config = _try_validate_config(config)
    if verbose:
        log.success("Config is correct")
    return config


def _load_config_from_yaml(path, config_class=Config, loader=yaml.FullLoader):
    """Load config object from yaml file

    Args:
        path (str): path to yaml file.
        loader (yaml.Loader, optional): yaml loader. Defaults to yaml.FullLoader.
            for custom objects, you might want to use yaml.UnsafeLoader.
        config_class (type, optional): Config class to instantiate. Defaults to Config.

    Returns:
        Config: config object.
    """
    with open(Path(path), "r", encoding="utf-8") as file:
        dictionary = yaml.load(file, Loader=loader)
    if dictionary:
        return config_class(dictionary)
    else:
        return config_class()


def _path_to_str(path):
    """Convert a Path object to a string."""
    if hasattr(path, "as_posix"):
        # If path is a Path object, convert to string
        path = path.as_posix()
    return path
