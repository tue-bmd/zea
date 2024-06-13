"""Test configs"""

import shutil
import sys
from pathlib import Path

import pytest
import yaml
from schema import SchemaError

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from usbmd import Config, load_config_from_yaml
from usbmd.config.validation import check_config

# Define some dictionaries to test the Config class
simple_dict = {"a": 1, "b": 2, "c": 3}
nested_dict = {"a": 1, "nested_dictionary": {"b": 2, "c": 3}}
doubly_nested_dict = {
    "a": 1,
    "nested_dictionary": {"b": 2, "doubly_nested_dictionary": 4},
}
dict_strings = {"a": "first", "b": "second"}
dict_none = {"a": 1, "b": None, "c": 3}
# Bundle all dictionaries in a list
config_initializers = [
    simple_dict,
    nested_dict,
    doubly_nested_dict,
    dict_strings,
    dict_none,
]


@pytest.mark.parametrize(
    "file",
    [
        *list(Path("./configs").rglob("*.yaml")),
        *list(Path("./examples").rglob("*.yaml")),
    ],
)
def test_all_configs_valid(file):
    """Test if configs are valide according to schema"""
    if file.name == "probes.yaml":
        pytest.skip("probes.yaml is not checked here.")
    with open(file, "r", encoding="utf-8") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    try:
        configuration = check_config(configuration)
        # check another time, since defaults are now set, which are not
        # checked by the first check_config. Basically this checks if the
        # config_validation.py entries are correct.
        check_config(configuration)

    except SchemaError as se:
        raise ValueError(f"Error in config {file}") from se


def test_dot_indexing():
    """Tests if the dot indexing works for simple dictionaries."""
    dictionary = {"a": 3, "b": 4}
    config = Config(dictionary=dictionary)
    assert config.a == 3
    assert config.b == 4
    # Check if config raises an error when indexing key_not_in_config
    with pytest.raises(AttributeError):
        print(config.key_not_in_config)


def test_nested_dot_indexing():
    """Tests if the dot indexing works for nested dictionaries."""
    dictionary = {"a": 3, "subdict": {"b": 4, "c": 5}}
    config = Config(dictionary=dictionary)
    assert config.subdict.b == 4
    assert config.subdict.c == 5
    # Check if config raises an error when indexing key_not_in_config
    with pytest.raises(AttributeError):
        print(config.subdict.key_not_in_config)


@pytest.mark.parametrize("dictionary", config_initializers)
def test_recursive_config(dictionary):
    """Tests if all types in the config correspond to the ones in the
    dictionary except for the dictionaries, which are converted to Configs.
    """
    config = Config(dictionary=dictionary)
    config_check_equal_recursive(config, dictionary)


@pytest.mark.parametrize("dictionary", config_initializers)
def test_yaml_saving_loading(request, dictionary):
    """Tests if the config can be saved to a yaml file."""
    config = Config(dictionary=dictionary)

    # Get a uique name for every parameter set/run to avoid tests interfering
    test_id = request.node.name

    # Define the save path
    path = Path(f"temp_{test_id}", "config.yaml")

    # Create the directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the config to a yaml file
    config.save_to_yaml(path)

    # Load the config from the yaml file
    config2 = load_config_from_yaml(path)

    # Delete the directory and file
    shutil.rmtree(path.parent)

    try:
        # Check if the config is the same
        config_check_equal_recursive(config, config2)
    except AssertionError as exc:
        raise AssertionError("Config is not the same after saving and loading") from exc


@pytest.mark.parametrize("dictionary", config_initializers)
def test_serialize(dictionary):
    """Tests if the config can be serialized and deserialized without changing its contents."""
    config = Config(dictionary=dictionary)

    # Serialize the config
    serialized = config.serialize()

    # Check if the config is the same
    config_check_equal_recursive(config, serialized)


def config_check_equal_recursive(config, dictionary):
    """Recursively check if all values in config are of the correct type and
    equal as to corresponding key in the config.

    Args:
        config (utils.config.Config): The config to check.
        dictionary (dict): The dictionary to check against.

    Raises:
        AssertionError: If the types or values do not match.
    """
    for value1, value2 in zip(config.values(), dictionary.values()):
        if isinstance(value1, Config):
            config_check_equal_recursive(value1, value2)
        else:
            assert value1 == value2, "All values must be the same"
            assert isinstance(value1, type(value2)), "All types must be the same"


def test_check_equal():
    """Tests the config_check_equal_recursive function."""
    # Two configs with the same values
    config = Config(dictionary=simple_dict)
    config2 = Config(dictionary=simple_dict)
    # A different config
    config3 = Config(dictionary=nested_dict)
    # The same config but with a value changed
    config4 = Config(dictionary=simple_dict)
    config4.a = 2
    # The same config but with a value changed
    config5 = Config(dictionary=simple_dict)
    config5.b = "3"

    config_check_equal_recursive(config, config2)
    with pytest.raises(AssertionError):
        config_check_equal_recursive(config, config3)
    with pytest.raises(AssertionError):
        config_check_equal_recursive(config, config4)
    with pytest.raises(AssertionError):
        config_check_equal_recursive(config, config5)


def test_freeze():
    """Tests if the config can be frozen and no new attributes can be added."""
    config = Config(dictionary=simple_dict)
    config.freeze()
    with pytest.raises(TypeError):
        config.new_attribute = 1
    config.unfreeze()
    config.new_attribute = 1


@pytest.mark.parametrize("dictionary", [{"freeze": "Yes"}, {"save_to_yaml": "No"}])
def test_protected_attribute(dictionary):
    """Tests if protected attributes cannot be overridden."""
    with pytest.raises(AttributeError):
        Config(dictionary=dictionary)


@pytest.mark.parametrize("dictionary", config_initializers)
def test_dict_and_attributes_equal(dictionary):
    """Tests if the dictionary and attributes are equal."""

    def test_getitem(config):
        """Tests if the getitem method works for simple dictionaries."""
        for key, value in config.items():
            assert getattr(config, key) == value

        # Check if config raises an error when indexing a missing key
        with pytest.raises(KeyError):
            print(config["key_not_in_config"])

    config = Config(dictionary=dictionary)
    test_getitem(config)
    config["update_with_dict"] = 1
    config.update({"update_with_update": 2})
    config.update_with_attribute = 3
    test_getitem(config)
