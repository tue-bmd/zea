"""Test configs"""

import sys
from pathlib import Path

import pytest
import yaml
from schema import SchemaError

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from zea.config import Config  # noqa: E402
from zea.config.validation import check_config  # noqa: E402

# Define some dictionaries to test the Config class
simple_dict = {"a": 1, "b": 2, "c": 3}
nested_dict = {"a": 1, "nested_dictionary": {"b": 2, "c": 3}}
doubly_nested_dict = {
    "a": 1,
    "nested_dictionary": {"b": 2, "doubly_nested_dictionary": {"c": 3}},
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


def config_check_equal_recursive(config, dictionary):
    """Helper funtcion which recursively check if all values in config are
    of the correct type and equal as to corresponding key in the config.

    NOTE: This function is must only be used in the tests. Why? See:
    https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-and-checking-how-many-key-value-pairs-are-equal

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


@pytest.mark.parametrize(
    "file",
    [
        *list(Path("./configs").rglob("*.yaml")),
        *list(Path("./examples").rglob("*.yaml")),
    ],
)
def test_all_configs_valid(file):
    """Test if configs are valide according to schema"""
    if file.name in ["probes.yaml", "users.yaml"]:
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
def test_yaml_saving_loading(tmp_path, request, dictionary):
    """Tests if the config can be saved to a yaml file."""
    config = Config(dictionary=dictionary)

    # Get a uique name for every parameter set/run to avoid tests interfering
    test_id = request.node.name

    # Define the save path
    path = Path(tmp_path, f"temp_{test_id}", "config.yaml")

    # Create the directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the config to a yaml file
    config.save_to_yaml(path)

    # Load the config from the yaml file
    config2 = Config.from_yaml(path)

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


def test_config_accessed():
    """
    Tests if the _assert_all_accessed method works correctly.
    """
    # Case 1: access all attributes
    config = Config(**nested_dict)
    tmp = config.a
    tmp = config.nested_dictionary.get("b")
    tmp = config.nested_dictionary.pop("c")
    config._assert_all_accessed()  # should not raise an error

    # Case 2: access only some attributes
    config = Config(**nested_dict)
    tmp = config.nested_dictionary.b
    with pytest.raises(AssertionError):
        config._assert_all_accessed()  # should raise an error

    # Case 3: access all attributes using **kwargs
    config = Config(**simple_dict)
    Config(**config)
    config._assert_all_accessed()  # should not raise an error

    del tmp  # remove tmp to avoid unused variable warning


def test_config_update():
    """Tests if the update method works correctly."""
    config = Config(simple_dict)
    config.update(**nested_dict)  # update with kwargs
    config.update(nested_dict)  # update with dict
    assert isinstance(config.nested_dictionary, Config), (
        "config.nested_dictionary should be a Config object not just a dictionary"
    )


def test_config_recursive():
    """Tests if the update_recursive method works correctly."""
    config = Config({"a": 1, "b": {"c": 2, "d": 3}})
    config.update_recursive({"a": 4, "b": {"c": 5}})
    expected_config = Config({"a": 4, "b": {"c": 5, "d": 3}})

    config_check_equal_recursive(config, expected_config)
