"""Test configs"""

import sys
from pathlib import Path

import pytest
import yaml

from zea.config import Config, _compact_operation, check_config
from zea.internal.setup_zea import setup_config

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


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
        config (Config): The config to check.
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
    with open(file, "r", encoding="utf-8") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    try:
        configuration = check_config(configuration)
        # check another time, since defaults are now set, which are not
        # checked by the first check_config. Basically this checks if the
        # validation.py entries are correct.
        check_config(configuration)

    except ValueError as ve:
        raise ValueError(f"Error in config {file}") from ve


def test_config_rejects_string_path():
    """Config(path) must raise TypeError — use Config.from_path() instead."""
    with pytest.raises(TypeError, match="Config.from_path"):
        Config("configs/config_picmus_rf.yaml")


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
    config.to_yaml(path)

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


def test_config_from_path_forwards_hf_kwargs(tmp_path, monkeypatch):
    """Config.from_path should forward repo_type and other HF kwargs to the resolver."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a: 1\n", encoding="utf-8")

    calls = []

    def fake_resolve(path, **kwargs):
        calls.append((path, kwargs))
        return config_path

    monkeypatch.setattr("zea.config._hf_resolve_path", fake_resolve)

    config = Config.from_path(
        "hf://org/repo/config.yaml",
        repo_type="model",
        revision="my-branch",
    )

    assert config.a == 1
    assert calls == [
        (
            "hf://org/repo/config.yaml",
            {"repo_type": "model", "revision": "my-branch"},
        )
    ]


def test_setup_config_forwards_hf_kwargs(monkeypatch):
    """setup_config should pass HF kwargs through to Config.from_path."""
    calls = []

    def fake_from_path(path, **kwargs):
        calls.append((path, kwargs))
        return Config({"device": "cpu", "hide_devices": None, "data": {"local": True}})

    monkeypatch.setattr("zea.internal.setup_zea.Config.from_path", fake_from_path)
    monkeypatch.setattr("zea.internal.setup_zea.get_git_summary", lambda verbose: {"sha": "abc"})

    config = setup_config(
        "hf://org/repo/config.yaml",
        verbose=False,
        disable_config_check=True,
        repo_type="model",
        revision="my-branch",
    )

    assert config.git == {"sha": "abc"}
    assert calls == [
        (
            "hf://org/repo/config.yaml",
            {
                "loader": yaml.FullLoader,
                "repo_type": "model",
                "revision": "my-branch",
            },
        )
    ]


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


def test_config_repr():
    """Config repr is Config({...}) with no angle brackets."""
    config = Config(simple_dict)
    r = repr(config)
    assert r.startswith("Config(")
    assert r.endswith(")")
    assert "<" not in r


def test_pipeline_operations_compact_form():
    """``pipeline`` and ``parameters`` stay Config objects, but the individual
    ``pipeline.operations`` entries are kept as plain ``str``/``dict`` rather
    than nested Config objects: a name-only operation collapses to its bare
    name string, while an operation with ``params`` stays a plain dict.
    """
    config = Config(
        {
            "parameters": {"grid_size_x": 400},
            "pipeline": {
                "operations": [
                    {"name": "demodulate"},
                    {"name": "downsample", "params": {"factor": 4}},
                    {"name": "normalize", "params": {}},
                    "log_compress",
                ],
            },
        }
    )

    # Nested mappings are still Config objects ...
    assert isinstance(config.pipeline, Config)
    assert isinstance(config.parameters, Config)

    # ... but operations are plain str/dict, never Config.
    operations = config.pipeline.operations
    assert operations == [
        "demodulate",
        {"name": "downsample", "params": {"factor": 4}},
        "normalize",
        "log_compress",
    ]
    for operation in operations:
        assert isinstance(operation, (str, dict))
        assert not isinstance(operation, Config)


def test_compact_operation_helper():
    """``_compact_operation`` correctly handles all input shapes, including
    a plain dict whose ``params`` value is a nested Config object (the fix
    added in config.py lines 598-599)."""
    # bare string passes through unchanged
    assert _compact_operation("demodulate") == "demodulate"

    # name-only dict collapses to bare string
    assert _compact_operation({"name": "demodulate"}) == "demodulate"

    # name-only Config collapses to bare string
    assert _compact_operation(Config({"name": "demodulate"})) == "demodulate"

    # dict with params stays a plain dict, params value already plain
    assert _compact_operation({"name": "downsample", "params": {"factor": 4}}) == {
        "name": "downsample",
        "params": {"factor": 4},
    }

    # dict whose "params" is a Config — must be unwrapped to a plain dict
    op_with_config_params = {"name": "downsample", "params": Config({"factor": 4})}
    result = _compact_operation(op_with_config_params)
    assert result == {"name": "downsample", "params": {"factor": 4}}
    assert not isinstance(result["params"], Config)

    # Config whose "params" is itself a Config — same unwrap must happen
    op_config = Config({"name": "beamform", "params": Config({"num_patches": 200})})
    result = _compact_operation(op_config)
    assert result == {"name": "beamform", "params": {"num_patches": 200}}
    assert not isinstance(result["params"], Config)

    # empty params dict collapses to bare string (name-only semantics)
    assert _compact_operation({"name": "normalize", "params": {}}) == "normalize"


def test_pipeline_operations_compact_after_check_config():
    """``check_config`` re-wraps the validated dict into a Config; operations
    must remain in the compact str/dict form afterwards (regression test for
    operations being turned back into a list of Config objects)."""
    config = Config(
        {
            "pipeline": {
                "operations": [
                    {"name": "demodulate"},
                    {"name": "downsample", "params": {"factor": 4}},
                ]
            },
        }
    )
    config = check_config(config)
    assert config.pipeline.operations == [
        "demodulate",
        {"name": "downsample", "params": {"factor": 4}},
    ]
    assert not any(isinstance(op, Config) for op in config.pipeline.operations)


def test_config_operations_string_indexing():
    """config.pipeline.operations supports name-based string indexing."""
    config = Config(
        {
            "pipeline": {
                "operations": [
                    "demodulate",
                    {"name": "normalize", "params": {"output_range": [0, 1]}},
                ],
            }
        }
    )

    assert config.pipeline.operations["demodulate"] == "demodulate"
    result = config.pipeline.operations["normalize"]
    assert result == {"name": "normalize", "params": {"output_range": [0, 1]}}


def test_config_operations_int_indexing_unchanged():
    """Integer indexing on config.pipeline.operations is unaffected."""
    config = Config({"pipeline": {"operations": ["demodulate", "normalize"]}})

    assert config.pipeline.operations[0] == "demodulate"
    assert config.pipeline.operations[1] == "normalize"


def test_config_operations_is_list():
    """config.pipeline.operations is still a list (backward compat)."""
    from zea.internal.ops_list import OperationList

    config = Config({"pipeline": {"operations": ["demodulate"]}})
    assert isinstance(config.pipeline.operations, list)
    assert isinstance(config.pipeline.operations, OperationList)


def test_config_operations_not_found():
    """KeyError lists available names when key is missing."""
    config = Config({"pipeline": {"operations": ["demodulate"]}})

    with pytest.raises(KeyError, match="Available"):
        config.pipeline.operations["nonexistent"]


def test_config_operations_close_match():
    """KeyError suggests a close match when the name is slightly wrong."""
    config = Config({"pipeline": {"operations": ["demodulate"]}})

    with pytest.raises(KeyError, match="demodulate"):
        config.pipeline.operations["demoulate"]  # intentional typo


def test_config_operations_duplicate_raises():
    """Ambiguous bare name raises with numbered hints when duplicates exist."""
    config = Config({"pipeline": {"operations": ["normalize", "normalize"]}})

    with pytest.raises(KeyError, match="normalize_0"):
        config.pipeline.operations["normalize"]


def test_config_operations_numbered_suffix():
    """Numbered suffix resolves duplicate operations in a config."""
    config = Config(
        {
            "pipeline": {
                "operations": [
                    {"name": "normalize", "params": {"output_range": [0, 1]}},
                    {"name": "normalize", "params": {"output_range": [-1, 1]}},
                ]
            }
        }
    )

    assert config.pipeline.operations["normalize_0"]["params"]["output_range"] == [0, 1]
    assert config.pipeline.operations["normalize_1"]["params"]["output_range"] == [-1, 1]


def test_config_operations_mutate_via_index():
    """Modifying the returned dict mutates the entry in-place."""
    config = Config(
        {
            "pipeline": {
                "operations": [
                    {"name": "beamform", "params": {"enable_pfield": False}},
                ]
            }
        }
    )

    config.pipeline.operations["beamform"]["params"]["enable_pfield"] = True
    assert config.pipeline.operations["beamform"]["params"]["enable_pfield"] is True


def test_config_pickle():
    """Tests if the config can be pickled and unpickled without changing its contents."""
    import pickle

    config = Config(dictionary=doubly_nested_dict)

    # Pickle the config
    pickled = pickle.dumps(config)

    # Unpickle the config
    unpickled = pickle.loads(pickled)

    # Check if the config is the same
    config_check_equal_recursive(config, unpickled)
