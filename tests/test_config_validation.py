"""Tests for the dataclass-based config validation (zea.internal.config.validation)."""

import pytest

from zea.config import Config, _migrate_legacy_config, check_config
from zea.internal.config.validation import (
    ConfigSchema,
    ParametersConfig,
    validate_config,
)


def test_defaults_are_filled():
    """Validation fills in defaults for all optional sections."""
    result = validate_config({})

    assert result["device"] == "auto:1"
    assert result["git"] is None
    assert result["pipeline"]["operations"] == ["identity"]
    # data defaults
    assert result["data"]["local"] is True
    assert result["data"]["path"] is None
    assert result["data"]["indices"] is None


def test_validation_is_idempotent():
    """Validating an already-validated config yields the same dict."""
    once = validate_config({"data": {"path": "hf://zeahub/picmus/file.hdf5", "local": False}})
    twice = validate_config(once)
    assert once == twice


def test_empty_config_is_valid():
    """An empty config is valid — no required fields in ConfigSchema."""
    result = validate_config({})
    assert result["device"] == "auto:1"
    assert result["data"]["local"] is True


def test_missing_required_data_field_does_not_raise():
    """All data fields are optional — an empty data: section is valid."""
    result = validate_config({"data": {}})
    assert result["data"]["path"] is None
    assert result["data"]["local"] is True


@pytest.mark.parametrize(
    "config",
    [
        {"device": "tpu:0"},  # invalid device
        {"pipeline": {"jit_options": "bad_option"}},  # enum
        {"data": {"local": "yes"}},  # must be bool
        {"data": {"indices": {"bad": "type"}}},  # invalid indices type
    ],
)
def test_invalid_values_raise(config):
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("device", ["cpu", "gpu", "cuda", "cuda:0", "gpu:1", "auto:1", "auto:-1"])
def test_valid_devices(device):
    result = validate_config({"device": device})
    assert result["device"] == device


def test_arbitrary_parameters_keys_pass_through():
    """The parameters section accepts and round-trips arbitrary custom keys."""
    config = {"parameters": {"grid_size_x": 128, "my_custom_param": 42}}
    result = validate_config(config)
    assert result["parameters"]["grid_size_x"] == 128
    assert result["parameters"]["my_custom_param"] == 42


def test_arbitrary_top_level_keys_preserved():
    """Unknown top-level sections (e.g. model:) are preserved unchanged."""
    config = {"model": {"name": "diffusion", "steps": 100}}
    result = validate_config(config)
    assert result["model"] == {"name": "diffusion", "steps": 100}


def test_parameters_config_is_open():
    assert ParametersConfig.ALLOW_EXTRA is True
    assert ConfigSchema.ALLOW_EXTRA is True


def test_all_field_paths_includes_nested():
    paths = ConfigSchema.all_field_paths()
    assert "data.path" in paths
    assert "data.local" in paths
    assert "data.indices" in paths
    assert "pipeline.operations" in paths
    assert "pipeline.jit_options" in paths
    assert "device" in paths
    assert "git" in paths
    assert "plot.plot_lib" not in paths
    assert "data.dtype" not in paths
    assert "data.dynamic_range" not in paths


def test_scan_alias_migrated_to_parameters():
    """The deprecated scan: section is aliased to parameters: on load."""
    migrated = _migrate_legacy_config({"scan": {"grid_size_x": 64}})
    assert "scan" not in migrated
    assert migrated["parameters"] == {"grid_size_x": 64}


def test_check_config_freezes_config_object():
    config = Config({})
    checked = check_config(config)
    assert isinstance(checked, Config)
    assert checked.__frozen__ is True
    assert checked.pipeline.operations == ["identity"]
    assert checked.data.local is True


def test_data_config_local_default():
    """DataConfig local defaults to True even without data: in the config."""
    result = validate_config({})
    assert result["data"]["local"] is True


def test_data_config_passthrough_with_full_section():
    """A full data: section validates correctly."""
    config = {"data": {"path": "hf://zeahub/picmus/file.hdf5", "local": False, "indices": "all"}}
    result = validate_config(config)
    assert result["data"]["path"] == "hf://zeahub/picmus/file.hdf5"
    assert result["data"]["local"] is False
    assert result["data"]["indices"] == "all"
