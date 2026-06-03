"""Tests for the dataclass-based config validation (zea.internal.config.validation)."""

import pytest

from zea.config import Config, _migrate_legacy_config, check_config
from zea.internal.config.validation import (
    ConfigSchema,
    ParametersConfig,
    validate_config,
)

MINIMAL = {"data": {"dtype": "image", "dataset_folder": "some/folder"}}


def test_defaults_are_filled():
    """Validation fills in defaults for all optional sections."""
    result = validate_config(MINIMAL)

    assert result["device"] == "auto:1"
    assert result["git"] is None
    # Nested sections get their own defaults.
    assert result["plot"]["plot_lib"] == "opencv"
    assert result["pipeline"]["operations"] == ["identity"]
    assert result["parameters"]["lens_thickness"] == 1e-3
    # data defaults
    assert result["data"]["to_dtype"] == "image"
    assert result["data"]["dynamic_range"] == [-60, 0]


def test_validation_is_idempotent():
    """Validating an already-validated config yields the same dict."""
    once = validate_config(MINIMAL)
    twice = validate_config(once)
    assert once == twice


def test_missing_required_section_raises():
    with pytest.raises(ValueError, match="missing required keys"):
        validate_config({})


def test_missing_required_data_field_raises():
    with pytest.raises(ValueError, match="missing required keys"):
        validate_config({"data": {"dtype": "image"}})


@pytest.mark.parametrize(
    "config",
    [
        {**MINIMAL, "plot": {"plot_lib": "not_a_lib"}},  # enum
        {**MINIMAL, "device": "tpu:0"},  # regex/enum
        {**MINIMAL, "parameters": {"lens_thickness": -1.0}},  # positive float
        {**MINIMAL, "parameters": {"grid_size_x": 0}},  # positive integer
        {"data": {"dtype": "not_a_dtype", "dataset_folder": "x"}},  # enum
        {**MINIMAL, "data": {**MINIMAL["data"], "dynamic_range": [1, 2, 3]}},  # list len
    ],
)
def test_invalid_values_raise(config):
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("device", ["cpu", "gpu", "cuda", "cuda:0", "gpu:1", "auto:1", "auto:-1"])
def test_valid_devices(device):
    result = validate_config({**MINIMAL, "device": device})
    assert result["device"] == device


def test_arbitrary_parameters_keys_pass_through():
    """The parameters section accepts and round-trips arbitrary custom keys."""
    config = {**MINIMAL, "parameters": {"grid_size_x": 128, "my_custom_param": 42}}
    result = validate_config(config)
    assert result["parameters"]["grid_size_x"] == 128
    assert result["parameters"]["my_custom_param"] == 42


def test_arbitrary_top_level_keys_preserved():
    """Unknown top-level sections (e.g. model:) are preserved unchanged."""
    config = {**MINIMAL, "model": {"name": "diffusion", "steps": 100}}
    result = validate_config(config)
    assert result["model"] == {"name": "diffusion", "steps": 100}


def test_parameters_config_is_open():
    assert ParametersConfig.ALLOW_EXTRA is True
    assert ConfigSchema.ALLOW_EXTRA is True


def test_all_field_paths_includes_nested():
    paths = ConfigSchema.all_field_paths()
    assert "data.dtype" in paths
    assert "plot.plot_lib" in paths
    assert "parameters.grid_size_x" in paths
    assert "pipeline.operations" in paths
    assert "device" in paths


def test_scan_alias_migrated_to_parameters():
    """The deprecated scan: section is aliased to parameters: on load."""
    migrated = _migrate_legacy_config({**MINIMAL, "scan": {"grid_size_x": 64}})
    assert "scan" not in migrated
    assert migrated["parameters"] == {"grid_size_x": 64}


def test_check_config_freezes_config_object():
    config = Config(MINIMAL)
    checked = check_config(config)
    assert isinstance(checked, Config)
    assert checked.__frozen__ is True
    assert checked.plot.plot_lib == "opencv"
