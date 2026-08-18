"""Tests for the dataclass-based config validation (zea.internal.config.validation)."""

import pytest

from zea.config import Config, _migrate_legacy_config, check_config
from zea.internal.config.users import (
    UserProfileSpec,
    local_remote_paths,
    validate_users_config,
)
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


# ---------------------------------------------------------------------------
# users.yaml schema (zea.internal.config.users)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "users_config",
    [
        pytest.param({}, id="empty"),
        pytest.param(None, id="none"),
        pytest.param({"data_root": "/mnt/shared/data"}, id="shared-root"),
        pytest.param({"data_root": {"local": "/l", "remote": "/r"}}, id="local-remote"),
        pytest.param({"data_root": {"local": "/l"}}, id="local-only"),
        pytest.param({"data_root": "/d", "output": {"remote": "/o"}}, id="output-mapping"),
        pytest.param(
            {
                "alice": {
                    "workstation": {
                        "system": "linux",
                        "data_root": {"local": "/l", "remote": "/r"},
                    },
                    "data_root": "/mnt/data/alice",
                },
                "bob": {"data_root": "/mnt/data/bob"},
                "data_root": "/mnt/shared/data",
            },
            id="users-and-machines",
        ),
    ],
)
def test_valid_users_configs(users_config):
    """A well-formed users.yaml validates and survives a round-trip."""
    result = validate_users_config(users_config)
    assert validate_users_config(result) == result


def test_users_config_keeps_only_the_keys_that_are_set():
    """Defaults are not filled in: absent means "fall back", unlike an explicit null."""
    result = validate_users_config({"data_root": "/mnt/shared/data"})
    assert result == {"data_root": "/mnt/shared/data"}
    assert "output" not in result and "system" not in result


def test_users_config_expands_nested_sections():
    """Nested user / machine sections come back as plain dicts, not spec objects."""
    result = validate_users_config({"alice": {"laptop": {"data_root": "/d"}}})
    assert result == {"alice": {"laptop": {"data_root": "/d"}}}
    assert isinstance(result["alice"], dict)
    assert not isinstance(result["alice"], UserProfileSpec)


@pytest.mark.parametrize(
    ("users_config", "match"),
    [
        pytest.param({"data_root": 42}, "must be a string or path", id="non-path-root"),
        pytest.param(
            {"data_root": {"nas": "/d"}}, r"unexpected keys \['nas'\]", id="unknown-subkey"
        ),
        pytest.param({"data_root": {}}, "at least one of", id="empty-mapping"),
        pytest.param(
            {"data_root": {"local": 3}}, "local: must be a string or path", id="non-path-subkey"
        ),
        pytest.param({"system": 3}, "must be a string", id="non-string-system"),
        pytest.param(
            {"notes": "my machine"},
            "expected a mapping for a user or machine section",
            id="stray-scalar",
        ),
        pytest.param(
            {"alice": {"laptop": {"data_root": {"nas": "/d"}}}},
            r"unexpected keys \['nas'\]",
            id="error-inside-nested-section",
        ),
    ],
)
def test_invalid_users_configs_raise(users_config, match):
    with pytest.raises(ValueError, match=match):
        validate_users_config(users_config)


def test_users_config_error_names_the_offending_section():
    """Errors are prefixed with the section they came from, to locate them in the file."""
    with pytest.raises(ValueError, match="alice"):
        validate_users_config({"alice": {"data_root": 42}})


def test_local_remote_paths_rejects_a_non_mapping():
    """The local/remote validator is also usable on its own."""
    with pytest.raises(ValueError, match="must be a mapping"):
        local_remote_paths("/not/a/mapping")


def test_users_config_rejects_a_non_mapping():
    with pytest.raises(ValueError, match="expected a mapping"):
        validate_users_config(["not", "a", "mapping"])


@pytest.mark.parametrize("local", [True, False, None])
def test_data_local_accepts_none(local):
    """`set_data_paths(local=None)` is valid, so the config must be able to say so."""
    assert validate_config({"data": {"local": local}})["data"]["local"] is local
