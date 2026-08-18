"""Tests for :mod:`zea.datapaths`, which resolves local (and remote) data paths."""

import builtins
import getpass
import socket
import warnings
from pathlib import Path

import pytest
import yaml

from zea.config import Config
from zea.datapaths import (
    DEFAULT_DATA_ROOT,
    DEFAULT_USERS_CONFIG_PATH,
    NoYamlFileError,
    UnknownHostnameWarning,
    UnknownLocalRemoteWarning,
    UnknownUsernameWarning,
    _build_user_profile_string,
    _check_for_comments_yaml_file,
    _default_output_path,
    _fallback_to_default_data_root,
    _is_interactive,
    _resolve_config_section,
    _to_read_yaml_file,
    _to_write_yaml_file,
    _try,
    _warning_type_was_thrown,
    create_new_user,
    format_data_path,
    set_data_paths,
)
from zea.tools.hf import HFPath

USERNAME = getpass.getuser()
HOSTNAME = socket.gethostname()

user_config0 = {
    USERNAME: {
        HOSTNAME: {
            "data_root": "C:/path_to_my_data_root/",
            "output": {
                "local": "C:/path_to_my_output/",
                "remote": "Z:/path_to_my_output/",
            },
        }
    }
}

user_config1 = {
    USERNAME: {
        HOSTNAME: {
            "data_root": {
                "local": "C:/path_to_my_output/",
                "remote": "Z:/path_to_my_output/",
            },
        }
    }
}

user_config2 = {
    "data_root": {
        "local": "C:/path_to_my_data_root/",
        "remote": "Z:/path_to_my_data_root/",
    },
    "output": {
        "local": "C:/path_to_my_output/",
        "remote": "Z:/path_to_my_output/",
    },
}

user_config3 = {
    "data_root": {
        "local": "C:/path_to_my_data_root/",
        "remote": "Z:/path_to_my_data_root/",
    },
    "output": {
        "local": "C:/path_to_my_output/",
        "remote": "Z:/path_to_my_output/",
    },
    "user_not_on_this_machine": {
        "data_root": {
            "local": "C:/path_to_my_output/",
            "remote": "Z:/path_to_my_output/",
        },
    },
}


def _write_yaml(path, data):
    """Dump ``data`` to ``path`` and return the path as a string."""
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(data, file)
    return str(path)


@pytest.fixture(name="answers")
def _answers(monkeypatch):
    """Feeds scripted answers to :func:`input` and records the prompts shown."""

    class Answers:
        def __init__(self):
            self.queue = []
            self.prompts = []

        def set(self, *responses):
            self.queue = list(responses)

        def _input(self, prompt=""):
            self.prompts.append(prompt)
            assert self.queue, f"Unexpected prompt for input: {prompt}"
            return self.queue.pop(0)

    answers = Answers()
    monkeypatch.setattr(builtins, "input", answers._input)
    return answers


# --------------------------------------------------------------------------------------
# set_data_paths
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_config",
    [user_config0, user_config1, user_config2, user_config3],
)
def test_set_data_paths(user_config):
    """Test set data paths"""

    for local in [True, False]:
        data_paths = set_data_paths(user_config, local=local)
        assert "data_root" in data_paths, f"data_root not in data_paths for local={local}"
        assert "output" in data_paths, f"output not in data_paths for local={local}"


@pytest.mark.parametrize(
    "user_config",
    ["users.test.yaml"],  # non-existing file
)
def test_set_data_paths_defaults(tmp_path, user_config):
    """Test set data paths"""

    if isinstance(user_config, str):
        # Add temp path and set as string
        user_config = str(tmp_path / user_config)

    for local in [True, False]:
        with pytest.warns((UnknownUsernameWarning, NoYamlFileError)):
            data_paths = set_data_paths(user_config, local=local)
        assert "data_root" in data_paths, f"data_root not in data_paths for local={local}"
        assert "output" in data_paths, f"output not in data_paths for local={local}"


def test_set_data_paths_returns_config_with_metadata(tmp_path):
    """All documented keys are present and reachable through dot notation."""
    data_paths = set_data_paths({"data_root": str(tmp_path)})

    assert isinstance(data_paths, Config)
    for key in ["data_root", "zea_root", "output", "system", "username", "hostname"]:
        assert key in data_paths, f"{key} missing from data paths"

    assert data_paths.data_root == Path(tmp_path)
    assert data_paths.username == USERNAME
    assert data_paths.hostname == HOSTNAME
    assert Path(data_paths.zea_root).name == "zea"


def test_set_data_paths_from_yaml_file(tmp_path):
    """A path to a YAML file gives the same result as the equivalent dictionary."""
    config = {"data_root": "/some/data", "output": "/some/output"}
    config_path = _write_yaml(tmp_path / "users.yaml", config)

    from_file = set_data_paths(config_path)
    from_dict = set_data_paths(config)

    assert from_file.data_root == from_dict.data_root == Path("/some/data")
    assert from_file.output == from_dict.output == Path("/some/output")


def test_set_data_paths_default_output_is_data_root_subfolder():
    """Without an ``output`` key the output defaults to ``data_root/output``."""
    data_paths = set_data_paths({"data_root": "/some/data"})
    assert data_paths.output == Path("/some/data/output")
    assert data_paths.output == _default_output_path("/some/data")


@pytest.mark.parametrize(
    ("local", "expected"),
    [(True, Path("/local/data")), (False, Path("/remote/data"))],
)
def test_set_data_paths_selects_local_or_remote(local, expected):
    """``local`` picks between the ``local`` and ``remote`` sub keys."""
    config = {"data_root": {"local": "/local/data", "remote": "/remote/data"}}
    assert set_data_paths(config, local=local).data_root == expected


@pytest.mark.parametrize(
    ("config", "local"),
    [
        ({"data_root": {"remote": "/remote/data"}}, True),
        ({"data_root": {"local": "/local/data"}}, False),
    ],
)
def test_set_data_paths_missing_local_remote_key_warns(config, local):
    """A missing local/remote sub key falls back to the default data root."""
    with pytest.warns(UnknownLocalRemoteWarning):
        data_paths = set_data_paths(config, local=local)
    assert data_paths.data_root == Path(_fallback_to_default_data_root(data_paths.system))


def test_set_data_paths_local_none_with_local_remote_keys_raises():
    """``local=None`` is only valid when the paths are plain strings."""
    config = {"data_root": {"local": "/local/data", "remote": "/remote/data"}}
    with pytest.raises(ValueError, match="Please set local to True or False"):
        set_data_paths(config, local=None)


def test_set_data_paths_local_none_with_string_path():
    """``local=None`` works fine for a shared (string) data root."""
    assert set_data_paths({"data_root": "/some/data"}, local=None).data_root == Path("/some/data")


def test_set_data_paths_system_mismatch_raises():
    """A ``system`` key that does not match the current OS is an error."""
    config = {"system": "some-other-os", "data_root": "/some/data"}
    with pytest.raises(AssertionError, match="does not match user settings"):
        set_data_paths(config)


def test_set_data_paths_ignores_unknown_keys():
    """Keys other than ``data_root`` and ``output`` are ignored, not validated."""
    config = {"data_root": "/some/data", "some_other_key": ["not", "a", "path"]}
    assert set_data_paths(config).data_root == Path("/some/data")


def test_set_data_paths_invalid_path_type_raises():
    """A data root that is neither a string nor a dict is rejected."""
    with pytest.raises(AssertionError, match="should be either a string or a dict"):
        set_data_paths({"data_root": 42})


def test_set_data_paths_invalid_local_remote_subkey_raises():
    """Only ``local`` and ``remote`` are accepted as sub keys."""
    with pytest.raises(AssertionError, match="should be either a string or a dict"):
        set_data_paths({"data_root": {"somewhere_else": "/some/data"}})


@pytest.mark.parametrize("user_config", [42, ["data_root"], Path("users.yaml")])
def test_set_data_paths_invalid_user_config_type_raises(user_config):
    """``user_config`` must be a string, a dictionary or None."""
    with pytest.raises(ValueError, match="user_config should be a string or dictionary"):
        set_data_paths(user_config)


def test_set_data_paths_does_not_mutate_input():
    """The user config passed in by the caller is left untouched."""
    config = {
        USERNAME: {HOSTNAME: {"system": "some-other-os", "data_root": "/some/data"}},
        "data_root": "/fallback/data",
    }
    original = yaml.safe_dump(config)

    with pytest.raises(AssertionError):
        set_data_paths(config)

    assert yaml.safe_dump(config) == original


def test_set_data_paths_username_takes_precedence():
    """A user specific entry wins from the userless fallback."""
    config = {"data_root": "/fallback/data", USERNAME: {"data_root": "/user/data"}}
    assert set_data_paths(config).data_root == Path("/user/data")


def test_set_data_paths_hostname_takes_precedence():
    """A machine specific entry wins from the user level fallback."""
    config = {
        USERNAME: {
            "data_root": "/user/data",
            HOSTNAME: {"data_root": "/machine/data"},
        }
    }
    assert set_data_paths(config).data_root == Path("/machine/data")


def test_set_data_paths_falls_back_to_user_level_data_root():
    """An unknown hostname falls back to the user level data root without warning."""
    config = {
        USERNAME: {
            "some-other-machine": {"data_root": "/other/data"},
            "data_root": "/user/data",
        }
    }
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        data_paths = set_data_paths(config)
    assert data_paths.data_root == Path("/user/data")
    assert not _warning_type_was_thrown(UnknownHostnameWarning, recorded)


def test_set_data_paths_unknown_username_warns():
    """An unknown user without fallback warns and uses the OS default."""
    config = {"some_other_user": {"data_root": "/other/data"}}
    with pytest.warns(UnknownUsernameWarning):
        data_paths = set_data_paths(config)
    assert data_paths.data_root == Path(_fallback_to_default_data_root(data_paths.system))


def test_set_data_paths_unknown_hostname_warns():
    """A known user on an unknown machine raises the hostname specific warning."""
    config = {USERNAME: {"some-other-machine": {"data_root": "/other/data"}}}
    with pytest.warns(UnknownHostnameWarning, match=f"hostname={HOSTNAME}"):
        data_paths = set_data_paths(config)
    assert data_paths.data_root == Path(_fallback_to_default_data_root(data_paths.system))


def test_set_data_paths_unknown_system_falls_back(monkeypatch):
    """An OS without a default entry falls back to the generic default data root."""
    monkeypatch.setattr("zea.datapaths.platform.system", lambda: "Plan9")
    with pytest.warns(UnknownUsernameWarning):
        data_paths = set_data_paths({"some_other_user": {"data_root": "/other/data"}})
    assert data_paths.data_root == Path(DEFAULT_DATA_ROOT[None])


def test_set_data_paths_creates_missing_yaml_file(tmp_path):
    """A missing users.yaml is created (empty) and the defaults are used."""
    config_path = tmp_path / "users.yaml"
    with pytest.warns((NoYamlFileError, UnknownUsernameWarning)):
        data_paths = set_data_paths(str(config_path))

    assert config_path.is_file(), "users.yaml should have been created"
    assert config_path.read_text(encoding="utf-8") == ""
    assert data_paths.data_root == Path(_fallback_to_default_data_root(data_paths.system))


def test_set_data_paths_does_not_prompt_when_not_interactive(tmp_path, monkeypatch):
    """Non-interactive runs (CI, notebooks) must never block on a prompt."""
    monkeypatch.setattr("zea.datapaths._is_interactive", lambda: False)
    monkeypatch.setattr(
        builtins, "input", lambda *args, **kwargs: pytest.fail("input() should not be called")
    )
    with pytest.warns((NoYamlFileError, UnknownUsernameWarning)):
        set_data_paths(str(tmp_path / "users.yaml"))


def test_set_data_paths_prompts_for_the_requested_file(tmp_path, monkeypatch):
    """Interactive runs offer to create a profile in the *requested* users.yaml."""
    config_path = tmp_path / "elsewhere.yaml"
    calls = {}

    def _fake_create_new_user(user_config_path=None, local=None):
        calls["user_config_path"] = user_config_path
        calls["local"] = local

    monkeypatch.setattr("zea.datapaths._is_interactive", lambda: True)
    monkeypatch.setattr("zea.datapaths.create_new_user", _fake_create_new_user)

    with pytest.warns((NoYamlFileError, UnknownUsernameWarning)):
        set_data_paths(str(config_path), local=False)

    assert Path(calls["user_config_path"]) == config_path
    assert calls["local"] is False


def test_set_data_paths_none_uses_default_users_yaml(tmp_path, monkeypatch):
    """``user_config=None`` falls back to ``./users.yaml``."""
    monkeypatch.chdir(tmp_path)
    _write_yaml(tmp_path / DEFAULT_USERS_CONFIG_PATH, {"data_root": str(tmp_path)})
    assert set_data_paths(None).data_root == Path(tmp_path)


def test_set_data_paths_survives_a_failing_prompt(tmp_path, monkeypatch):
    """A failed profile creation is reported, but never breaks ``set_data_paths``."""
    messages = []
    monkeypatch.setattr("zea.datapaths.log.warning", messages.append)
    monkeypatch.setattr("zea.datapaths._is_interactive", lambda: True)
    monkeypatch.setattr(
        builtins, "input", lambda *args, **kwargs: (_ for _ in ()).throw(EOFError("no stdin"))
    )

    with pytest.warns((NoYamlFileError, UnknownUsernameWarning)):
        data_paths = set_data_paths(str(tmp_path / "users.yaml"))

    assert any("Could not create user profile" in message for message in messages)
    assert data_paths.data_root == Path(_fallback_to_default_data_root(data_paths.system))


def test_set_data_paths_empty_yaml_file(tmp_path):
    """An existing but empty users.yaml behaves like an empty config."""
    config_path = tmp_path / "users.yaml"
    config_path.write_text("", encoding="utf-8")
    with pytest.warns(UnknownUsernameWarning):
        data_paths = set_data_paths(str(config_path))
    assert data_paths.data_root == Path(_fallback_to_default_data_root(data_paths.system))


def test_set_data_paths_corrupt_yaml_file_raises(tmp_path):
    """A users.yaml that is not a mapping is reported instead of being overwritten."""
    config_path = tmp_path / "users.yaml"
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML file should contain a dictionary"):
        set_data_paths(str(config_path))
    # The corrupt file is left alone so the user can fix it.
    assert config_path.read_text(encoding="utf-8") == "- just\n- a\n- list\n"


def test_set_data_paths_verify_warns_for_missing_directories(tmp_path, monkeypatch):
    """``verify=True`` warns about paths that do not exist, ``verify=False`` stays quiet."""
    messages = []
    monkeypatch.setattr("zea.datapaths.log.warning", messages.append)

    set_data_paths({"data_root": str(tmp_path / "nope"), "output": str(tmp_path)}, verify=True)
    assert any("data_root" in message for message in messages)
    assert not any("output path" in message for message in messages)

    messages.clear()
    set_data_paths({"data_root": str(tmp_path / "nope"), "output": str(tmp_path)}, verify=False)
    assert not messages


def test_set_data_paths_is_exported_from_zea():
    """``set_data_paths`` is part of the public ``zea`` namespace."""
    import zea

    assert zea.set_data_paths is set_data_paths


# --------------------------------------------------------------------------------------
# format_data_path
# --------------------------------------------------------------------------------------


def test_format_data_path_absolute_is_unchanged():
    """Absolute paths do not need a user and are returned as a Path."""
    assert format_data_path("/data/dataset") == Path("/data/dataset")
    assert format_data_path(Path("/data/dataset")) == Path("/data/dataset")


def test_format_data_path_relative_uses_data_root():
    """Relative paths are resolved against the user's data root."""
    user = set_data_paths({"data_root": "/data"})
    assert format_data_path("dataset/train", user) == Path("/data/dataset/train")


def test_format_data_path_relative_without_user_raises():
    """Resolving a relative path without a user is an error."""
    with pytest.raises(AssertionError, match="no user is provided"):
        format_data_path("dataset/train")


@pytest.mark.parametrize("path", ["hf://zeahub/camus-sample", HFPath("hf://zeahub/camus-sample")])
def test_format_data_path_huggingface(path):
    """``hf://`` paths are returned as HFPath, with or without a user."""
    resolved = format_data_path(path)
    assert isinstance(resolved, HFPath)
    assert str(resolved) == "hf://zeahub/camus-sample"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("system", ["windows", "linux", "darwin"])
def test_fallback_to_default_data_root_known_system(system):
    """Known systems get their own default."""
    assert _fallback_to_default_data_root(system) == DEFAULT_DATA_ROOT[system]


@pytest.mark.parametrize("system", ["Plan9", None, ""])
def test_fallback_to_default_data_root_unknown_system(system):
    """Unknown systems fall back to the generic default."""
    assert _fallback_to_default_data_root(system) == DEFAULT_DATA_ROOT[None]


@pytest.mark.parametrize("local", [None, True, False])
def test_build_user_profile_string(local):
    """The generated profile is valid YAML that ``set_data_paths`` understands."""
    data_paths = {
        "username": USERNAME,
        "hostname": HOSTNAME,
        "system": "linux",
        "data_root": "/some/data",
    }
    profile = yaml.safe_load(_build_user_profile_string(data_paths, local=local))

    entry = profile[USERNAME][HOSTNAME]
    assert entry["system"] == "linux"
    if local is None:
        assert entry["data_root"] == "/some/data"
    else:
        assert entry["data_root"] == {"local" if local else "remote": "/some/data"}


def test_build_user_profile_string_invalid_local_raises():
    """``local`` has to be a boolean or None."""
    data_paths = {
        "username": USERNAME,
        "hostname": HOSTNAME,
        "system": "linux",
        "data_root": "/some/data",
    }
    with pytest.raises(ValueError, match="local should set to a boolean or None"):
        _build_user_profile_string(data_paths, local="yes")


def test_warning_type_was_thrown():
    """Detects a single warning type among warnings of mixed types."""
    with pytest.warns((NoYamlFileError, UnknownUsernameWarning)) as recorded:
        warnings.warn("no file", NoYamlFileError)
        warnings.warn("no user", UnknownUsernameWarning)

    assert _warning_type_was_thrown(NoYamlFileError, recorded.list)
    assert _warning_type_was_thrown(UnknownUsernameWarning, recorded.list)
    assert not _warning_type_was_thrown(UnknownHostnameWarning, recorded.list)
    assert not _warning_type_was_thrown(NoYamlFileError, [])


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"data_root": "/top"}, {"data_root": "/top"}),
        ({USERNAME: {"data_root": "/user"}}, {"data_root": "/user"}),
        ({USERNAME: {HOSTNAME: {"data_root": "/machine"}}}, {"data_root": "/machine"}),
        ({USERNAME: {"other-machine": {"data_root": "/x"}, "data_root": "/user"}}, None),
    ],
)
def test_resolve_config_section(config, expected):
    """The section holding the paths is found the same way ``set_data_paths`` does."""
    section = _resolve_config_section(config, USERNAME, HOSTNAME)
    if expected is None:
        assert section["data_root"] == "/user"
    else:
        assert section == expected


def test_yaml_read_write_roundtrip(tmp_path):
    """``_to_read_yaml_file``/``_to_write_yaml_file`` round-trip a config."""
    config_path = tmp_path / "users.yaml"
    config = {USERNAME: {HOSTNAME: {"data_root": "/some/data"}}}
    _write_yaml(config_path, {})

    _to_write_yaml_file(config, str(config_path))
    assert _to_read_yaml_file(str(config_path)) == config


@pytest.mark.parametrize("function", [_to_read_yaml_file, _check_for_comments_yaml_file])
def test_yaml_helpers_require_an_existing_file(tmp_path, function):
    """Reading a users.yaml that does not exist is an error."""
    with pytest.raises(ValueError, match="does not lead to a file"):
        function(str(tmp_path / "does_not_exist.yaml"))


def test_to_write_yaml_file_requires_an_existing_file(tmp_path):
    """Writing only happens to a users.yaml that already exists."""
    with pytest.raises(ValueError, match="does not lead to a file"):
        _to_write_yaml_file({"data_root": "/some/data"}, str(tmp_path / "does_not_exist.yaml"))


def test_check_for_comments_yaml_file(tmp_path, answers, monkeypatch):
    """Comments are detected and the user is warned before they get dropped."""
    plain = tmp_path / "plain.yaml"
    plain.write_text("data_root: /some/data\n", encoding="utf-8")
    commented = tmp_path / "commented.yaml"
    commented.write_text("# my data\ndata_root: /some/data\n", encoding="utf-8")

    assert not _check_for_comments_yaml_file(str(plain))
    assert _check_for_comments_yaml_file(str(commented))

    messages = []
    monkeypatch.setattr("zea.datapaths.log.warning", messages.append)
    answers.set("")  # confirm that the comments may be dropped
    _to_write_yaml_file({"data_root": "/other/data"}, str(commented))

    assert any("contains comments" in message for message in messages)
    assert _to_read_yaml_file(str(commented)) == {"data_root": "/other/data"}


@pytest.mark.parametrize(
    ("stdin", "expected"),
    [(None, False), (object(), False), ("closed", False)],
)
def test_is_interactive_without_a_terminal(monkeypatch, tmp_path, stdin, expected):
    """Anything that is not a terminal counts as non-interactive."""
    if stdin == "closed":
        stdin = open(tmp_path / "stdin.txt", "w", encoding="utf-8")
        stdin.close()
    monkeypatch.setattr("zea.datapaths.sys.stdin", stdin)
    assert _is_interactive() is expected


def test_is_interactive_with_a_terminal(monkeypatch):
    """A tty means we can prompt the user."""

    class FakeTTY:
        def isatty(self):
            return True

    monkeypatch.setattr("zea.datapaths.sys.stdin", FakeTTY())
    assert _is_interactive() is True


def test_try_reports_errors_without_raising(capsys):
    """``_try`` swallows errors so a failed write does not crash the CLI."""

    def _boom(message):
        raise RuntimeError(message)

    assert _try(_boom, {"message": "kaboom"}) is None
    assert "kaboom" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# create_new_user
# --------------------------------------------------------------------------------------


def test_create_new_user_writes_profile(tmp_path, answers, monkeypatch):
    """A brand new users.yaml gets a profile for the current user and machine."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()

    answers.set(str(data_root), "")  # data root, then confirm
    create_new_user("users.yaml", local=None)

    assert not answers.queue, "not all scripted answers were used"
    config = _to_read_yaml_file("users.yaml")
    assert config[USERNAME][HOSTNAME]["data_root"] == str(data_root)
    # The written profile is picked up on the next run, without any warning.
    assert set_data_paths("users.yaml").data_root == data_root


def test_create_new_user_defaults_to_users_yaml(tmp_path, answers, monkeypatch):
    """Without a path, ``./users.yaml`` is used (as documented)."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()

    answers.set(str(data_root), "")
    create_new_user(local=None)

    assert (tmp_path / DEFAULT_USERS_CONFIG_PATH).is_file()
    assert _to_read_yaml_file(DEFAULT_USERS_CONFIG_PATH)[USERNAME][HOSTNAME]["data_root"] == str(
        data_root
    )


def test_create_new_user_declined(tmp_path, answers, monkeypatch):
    """Answering no leaves the users.yaml untouched."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()

    answers.set(str(data_root), "n")
    create_new_user("users.yaml", local=None)

    assert (tmp_path / "users.yaml").read_text(encoding="utf-8") == ""


def test_create_new_user_retries_on_invalid_data_root(tmp_path, answers, monkeypatch):
    """A path that is not a directory is rejected and asked again."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()

    answers.set(str(tmp_path / "does_not_exist"), str(data_root), "")
    create_new_user("users.yaml", local=None)

    assert not answers.queue
    assert _to_read_yaml_file("users.yaml")[USERNAME][HOSTNAME]["data_root"] == str(data_root)


def test_create_new_user_adds_missing_hostname(tmp_path, answers, monkeypatch):
    """A known user on a new machine gets an extra hostname entry, keeping the old one."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_yaml(
        tmp_path / "users.yaml",
        {USERNAME: {"some-other-machine": {"data_root": "/other/data"}}},
    )

    answers.set(str(data_root), "")
    create_new_user("users.yaml", local=None)

    config = _to_read_yaml_file("users.yaml")
    assert config[USERNAME][HOSTNAME]["data_root"] == str(data_root)
    assert config[USERNAME]["some-other-machine"]["data_root"] == "/other/data"


def test_create_new_user_adds_missing_remote_data_root(tmp_path, answers, monkeypatch):
    """A missing ``remote`` data root is added next to the existing ``local`` one."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_yaml(
        tmp_path / "users.yaml",
        {USERNAME: {HOSTNAME: {"data_root": {"local": "/local/data"}}}},
    )

    answers.set(str(data_root), "")
    create_new_user("users.yaml", local=False)

    config = _to_read_yaml_file("users.yaml")
    assert config[USERNAME][HOSTNAME]["data_root"] == {
        "local": "/local/data",
        "remote": str(data_root),
    }
    assert set_data_paths("users.yaml", local=False).data_root == data_root


def test_create_new_user_adds_missing_local_data_root_at_top_level(tmp_path, answers, monkeypatch):
    """The local/remote entry is also found for a userless, machineless users.yaml."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_yaml(tmp_path / "users.yaml", {"data_root": {"remote": "/remote/data"}})

    answers.set(str(data_root), "")
    create_new_user("users.yaml", local=True)

    assert _to_read_yaml_file("users.yaml")["data_root"] == {
        "remote": "/remote/data",
        "local": str(data_root),
    }


def test_create_new_user_declined_local_remote_update(tmp_path, answers, monkeypatch):
    """Declining the local/remote update leaves the users.yaml untouched."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_yaml(tmp_path / "users.yaml", {"data_root": {"remote": "/remote/data"}})
    before = (tmp_path / "users.yaml").read_text(encoding="utf-8")

    answers.set(str(data_root), "no")
    create_new_user("users.yaml", local=True)

    assert (tmp_path / "users.yaml").read_text(encoding="utf-8") == before


def test_create_new_user_declined_hostname_update(tmp_path, answers, monkeypatch):
    """Declining the hostname update leaves the users.yaml untouched."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_yaml(
        tmp_path / "users.yaml",
        {USERNAME: {"some-other-machine": {"data_root": "/other/data"}}},
    )
    before = (tmp_path / "users.yaml").read_text(encoding="utf-8")

    answers.set(str(data_root), "no")
    create_new_user("users.yaml", local=None)

    assert (tmp_path / "users.yaml").read_text(encoding="utf-8") == before


def test_create_new_user_existing_profile_is_left_alone(tmp_path, answers, monkeypatch):
    """Nothing is asked or changed when the current user already has a data root."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_yaml(tmp_path / "users.yaml", {USERNAME: {HOSTNAME: {"data_root": str(data_root)}}})
    before = (tmp_path / "users.yaml").read_text(encoding="utf-8")

    data_paths = create_new_user("users.yaml", local=None)

    assert not answers.prompts, "the user should not have been prompted"
    assert (tmp_path / "users.yaml").read_text(encoding="utf-8") == before
    assert data_paths.data_root == data_root


if __name__ == "__main__":
    pytest.main(["-v", __file__])
