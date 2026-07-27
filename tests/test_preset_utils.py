"""Tests for the preset utilities shared by zea models and datasets.

Covers the Hugging Face plumbing in ``zea.internal.preset_utils`` (used by both the
model and the data stack) and the Keras preset loading/saving in
``zea.models.preset_utils``. Everything here runs offline: hub calls are faked.
"""

import json
import logging
import threading
import time
from pathlib import Path

import httpx
import keras
import numpy as np
import pytest
from huggingface_hub import RepoFile
from huggingface_hub.utils import (
    EntryNotFoundError,
    HFValidationError,
    RepositoryNotFoundError,
)

import zea
from zea import log
from zea.internal import preset_utils as ipu
from zea.internal.registry import model_registry
from zea.models import preset_utils as mpu
from zea.models.base import BaseModel, deserialize_zea_object
from zea.models.dense import DenseNet

REPO_ID = "zeahub/pytest-preset"


def _repo_not_found(message="404 Client Error"):
    """A ``RepositoryNotFoundError`` as huggingface_hub raises it (needs a response)."""
    request = httpx.Request("GET", "https://huggingface.co")
    return RepositoryNotFoundError(message, response=httpx.Response(404, request=request))


@pytest.fixture(autouse=True)
def _clear_hf_caches():
    """Keep the memoized listings/downloads from leaking between tests."""
    ipu._hf_clear_caches()
    yield
    ipu._hf_clear_caches()


# --------------------------------------------------------------------------------------
# _hf_call: login-and-retry
# --------------------------------------------------------------------------------------


def test_hf_call_returns_without_login(monkeypatch):
    """A successful call does not attempt to log in."""
    logins = []
    monkeypatch.setattr(ipu, "_hf_login", lambda: logins.append(1))

    assert ipu._hf_call(lambda x: x + 1, 1) == 2
    assert logins == []


def test_hf_call_retries_once_after_login(monkeypatch):
    """A 404-style failure is retried once, after a login attempt."""
    logins = []
    monkeypatch.setattr(ipu, "_hf_login", lambda: logins.append(1))
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise _repo_not_found()
        return "ok"

    assert ipu._hf_call(flaky) == "ok"
    assert len(calls) == 2
    assert len(logins) == 1


def test_hf_call_reraises_when_login_does_not_help(monkeypatch):
    """Without a usable token the retry raises the original error type."""
    monkeypatch.setattr(ipu, "_hf_login", lambda: None)

    def always_fails():
        raise _repo_not_found()

    with pytest.raises(RepositoryNotFoundError):
        ipu._hf_call(always_fails)


def test_hf_call_does_not_retry_other_errors(monkeypatch):
    """Errors a login cannot fix are propagated without a second attempt."""
    logins = []
    monkeypatch.setattr(ipu, "_hf_login", lambda: logins.append(1))
    calls = []

    def boom():
        calls.append(1)
        raise OSError("connection reset")

    with pytest.raises(OSError):
        ipu._hf_call(boom)
    assert len(calls) == 1
    assert logins == []


def test_hf_call_download_retry_set_excludes_missing_entries(monkeypatch):
    """`check_file_exists` must fail fast: a missing file is not retried."""
    logins = []
    monkeypatch.setattr(ipu, "_hf_login", lambda: logins.append(1))
    calls = []

    def missing():
        calls.append(1)
        raise EntryNotFoundError("404 Client Error")

    with pytest.raises(EntryNotFoundError):
        ipu._hf_call(missing, retry_on=ipu._HF_DOWNLOAD_RETRY_ERRORS)
    assert len(calls) == 1
    assert logins == []


# --------------------------------------------------------------------------------------
# _hf_login
# --------------------------------------------------------------------------------------


def test_hf_login_without_token_is_noop(monkeypatch):
    """No token in the environment means no (potentially interactive) login."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(ipu, "login", lambda **kwargs: calls.append(kwargs))

    ipu._hf_login()
    assert calls == []


@pytest.mark.parametrize("env_var", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_hf_login_uses_token_from_env(monkeypatch, env_var):
    """Either token variable is picked up, and login is skipped when already done."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv(env_var, "hf_dummy_token")
    calls = []
    monkeypatch.setattr(ipu, "login", lambda **kwargs: calls.append(kwargs))

    ipu._hf_login()
    assert calls == [{"token": "hf_dummy_token", "skip_if_logged_in": True}]


# --------------------------------------------------------------------------------------
# _TTLCache / _cache_key
# --------------------------------------------------------------------------------------


def test_ttl_cache_returns_cached_value():
    cache = ipu._TTLCache(ttl=60)
    calls = []

    def compute():
        calls.append(1)
        return "value"

    assert cache.get_or_call("k", compute) == "value"
    assert cache.get_or_call("k", compute) == "value"
    assert len(calls) == 1


def test_ttl_cache_expires():
    cache = ipu._TTLCache(ttl=0.02)
    calls = []
    cache.get_or_call("k", lambda: calls.append(1))
    time.sleep(0.05)
    cache.get_or_call("k", lambda: calls.append(1))
    assert len(calls) == 2


def test_ttl_cache_disabled_with_zero_ttl():
    cache = ipu._TTLCache(ttl=0)
    calls = []
    for _ in range(2):
        cache.get_or_call("k", lambda: calls.append(1))
    assert len(calls) == 2


def test_ttl_cache_bypassed_for_unhashable_key():
    cache = ipu._TTLCache(ttl=60)
    calls = []
    for _ in range(2):
        cache.get_or_call(None, lambda: calls.append(1))
    assert len(calls) == 2


def test_ttl_cache_revalidates_entry():
    """A cached value the predicate rejects (e.g. a deleted file) is recomputed."""
    cache = ipu._TTLCache(ttl=60)
    calls = []

    def compute():
        calls.append(1)
        return "gone"

    cache.get_or_call("k", compute, valid=lambda _: False)
    cache.get_or_call("k", compute, valid=lambda _: False)
    assert len(calls) == 2


def test_ttl_cache_clear():
    cache = ipu._TTLCache(ttl=60)
    calls = []
    cache.get_or_call("k", lambda: calls.append(1))
    cache.clear()
    cache.get_or_call("k", lambda: calls.append(1))
    assert len(calls) == 2


def test_cache_key_is_none_for_unhashable_arguments():
    assert ipu._cache_key("repo", "dataset") is not None
    assert ipu._cache_key("repo", allow_patterns=["*.h5"]) is None


# --------------------------------------------------------------------------------------
# Repository listings
# --------------------------------------------------------------------------------------


@pytest.fixture
def fake_tree(monkeypatch):
    """Patch ``list_repo_tree`` with a fixed repo layout and count the calls."""
    entries = [
        RepoFile(path="val/a.hdf5", size=10, oid="1"),
        RepoFile(path="val/b.h5", size=20, oid="2"),
        RepoFile(path="val/notes.txt", size=1, oid="3"),
        RepoFile(path="train/c.hdf5", size=30, oid="4"),
    ]
    calls = []

    def fake_list_repo_tree(repo_id, recursive=True, repo_type="dataset", **kwargs):
        calls.append((repo_id, repo_type, tuple(sorted(kwargs.items()))))
        return entries

    monkeypatch.setattr(ipu, "list_repo_tree", fake_list_repo_tree)
    return calls


def test_repo_listing_is_memoized(fake_tree):
    """Repeated listings of the same repo cost a single hub round trip."""
    assert ipu._hf_list_files(REPO_ID) == [
        "val/a.hdf5",
        "val/b.h5",
        "val/notes.txt",
        "train/c.hdf5",
    ]
    # Both the name-only and the size-carrying view share one listing.
    ipu._hf_list_files(REPO_ID)
    ipu._hf_list_h5_files(f"hf://{REPO_ID}")
    assert len(fake_tree) == 1


def test_repo_listing_keyed_on_revision_and_repo_type(fake_tree):
    """Different revisions/repo types are cached separately."""
    ipu._hf_list_files(REPO_ID)
    ipu._hf_list_files(REPO_ID, revision="v1")
    ipu._hf_list_files(REPO_ID, repo_type="model")
    assert len(fake_tree) == 3


def test_list_h5_files_filters_extensions(fake_tree):
    """Only .h5/.hdf5 files are returned, with their sizes."""
    assert ipu._hf_list_h5_files(f"hf://{REPO_ID}") == [
        ("val/a.hdf5", 10),
        ("val/b.h5", 20),
        ("train/c.hdf5", 30),
    ]


def test_list_h5_files_filters_subdirectory(fake_tree):
    assert ipu._hf_list_h5_files(f"hf://{REPO_ID}/val") == [
        ("val/a.hdf5", 10),
        ("val/b.h5", 20),
    ]


def test_list_h5_files_single_file(fake_tree):
    assert ipu._hf_list_h5_files(f"hf://{REPO_ID}/val/a.hdf5") == [("val/a.hdf5", 10)]


def test_list_h5_files_unknown_subdirectory_is_empty(fake_tree):
    assert ipu._hf_list_h5_files(f"hf://{REPO_ID}/nope") == []


# --------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------


@pytest.fixture
def fake_hub_download(monkeypatch, tmp_path):
    """Patch ``hf_hub_download`` to materialize files in ``tmp_path``."""
    calls = []

    def fake_download(*, repo_id, filename, cache_dir=None, repo_type=None, **kwargs):
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "cache_dir": cache_dir,
                "repo_type": repo_type,
                **kwargs,
            }
        )
        path = tmp_path / repo_id.replace("/", "--") / "snapshots" / "abc123" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data")
        return str(path)

    monkeypatch.setattr(ipu, "hf_hub_download", fake_download)
    return calls


def test_hf_download_is_memoized(fake_hub_download):
    first = ipu._hf_download(REPO_ID, "config.json")
    second = ipu._hf_download(REPO_ID, "config.json")
    assert first == second
    assert len(fake_hub_download) == 1


def test_hf_download_repeats_when_cached_file_disappeared(fake_hub_download):
    """A memoized path that no longer exists is downloaded again."""
    path = ipu._hf_download(REPO_ID, "config.json")
    Path(path).unlink()
    assert ipu._hf_download(REPO_ID, "config.json") == path
    assert len(fake_hub_download) == 2


@pytest.mark.parametrize(
    "repo_type, expected_dir",
    [("dataset", ipu.HF_DATASETS_DIR), ("model", ipu.HF_MODELS_DIR)],
)
def test_hf_download_default_cache_dir_per_repo_type(fake_hub_download, repo_type, expected_dir):
    """Models and datasets land in their own cache directory by default."""
    ipu._hf_download(REPO_ID, "config.json", repo_type=repo_type)
    assert fake_hub_download[0]["cache_dir"] == expected_dir


def test_download_files_in_path_filters_and_keeps_order(monkeypatch):
    downloaded = []

    def fake_download(repo_id, filename, cache_dir=None, repo_type="dataset", **kwargs):
        downloaded.append(filename)
        return f"/local/{filename}"

    monkeypatch.setattr(ipu, "_hf_download", fake_download)

    files = ["val/a.hdf5", "train/c.hdf5", "val/b.h5"]
    result = ipu._download_files_in_path(REPO_ID, files, "val/")

    assert result == ["/local/val/a.hdf5", "/local/val/b.h5"]
    assert sorted(downloaded) == ["val/a.hdf5", "val/b.h5"]


def test_download_files_in_path_runs_concurrently(monkeypatch):
    """Multiple files are fetched in parallel rather than one after another."""
    lock = threading.Lock()
    state = {"running": 0, "peak": 0}

    def fake_download(repo_id, filename, cache_dir=None, repo_type="dataset", **kwargs):
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        time.sleep(0.05)
        with lock:
            state["running"] -= 1
        return f"/local/{filename}"

    monkeypatch.setattr(ipu, "_hf_download", fake_download)

    files = [f"val/{i}.hdf5" for i in range(4)]
    result = ipu._download_files_in_path(REPO_ID, files)

    assert result == [f"/local/val/{i}.hdf5" for i in range(4)]
    assert state["peak"] > 1


# --------------------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------------------


@pytest.fixture
def fake_resolve(monkeypatch, tmp_path):
    """Patch listing + download so ``_hf_resolve_path`` works against tmp_path."""
    files = ["val/a.hdf5", "val/b.hdf5", "train/c.hdf5"]
    snapshot = tmp_path / f"datasets--{REPO_ID.replace('/', '--')}" / "snapshots" / "abc123"

    monkeypatch.setattr(ipu, "_hf_list_files", lambda repo_id, **kwargs: files)

    def fake_download(repo_id, filename, cache_dir=None, repo_type="dataset", **kwargs):
        path = snapshot / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data")
        return str(path)

    monkeypatch.setattr(ipu, "_hf_download", fake_download)
    return snapshot


def test_resolve_path_directory(fake_resolve):
    result = ipu._hf_resolve_path(f"hf://{REPO_ID}/val")
    assert result == fake_resolve / "val"
    assert sorted(p.name for p in result.iterdir()) == ["a.hdf5", "b.hdf5"]


def test_resolve_path_single_file(fake_resolve):
    result = ipu._hf_resolve_path(f"hf://{REPO_ID}/val/a.hdf5")
    assert result == fake_resolve / "val" / "a.hdf5"


def test_resolve_path_whole_repo(fake_resolve):
    assert ipu._hf_resolve_path(f"hf://{REPO_ID}") == fake_resolve


def test_resolve_path_missing_subpath(fake_resolve):
    with pytest.raises(FileNotFoundError, match="not found in"):
        ipu._hf_resolve_path(f"hf://{REPO_ID}/nope.hdf5")


def test_resolve_path_empty_repo(monkeypatch):
    monkeypatch.setattr(ipu, "_hf_list_files", lambda repo_id, **kwargs: [])
    with pytest.raises(FileNotFoundError, match="No files found in repository"):
        ipu._hf_resolve_path(f"hf://{REPO_ID}")


def test_snapshot_dir_lookup_without_snapshots_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not find snapshot directory"):
        ipu._get_snapshot_dir_from_downloaded_file(tmp_path / "a.hdf5")


# --------------------------------------------------------------------------------------
# repo_type prefixes and stream URLs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repo_type, prefix", [("dataset", "datasets/"), ("model", ""), ("space", "spaces/")]
)
def test_repo_type_prefix(repo_type, prefix):
    assert ipu._hf_repo_type_prefix(repo_type) == prefix


def test_repo_type_prefix_rejects_unknown():
    with pytest.raises(ValueError, match="repo_type"):
        ipu._hf_repo_type_prefix("banana")


def test_stream_url():
    url = ipu._hf_stream_url(f"hf://{REPO_ID}/val/a.hdf5")
    assert url == f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/val/a.hdf5"


def test_stream_url_with_revision_and_repo_type():
    url = ipu._hf_stream_url(f"hf://{REPO_ID}/model.weights.h5", revision="v1", repo_type="model")
    assert url == f"https://huggingface.co/{REPO_ID}/resolve/v1/model.weights.h5"


def test_stream_url_requires_file():
    with pytest.raises(ValueError, match="single file"):
        ipu._hf_stream_url(f"hf://{REPO_ID}")


def test_stream_url_rejects_bad_repo_type():
    with pytest.raises(ValueError, match="repo_type"):
        ipu._hf_stream_url(f"hf://{REPO_ID}/a.hdf5", repo_type="banana")


def test_stream_open_retries_after_login(monkeypatch):
    """A stream open that 404s anonymously is retried once after logging in."""
    logins = []
    monkeypatch.setattr(ipu, "_hf_login", lambda: logins.append(1))
    attempts = []

    class FakeFS:
        def open(self, path, mode, **kwargs):
            attempts.append(path)
            if len(attempts) == 1:
                raise _repo_not_found()
            return "FILEOBJ"

    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)

    assert ipu._hf_stream_open(f"hf://{REPO_ID}/val/a.hdf5") == "FILEOBJ"
    assert len(attempts) == 2
    assert len(logins) == 1


# --------------------------------------------------------------------------------------
# zea.models.preset_utils — get_file
# --------------------------------------------------------------------------------------


@pytest.fixture
def local_preset(tmp_path):
    """A minimal local preset directory."""
    (tmp_path / "config.json").write_text(json.dumps({"registered_name": "Functional"}))
    return tmp_path


def test_get_file_requires_string_preset():
    with pytest.raises(ValueError, match="must be a string"):
        mpu.get_file(Path("some/dir"), "config.json")


def test_get_file_local(local_preset):
    assert mpu.get_file(str(local_preset), "config.json") == str(local_preset / "config.json")


def test_get_file_local_missing(local_preset):
    with pytest.raises(FileNotFoundError, match="doesn't exist in preset directory"):
        mpu.get_file(str(local_preset), "missing.json")


def test_get_file_local_rejects_escaping_path(local_preset):
    """A path that walks out of the preset directory is refused."""
    with pytest.raises(ValueError, match="escapes the preset directory"):
        mpu.get_file(str(local_preset), "../secret.json")


def test_get_file_unknown_identifier():
    with pytest.raises(ValueError, match="Unknown preset identifier"):
        mpu.get_file("not-a-preset", "config.json")


@pytest.fixture
def fake_model_download(monkeypatch):
    """Patch the shared HF download used by ``zea.models.preset_utils``."""
    calls = []

    def fake_download(repo_id, filename, cache_dir=None, repo_type="dataset", **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, "repo_type": repo_type})
        return f"/local/{repo_id}/{filename}"

    monkeypatch.setattr(mpu, "_hf_download", fake_download)
    return calls


def test_get_file_hf(fake_model_download):
    """An ``hf://`` preset downloads from the model repo cache."""
    assert mpu.get_file("hf://zeahub/taesdxl", "config.json") == "/local/zeahub/taesdxl/config.json"
    assert fake_model_download == [
        {"repo_id": "zeahub/taesdxl", "filename": "config.json", "repo_type": "model"}
    ]


def test_get_file_hf_with_subpath(fake_model_download):
    """A subdirectory in the handle prefixes the requested file."""
    mpu.get_file("hf://zeahub/models/taesdxl_encoder", "config.json")
    assert fake_model_download[0]["filename"] == "taesdxl_encoder/config.json"


def test_get_file_hf_validation_error(monkeypatch):
    def boom(*args, **kwargs):
        raise HFValidationError("bad handle")

    monkeypatch.setattr(mpu, "_hf_download", boom)
    with pytest.raises(ValueError, match="Unexpected Hugging Face preset"):
        mpu.get_file("hf://taesdxl", "config.json")


def test_get_file_hf_missing_entry(monkeypatch):
    def boom(*args, **kwargs):
        raise EntryNotFoundError("404 Client Error")

    monkeypatch.setattr(mpu, "_hf_download", boom)
    with pytest.raises(FileNotFoundError, match="doesn't exist in preset directory"):
        mpu.get_file("hf://zeahub/taesdxl", "config.json")


# --------------------------------------------------------------------------------------
# Built-in preset registry
# --------------------------------------------------------------------------------------


@pytest.fixture
def registered_presets():
    """Register throwaway presets and clean up the global registry afterwards."""

    class _Dummy:
        pass

    presets = {
        "_pytest_hf": {"hf_handle": "hf://zeahub/_pytest", "metadata": {}},
        "_pytest_local": {"path": "/tmp/_pytest_preset", "metadata": {}},
    }
    mpu.register_presets(presets, _Dummy)
    yield _Dummy, presets
    for name in presets:
        mpu.BUILTIN_PRESETS.pop(name, None)
    mpu.BUILTIN_PRESETS_FOR_MODEL.pop(_Dummy, None)


def test_builtin_presets_are_registered_per_class(registered_presets):
    cls, presets = registered_presets
    assert mpu.builtin_presets(cls) == presets
    assert mpu.builtin_presets(object) == {}


def test_builtin_preset_resolves_to_hf_handle(registered_presets, fake_model_download):
    mpu.get_file("_pytest_hf", "config.json")
    assert fake_model_download[0]["repo_id"] == "zeahub/_pytest"


def test_builtin_preset_resolves_to_path(registered_presets):
    """A preset registered with a `path` resolves to that (here missing) directory."""
    with pytest.raises(ValueError, match="Unknown preset identifier"):
        mpu.get_file("_pytest_local", "config.json")


def test_model_presets_property_lists_builtins():
    assert "dense" in DenseNet.presets or DenseNet.presets == {}


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------


def test_load_json(local_preset):
    assert mpu.load_json(str(local_preset)) == {"registered_name": "Functional"}


def test_check_file_exists(local_preset):
    assert mpu.check_file_exists(str(local_preset), "config.json") is True
    assert mpu.check_file_exists(str(local_preset), "nope.json") is False


def test_assert_file_exists(local_preset):
    mpu._assert_file_exists(str(local_preset), "config.json")
    with pytest.raises(ValueError, match="has no nope.json"):
        mpu._assert_file_exists(str(local_preset), "nope.json")


def test_set_dtype_in_config_without_dtype_is_identity():
    config = {"config": {}}
    assert mpu.set_dtype_in_config(config) is config


def test_set_dtype_in_config_forwards_dtype():
    config = mpu.set_dtype_in_config({"config": {}}, "float16")
    assert config["config"]["dtype"] == "float16"


def test_set_dtype_in_config_updates_policy_map():
    config = {
        "config": {
            "dtype": {
                "class_name": "DTypePolicyMap",
                "config": {
                    "default_policy": "float32",
                    "policy_map": {"layer": {"config": {"source_name": "float32"}}},
                },
            }
        }
    }
    updated = mpu.set_dtype_in_config(config, "bfloat16")
    policy_map_config = updated["config"]["dtype"]["config"]
    assert policy_map_config["default_policy"] == "bfloat16"
    assert policy_map_config["policy_map"]["layer"]["config"]["source_name"] == "bfloat16"


@pytest.mark.parametrize("registered_name", ["Functional", "Sequential"])
def test_check_config_class_functional(registered_name):
    assert mpu.check_config_class({"registered_name": registered_name}) is keras.Model


def test_check_config_class_zea_model():
    assert mpu.check_config_class({"registered_name": "DenseNet"}) is DenseNet


def test_check_config_class_unknown():
    with pytest.raises(ValueError, match="not found in `zea` registry"):
        mpu.check_config_class({"registered_name": "NotARegisteredModel"})


def test_keras_to_zea_registry():
    assert model_registry[mpu.keras_to_zea_registry("DenseNet", model_registry)] is DenseNet


# --------------------------------------------------------------------------------------
# jax_memory_cleanup
# --------------------------------------------------------------------------------------


class _FakeValue:
    def __init__(self, sharding=None):
        self.sharding = sharding
        self.deleted = False

    def delete(self):
        self.deleted = True


class _FakeWeight:
    def __init__(self, value):
        self._value = value


class _FakeLayer:
    def __init__(self, weights):
        self.weights = weights


def test_jax_memory_cleanup_deletes_unsharded_values(monkeypatch):
    """Sharded arrays are left alone; deleting them breaks distributed setups."""
    monkeypatch.setattr(keras.config, "backend", lambda: "jax")
    plain, sharded = _FakeValue(), _FakeValue(sharding=object())
    layer = _FakeLayer([_FakeWeight(plain), _FakeWeight(sharded), _FakeWeight(None)])

    mpu.jax_memory_cleanup(layer)

    assert plain.deleted is True
    assert sharded.deleted is False


def test_jax_memory_cleanup_noop_on_other_backends(monkeypatch):
    monkeypatch.setattr(keras.config, "backend", lambda: "torch")
    plain = _FakeValue()
    mpu.jax_memory_cleanup(_FakeLayer([_FakeWeight(plain)]))
    assert plain.deleted is False


# --------------------------------------------------------------------------------------
# Loader / saver
# --------------------------------------------------------------------------------------


def test_get_preset_loader_returns_keras_loader(local_preset):
    loader = mpu.get_preset_loader(str(local_preset))
    assert isinstance(loader, mpu.KerasPresetLoader)
    assert loader.check_model_class() is keras.Model


def test_get_preset_loader_missing_config(tmp_path):
    with pytest.raises(ValueError, match="has no config.json"):
        mpu.get_preset_loader(str(tmp_path))


def test_get_preset_loader_unrecognized_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    with pytest.raises(ValueError, match="Unrecognized format"):
        mpu.get_preset_loader(str(tmp_path))


def test_loader_get_file(local_preset):
    loader = mpu.get_preset_loader(str(local_preset))
    assert loader.get_file("config.json") == str(local_preset / "config.json")


def test_preset_saver_creates_directory(tmp_path):
    preset_dir = tmp_path / "nested" / "preset"
    mpu.get_preset_saver(str(preset_dir))
    assert preset_dir.is_dir()


def test_recursive_pop_removes_nested_keys(tmp_path):
    saver = mpu.KerasPresetSaver(str(tmp_path))
    config = {"build_config": 1, "config": {"build_config": 2, "inner": {"build_config": 3}}}
    saver._recursive_pop(config, "build_config")
    assert config == {"config": {"inner": {}}}


@model_registry(name="_preset_utils_test_model")
class _TinyPresetModel(BaseModel):
    """Small registered model used for preset save/load round trips."""

    def __init__(self, units=3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = keras.layers.Dense(units)

    @property
    def image_shape(self):
        return (None, 5)

    def build(self, input_shape):
        self.dense.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def test_save_and_load_preset_round_trip(tmp_path):
    """A saved preset reloads with identical weights."""
    model = _TinyPresetModel(units=3)
    inputs = np.random.rand(2, 5).astype("float32")
    expected = np.array(model(inputs))

    model.save_to_preset(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "config.json",
        "metadata.json",
        "model.weights.h5",
    ]

    reloaded = _TinyPresetModel.from_preset(str(tmp_path))
    assert np.allclose(expected, np.array(reloaded(inputs)))


def test_saved_config_drops_build_and_compile_config(tmp_path):
    model = _TinyPresetModel(units=3)
    model(np.zeros((1, 5), dtype="float32"))
    model.save_to_preset(str(tmp_path))

    config = json.loads((tmp_path / "config.json").read_text())
    assert "build_config" not in config
    assert "compile_config" not in config
    assert config["registered_name"] == "_TinyPresetModel"


def test_saved_metadata(tmp_path):
    model = _TinyPresetModel(units=3)
    model(np.zeros((1, 5), dtype="float32"))
    model.save_to_preset(str(tmp_path))

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["zea_version"] == zea.__version__
    assert metadata["parameter_count"] == model.count_params()
    assert metadata["keras_version"] == keras.version()
    assert metadata["date_saved"]


def test_load_preset_without_weights(tmp_path):
    """A model that cannot be rebuilt still loads when weights are not requested."""
    model = DenseNet(input_dim=4, widths=[8], output_dim=2)
    model(np.zeros((1, 4), dtype="float32"))
    model.save_to_preset(str(tmp_path))

    reloaded = DenseNet.from_preset(str(tmp_path), load_weights=False)
    assert isinstance(reloaded, DenseNet)


def test_load_preset_unbuildable_model_raises(tmp_path):
    """Without a build hint, loading weights reports how to fix the preset."""
    model = DenseNet(input_dim=4, widths=[8], output_dim=2)
    model(np.zeros((1, 4), dtype="float32"))
    model.save_to_preset(str(tmp_path))

    with pytest.raises(ValueError, match="Model could not be built"):
        DenseNet.from_preset(str(tmp_path))


def test_load_model_builds_from_input_shape(monkeypatch, local_preset):
    """An unbuilt model is built from its `input_shape` before weights are loaded."""
    built_with, loaded_from = [], []

    class _Unbuilt:
        weights = []
        built = False
        input_shape = (None, 5)

        def build(self, input_shape):
            built_with.append(input_shape)

        def load_weights(self, path):
            loaded_from.append(path)

    (local_preset / mpu.MODEL_WEIGHTS_FILE).write_text("weights")
    monkeypatch.setattr(mpu, "load_serialized_object", lambda *a, **kw: _Unbuilt())

    loader = mpu.get_preset_loader(str(local_preset))
    loader.load_model(cls=object, load_weights=True)

    assert built_with == [(None, 5)]
    assert loaded_from == [str(local_preset / mpu.MODEL_WEIGHTS_FILE)]


# --------------------------------------------------------------------------------------
# Converters and preprocessors
# --------------------------------------------------------------------------------------


def test_get_model_kwargs_forwards_dtype_and_image_shape():
    loader = mpu.PresetLoader("preset", {})
    model_kwargs, kwargs = loader.get_model_kwargs(
        dtype="float16", image_shape=(4, 4), other="keep"
    )
    assert model_kwargs == {"dtype": "float16", "image_shape": (4, 4)}
    assert kwargs == {"other": "keep"}


def test_base_loader_load_model_is_abstract():
    with pytest.raises(NotImplementedError):
        mpu.PresetLoader("preset", {}).load_model(cls=object, load_weights=False)


def test_base_loader_load_preprocessor_uses_add_missing_kwargs():
    """The fallback builds the preprocessor from defaults filled in by the class."""

    class _Preprocessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def _add_missing_kwargs(cls, loader, kwargs):
            return {**kwargs, "added": True}

    preprocessor = mpu.PresetLoader("preset", {}).load_preprocessor(_Preprocessor, given=1)
    assert preprocessor.kwargs == {"given": 1, "added": True}


def test_image_converter_round_trip(tmp_path):
    """An image converter saved to a preset is loaded back with its config."""
    saver = mpu.get_preset_saver(str(tmp_path))
    saver.save_image_converter(keras.layers.Rescaling(scale=1 / 255))
    assert (tmp_path / mpu.IMAGE_CONVERTER_CONFIG_FILE).exists()

    loader = mpu.KerasPresetLoader(str(tmp_path), {})
    converter = loader.load_image_converter(keras.layers.Rescaling)
    assert isinstance(converter, keras.layers.Rescaling)
    assert converter.scale == 1 / 255


@model_registry(name="_preset_utils_test_preprocessor")
class _TinyPreprocessor(BaseModel):
    """Registered stand-in for a preprocessor with preset assets."""

    def __init__(self, factor=2, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.assets_from = None

    def load_preset_assets(self, preset):
        self.assets_from = preset

    def call(self, x):
        return x * self.factor

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config


def test_load_preprocessor_from_preset(tmp_path):
    """A `preprocessor.json` for the right class is deserialized and gets its assets."""
    saver = mpu.get_preset_saver(str(tmp_path))
    saver.save_preprocessor(_TinyPreprocessor(factor=7))

    loader = mpu.KerasPresetLoader(str(tmp_path), {})
    preprocessor = loader.load_preprocessor(_TinyPreprocessor)

    assert preprocessor.factor == 7
    assert preprocessor.assets_from == str(tmp_path)


def test_load_preprocessor_falls_back_without_config(tmp_path):
    """Without a `preprocessor.json` the loader falls back to class defaults."""

    class _Preprocessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def _add_missing_kwargs(cls, loader, kwargs):
            return kwargs

    loader = mpu.KerasPresetLoader(str(tmp_path), {})
    assert isinstance(loader.load_preprocessor(_Preprocessor), _Preprocessor)


def test_load_preprocessor_falls_back_for_other_class(tmp_path):
    """A `preprocessor.json` belonging to another class is ignored."""
    saver = mpu.get_preset_saver(str(tmp_path))
    saver.save_preprocessor(_TinyPreprocessor(factor=7))

    class _Other(_TinyPresetModel):
        @classmethod
        def _add_missing_kwargs(cls, loader, kwargs):
            return kwargs

    loader = mpu.KerasPresetLoader(str(tmp_path), {})
    assert isinstance(loader.load_preprocessor(_Other), _Other)


def test_save_preprocessor_uses_config_file_and_saves_sublayers(tmp_path):
    """A preprocessor's own `config_file` is honoured and its layers save their assets."""
    saved_to = []

    class _AssetLayer(keras.layers.Layer):
        def save_to_preset(self, preset_dir):
            saved_to.append(preset_dir)

    class _Preprocessor(keras.layers.Layer):
        config_file = "custom_preprocessor.json"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.asset_layer = _AssetLayer()

    mpu.get_preset_saver(str(tmp_path)).save_preprocessor(_Preprocessor())

    assert (tmp_path / "custom_preprocessor.json").exists()
    assert saved_to == [str(tmp_path)]


# --------------------------------------------------------------------------------------
# zea.models.base — the preset entry points
# --------------------------------------------------------------------------------------


@model_registry(name="_preset_utils_test_submodel")
class _SubTinyPresetModel(_TinyPresetModel):
    """Registered subclass, to exercise the class-mismatch branches of from_preset."""


@pytest.fixture
def attach_caplog(caplog):
    """Attach pytest's caplog handler to zea's (non-propagating) logger."""
    caplog.set_level(logging.DEBUG)
    log.logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        log.logger.removeHandler(caplog.handler)


@pytest.fixture
def tiny_preset(tmp_path):
    """A saved preset for `_TinyPresetModel`."""
    model = _TinyPresetModel(units=3)
    model(np.zeros((1, 5), dtype="float32"))
    model.save_to_preset(str(tmp_path))
    return str(tmp_path)


def test_presets_property_is_empty_for_unregistered_model():
    assert _TinyPresetModel.presets == {}


def test_from_preset_on_subclass_warns_and_returns_subclass(tiny_preset, attach_caplog):
    """Loading a parent's preset from a subclass returns the subclass, with a warning."""
    model = _SubTinyPresetModel.from_preset(tiny_preset)

    assert isinstance(model, _SubTinyPresetModel)
    assert "you are calling from a subclass" in attach_caplog.text


def test_from_preset_on_parent_class_warns(tmp_path, attach_caplog):
    """Loading a subclass' preset from the parent returns the parent, with a warning."""
    model = _SubTinyPresetModel(units=3)
    model(np.zeros((1, 5), dtype="float32"))
    model.save_to_preset(str(tmp_path))

    reloaded = _TinyPresetModel.from_preset(str(tmp_path))

    assert type(reloaded) is _TinyPresetModel
    assert "which is a subclass of the calling class" in attach_caplog.text


def test_from_preset_incompatible_class_raises(tiny_preset):
    with pytest.raises(ValueError, match="not compatible with the calling class"):
        _TinyPreprocessor.from_preset(tiny_preset)


def test_deserialize_zea_object_requires_from_config():
    """A class without `from_config()` cannot be reconstructed."""
    config = {"class_name": "_NoFromConfig", "config": {}}
    with pytest.raises(TypeError, match="missing a `from_config\\(\\)` method"):
        deserialize_zea_object(config, cls=type("_NoFromConfig", (), {}))


def test_deserialize_zea_object_reports_bad_config():
    """A config that does not match the constructor is reported with the config."""

    class _NeedsArgument(BaseModel):
        def __init__(self, required, **kwargs):
            super().__init__(**kwargs)
            self.required = required

    config = {"class_name": "_NeedsArgument", "config": {}}
    with pytest.raises(TypeError, match="could not be deserialized properly"):
        deserialize_zea_object(config, cls=_NeedsArgument)


def test_deserialize_zea_object_resolves_class_from_module():
    """Without an explicit class, the config's module/registered_name is imported."""
    config = {
        "class_name": "DenseNet",
        "registered_name": "DenseNet",
        "module": "zea.models.dense",
        "config": {"input_dim": 4, "widths": [8], "output_dim": 2},
    }
    assert isinstance(deserialize_zea_object(config), DenseNet)


def test_deserialize_zea_object_rejects_foreign_module():
    config = {"class_name": "OrderedDict", "module": "collections", "config": {}}
    with pytest.raises(TypeError, match="Could not locate class"):
        deserialize_zea_object(config)


def test_deserialize_zea_object_missing_zea_module():
    config = {"class_name": "Nope", "module": "zea.does_not_exist", "config": {}}
    with pytest.raises(TypeError, match="cannot be imported"):
        deserialize_zea_object(config)


def test_deserialize_zea_object_applies_build_and_compile_config():
    """`build_config`/`compile_config`/`shared_object_id` in a config are applied."""

    class _Recorder:
        built = False
        compiled = False

        @classmethod
        def from_config(cls, config):
            return cls()

        def build_from_config(self, config):
            self.build_config = config

        def compile_from_config(self, config):
            self.compile_config = config

    config = {
        "class_name": "_Recorder",
        "config": {},
        "build_config": {"input_shape": (None, 5)},
        "compile_config": {"optimizer": "adam"},
        "shared_object_id": 7,
    }
    instance = deserialize_zea_object(config, cls=_Recorder)

    assert instance.build_config == {"input_shape": (None, 5)}
    assert instance.built is True
    assert instance.compile_config == {"optimizer": "adam"}
    assert instance.compiled is True


def test_preset_with_build_config_loads_weights(tmp_path):
    """The documented way to make a preset loadable: a `build_config` in config.json."""
    model = _TinyPresetModel(units=3)
    inputs = np.random.rand(2, 5).astype("float32")
    expected = np.array(model(inputs))
    model.save_to_preset(str(tmp_path))

    # The saver strips build_config; presets add it back by hand (see the error
    # raised by KerasPresetLoader.load_model when a model cannot be built).
    config_path = tmp_path / mpu.CONFIG_FILE
    config = json.loads(config_path.read_text())
    config["build_config"] = {"input_shape": [None, 5]}
    config_path.write_text(json.dumps(config))

    reloaded = _TinyPresetModel.from_preset(str(tmp_path))
    assert reloaded.built is True
    assert np.allclose(expected, np.array(reloaded(inputs)))


# Sharded weights (mirrors keras-team/keras-hub#2218).


@pytest.mark.parametrize(
    "dtype, bits", [("float32", 32), ("bfloat16", 16), ("int8", 8), ("uint8", 8), ("bool", 1)]
)
def test_dtype_size_in_bits(dtype, bits):
    assert mpu._dtype_size_in_bits(dtype) == bits


def test_variables_size_in_bytes_counts_shared_variables_once():
    variable = keras.Variable(np.zeros((4, 8), dtype="float32"))
    assert mpu._variables_size_in_bytes([variable]) == 4 * 8 * 4
    assert mpu._variables_size_in_bytes([variable, variable]) == 4 * 8 * 4


def test_get_sharded_filenames_handles_list_values(tmp_path):
    """A weight can be spread over several shards, so values may be lists."""
    config_path = tmp_path / mpu.SHARDED_MODEL_WEIGHTS_CONFIG_FILE
    config_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model_00000.weights.h5",
                    "b": ["model_00000.weights.h5", "model_00001.weights.h5"],
                }
            }
        )
    )
    loader = mpu.KerasPresetLoader(str(tmp_path), {})
    assert loader._get_sharded_filenames(str(config_path)) == [
        "model_00000.weights.h5",
        "model_00001.weights.h5",
    ]


@pytest.mark.parametrize("max_shard_size", [10, None])
def test_small_model_is_saved_as_a_single_weights_file(tmp_path, max_shard_size):
    model = _TinyPresetModel(units=3)
    model(np.zeros((1, 5), dtype="float32"))
    model.save_to_preset(str(tmp_path), max_shard_size=max_shard_size)

    assert (tmp_path / mpu.MODEL_WEIGHTS_FILE).exists()
    assert not (tmp_path / mpu.SHARDED_MODEL_WEIGHTS_CONFIG_FILE).exists()


def test_sharded_save_and_load_round_trip(tmp_path, monkeypatch):
    """A model saved as shards reloads with identical weights."""
    model = _TinyPresetModel(units=3)
    inputs = np.random.rand(2, 5).astype("float32")
    expected = np.array(model(inputs))

    # 64 bytes: above the largest single variable, below the model total.
    model.save_to_preset(str(tmp_path), max_shard_size=64 / 1024**3)

    assert (tmp_path / mpu.SHARDED_MODEL_WEIGHTS_CONFIG_FILE).exists()
    assert not (tmp_path / mpu.MODEL_WEIGHTS_FILE).exists()
    shards = sorted(p.name for p in tmp_path.glob("*.weights.h5"))
    assert len(shards) > 1

    # Every shard has to be fetched, not just the index (they are separate
    # downloads for an `hf://` preset).
    requested = []
    real_get_file = mpu.get_file
    monkeypatch.setattr(
        mpu,
        "get_file",
        lambda preset, path: (requested.append(path), real_get_file(preset, path))[1],
    )

    reloaded = _TinyPresetModel.from_preset(str(tmp_path))

    assert np.allclose(expected, np.array(reloaded(inputs)))
    assert set(shards).issubset(requested)
