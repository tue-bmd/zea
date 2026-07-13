"""Tests for ``zea.data.publish``: the HF publishing / migration flow.

The Hub calls are stubbed — these tests check what *would* be uploaded (resaved with the
current codec and chunking) and that the virtual reference is pinned to the data commit.
"""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from zea.data.file import File
from zea.data.spec import DEFAULT_CHUNK_AXES

from . import generate_dummy_scan

pytest.importorskip("virtualizarr", reason="needs the 'zea[virtual]' extra")

from zea.data import publish as publish_module  # noqa: E402
from zea.data.publish import publish_dataset  # noqa: E402

# h5py reports filters by name for the ones it knows, by id for plugins.
BLOSC_FILTER = "32001"
LZF_FILTER = "lzf"
N_TX, N_AX, N_EL = 3, 64, 16


@pytest.fixture
def legacy_dir(tmp_path):
    """A dataset in the pre-0.1.3 layout: lzf, chunked per (frame, transmit)."""
    source = tmp_path / "legacy"
    source.mkdir()
    for i in range(2):
        File.create(
            source / f"file_{i}.hdf5",
            data={"raw_data": np.zeros((2, N_TX, N_AX, N_EL, 1), dtype=np.float32)},
            scan=generate_dummy_scan(n_tx=N_TX, n_el=N_EL),
            probe={"name": "generic", "probe_geometry": np.zeros((N_EL, 3), dtype=np.float32)},
            description="legacy test file",
            compression="lzf",
            chunk_axes=("n_frames", "n_tx"),
            overwrite=True,
            ignore_warnings=True,
        )
    return source


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Stub the Hub: record the uploads, and fake the reference build against it."""
    calls = SimpleNamespace(repos=[], uploads=[], virtualized=[])

    def create_repo(repo_id, **kwargs):
        calls.repos.append((repo_id, kwargs))

    def upload_folder(*, repo_id, folder_path, **kwargs):
        # Record what is in the folder now: publish uploads from a temporary directory
        # that no longer exists once it returns.
        contents = sorted(path.name for path in Path(folder_path).rglob("*") if path.is_file())
        calls.uploads.append(
            {"repo_id": repo_id, "folder_path": folder_path, "contents": contents, **kwargs}
        )
        return SimpleNamespace(oid=f"commit{len(calls.uploads)}" + "0" * 34)

    def build_virtual_reference(paths, output_path, revision=None, **kwargs):
        calls.virtualized.append({"paths": paths, "revision": revision})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")
        (output_path.parent / "params.json").write_text("{}")
        return output_path

    monkeypatch.setattr("huggingface_hub.create_repo", create_repo)
    monkeypatch.setattr("huggingface_hub.upload_folder", upload_folder)
    monkeypatch.setattr(publish_module, "build_virtual_reference", build_virtual_reference)
    return calls


def _filters(path):
    with h5py.File(path) as file:
        dataset = file["tracks/track_0/data/raw_data"]
        return dataset.chunks, set(dataset._filters)


def test_publish_resaves_uploads_and_virtualizes(legacy_dir, hub, tmp_path):
    workdir = tmp_path / "resaved"
    result = publish_dataset(legacy_dir, "zeahub/test-dataset", workdir=workdir)

    # 1. the files are rewritten with the current codec and chunking (lzf cannot be
    #    virtualized, and per-(frame, tx) chunks make for many tiny range requests)
    resaved = sorted(workdir.rglob("*.hdf5"))
    assert len(resaved) == 2
    for path in resaved:
        chunks, filters = _filters(path)
        assert chunks == (1, N_TX, N_AX, N_EL, 1)  # one frame per chunk
        assert BLOSC_FILTER in filters and LZF_FILTER not in filters
    assert DEFAULT_CHUNK_AXES == ("n_frames",)  # what the assertion above encodes

    # 2. data first, then the reference and its parameter sidecar under virtual/
    data_upload, virtual_upload = hub.uploads
    assert data_upload["repo_id"] == "zeahub/test-dataset"
    assert data_upload["folder_path"] == str(workdir)
    assert virtual_upload["path_in_repo"] == "virtual"
    assert virtual_upload["contents"] == ["index.json", "params.json"]

    # 3. the reference is built against the *published* files, pinned to their commit,
    #    so it cannot drift from the data it describes
    assert hub.virtualized == [
        {"paths": "hf://zeahub/test-dataset", "revision": result["data_commit"]}
    ]
    assert result["data_commit"] != result["virtual_commit"]
    assert result["n_files"] == 2


def test_publish_creates_repo_with_visibility(legacy_dir, hub):
    publish_dataset(legacy_dir, "zeahub/test-dataset", private=True, branch="staging")

    repo_id, kwargs = hub.repos[0]
    assert repo_id == "zeahub/test-dataset"
    assert kwargs["private"] is True and kwargs["exist_ok"] is True
    assert all(upload["revision"] == "staging" for upload in hub.uploads)


def test_publish_without_resave_uploads_as_is(legacy_dir, hub):
    """--no-resave skips the rewrite: the files are uploaded exactly as they are."""
    result = publish_dataset(legacy_dir, "zeahub/test-dataset", resave=False)

    assert hub.uploads[0]["folder_path"] == str(legacy_dir)
    assert result["n_files"] == 2
    # untouched: still lzf (which is why --no-resave is only for already-current files)
    assert LZF_FILTER in _filters(legacy_dir / "file_0.hdf5")[1]


def test_publish_missing_input(tmp_path, hub):
    with pytest.raises(FileNotFoundError):
        publish_dataset(tmp_path / "nope", "zeahub/test-dataset")
    assert not hub.uploads
