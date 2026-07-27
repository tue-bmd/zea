"""Minimal tests for the HFPath class in zea.tools.hf module."""

# Create the directory structure for testing
import tempfile
from pathlib import Path

import pytest

from zea.internal.preset_utils import (
    _download_files_in_path,
    _get_snapshot_dir_from_downloaded_file,
    _hf_parse_path,
    _hf_resolve_path,
    _hf_stream_open,
)
from zea.tools.hf import HFPath

REPO_ID = "zeahub/camus-sample"
FOLDER_STR = f"hf://{REPO_ID}"
FILE_SUBPATH = "val/patient0401/patient0401_4CH_half_sequence.hdf5"
FILE_STR = f"{FOLDER_STR}/{FILE_SUBPATH}"


@pytest.fixture
def folder():
    return HFPath(FOLDER_STR)


@pytest.fixture
def file(folder):
    return folder / FILE_SUBPATH


@pytest.fixture
def fake_files():
    return [
        FILE_SUBPATH,
        "val/patient0401/patient0401_2CH_full_sequence.hdf5",
        "val/patient0402/patient0402_4CH_half_sequence.hdf5",
    ]


def test_str_folder(folder):
    assert str(folder) == FOLDER_STR


def test_str_file(file):
    assert str(file) == FILE_STR


def test_repo_id(file):
    assert file.repo_id == REPO_ID


def test_subpath(file):
    assert file.subpath == FILE_SUBPATH


def test_path_joining(folder):
    # HFPath / string
    f = folder / FILE_SUBPATH
    assert isinstance(f, HFPath)
    assert str(f) == FILE_STR

    # HFPath / Path-like
    from pathlib import PurePosixPath

    f2 = folder / PurePosixPath(FILE_SUBPATH)
    assert isinstance(f2, HFPath)
    assert str(f2) == FILE_STR

    # HFPath / HFPath (should just append as string)
    f3 = folder / HFPath(FILE_SUBPATH)
    assert isinstance(f3, HFPath)
    assert str(f3) == FILE_STR


def test_is_file_and_is_dir(file, folder, fake_files, monkeypatch):
    # Patch _hf_parse_path and _hf_list_files to simulate HF repo
    def fake_parse_path(path_str):
        if path_str == FOLDER_STR:
            return REPO_ID, ""
        if path_str.startswith(FOLDER_STR + "/"):
            return REPO_ID, path_str[len(FOLDER_STR) + 1 :]
        return REPO_ID, ""

    def fake_list_files(repo_id, repo_type="dataset", **kwargs):
        assert repo_id == REPO_ID
        assert repo_type == "dataset"
        return fake_files

    monkeypatch.setattr("zea.tools.hf._hf_parse_path", fake_parse_path)
    monkeypatch.setattr("zea.tools.hf._hf_list_files", fake_list_files)

    # file is a file
    assert file.is_file() is True
    # file is not a dir
    assert file.is_dir() is False
    # folder is a dir
    assert folder.is_dir() is True
    # folder is not a file
    assert folder.is_file() is False
    # non-existent file
    non_file = folder / "val/patient0401/doesnotexist.hdf5"
    assert non_file.is_file() is False
    # non-existent dir
    non_dir = folder / "notareal"
    assert non_dir.is_dir() is False


def test_hf_resolve_path(folder, fake_files, monkeypatch):
    """Test _hf_resolve_path function with mocked HF calls."""

    def fake_parse_path(path_str):
        if path_str == FOLDER_STR:
            return REPO_ID, None
        if path_str == f"{FOLDER_STR}/val":
            return REPO_ID, "val"
        if path_str.startswith(FOLDER_STR + "/"):
            return REPO_ID, path_str[len(FOLDER_STR) + 1 :]
        return REPO_ID, None

    def fake_list_files(repo_id, repo_type="dataset", **kwargs):
        assert repo_id == REPO_ID
        assert repo_type == "dataset"
        return fake_files

    def fake_download(repo_id, filename, cache_dir, repo_type="dataset", **kwargs):
        assert repo_type == "dataset"
        # Simulate HF Hub download path structure
        mock_path = (
            cache_dir
            / f"datasets--{repo_id.replace('/', '--')}"
            / "snapshots"
            / "abc123"
            / filename
        )
        return str(mock_path)

    monkeypatch.setattr("zea.internal.preset_utils._hf_parse_path", fake_parse_path)
    monkeypatch.setattr("zea.internal.preset_utils._hf_list_files", fake_list_files)
    monkeypatch.setattr("zea.internal.preset_utils._hf_download", fake_download)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)

        # Create mock directory structure
        snapshot_dir = (
            cache_dir / f"datasets--{REPO_ID.replace('/', '--')}" / "snapshots" / "abc123"
        )
        val_dir = snapshot_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)

        result = _hf_resolve_path(f"{FOLDER_STR}/val", cache_dir)
        assert isinstance(result, Path)
        assert result.name == "val"


def test_hf_parse_path():
    """Test HF path parsing."""

    # Test repo only
    repo_id, subpath = _hf_parse_path("hf://zeahub/camus-sample")
    assert repo_id == "zeahub/camus-sample"
    assert subpath is None

    # Test repo with subpath
    repo_id, subpath = _hf_parse_path("hf://zeahub/camus-sample/val/patient0401")
    assert repo_id == "zeahub/camus-sample"
    assert subpath == "val/patient0401"

    # Test invalid path
    with pytest.raises(ValueError):
        _hf_parse_path("invalid://path")


def test_download_files_in_path(fake_files, monkeypatch):
    """Test file filtering and download logic."""

    downloaded_files = []

    def fake_download(repo_id, filename, cache_dir, repo_type="dataset", **kwargs):
        assert repo_type == "dataset"
        downloaded_files.append(filename)
        return f"/mock/path/{filename}"

    monkeypatch.setattr("zea.internal.preset_utils._hf_download", fake_download)

    # Test downloading files with path filter
    result = _download_files_in_path(REPO_ID, fake_files, "val/patient0401/", "/tmp")

    # Should download 2 files that start with "val/patient0401/"
    assert len(result) == 2
    assert len(downloaded_files) == 2
    assert all(f.startswith("val/patient0401/") for f in downloaded_files)


@pytest.mark.parametrize(
    "hf_path, kwargs, expected_fs_path",
    [
        (FILE_STR, {}, f"datasets/{REPO_ID}/{FILE_SUBPATH}"),
        (FILE_STR, {"repo_type": "model"}, f"{REPO_ID}/{FILE_SUBPATH}"),
        (FILE_STR, {"repo_type": "space"}, f"spaces/{REPO_ID}/{FILE_SUBPATH}"),
        (
            FILE_STR,
            {"revision": "v0.1.0"},
            f"datasets/{REPO_ID}@v0.1.0/{FILE_SUBPATH}",
        ),
    ],
)
def test_hf_stream_open_path_construction(hf_path, kwargs, expected_fs_path, monkeypatch):
    """The fsspec path (prefix + repo + revision + subpath) is built correctly."""
    opened = {}

    class FakeFS:
        def open(self, path, mode, block_size=None, **kw):
            opened["path"] = path
            opened["mode"] = mode
            opened["block_size"] = block_size
            return "FAKE_FILEOBJ"

    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)

    result = _hf_stream_open(hf_path, block_size=1234, **kwargs)
    assert result == "FAKE_FILEOBJ"
    assert opened["path"] == expected_fs_path
    assert opened["mode"] == "rb"
    assert opened["block_size"] == 1234


def test_hf_stream_open_requires_file(monkeypatch):
    """Streaming a repo root / directory (no subpath) raises a clear error."""
    monkeypatch.setattr("huggingface_hub.HfFileSystem", object)
    with pytest.raises(ValueError, match="single file"):
        _hf_stream_open(FOLDER_STR)


def test_hf_stream_open_rejects_bad_repo_type(monkeypatch):
    """An unknown repo_type is rejected before any network access."""
    monkeypatch.setattr("huggingface_hub.HfFileSystem", object)
    with pytest.raises(ValueError, match="repo_type"):
        _hf_stream_open(FILE_STR, repo_type="banana")


def test_get_snapshot_dir_from_downloaded_file():
    """Test snapshot directory extraction from file path."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        snapshots_dir = tmp_path / "snapshots"
        snapshot_hash_dir = snapshots_dir / "abc123def"
        file_dir = snapshot_hash_dir / "val" / "patient0401"
        file_dir.mkdir(parents=True)

        # Create the mock file
        mock_file = file_dir / "file.hdf5"
        mock_file.touch()

        result = _get_snapshot_dir_from_downloaded_file(str(mock_file))
        assert result == snapshot_hash_dir
        assert result.name == "abc123def"


def test_load_model_from_hf(monkeypatch, tmp_path):
    """The model snapshot is downloaded and its directory returned."""
    from datetime import datetime, timezone

    import zea.tools.hf as hf

    logins = []
    monkeypatch.setattr(hf, "_hf_login", lambda: logins.append(1))
    monkeypatch.setattr(hf, "snapshot_download", lambda **kwargs: str(tmp_path))

    class FakeCommit:
        title = "Add weights"
        created_at = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)

    class FakeApi:
        def list_repo_commits(self, repo_id, revision=None):
            return [FakeCommit()]

    monkeypatch.setattr(hf, "HfApi", FakeApi)

    model_dir = hf.load_model_from_hf("zeahub/taesdxl")

    assert model_dir == Path(tmp_path)
    assert logins == [1]


def test_upload_folder_to_hf(monkeypatch, tmp_path):
    """A local directory is uploaded to the given repo, branch and tag."""
    import zea.tools.hf as hf

    monkeypatch.setattr(hf, "_hf_login", lambda: None)
    calls = {}

    class FakeApi:
        def create_branch(self, repo_id, repo_type=None, branch=None, exist_ok=False):
            calls["branch"] = (repo_id, repo_type, branch, exist_ok)

        def upload_folder(self, folder_path=None, repo_id=None, **kwargs):
            calls["upload"] = (str(folder_path), repo_id, kwargs.get("commit_message"))

        def create_tag(self, repo_id, repo_type=None, tag=None):
            calls["tag"] = (repo_id, repo_type, tag)

    monkeypatch.setattr(hf, "HfApi", FakeApi)

    # A stale listing must not survive the upload that invalidates it.
    from zea.internal import preset_utils as ipu

    ipu._LISTING_CACHE.get_or_call(("zeahub/taesdxl", "model"), lambda: {"old.txt": 1})
    assert ipu._LISTING_CACHE._entries, "listing was not seeded, the assert below is vacuous"

    url = hf.upload_folder_to_hf(tmp_path, "zeahub/taesdxl", tag="v1")

    assert url == "https://huggingface.co/zeahub/taesdxl"
    assert ipu._LISTING_CACHE._entries == {}
    assert calls["branch"] == ("zeahub/taesdxl", "model", "main", True)
    assert calls["upload"][0] == str(tmp_path)
    assert calls["upload"][2] == f"Upload files from {tmp_path.name}"
    assert calls["tag"] == ("zeahub/taesdxl", "model", "v1")


def test_hfpath_joinpath_and_scheme_handling(folder):
    """joinpath keeps the hf:// scheme, and an already-prefixed part is not doubled."""
    joined = folder.joinpath("val", "patient0401")
    assert str(joined) == f"{FOLDER_STR}/val/patient0401"
    assert str(HFPath(FOLDER_STR, f"{HFPath._scheme}val")) == f"{FOLDER_STR}/val"


def test_hfpath_repo_id_requires_org_and_repo():
    with pytest.raises(ValueError, match="cannot extract repo_id"):
        HFPath("hf://only-org").repo_id
