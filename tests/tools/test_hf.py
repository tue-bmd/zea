"""Minimal tests for the HFPath class in zea.tools.hf module."""

# Create the directory structure for testing
import tempfile
from pathlib import Path

import pytest

from zea.data.preset_utils import (
    _download_files_in_path,
    _get_snapshot_dir_from_downloaded_file,
    _hf_parse_path,
    _hf_resolve_path,
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

    def fake_list_files(repo_id):
        assert repo_id == REPO_ID
        return fake_files

    monkeypatch.setattr("zea.data.preset_utils._hf_parse_path", fake_parse_path)
    monkeypatch.setattr("zea.data.preset_utils._hf_list_files", fake_list_files)

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

    def fake_list_files(repo_id):
        assert repo_id == REPO_ID
        return fake_files

    def fake_download(repo_id, filename, cache_dir):
        # Simulate HF Hub download path structure
        mock_path = (
            cache_dir
            / f"datasets--{repo_id.replace('/', '--')}"
            / "snapshots"
            / "abc123"
            / filename
        )
        return str(mock_path)

    monkeypatch.setattr("zea.data.preset_utils._hf_parse_path", fake_parse_path)
    monkeypatch.setattr("zea.data.preset_utils._hf_list_files", fake_list_files)
    monkeypatch.setattr("zea.data.preset_utils._hf_download", fake_download)

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

    def fake_download(repo_id, filename, cache_dir):
        downloaded_files.append(filename)
        return f"/mock/path/{filename}"

    monkeypatch.setattr("zea.data.preset_utils._hf_download", fake_download)

    # Test downloading files with path filter
    result = _download_files_in_path(REPO_ID, fake_files, "val/patient0401/", "/tmp")

    # Should download 2 files that start with "val/patient0401/"
    assert len(result) == 2
    assert len(downloaded_files) == 2
    assert all(f.startswith("val/patient0401/") for f in downloaded_files)


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
