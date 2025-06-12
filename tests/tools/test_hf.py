"""Minimal tests for the HFPath class in zea.tools.hf module."""

import pytest

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
