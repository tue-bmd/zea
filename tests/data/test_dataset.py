"""Testing for `zea.data.datasets` module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from zea.config import Config, check_config
from zea.data.datasets import Dataset, Folder, split_files_by_directory
from zea.internal.checks import _IMAGE_DATA_TYPES, _NON_IMAGE_DATA_TYPES

from .. import DUMMY_DATASET_GRID_SIZE_X, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_N_FRAMES
from . import generate_example_dataset

_ALL_DATA_TYPES = _IMAGE_DATA_TYPES + _NON_IMAGE_DATA_TYPES


@pytest.mark.parametrize(
    "file_idx, idx, expected_shape",
    [
        (
            0,
            slice(None),
            (DUMMY_DATASET_N_FRAMES, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            -1,
            (1, 2, 3),
            (),
        ),
        (
            0,
            [1, 2, 3],
            (3, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            -1,
            np.array([1, 2, 3]),
            (3, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            0,
            slice(1, 3),
            (2, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            -1,
            (0, list(range(5))),
            (5, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            0,
            (np.array([1, 2]), slice(10)),
            (2, 10, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            0,
            (slice(None), np.arange(10)),
            (DUMMY_DATASET_N_FRAMES, 10, DUMMY_DATASET_GRID_SIZE_X),
        ),
    ],
)
def test_dataset_indexing(file_idx, idx, expected_shape, dummy_dataset_path):
    """Test ui initialization function"""
    config = {"data": {"path": str(dummy_dataset_path)}}
    config = check_config(Config(config))
    dataset = Dataset.from_config(**config.data)

    file = dataset[file_idx]
    data = file[file.format_key("image")]["values"][idx]

    assert data.shape == expected_shape, (
        f"Data shape {data.shape} does not match expected shape {expected_shape}"
    )


def test_folder_copy_key_by_key(dummy_dataset_path, tmp_path):
    """Test copying a `zea.Folder` key by key."""

    _copied_keys = []

    for key in _ALL_DATA_TYPES:
        _copied_keys.append(key)
        _other_keys = [k for k in _ALL_DATA_TYPES if k not in _copied_keys]

        # Copy the folder with the specified key
        folder = Folder(dummy_dataset_path, validate=False)
        folder.copy(tmp_path / "copy", key)

        # Check required keys in the copied folder
        with Dataset(tmp_path / "copy", validate=False) as copied_folder:
            for file in iter(copied_folder):
                for key in _copied_keys:
                    assert key in file["data"], f"Copied folder does not contain {key} key"
                assert "scan" in file, "Copied folder does not contain 'scan' key"

            # Check that the copied folder does not contain other keys
            for file in iter(copied_folder):
                for other_key in _other_keys:
                    assert other_key not in file["data"], (
                        f"Copied folder should not contain {other_key} key"
                    )


def test_folder_copy_all_keys(dummy_dataset_path, tmp_path):
    """Test copying a `zea.Folder` with all keys."""

    # Copy the folder
    folder = Folder(dummy_dataset_path, validate=False)
    folder.copy(tmp_path / "copy", key="all")

    # Check required keys in the copied folder
    with Dataset(tmp_path / "copy", validate=False) as copied_folder:
        for file in iter(copied_folder):
            for key in _ALL_DATA_TYPES:
                assert key in file["data"], f"Copied folder does not contain {key} key"
            assert "scan" in file, "Copied folder does not contain 'scan' key"


@pytest.mark.parametrize(
    "dir_sizes, splits, expected_counts",
    [
        # full split returns all files from both directories
        ([10, 20], [1.0, 1.0], [10, 20]),
        # half split from each directory
        ([10, 20], [0.5, 0.5], [5, 10]),
        # zero split from first directory, all from second
        ([10, 20], [0.0, 1.0], [0, 20]),
        # all from first, none from second
        ([10, 20], [1.0, 0.0], [10, 0]),
        # three directories, full split
        ([5, 5, 5], [1.0, 1.0, 1.0], [5, 5, 5]),
        # three directories, partial split (int truncation: int(0.6*5)=3)
        ([5, 5, 5], [0.6, 0.6, 0.6], [3, 3, 3]),
        # single directory
        ([8], [0.25], [2]),
    ],
)
def test_split_files_by_directory(dir_sizes, splits, expected_counts, tmp_path):
    """Test that split_files_by_directory returns the correct number of files per directory."""

    # Build fake file paths (no real files needed)
    directories = [str(tmp_path / f"dir{i}") for i in range(len(dir_sizes))]
    file_names = []
    for dir_path, n_files in zip(directories, dir_sizes):
        for j in range(n_files):
            file_names.append(str(Path(dir_path) / f"file{j:04d}.hdf5"))

    result = split_files_by_directory(file_names, directories, splits)

    assert len(result) == sum(expected_counts), (
        f"Expected {sum(expected_counts)} files, got {len(result)}"
    )

    # Verify the correct number of files was taken from each directory
    for dir_path, expected in zip(directories, expected_counts):
        count = sum(1 for f in result if f.startswith(dir_path))
        assert count == expected, f"Expected {expected} files from '{dir_path}', got {count}"


def test_find_h5_files_finds_only_h5(tmp_path):
    """Folder.find_h5_files returns only .hdf5 / .h5 files, ignoring others."""

    # Create two HDF5 files and one unrelated file
    generate_example_dataset(tmp_path / "a.hdf5")
    generate_example_dataset(tmp_path / "b.h5")
    (tmp_path / "notes.txt").write_text("ignore me")

    folder = Folder(tmp_path, validate=False)
    found = folder.find_h5_files()

    assert len(found) == 2
    assert all(f.endswith((".hdf5", ".h5")) for f in found)
    assert not any(f.endswith(".txt") for f in found)


def test_find_h5_files_recurses_subdirectories(tmp_path):
    """find_h5_files discovers files in nested sub-directories."""

    (tmp_path / "sub").mkdir()
    generate_example_dataset(tmp_path / "root.hdf5")
    generate_example_dataset(tmp_path / "sub" / "nested.hdf5")

    folder = Folder(tmp_path, validate=False)
    found = folder.find_h5_files()

    assert len(found) == 2


def test_folder_properties(dummy_dataset_path):
    """Folder exposes correct n_files, __len__, __repr__ and __str__."""
    folder = Folder(dummy_dataset_path, validate=False)

    assert folder.n_files == 2
    assert len(folder) == 2
    assert repr(folder) == f"Folder(n_files=2, folder='{dummy_dataset_path}')"
    assert str(dummy_dataset_path) in repr(folder)
    assert str(folder) == f"Folder with 2 files in '{dummy_dataset_path}'"


def test_dataset_properties(dummy_dataset_path):
    """Dataset exposes correct n_files, __len__, __repr__ and __str__."""
    with Dataset(dummy_dataset_path, validate=False) as dataset:
        assert dataset.n_files == 2
        assert len(dataset) == 2
        assert repr(dataset) == "Dataset(n_files=2)"
        assert str(dataset) == "Dataset with 2 files"


def test_folder_rejects_invalid_type():
    """Folder raises ValueError when given a non-string/Path argument."""
    with pytest.raises(ValueError, match="Invalid folder path"):
        Folder(12345, validate=False)


def test_folder_rejects_single_file(dummy_dataset_path):
    """Folder raises ValueError when given a path to a single file."""
    file_path = next(Path(dummy_dataset_path).glob("*.hdf5"))
    with pytest.raises(ValueError, match="Use File class instead"):
        Folder(file_path, validate=False)


def test_dataset_lazy_hf_defers_download(tmp_path):
    """lazy=True stores hf:// pointers at init and downloads each file on first access."""
    f1 = tmp_path / "file1.hdf5"
    f2 = tmp_path / "file2.hdf5"
    generate_example_dataset(f1)
    generate_example_dataset(f2)

    hf_files = [("file1.hdf5", 1024), ("file2.hdf5", 2048)]

    with (
        patch("zea.data.datasets._hf_list_h5_files", return_value=hf_files),
        patch("zea.data.datasets._hf_resolve_path", return_value=f1) as mock_resolve,
    ):
        ds = Dataset("hf://org/myrepo", lazy=True)

        # No download at init
        mock_resolve.assert_not_called()
        assert len(ds) == 2
        assert ds.file_paths[0] == "hf://org/myrepo/file1.hdf5"
        assert ds.file_paths[1] == "hf://org/myrepo/file2.hdf5"

        # First access triggers download of that file only
        _ = ds[0]
        mock_resolve.assert_called_once_with("hf://org/myrepo/file1.hdf5")
        assert ds.file_paths[0] == str(f1)  # pointer replaced with local path
        assert ds.file_paths[1] == "hf://org/myrepo/file2.hdf5"  # untouched

        # Second access to the same index does not re-download
        mock_resolve.reset_mock()
        _ = ds[0]
        mock_resolve.assert_not_called()

        ds.close()


def test_dataloader_rejects_lazy():
    """H5DataSource raises ValueError when lazy=True is passed."""
    from zea.data.dataloader import H5DataSource

    with pytest.raises(ValueError, match="lazy=True is not supported"):
        H5DataSource("nonexistent_path", lazy=True)


def test_dataset_hf_large_warns_about_gb_size(tmp_path):
    """_find_hf_files logs a warning with GB size when >10 non-lazy files are found."""
    f = tmp_path / "file00.hdf5"
    generate_example_dataset(f)

    # 12 files × 100 MB each = 1.2 GB
    hf_files = [(f"file{i:02d}.hdf5", 100_000_000) for i in range(12)]

    with (
        patch("zea.data.datasets._hf_list_h5_files", return_value=hf_files),
        patch("zea.data.datasets._hf_resolve_path", return_value=tmp_path),
        patch("zea.data.datasets.search_file_tree", return_value=[f]),
        patch("zea.data.datasets.log") as mock_log,
    ):
        ds = Dataset("hf://org/myrepo", lazy=False, _suggest_lazy=True)

    msgs = [str(c.args[0]) for c in mock_log.warning.call_args_list]
    assert any("GB" in m for m in msgs), "expected GB size in warning"
    assert any("lazy=True" in m for m in msgs), "expected lazy suggestion in warning"
    ds.close()


def test_dataset_hf_large_no_lazy_suggestion(tmp_path):
    """_find_hf_files omits the lazy hint when _suggest_lazy=False."""
    f = tmp_path / "file00.hdf5"
    generate_example_dataset(f)

    hf_files = [(f"file{i:02d}.hdf5", 50_000_000) for i in range(12)]  # 12 × 50 MB

    with (
        patch("zea.data.datasets._hf_list_h5_files", return_value=hf_files),
        patch("zea.data.datasets._hf_resolve_path", return_value=tmp_path),
        patch("zea.data.datasets.search_file_tree", return_value=[f]),
        patch("zea.data.datasets.log") as mock_log,
    ):
        ds = Dataset("hf://org/myrepo", lazy=False, _suggest_lazy=False)

    msgs = [str(c.args[0]) for c in mock_log.warning.call_args_list]
    assert any("GB" in m for m in msgs)
    assert not any("lazy=True" in m for m in msgs)
    ds.close()
