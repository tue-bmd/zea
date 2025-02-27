"""Tests for the read_h5 module."""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from usbmd.data.read_h5 import (
    ReadH5,
    print_hdf5_attrs,
    recursively_load_dict_contents_from_group,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path("temp")
    temp_path.mkdir(exist_ok=True)
    yield temp_path
    # Cleanup is handled by h5_file fixture


@pytest.fixture
def h5_filepath(temp_dir):
    """Create path for the H5 test file."""
    path = temp_dir / "dummy_dataset.hdf5"
    yield path
    # Clean up the file after the test
    if path.exists():
        os.remove(path)

    # Remove temp directory if empty
    if not any(temp_dir.iterdir()):
        temp_dir.rmdir()


@pytest.fixture
def simple_h5_file(h5_filepath):
    """Create a simple H5 file with only attributes."""
    with h5py.File(h5_filepath, "w") as dataset:
        dataset.attrs["dummy_attr"] = "dummy_value"
        dataset.attrs["dummy_attr2"] = "dummy_value2"
        dataset.attrs["dummy_attr3"] = ["dummy_value3"]
    yield h5_filepath


@pytest.fixture
def complex_h5_file(h5_filepath):
    """Create an H5 file with attributes and datasets."""
    with h5py.File(h5_filepath, "w") as dataset:
        dataset.attrs["dummy_attr"] = "dummy_value"
        dataset.create_dataset("dummy_dataset", data=np.random.randn(10, 20))
        dataset.create_dataset("dummy_dataset2", data=np.arange(5))
    yield h5_filepath


def test_readh5_basic_properties(simple_h5_file):
    """Test basic properties of ReadH5 class."""
    h5_reader = ReadH5(simple_h5_file)

    with h5_reader.open() as file:
        assert file.attrs["dummy_attr"] == "dummy_value"

        # Get length of file (should be 0 as there are no datasets)
        assert len(h5_reader) == 0


def test_readh5_extension(simple_h5_file):
    """Test file extension retrieval."""
    h5_reader = ReadH5(simple_h5_file)
    assert h5_reader.get_extension() == ".hdf5"


def test_readh5_with_datasets(complex_h5_file):
    """Test ReadH5 features with datasets."""
    h5_reader = ReadH5(complex_h5_file)

    with h5_reader.open():
        # Get length of file
        assert len(h5_reader) == 10

        # Get shape of file
        assert h5_reader.shape == (10, 20)

        # Get largest group name
        assert h5_reader.get_largest_group_name() == "dummy_dataset"

        # Get keys in file
        assert list(h5_reader.keys()) == ["dummy_dataset", "dummy_dataset2"]


def test_recursively_load_dict(complex_h5_file):
    """Test recursively loading dict contents from group."""
    h5_reader = ReadH5(complex_h5_file)

    with h5_reader.open() as file:
        dict_contents = recursively_load_dict_contents_from_group(file, "/")
        assert list(dict_contents.keys()) == ["dummy_dataset", "dummy_dataset2"]
        assert dict_contents["dummy_dataset"].shape == (10, 20)
        assert dict_contents["dummy_dataset2"].shape == (5,)
        assert np.array_equal(dict_contents["dummy_dataset2"], np.arange(5))


def test_print_hdf5_attrs(complex_h5_file, capsys):
    """Test printing HDF5 attributes."""
    h5_reader = ReadH5(complex_h5_file)

    with h5_reader.open() as file:
        print_hdf5_attrs(file)

    captured = capsys.readouterr()
    assert "dummy_attr" in captured.out
