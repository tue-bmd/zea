"""Tests for the read_h5 module."""

import numpy as np
import pytest

from usbmd.data.file import File


@pytest.fixture
def h5_filepath(tmp_path):
    """Create path for the H5 test file."""
    path = tmp_path / "dummy_dataset.hdf5"
    yield path


@pytest.fixture
def simple_h5_file(h5_filepath):
    """Create a simple H5 file with only attributes."""
    with File(h5_filepath, "w") as dataset:
        dataset.attrs["dummy_attr"] = "dummy_value"
        dataset.attrs["dummy_attr2"] = "dummy_value2"
        dataset.attrs["dummy_attr3"] = ["dummy_value3"]
    yield h5_filepath


@pytest.fixture
def complex_h5_file(h5_filepath):
    """Create an H5 file with attributes and datasets."""
    with File(h5_filepath, "w") as dataset:
        dataset.attrs["dummy_attr"] = "dummy_value"
        dataset.create_dataset("dummy_dataset", data=np.random.randn(10, 20))
        dataset.create_dataset("dummy_dataset2", data=np.arange(5))
    yield h5_filepath


def test_readh5_basic_properties(simple_h5_file):
    """Test basic properties of ReadH5 class."""

    with File(simple_h5_file) as file:
        assert file.attrs["dummy_attr"] == "dummy_value"

        # Get length of file (should be 0 as there are no datasets)
        assert len(file) == 0


def test_readh5_with_datasets(complex_h5_file):
    """Test ReadH5 features with datasets."""
    with File(complex_h5_file) as file:
        # Get length of file
        assert len(file) == 10

        # Get shape of file
        assert file.shape("data") == (10, 20)

        # Get keys in file
        assert list(file.keys()) == ["dummy_dataset", "dummy_dataset2"]


def test_recursively_load_dict(complex_h5_file):
    """Test recursively loading dict contents from group."""

    with File(complex_h5_file) as file:
        dict_contents = file.recursively_load_dict_contents_from_group("/")
        assert list(dict_contents.keys()) == ["dummy_dataset", "dummy_dataset2"]
        assert dict_contents["dummy_dataset"].shape == (10, 20)
        assert dict_contents["dummy_dataset2"].shape == (5,)
        assert np.array_equal(dict_contents["dummy_dataset2"], np.arange(5))


def test_print_hdf5_attrs(complex_h5_file, capsys):
    """Test printing HDF5 attributes."""

    with File(complex_h5_file) as file:
        file.print()

    captured = capsys.readouterr()
    assert "dummy_attr" in captured.out
