"""Tests for the File module."""

from pathlib import Path

import numpy as np
import pytest

from usbmd.data.file import File
from usbmd.probes import Probe
from usbmd.scan import Scan

from . import data_root

DATASET_PATH = f"{data_root}/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5"
FILE_NAME = "20240701_P1_A4CH_0000.hdf5"
FILE_PATH = DATASET_PATH + "/" + FILE_NAME
FILE_HAS_EVENTS = False
FILE_NUM_FRAMES = 100
FILE_PROBE_NAME = "generic"


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


def test_basic_properties(simple_h5_file):
    """Test basic properties of File class."""

    with File(simple_h5_file) as file:
        assert file.attrs["dummy_attr"] == "dummy_value"

        # Get length of file (should be 0 as there are no datasets)
        assert len(file) == 0


def test_with_datasets(complex_h5_file):
    """Test File features with datasets."""
    with File(complex_h5_file) as file:
        # Get length of file
        assert len(file) == 2

        # Get shape of file
        assert file.shape("dummy_dataset") == (10, 20)

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


def test_file_attributes():
    if not Path(DATASET_PATH).exists():
        pytest.skip("The dataset path is unavailable.")

    # Test the file attributes
    with File(FILE_PATH) as file:
        assert file.name == FILE_NAME
        assert file.path == Path(FILE_PATH)
        assert file.has_events == FILE_HAS_EVENTS
        assert file.num_frames == FILE_NUM_FRAMES
        assert file.probe_name == FILE_PROBE_NAME
        assert isinstance(file.probe(), Probe)
        assert isinstance(file.scan(), Scan)

        file.validate()
