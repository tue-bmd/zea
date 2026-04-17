"""Tests for the File module."""

import numpy as np
import pytest

from zea.data.file import File, dict_to_sorted_list, load_file
from zea.probes import Probe
from zea.scan import Scan


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
        file.summary()

    captured = capsys.readouterr()
    assert "dummy_attr" in captured.out


def test_file_attributes():
    """Test file attributes."""

    DATASET_PATH = (
        "hf://zeahub/picmus/database/simulation/contrast_speckle/contrast_speckle_simu_dataset_iq"
    )

    FILE_NAME = "contrast_speckle_simu_dataset_iq.hdf5"
    FILE_PATH = DATASET_PATH + "/" + FILE_NAME
    FILE_HAS_EVENTS = False
    FILE_N_FRAMES = 1
    FILE_PROBE_NAME = "verasonics_l11_4v"

    with File(FILE_PATH) as file:
        assert file.name == FILE_NAME, "File name should match expected value"
        assert file.has_events == FILE_HAS_EVENTS, "File should not have events"
        assert file.n_frames == FILE_N_FRAMES, "Number of frames should match expected value"
        assert file.probe_name == FILE_PROBE_NAME, "Probe name should match expected value"
        assert isinstance(file.probe(), Probe), "Probe should be an instance of Probe class"
        assert isinstance(file.scan(), Scan), "Scan should be an instance of Scan class"

        file.validate()


def test_load_file_function(dummy_file):
    """Test the load_file function."""

    selected_transmits = [0, 2, 4]
    data, scan, probe = load_file(dummy_file, indices=(slice(2), selected_transmits))

    assert data.shape[0] == 2, "Data should have 2 frames"
    assert data.shape[1] == 3, "Data should have 3 selected transmits"
    assert isinstance(scan, Scan), "Scan should be an instance of Scan class"
    assert isinstance(probe, Probe), "Probe should be an instance of Probe class"
    assert scan.selected_transmits == selected_transmits, (
        "Selected transmits should match expected value"
    )


def test_dict_to_sorted_list():
    """Test dict_to_sorted_list utility function."""

    test_dict = {"b": 2, "a": 1, "c": 3}
    sorted_list = dict_to_sorted_list(test_dict)

    assert sorted_list == [1, 2, 3], "The sorted list should be [1, 2, 3]"

    assert dict_to_sorted_list({}) == [], "The sorted list of an empty dict should be []"
