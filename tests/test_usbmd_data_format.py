"""Test generating and validating zea data format."""

from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from zea.data.data_format import (
    DatasetElement,
    generate_example_dataset,
    generate_zea_dataset,
    load_additional_elements,
)
from zea.data.file import File, validate_file
from zea.internal.checks import _REQUIRED_SCAN_KEYS
from . import DEFAULT_TEST_SEED

n_frames = 2
n_tx = 4
n_el = 16
n_ax = 128
n_ch = 1

DATASET_PARAMETERS = {
    "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
    "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
    "sampling_frequency": 30e6,
    "center_frequency": 6e6,
    "initial_times": np.zeros((n_tx), dtype=np.float32),
    "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
    "sound_speed": 1540.0,
    "probe_name": "generic",
    "description": "Dataset parameters for testing",
    "focus_distances": np.zeros((n_tx,), dtype=np.float32),
    "polar_angles": np.linspace(-np.pi / 2, np.pi / 2, n_tx, dtype=np.float32),
    "azimuth_angles": np.zeros((n_tx), np.float32),
    "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
    "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32),
    "bandwidth_percent": 200.0,
    "waveforms_one_way": [np.zeros((512), dtype=np.float32)],
    "waveforms_two_way": [np.zeros((512,), dtype=np.float32)],
    "tx_waveform_indices": np.zeros((n_tx,), dtype=np.int32),
}


@pytest.fixture
def tmp_hdf5_path(tmp_path) -> Generator[Path, None, None]:
    """Fixture to create a temporary HDF5 file."""
    yield Path(tmp_path, "test_case_dataset.hdf5")


@pytest.fixture
def example_dataset_path(tmp_hdf5_path):
    """Fixture to create a temporary dataset for testing."""
    generate_example_dataset(tmp_hdf5_path)
    yield tmp_hdf5_path


def test_example_dataset(example_dataset_path):
    """Tests the generate_example_dataset function by calling it and then
    validating it using the validate_file function.
    """
    # Validate the dataset
    validate_file(example_dataset_path)

    # Check if the dataset can be loaded correctly
    with File(example_dataset_path) as dataset:
        raw_data = dataset.load_data("raw_data", 0)
        assert raw_data is not None, "Dataset not loaded correctly"


@pytest.mark.parametrize(
    "key",
    [key for key in DATASET_PARAMETERS if key not in _REQUIRED_SCAN_KEYS],
)
def test_omit_key(key, tmp_hdf5_path):
    """Tests if omitting an optional key in the dataset_parameters dictionary
    does not raise an error.

    Args:
        key (str): The key to omit from the dataset_parameters dictionary.
    """
    reduced_parameters = DATASET_PARAMETERS.copy()
    reduced_parameters.pop(key)
    generate_zea_dataset(path=tmp_hdf5_path, **DATASET_PARAMETERS)


@pytest.mark.parametrize(
    "key",
    [
        "raw_data",
        "probe_geometry",
        "sampling_frequency",
        "center_frequency",
        "initial_times",
        "t0_delays",
        "sound_speed",
        "probe_name",
        "description",
        "focus_distances",
        "polar_angles",
        "azimuth_angles",
        "tx_apodizations",
        "bandwidth_percent",
        "time_to_next_transmit",
    ],
)
def test_wrong_shape(key, tmp_hdf5_path):
    """Tests if passing a parameter with the wrong shape raises an error.

    Args:
        key (str): The key to change in the dataset_parameters dictionary.
    """
    wrong_parameters = DATASET_PARAMETERS.copy()
    wrong_parameters[key] = np.zeros((n_frames, n_tx + 7, n_el + 1), dtype=np.float32)
    with pytest.raises(AssertionError):
        generate_zea_dataset(path=tmp_hdf5_path, **wrong_parameters)


def test_existing_path(tmp_hdf5_path):
    """Tests if passing a path that already exists raises an error."""
    # Ensure that the file exists
    tmp_hdf5_path.touch()

    with pytest.raises(FileExistsError):
        generate_zea_dataset(path=tmp_hdf5_path, **DATASET_PARAMETERS)


def test_only_waveforms_one_way(tmp_hdf5_path):
    """Tests if passing only waveforms_one_way works correctly."""
    parameters = DATASET_PARAMETERS.copy()
    parameters["waveforms_one_way"] = None
    generate_zea_dataset(path=tmp_hdf5_path, **parameters)


def test_only_waveforms_two_way(tmp_hdf5_path):
    """Tests if passing only waveforms_two_way works correctly."""
    parameters = DATASET_PARAMETERS.copy()
    parameters["waveforms_two_way"] = None
    generate_zea_dataset(path=tmp_hdf5_path, **parameters)


def test_additional_dataset_element(tmp_hdf5_path):
    """Tests the functionality of the additional_elements parameter in the
    generate_zea_dataset function by adding additional elements to the
    dataset."""

    elements = []
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    elements.append(
        DatasetElement(
            dataset_name="lens_correction",
            data=np.array(0.1),
            description="The additional path length due to the lens in wavelengths.",
            unit="wavelengths",
        )
    )
    elements.append(
        DatasetElement(
            dataset_name="sound_speed_map",
            data=rng.standard_normal((10, 10)),
            description="The local speed of sound in the medium.",
            unit="m/s",
        )
    )

    # Add elements to subgroup
    t = np.arange(100) / DATASET_PARAMETERS["sampling_frequency"]
    for n in range(4):
        elements.append(
            DatasetElement(
                group_name="functions",
                dataset_name=f"functions{n:02d}",
                data=np.sin(2 * np.pi * 1e6 * n * t),
                description="element3 description",
                unit="m",
            )
        )

    generate_zea_dataset(path=tmp_hdf5_path, **DATASET_PARAMETERS, additional_elements=elements)

    elements = load_additional_elements(tmp_hdf5_path)

    assert len(elements) == 6, "Not all additional elements were saved correctly."
