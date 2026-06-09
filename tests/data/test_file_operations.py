"""Test the file operations module."""

import os
from pathlib import Path
from typing import Generator

import h5py
import pytest

from zea.data.data_format import (
    generate_example_dataset,
    load_additional_elements,
    load_description,
)
from zea.data.file import File, load_file, load_file_all_data_types, validate_file
from zea.data.file_operations import (
    compound_frames,
    compound_transmits,
    extract_frames_transmits,
    resave,
    sum_data,
)


@pytest.fixture
def tmp_hdf5_path(tmp_path) -> Generator[Path, None, None]:
    """Fixture to create a temporary HDF5 file."""
    yield Path(tmp_path, "test_case_dataset.hdf5")


def test_file_operations_sum(tmp_hdf5_path):
    """Tests the sum_data function by creating two example datasets,
    summing them and checking if the result is correct."""

    # Create two example datasets
    input_path1 = tmp_hdf5_path.parent / "test_case_dataset1.hdf5"
    input_path2 = tmp_hdf5_path.parent / "test_case_dataset2.hdf5"
    generate_example_dataset(input_path1, add_optional_dtypes=True)
    generate_example_dataset(input_path2, add_optional_dtypes=True)

    data1, scan1, probe1 = load_file(input_path1)
    data2, scan2, probe2 = load_file(input_path2)

    # Sum the datasets
    output_path = tmp_hdf5_path.parent / "summed_dataset.hdf5"

    sum_data([input_path1, input_path2], output_path)

    _assert_descriptions_and_additional_elements_equal(input_path1, output_path)

    # Load the summed dataset and check if the data is correct
    with File(output_path) as f:
        raw_data = f["data/raw_data"][:]
        assert raw_data[0, 0, 0, 0, 0] == data1[0, 0, 0, 0, 0] + data2[0, 0, 0, 0, 0]


def test_file_operations_extract(tmp_hdf5_path):
    """Tests the load_data function by creating an example dataset and
    loading a subset of the data."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "extracted_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True)

    extract_frames_transmits(
        input_path, output_path, frame_indices=slice(1), transmit_indices=[0, 3]
    )
    data_dict, scan, probe = load_file_all_data_types(output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    assert data_dict["raw_data"].shape[0] == 1
    assert data_dict["raw_data"].shape[1] == 2
    assert data_dict["aligned_data"].shape[0] == 1
    assert data_dict["aligned_data"].shape[1] == 2
    assert data_dict["beamformed_data"].shape[0] == 1
    assert data_dict["image"].shape[0] == 1
    assert data_dict["image_sc"].shape[0] == 1

    _assert_beamformed_data_still_exists(output_path)
    _assert_descriptions_and_additional_elements_equal(input_path, output_path)


def test_file_operations_resave(tmp_hdf5_path):
    """Tests the resave operation by creating an example dataset and
    resaving it to a new file."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "resaved_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True)

    resave(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    # Validate the resaved dataset
    validate_file(output_path)


def test_file_operations_compound_frames(tmp_hdf5_path):
    """Tests the compound_frames function by creating an example dataset and
    compounding frames."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_frames_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True)

    compound_frames(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    data_dict, scan, probe = load_file_all_data_types(output_path)
    for key in data_dict.keys():
        dataset = data_dict[key]
        if dataset is None:
            continue
        assert dataset.shape[0] == 1  # Only one frame should remain


def test_file_operations_compound_transmits(tmp_hdf5_path):
    """Tests the compound_transmits function by creating an example dataset and
    compounding transmits."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_transmits_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True)

    compound_transmits(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    data, scan, probe = load_file(output_path)
    assert data.shape[1] == 1  # Only one transmit should remain
    assert scan["initial_times"].shape[0] == 1
    assert scan["t0_delays"].shape[0] == 1
    assert scan["azimuth_angles"].shape[0] == 1
    assert scan["tx_apodizations"].shape[0] == 1


def test_file_operations_cli_sum(tmp_hdf5_path):
    """Tests the sum_data function CLI by creating two example datasets,
    summing them and checking if the result is correct."""

    # Create two example datasets
    path1 = tmp_hdf5_path.parent / "test_case_dataset1.hdf5"
    path2 = tmp_hdf5_path.parent / "test_case_dataset2.hdf5"
    generate_example_dataset(path1, add_optional_dtypes=True)
    generate_example_dataset(path2, add_optional_dtypes=True)

    data1, scan1, probe1 = load_file(path1)
    data2, scan2, probe2 = load_file(path2)

    # Sum the datasets
    output_path = tmp_hdf5_path.parent / "summed_dataset.hdf5"

    os.system(
        "python -m zea.data.file_operations sum "
        + str(path1)
        + " "
        + str(path2)
        + " "
        + str(output_path)
    )

    # Load the summed dataset and check if the data is correct
    with File(output_path) as f:
        raw_data = f["data/raw_data"][:]
        assert raw_data[0, 0, 0, 0, 0] == data1[0, 0, 0, 0, 0] + data2[0, 0, 0, 0, 0]


def test_file_operations_cli_extract(tmp_hdf5_path):
    """Tests the load_data function CLI by creating an example dataset and
    loading a subset of the data."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "extracted_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True)

    os.system(
        "python -m zea.data.file_operations extract "
        + str(input_path)
        + " "
        + str(output_path)
        + " --frames 0-1 --transmits 0 3 4"
    )

    data_dict, scan, probe = load_file_all_data_types(output_path)
    assert data_dict["raw_data"].shape[0] == 2
    assert data_dict["raw_data"].shape[1] == 3
    assert data_dict["aligned_data"].shape[0] == 2
    assert data_dict["aligned_data"].shape[1] == 3
    assert data_dict["beamformed_data"].shape[0] == 2
    assert data_dict["image"].shape[0] == 2
    assert data_dict["image_sc"].shape[0] == 2


def test_file_operations_cli_resave(tmp_hdf5_path):
    """Tests the resave operation CLI by creating an example dataset and
    resaving it to a new file."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "resaved_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True)

    os.system(
        "python -m zea.data.file_operations resave " + str(input_path) + " " + str(output_path)
    )

    # Validate the resaved dataset
    validate_file(output_path)


def test_file_operations_cli_compound_frames(tmp_hdf5_path):
    """Tests the compound_frames function CLI by creating an example dataset and
    compounding frames."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_frames_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True)

    os.system(
        "python -m zea.data.file_operations compound_frames "
        + str(input_path)
        + " "
        + str(output_path)
    )

    data_dict, scan, probe = load_file_all_data_types(output_path)
    assert data_dict["raw_data"].shape[0] == 1  # Only one frame should remain
    assert data_dict["aligned_data"].shape[0] == 1
    assert data_dict["beamformed_data"].shape[0] == 1
    assert data_dict["image"].shape[0] == 1
    assert data_dict["image_sc"].shape[0] == 1


def test_file_operations_cli_compound_transmits(tmp_hdf5_path):
    """Tests the compound_transmits function CLI by creating an example dataset and
    compounding transmits."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_transmits_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True)

    os.system(
        "python -m zea.data.file_operations compound_transmits "
        + str(input_path)
        + " "
        + str(output_path)
    )

    data_dict, scan, probe = load_file_all_data_types(output_path)
    assert data_dict["raw_data"].shape[1] == 1  # Only one transmit should remain
    assert data_dict["aligned_data"].shape[1] == 1


def test_file_operations_folder_resave(tmp_path):
    """Tests that resave works on a folder of files, mirroring the folder structure."""

    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"

    # Create a folder of example datasets, including a nested subfolder
    input_paths = [
        input_folder / "case_0.hdf5",
        input_folder / "case_1.hdf5",
        input_folder / "nested" / "case_2.hdf5",
    ]
    for path in input_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_example_dataset(path, add_optional_dtypes=True)

    resave(input_folder, output_folder)

    # Each input file should have a matching output file at the mirrored location
    for input_path in input_paths:
        output_path = output_folder / input_path.relative_to(input_folder)
        assert output_path.is_file()
        validate_file(output_path)
        _assert_descriptions_and_additional_elements_equal(input_path, output_path)


def test_file_operations_folder_compound_frames(tmp_path):
    """Tests that compound_frames works on a folder of files."""

    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    input_folder.mkdir()

    input_paths = [input_folder / "case_0.hdf5", input_folder / "case_1.hdf5"]
    for path in input_paths:
        generate_example_dataset(path, add_optional_dtypes=True)

    compound_frames(input_folder, output_folder)

    for input_path in input_paths:
        output_path = output_folder / input_path.name
        assert output_path.is_file()
        data_dict, _, _ = load_file_all_data_types(output_path)
        for dataset in data_dict.values():
            if dataset is not None:
                assert dataset.shape[0] == 1  # Only one frame should remain


def test_file_operations_folder_sum(tmp_path):
    """Tests that sum_data accepts a folder and sums all files it contains."""

    input_folder = tmp_path / "input"
    input_folder.mkdir()
    output_path = tmp_path / "summed.hdf5"

    input_paths = [input_folder / "case_0.hdf5", input_folder / "case_1.hdf5"]
    for path in input_paths:
        generate_example_dataset(path, add_optional_dtypes=True)

    data0, _, _ = load_file(input_paths[0])
    data1, _, _ = load_file(input_paths[1])

    sum_data(input_folder, output_path)

    with File(output_path) as f:
        raw_data = f["data/raw_data"][:]
        assert raw_data[0, 0, 0, 0, 0] == data0[0, 0, 0, 0, 0] + data1[0, 0, 0, 0, 0]


def _load_description_and_additional_elements(path: Path):
    description = load_description(path)
    additional_elements = load_additional_elements(path)
    return description, additional_elements


def _assert_descriptions_and_additional_elements_equal(path, other_path: Path):
    description, additional_elements = _load_description_and_additional_elements(path)
    other_description, other_additional_elements = _load_description_and_additional_elements(
        other_path
    )
    assert description == other_description
    assert len(additional_elements) == len(other_additional_elements)


def _assert_beamformed_data_still_exists(path: Path):
    with h5py.File(path, "r") as f:
        assert "data/beamformed_data" in f
