"""Basic testing datasets"""

import sys
from pathlib import Path

import numpy as np
import pytest

from usbmd.config.validation import check_config
from usbmd.data import generate_usbmd_dataset
from usbmd.data.datasets import Dataset
from usbmd.generate import GenerateDataSet
from usbmd.setup_usbmd import setup_config

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


@pytest.fixture
def dataset_path(tmp_path):
    """Fixture to create a temporary dataset"""
    for i in range(2):
        temp_file = tmp_path / f"test{i}.hdf5"
        dummy_data = np.random.rand(10, 20, 30)

        if not temp_file.exists():
            generate_usbmd_dataset(
                path=temp_file,
                image=dummy_data,
                probe_name="dummy",
                description="dummy dataset",
            )

    return str(tmp_path)


@pytest.mark.parametrize(
    "file_idx, frame_idx",
    [
        (0, "all"),
        (-1, (1, 2, 3)),
        (0, [1, 2, 3]),
        (0, np.array([1, 2, 3])),
    ],
)
def test_dataset_indexing(file_idx, frame_idx, dataset_path):
    """Test ui initialization function"""
    config = {"data": {"dataset_folder": dataset_path, "dtype": "image"}}
    config = check_config(config)
    dataset = Dataset.from_config(**config.data)

    file = dataset[file_idx]
    data = file.load_data(config.data.dtype, frame_idx=frame_idx)

    if isinstance(frame_idx, (list, tuple, np.ndarray)):
        assert len(data) == len(
            frame_idx
        ), f"Data length {data.shape} does not match frame_idx length {len(frame_idx)}"
    elif frame_idx == "all":
        assert (
            len(data) == file.num_frames
        ), f"Data length {data.shape} does not match file length {file.num_frames}"


@pytest.mark.parametrize(
    "dtype, to_dtype, filetype",
    [
        ("raw_data", "image", "png"),
        ("beamformed_data", "image", "hdf5"),
    ],
)
def test_generate(dtype, to_dtype, filetype, tmp_path):
    """Test generate class"""
    config = setup_config("./tests/config_test.yaml")
    config.data.dtype = dtype

    config.pipeline.operations = [
        {"name": "demodulate"},
        {"name": "tof_correction"},
        {"name": "delay_and_sum"},
        {"name": "envelope_detect"},
        {"name": "normalize"},
        {"name": "log_compress"},
    ]
    if dtype == "beamformed_data":
        config.pipeline.operations = config.pipeline.operations[3:]

    generator = GenerateDataSet(
        config,
        destination_folder=tmp_path,
        to_dtype=to_dtype,
        retain_folder_structure=True,
        filetype=filetype,
        overwrite=True,
        verbose=False,
    )
    generator.generate()
