"""Basic testing datasets"""

import numpy as np
import pytest

from zea.config import Config
from zea.config.validation import check_config
from zea.data.datasets import Dataset
from zea.generate import GenerateDataSet
from zea.internal.setup_zea import setup_config

from .. import DUMMY_DATASET_N_FRAMES, DUMMY_DATASET_N_X, DUMMY_DATASET_N_Z


@pytest.mark.parametrize(
    "file_idx, idx, expected_shape",
    [
        (
            0,
            "all",
            (DUMMY_DATASET_N_FRAMES, DUMMY_DATASET_N_Z, DUMMY_DATASET_N_X),
        ),
        (
            -1,
            (1, 2, 3),
            (3, DUMMY_DATASET_N_Z, DUMMY_DATASET_N_X),
        ),
        (
            0,
            [1, 2, 3],
            (3, DUMMY_DATASET_N_Z, DUMMY_DATASET_N_X),
        ),
        (
            -1,
            np.array([1, 2, 3]),
            (3, DUMMY_DATASET_N_Z, DUMMY_DATASET_N_X),
        ),
        (
            0,
            slice(1, 3),
            (2, DUMMY_DATASET_N_Z, DUMMY_DATASET_N_X),
        ),
        (
            -1,
            [0, range(5)],
            (5, DUMMY_DATASET_N_X),
        ),
        (
            0,
            (np.array([1, 2]), slice(10)),
            (2, 10, DUMMY_DATASET_N_X),
        ),
    ],
)
def test_dataset_indexing(file_idx, idx, expected_shape, dummy_dataset_path):
    """Test ui initialization function"""
    config = {"data": {"dataset_folder": dummy_dataset_path, "dtype": "image"}}
    config = check_config(Config(config))
    dataset = Dataset.from_config(
        **config.data, search_file_tree_kwargs={"parallel": False, "verbose": False}
    )

    file = dataset[file_idx]
    data = file.load_data(config.data.dtype, idx)

    assert data.shape == expected_shape, (
        f"Data shape {data.shape} does not match expected shape {expected_shape}"
    )


@pytest.mark.parametrize(
    "dtype, to_dtype, filetype",
    [
        ("raw_data", "image", "png"),
        ("beamformed_data", "image", "hdf5"),
    ],
)
def test_generate(dtype, to_dtype, filetype, tmp_path, dummy_dataset_path):
    """Test generate class"""
    config = setup_config("./tests/config_test.yaml")
    config.data.dtype = dtype
    config.data.dataset_folder = dummy_dataset_path
    config.data.file_path = "test0.hdf5"

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
        destination_folder=tmp_path / "test",
        to_dtype=to_dtype,
        retain_folder_structure=True,
        filetype=filetype,
        overwrite=True,
        verbose=False,
        search_file_tree_kwargs={"parallel": False, "verbose": False},
        # jit_options=None, # uncomment for debugging
    )
    generator.generate()
