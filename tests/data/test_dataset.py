"""Basic testing datasets"""

import numpy as np
import pytest

from zea.config import Config
from zea.config.validation import check_config
from zea.data.datasets import Dataset

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
