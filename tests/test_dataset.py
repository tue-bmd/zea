"""Basic testing datasets
"""

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from usbmd.config import Config
from usbmd.config.validation import check_config
from usbmd.data import generate_usbmd_dataset, get_dataset
from usbmd.generate import GenerateDataSet
from usbmd.setup_usbmd import setup_config

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


plt.rcParams["backend"] = "agg"


@pytest.mark.parametrize(
    "file_idx, frame_idx",
    [
        (0, "all"),
        (-1, (1, 2, 3)),
        (0, [1, 2, 3]),
        (0, np.array([1, 2, 3])),
    ],
)
def test_dataset_indexing(file_idx, frame_idx):
    """Test ui initialization function"""
    temp_folder = Path("./temp/temp_dataset")
    # if temp_folder.exists():
    #     shutil.rmtree(temp_folder)

    for i in range(2):
        temp_file = temp_folder / f"test{i}.hdf5"
        dummy_data = np.random.rand(10, 20, 30)

        if not temp_file.exists():
            generate_usbmd_dataset(
                path=temp_file,
                image=dummy_data,
                probe_name="dummy",
                description="dummy dataset",
            )

    config = {"data": {"dataset_folder": str(temp_folder), "dtype": "image"}}
    config = check_config(Config(config))
    dataset = get_dataset(config.data)

    data = dataset[(file_idx, frame_idx)]

    if isinstance(frame_idx, (list, tuple, np.ndarray)):
        assert len(data) == len(
            frame_idx
        ), f"Data length {data.shape} does not match frame_idx length {len(frame_idx)}"
    elif frame_idx == "all":
        assert (
            len(data) == dataset.num_frames
        ), f"Data length {data.shape} does not match file length {dataset.num_frames}"


@pytest.mark.parametrize(
    "dtype, to_dtype, filetype",
    [
        ("raw_data", "image", "png"),
        ("beamformed_data", "image", "hdf5"),
    ],
)
def test_generate(dtype, to_dtype, filetype):
    """Test generate class"""
    config = setup_config("./tests/config_test.yaml")
    config.data.dtype = dtype

    # setting sum_transmits to False and operation_chain to None
    # that way we can use the default operation chain that
    # automatically checks the dtype and to_dtype
    config.model.beamformer.sum_transmits = False
    config.preprocess.operation_chain = None

    temp_folder = Path("./tests/temp")
    shutil.rmtree(temp_folder, ignore_errors=True)

    generator = GenerateDataSet(
        config,
        destination_folder=temp_folder,
        to_dtype=to_dtype,
        retain_folder_structure=True,
        filetype=filetype,
        overwrite=True,
        verbose=False,
    )
    generator.generate()
    shutil.rmtree(temp_folder)
