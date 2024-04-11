"""Basic testing for ui / generate
"""
import shutil
import sys
from pathlib import Path
import pytest
import matplotlib.pyplot as plt
import numpy as np

from usbmd import ui
from usbmd.generate import GenerateDataSet
from usbmd.setup_usbmd import setup_config

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


plt.rcParams["backend"] = "agg"


def test_ui_initialization():
    """Test ui initialization function"""
    config = setup_config("./tests/config_test.yaml")
    config.ml_library = "torch"

    dataloader_ui = ui.DataLoaderUI(config)
    dataloader_ui.run()
    dataloader_ui.run(plot=True)

    config = setup_config("./tests/config_test.yaml")
    config.ml_library = "tensorflow"
    dataloader_ui = ui.DataLoaderUI(config)
    dataloader_ui.run()
    dataloader_ui.run(plot=True)


def test_get_data():
    """Test ui get_data function"""
    config = setup_config("./tests/config_test.yaml")
    dataloader_ui = ui.DataLoaderUI(config)
    data = dataloader_ui.get_data()
    assert data is not None
    assert isinstance(data, np.ndarray), "Data is not a numpy array"
    assert len(data.shape) == 4, "Data must be 4d (n_tx, n_el, n_ax, N_ch)"

@pytest.mark.parametrize(
    "ml_library, dtype, to_dtype, filetype",
    [
        ("torch", "raw_data", "image", "png"),
        ("torch", "beamformed_data", "image", "hdf5"),
        ("tensorflow", "raw_data", "image", "png"),
        ("tensorflow", "beamformed_data", "image", "hdf5"),
     ]
)
def test_generate(ml_library, dtype, to_dtype, filetype):
    """Test generate class"""
    config = setup_config("./tests/config_test.yaml")
    config.ml_library = ml_library
    config.data.dtype = dtype

    temp_folder = Path("./tests/temp")
    shutil.rmtree(temp_folder, ignore_errors=True)

    generator = GenerateDataSet(
        config,
        destination_folder=temp_folder,
        to_dtype=to_dtype,
        retain_folder_structure=True,
        filetype=filetype,
        overwrite=True,
    )
    generator.generate()
    shutil.rmtree(temp_folder)
