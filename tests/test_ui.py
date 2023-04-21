"""Basic testing for ui / generate
"""
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from usbmd import ui
from usbmd.generate import GenerateDataSet

plt.rcParams['backend'] = 'agg'

def test_ui_initialization():
    """Test ui initialization function"""
    config = ui.setup('./tests/config_test.yaml')
    config.ml_library = 'torch'

    dataloader_ui = ui.DataLoaderUI(config)
    dataloader_ui.run()

    config = ui.setup('./tests/config_test.yaml')
    config.ml_library = 'tensorflow'
    dataloader_ui = ui.DataLoaderUI(config)
    dataloader_ui.run()

def test_get_data():
    """Test ui get_data function"""
    config = ui.setup('./tests/config_test.yaml')
    dataloader_ui = ui.DataLoaderUI(config)
    data = dataloader_ui.get_data()
    assert data is not None
    assert isinstance(data, np.ndarray), 'Data is not a numpy array'
    assert len(data.shape) == 4, 'Data must be 4d (N_tx, N_el, N_ax, N_ch)'

def test_generate():
    """Test generate class"""
    config = ui.setup('./tests/config_test.yaml')
    config.ml_library = 'tensorflow'
    config.data.dtype = 'beamformed_data' # TODO: fix for raw_data

    temp_folder = Path('./tests/temp')

    generator = GenerateDataSet(
        config,
        to_dtype='image',
        destination_folder=temp_folder,
        retain_folder_structure=True,
        filetype='png',
        overwrite=True,
    )
    generator.generate()

    config.ml_library = 'torch'

    generator = GenerateDataSet(
        config,
        to_dtype='image',
        destination_folder=temp_folder,
        retain_folder_structure=True,
        filetype='hdf5',
        overwrite=True,
    )
    generator.generate()

    shutil.rmtree(temp_folder)
