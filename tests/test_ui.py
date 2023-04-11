"""Basic testing
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from usbmd import ui

def test_ui_initialization():
    """Test ui initialization function"""
    config = ui.setup('./tests/config_test.yaml')
    dataloader_ui = ui.DataLoaderUI(config)
    dataloader_ui.run()
    plt.close()

def test_get_data():
    """Test ui get_data function"""
    config = ui.setup('./tests/config_test.yaml')
    dataloader_ui = ui.DataLoaderUI(config)
    data = dataloader_ui.get_data()
    assert data is not None
    assert isinstance(data, np.ndarray), 'Data is not a numpy array'
    assert len(data.shape) == 4, 'Data must be 4d (N_tx, N_el, N_ax, N_ch)'
