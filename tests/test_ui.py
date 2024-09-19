"""Basic testing for ui / generate
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from usbmd import __main__
from usbmd.setup_usbmd import setup_config

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


plt.rcParams["backend"] = "agg"


def test_ui_initialization():
    """Test ui initialization function"""
    config = setup_config("./tests/config_test.yaml")
    config.ml_library = "torch"

    dataloader_ui = __main__.DataLoaderUI(config)
    dataloader_ui.run()
    dataloader_ui.run(plot=True)

    config = setup_config("./tests/config_test.yaml")
    config.ml_library = "tensorflow"
    dataloader_ui = __main__.DataLoaderUI(config)
    dataloader_ui.run()
    dataloader_ui.run(plot=True)


def test_get_data():
    """Test ui get_data function"""
    config = setup_config("./tests/config_test.yaml")
    dataloader_ui = __main__.DataLoaderUI(config)
    data = dataloader_ui.get_data()
    assert data is not None
    assert isinstance(data, np.ndarray), "Data is not a numpy array"
    assert len(data.shape) == 4, "Data must be 4d (n_tx, n_el, n_ax, N_ch)"
