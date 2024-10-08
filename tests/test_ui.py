"""Basic testing for ui / generate
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from usbmd import ui
from usbmd.processing import set_backend
from usbmd.setup_usbmd import setup_config

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


plt.rcParams["backend"] = "agg"


@pytest.mark.parametrize(
    "ml_library",
    [
        "numpy",
        "jax",
        "torch",
        "tensorflow",
    ],
)
def test_ui_initialization(ml_library):
    """Test ui initialization function"""
    config = setup_config("./tests/config_test.yaml")
    config.ml_library = ml_library
    set_backend(config.ml_library)

    dataloader_ui = ui.DataLoaderUI(config)
    dataloader_ui.run()
    dataloader_ui.run(plot=True)


def test_get_data():
    """Test ui get_data function"""
    config = setup_config("./tests/config_test.yaml")
    set_backend(config.ml_library)

    dataloader_ui = ui.DataLoaderUI(config)
    data = dataloader_ui.get_data()
    assert data is not None
    assert isinstance(data, np.ndarray), "Data is not a numpy array"
    assert len(data.shape) == 4, "Data must be 4d (n_tx, n_el, n_ax, N_ch)"
