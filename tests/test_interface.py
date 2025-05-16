"""Basic testing for interface / generate"""

import sys
from pathlib import Path

import numpy as np

from usbmd.interface import Interface
from usbmd.internal.setup_usbmd import setup_config

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


def test_interface_initialization(dummy_dataset_path):
    """Test interface initialization"""
    config = setup_config("./tests/config_test.yaml")
    config.update_recursive({"data": {"dataset_folder": dummy_dataset_path}})

    interface = Interface(config)
    interface.run()
    interface.run(plot=True)


def test_get_data(dummy_dataset_path):
    """Test interface get_data function"""
    config = setup_config("./tests/config_test.yaml")
    config.update_recursive({"data": {"dataset_folder": dummy_dataset_path}})

    interface = Interface(config)
    data = interface.get_data()
    assert data is not None
    assert isinstance(data, np.ndarray), "Data is not a numpy array"
    assert len(data.shape) == 4, "Data must be 4d (n_tx, n_el, n_ax, N_ch)"
