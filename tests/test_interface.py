"""Basic testing for interface / generate"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from zea.interface import Interface
from zea.internal.setup_zea import setup_config
from zea.internal.viewer import filename_from_window_dialog

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


def test_filename_from_window_dialog_no_tkinter():
    """Test that ImportError is raised when tkinter is not available."""
    with patch.dict(sys.modules, {"tkinter": None, "tkinter.filedialog": None}):
        with pytest.raises(ImportError, match="Tkinter"):
            filename_from_window_dialog()


def test_interface_initialization():
    """Test interface initialization"""
    config = setup_config("hf://zeahub/configs/config_camus.yaml")

    interface = Interface(config)
    interface.run(plot=True)

    data = interface.get_data()
    assert data is not None
    assert isinstance(data, np.ndarray), "Data is not a numpy array"
    assert len(data.shape) == 2, "Data must be 2d (grid_size_z, grid_size_x)"
