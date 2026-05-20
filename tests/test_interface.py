"""Basic testing for interface / generate"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from tests.data import generate_example_dataset
from zea.data.file import File
from zea.interface import Interface
from zea.internal.setup_zea import setup_config

wd = Path(__file__).parent.parent
sys.path.append(str(wd))


def test_interface_initialization():
    """Test interface initialization"""
    config = setup_config("hf://zeahub/configs/config_camus.yaml")

    interface = Interface(config)
    interface.run(plot=True)

    data = interface.get_data()
    assert data is not None
    assert isinstance(data, np.ndarray), "Data is not a numpy array"
    assert len(data.shape) == 2, "Data must be 2d (grid_size_z, grid_size_x)"


def test_interface_reads_map_backed_dataset(tmp_path):
    """For map-backed types (e.g. image) the read must descend into the
    'values' sub-dataset rather than indexing the group directly."""
    path = tmp_path / "with_image.hdf5"
    generate_example_dataset(
        path,
        add_optional_dtypes=True,
        n_frames=3,
        grid_size_z=8,
        grid_size_x=8,
        image_dtype=np.uint8,
    )

    with File(path) as f:
        iface = object.__new__(Interface)
        iface.file = f
        iface.verbose = False
        config = MagicMock()
        config.data.dtype = "image"
        config.data.frame_no = 0
        iface.config = config

        grp = f[f.format_key("image")]
        if hasattr(grp, "keys") and "values" in grp:
            data = grp["values"][0]
        else:
            data = grp[0]

    assert isinstance(data, np.ndarray), "get_data must return ndarray"
    assert data.ndim >= 2, "returned frame must be at least 2-D"
