"""Tests for the probes module."""

from pathlib import Path

import numpy as np
import pytest

from usbmd.config import Config
from usbmd.data.datasets import DataSet
from usbmd.probes import Probe, get_probe
from usbmd.registry import probe_registry


class DummyDataset(DataSet):
    """Dummy dataset."""

    def __init__(self, config=None, modtype="rf"):
        """
        Initializes dummy dataset which returns normally distributed random
        values.

        If no modtype nor config is provided the dataset will assume raw_data in rf
        modulated form of shape (75, 128, 2048, 1).

        If a config and modtype is supplied the ouptput depends on the dtype
        elements:
        - (dtype:raw_data, modtype:"rf") - (75, 128, 2048, 1)
        - (dtype:raw_data, modtype:"iq") - (75, 128, 2048, 2)
        - (dtype:beamformed_data) - (128, 356, 1)

        Args:
            config (Config, optional): config.data section from a config file.
            Defaults to None.
        """
        try:
            # Run baseclass init to define all the internal variables
            super().__init__(None, None, None, None)
        # The base init will raise an error because the __init__ method was
        # called with just None inputs, which is expected.
        except AssertionError:
            pass

        self.data_root = "dummy"
        self.file_paths = [Path(f"dummy/dummy_file_{i}.hdf5") for i in range(len(self))]
        # use raw rf data if no config is supplied
        if config is None:
            self.dtype = "raw_data"
        else:
            self.dtype = config.dtype

        if modtype == "rf":
            self.n_ch = 1
        elif modtype == "iq":
            self.n_ch = 2
        else:
            raise ValueError(
                f"Modulation type {modtype} not available for this dataset"
            )

        self.n_ax = 2048
        self.n_tx = 75
        self.Nz = 128

    @property
    def num_frames(self):
        """Return number of frames in current file."""
        return 2

    @property
    def total_num_frames(self):
        """Return total number of frames in dataset."""
        return 4

    @property
    def event_structure(self):
        """Whether the files in the dataset have an event structure."""
        return False

    def __len__(self):
        """
        Return number of files in dataset. The number is arbitrary for the dummy
        dataset because it can generate as much data as needed.
        """
        return 2

    def __getitem__(self, index):
        """Read file at index place in file_paths.

        Args:
            index (int): Index of which file in dataset to read.

        Raises:
            ValueError: Raises when filetype is not hdf5 or mat.

        Returns:
            file (h5py or mat): File container.

        """
        if isinstance(index, int):
            frame_no = None
        elif len(index) == 2:
            index, frame_no = index
        else:
            raise ValueError(
                "Index should either be an integer (indicating file index), "
                "or tuple containing file index and frame number!"
            )
        self.file = super().__getitem__(index)
        self.frame_no = self.get_frame_no(frame_no)

        self.file = self.get_file(index)
        if self.dtype == "raw_data":
            data = np.random.randn(
                self.num_frames, self.n_tx, self.n_ax, self.Nz, self.n_ch
            )
            return data[self.frame_no]

        elif self.dtype == "beamformed_data":
            data = np.random.randn(self.num_frames, self.Nz, 256, 1)
            return data[self.frame_no]

        else:
            raise ValueError(f"Data type {self.dtype} not available for this dataset")

    def get_file(self, index: int):
        """Returns fake file object because dummy dataset has no files."""
        file = {}
        file["attrs"] = {}
        file["attrs"]["probe"] = "generic"
        file["attrs"]["description"] = "Dummy dataset"
        file = Config(file)
        return file

    def get_default_scan_parameters(self):
        """Returns placeholder default scan parameters."""
        probe_parameters = get_probe(self.get_probe_name()).get_parameters()
        probe_parameters["n_ax"] = self.n_ax
        probe_parameters["n_tx"] = self.n_tx
        probe_parameters["Nz"] = self.Nz

        scan_parameters = {
            "angles": np.linspace(-0.27925268, 0.27925268, 75),
            **probe_parameters,
        }
        return scan_parameters

    # pylint: disable=unused-argument
    def get_probe_parameters_from_file(self, file=None, event=None):
        """Returns placeholder probe parameters."""
        return get_probe(self.get_probe_name()).get_parameters()

    # pylint: disable=unused-argument
    def get_scan_parameters_from_file(self, file=None, event=None):
        """Returns placeholder scan parameters."""
        return self.get_default_scan_parameters()


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_get_probe(probe_name):
    """Tests the get_probe function by calling it on all registered probes and
    checking that it returns a probe object."""
    probe = get_probe(probe_name)

    assert isinstance(probe, Probe), "get_probe must return a Probe object"


def test_get_probe_error():
    """Tests the get_probe function by calling it on a probe name that is not
    registered and checking that it raises a NotImplementedError."""
    with pytest.raises(NotImplementedError):
        get_probe("nonexistent_probe", fallback=False)


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_get_default_scan_paramters(probe_name):
    """Tests the get_probe function by calling it on all registered probes and
    calling their get_parameters() method."""
    probe = get_probe(probe_name)

    probe.get_parameters()


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_probe_attributes(probe_name):
    """Tests the get_probe function by calling it on all registered probes and
    checking that
    1. the element positions are of the correct shape
    2. the probe type is either 'linear' or 'phased'
    """
    if probe_name == "generic":
        dataset = DummyDataset()
        probe_params = dataset.get_probe_parameters_from_file()
        probe = get_probe(probe_name, **probe_params)
    else:
        probe = get_probe(probe_name)

    assert isinstance(
        probe.probe_geometry, np.ndarray
    ), "Element positions must be a numpy array"
    assert probe.probe_geometry.shape == (
        probe.n_el,
        3,
    ), "Element positions must be of shape (n_el, 3)"
    # assert probe.bandwidth is not None, 'Probe must have a bandwidth'
    assert probe.probe_type in ("linear", "phased"), (
        "Probe type must be" ' either "linear" or' ' "phased"'
    )
