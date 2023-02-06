"""Test the pytorch implementation of the beamformers.
"""
# pylint: disable=no-member
import sys
from pathlib import Path

import torch

from usbmd.datasets import DummyDataset
from usbmd.probes import Verasonics_l11_4v
from usbmd.pytorch_ultrasound.layers.beamformers import create_beamformer
from usbmd.pytorch_ultrasound.processing import on_device_torch
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.pixelgrid import make_pixel_grid_v2

# Add project folder to path to find config files
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

def test_das_beamforming():
    """
    Performs DAS beamforming on random data to verify that no errors occur. Does
    not check correctness of the output.
    """

    config = load_config_from_yaml(r'./tests/config_test.yaml')

    # Ensure DAS beamforming even if the config were to change
    config.model.type = 'das'
    config.data.dataset_name = 'picmus'
    config.data.n_angles = 1

    #probe = get_probe(config)
    probe = Verasonics_l11_4v(config)

    # Perform the beamforming on a small grid to ensure the test runs quickly
    grid = make_pixel_grid_v2(
        config.scan.xlims,
        config.scan.zlims,
        config.get('Nx', 64),
        config.get('Nz', 32))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    beamformer = create_beamformer(probe, grid, config)

    # Ensure reproducible results
    torch.random.manual_seed(0)

    dataset = DummyDataset()

    # Get dummy data from dataset and add batch dimension
    input_data = dataset[0][None]

    # Perform beamforming on device
    on_device_torch(beamformer, input_data, device=device, return_numpy=True)


if __name__ == '__main__':
    test_das_beamforming()
