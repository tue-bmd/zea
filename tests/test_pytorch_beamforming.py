"""Test the pytorch implementation of the beamformers.
"""
# pylint: skip-file
import sys
import torch
from pathlib import Path

from usbmd.pytorch_ultrasound.layers.beamformers import create_beamformer
from usbmd.probes import Verasonics_l11_4v
from usbmd.datasets import DummyDataset
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.pixelgrid import make_pixel_grid_v2
from usbmd.processing import Process

# Add project folder to path to find config files
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

def test_das_beamforming(debug=False):
    """
    Performs DAS beamforming on random data to verify that no errors occur. Does
    not check correctness of the output.
    """

    config = load_config_from_yaml(r'./configs/config_picmus.yaml')

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

    beamformer = create_beamformer(probe, grid, config).to(device)

    # Ensure reproducible results
    torch.random.manual_seed(0)

    dataset = DummyDataset()

    # Get dummy data from dataset and add batch dimension
    input_data = dataset[0][None]

    # Convert to tensor
    input_data = torch.from_numpy(input_data).to(device)

    # Perform beamforming and convert to numpy array
    beamformer(input_data)['beamformed'].cpu().numpy()


if __name__ == '__main__':
    test_das_beamforming(debug=True)