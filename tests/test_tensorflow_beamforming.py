"""Test the tensorflow implementation of the beamformers.
"""
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

from usbmd.tensorflow_ultrasound.layers.beamformers_v2 import create_beamformer
from usbmd.probes import get_probe
from usbmd.datasets import PICMUS
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.pixelgrid import make_pixel_grid_v2
from usbmd.processing import Process

# Add project folder to path to find config files
wd = Path(__file__).parent.parent
sys.path.append(str(wd))






def test_das_beamforming():
    """
    Performs DAS beamforming on random data to verify that no errors occur. Does
    not check correctness of the output.
    """

    config = load_config_from_yaml(r'./configs/config_picmus.yaml')

    # Ensure DAS beamforming even if the config were to change
    config.model.type = 'das'

    probe = get_probe(config)

    # Perform the beamforming on a small grid to ensure the test runs quickly
    grid = make_pixel_grid_v2(
        config.scan.xlims,
        config.scan.zlims,
        config.get('Nx', 64),
        config.get('Nz', 32))

    beamformer = create_beamformer(probe, grid, config)

    # Ensure reproducible results
    tf.random.set_seed(0)

    # Generate pseudorandom input tensor
    input_data = np.random.rand((1, 75, 128, 3328, 1))

    # Perform beamforming and convert to numpy array
    beamformer(input_data)['beamformed'].cpu().numpy()


if __name__ == '__main__':
    test_das_beamforming()
