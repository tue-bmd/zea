"""Test the tf implementation of the beamformers.
"""
# pylint: skip-file
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import pytest
import cv2

from usbmd.tensorflow_ultrasound.layers.beamformers import create_beamformer
from usbmd.probes import get_probe, Verasonics_l11_4v
from usbmd.datasets import PICMUS
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.pixelgrid import make_pixel_grid, make_pixel_grid_v2
from usbmd.processing import Process
from usbmd.utils.simulator import Ultrasound_Simulator

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

    probe = Verasonics_l11_4v(config)
    probe.N_ax = 2046
    wvln = probe.c/probe.fc
    grid = make_pixel_grid([-19e-3, 19e-3], [0, 63e-3], wvln/4, wvln/4)

    simulator = Ultrasound_Simulator(probe, grid)
    
    beamformer = create_beamformer(probe, grid, config)

    # Ensure reproducible results
    tf.random.set_seed(0)
    np.random.seed(0)

    # Generate pseudorandom input tensor
    data = simulator.generate(20)

    input = np.expand_dims(data[0], axis=(1,-1))
    input = np.transpose(input, axes=(0,1,3,2,4))

    # Perform beamforming and convert to numpy array
    output = beamformer(input)


    # plot results
    if debug:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,3)

        aspect_ratio = (data[1].shape[1]/data[1].shape[2])/(data[0].shape[1]/data[0].shape[2])
        axs[0].imshow(np.abs(input.squeeze().T), aspect=aspect_ratio)
        axs[0].set_title('RF data') 
        axs[1].imshow(np.squeeze(output))
        axs[1].set_title('Beamformed')
        axs[2].imshow(cv2.GaussianBlur(data[1].squeeze(), (5,5), cv2.BORDER_DEFAULT))
        axs[2].set_title('Ground Truth')
        fig.show()

    y_true = cv2.GaussianBlur(data[1].squeeze(), (5,5), cv2.BORDER_DEFAULT)
    y_pred = np.squeeze(output)

    y_true = y_true/y_true.max()
    y_pred = y_pred/y_pred.max()

    MSE = np.mean(np.square(y_true-y_pred))

    assert MSE < 0.001


if __name__ == '__main__':
    test_das_beamforming(debug=True)