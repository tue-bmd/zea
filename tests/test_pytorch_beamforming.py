"""Test the pytorch implementation of the beamformers.
"""
# pylint: disable=no-member
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from usbmd.probes import Verasonics_l11_4v
from usbmd.pytorch_ultrasound.layers.beamformers import create_beamformer
from usbmd.pytorch_ultrasound.processing import on_device_torch
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.pixelgrid import make_pixel_grid
from usbmd.utils.simulator import UltrasoundSimulator

# Add project folder to path to find config files
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

# def test_das_beamforming():
#     """
#     Performs DAS beamforming on random data to verify that no errors occur. Does
#     not check correctness of the output.
#     """

#     config = load_config_from_yaml(r'./tests/config_test.yaml')

#     # Ensure DAS beamforming even if the config were to change
#     config.model.type = 'das'
#     config.data.dataset_name = 'picmus'
#     config.data.n_angles = 1

#     #probe = get_probe(config)
#     probe = Verasonics_l11_4v(config)

#     # Perform the beamforming on a small grid to ensure the test runs quickly
#     grid = make_pixel_grid_v2(
#         config.scan.xlims,
#         config.scan.zlims,
#         config.get('Nx', 64),
#         config.get('Nz', 32))

#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#     beamformer = create_beamformer(probe, grid, config)

#     # Ensure reproducible results
#     torch.random.manual_seed(0)

#     dataset = DummyDataset()

#     # Get dummy data from dataset and add batch dimension
#     input_data = dataset[0][None]

#     # Perform beamforming on device
#     on_device_torch(beamformer, input_data, device=device, return_numpy=True)




def test_das_beamforming(debug=False, compare_gt=True):
    """Performs DAS beamforming on random data to verify that no errors occur. Does
    not check correctness of the output.

    Args:
        debug (bool, optional): Set to True to enable debugging options (plotting). Defaults to
        False. compare_gt (bool, optional): Set to True to compare against GT. Defaults to True.

    Returns:
        numpy array: beamformed output
    """


    config = load_config_from_yaml(r'./tests/config_test.yaml')
    config.ml_library = 'torch'
    probe = Verasonics_l11_4v(config)
    probe.N_ax = 2046
    wvln = probe.c/probe.fc
    grid = make_pixel_grid([-19e-3, 19e-3], [0, 63e-3], wvln/4, wvln/4)

    simulator = UltrasoundSimulator(probe, grid)
    beamformer = create_beamformer(probe, grid, config)

    # Ensure reproducible results
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Generate pseudorandom input tensor
    data = simulator.generate(200)

    inputs = np.expand_dims(data[0], axis=(1,-1))
    inputs = np.transpose(inputs, axes=(0,1,3,2,4))

    # Perform beamforming and convert to numpy array
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    outputs = on_device_torch(beamformer, inputs, device=device, return_numpy=True)

    # plot results
    if debug:
        fig, axs = plt.subplots(1,3)
        aspect_ratio = (data[1].shape[1]/data[1].shape[2])/(data[0].shape[1]/data[0].shape[2])
        axs[0].imshow(np.abs(inputs.squeeze().T), aspect=aspect_ratio)
        axs[0].set_title('RF data')
        axs[1].imshow(np.squeeze(outputs))
        axs[1].set_title('Beamformed')
        axs[2].imshow(cv2.GaussianBlur(data[1].squeeze(), (5,5), cv2.BORDER_DEFAULT))
        axs[2].set_title('Ground Truth')
        fig.show()

    y_true = cv2.GaussianBlur(data[1].squeeze(), (5,5), cv2.BORDER_DEFAULT)
    y_pred = np.squeeze(outputs)

    y_true = y_true/y_true.max()
    y_pred = y_pred/y_pred.max()

    MSE = np.mean(np.square(y_true-y_pred))
    print(f'MSE: {MSE}')

    if compare_gt:
        assert MSE < 0.01

    return y_pred

if __name__ == '__main__':
    test_das_beamforming(debug=True)
