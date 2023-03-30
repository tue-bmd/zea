"""Test the tf implementation of the beamformers.
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from usbmd.probes import Verasonics_l11_4v
from usbmd.scan import PlaneWaveScan
from usbmd.tensorflow_ultrasound.layers.beamformers import create_beamformer
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.pixelgrid import make_pixel_grid
from usbmd.utils.simulator import UltrasoundSimulator

# Add project folder to path to find config files
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

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
    config.ml_library = 'tensorflow'

    probe = Verasonics_l11_4v()
    probe_parameters = probe.get_default_scan_parameters()
    scan = PlaneWaveScan(N_tx=1,
                         xlims=(-19e-3, 19e-3),
                         zlims=(0, 63e-3),
                         N_ax=2046,
                         fs=probe_parameters['fs'],
                         fc=probe_parameters['fc'],
                         angles=np.array([0,]))

    scan.grid = make_pixel_grid(scan.xlims, scan.zlims, scan.wvln/4, scan.wvln/4)
    simulator = UltrasoundSimulator(probe, scan)
    beamformer = create_beamformer(probe, scan, config)

    # Ensure reproducible results
    tf.random.set_seed(0)
    np.random.seed(0)

    # Generate pseudorandom input tensor
    data = simulator.generate(200)

    inputs = np.expand_dims(data[0], axis=(1,-1))
    inputs = np.transpose(inputs, axes=(0,1,3,2,4))

    # Perform beamforming and convert to numpy array
    outputs = beamformer(inputs)

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
    else:
        return y_pred


if __name__ == '__main__':
    test_das_beamforming(debug=True)
