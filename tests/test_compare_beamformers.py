"""Compares the outputs of the beamformers
"""

import numpy as np

from tests.test_pytorch_beamforming import test_das_beamforming as test_torch
from tests.test_tensorflow_beamforming import test_das_beamforming as test_tf


def test_compare_beamformers():
    """Compares Torch and Tensorflow outputs, and checks if they are (almost) equal"""

    output_torch = test_torch()
    output_tf = test_tf()
    MSE = np.mean(np.square(output_torch-output_tf))
    print(f'MSE: {MSE}')
    np.testing.assert_almost_equal(output_torch, output_tf, decimal=4)

if __name__ == '__main__':
    test_compare_beamformers()
