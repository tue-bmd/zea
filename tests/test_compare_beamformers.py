"""Compares the outputs of the beamformers
"""

import numpy as np

from tests.test_pytorch_beamforming import test_das_beamforming as test_torch
from tests.test_tensorflow_beamforming import test_das_beamforming as test_tf


def test_compare_beamformers():
    """Compares Torch and Tensorflow outputs, and checks if they are (almost) equal"""
    output_torch_pw = test_torch(compare_gt=False, reconstruction_mode="pw")
    output_torch_gen = test_torch(compare_gt=False, reconstruction_mode="generic")
    output_tf_pw = test_tf(compare_gt=False, reconstruction_mode="pw")
    output_tf_gen = test_tf(compare_gt=False, reconstruction_mode="generic")

    MSE = np.mean(np.square(output_torch_pw - output_tf_pw))
    print(f"MSE: {MSE}")
    assert MSE < 1e-9
    np.testing.assert_almost_equal(output_torch_pw, output_tf_pw, decimal=2)
    np.testing.assert_almost_equal(output_torch_gen, output_tf_gen, decimal=2)
    np.testing.assert_almost_equal(output_torch_pw, output_torch_gen, decimal=2)
    np.testing.assert_almost_equal(output_tf_pw, output_tf_gen, decimal=2)


if __name__ == "__main__":
    test_compare_beamformers()
