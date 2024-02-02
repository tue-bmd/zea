"""Compares the outputs of the beamformers
"""

import numpy as np

from .test_pytorch_beamforming import test_das_beamforming as test_torch
from .test_tensorflow_beamforming import test_das_beamforming as test_tf


def test_compare_beamformers():
    """Compares Torch and Tensorflow outputs, and checks if they are (almost) equal"""
    output_torch_pw = test_torch(reconstruction_mode="pw", compare_gt=False)
    output_torch_gen = test_torch(reconstruction_mode="generic", compare_gt=False)
    output_tf_pw = test_tf(reconstruction_mode="pw", patches=None, compare_gt=False)
    output_tf_gen = test_tf(
        reconstruction_mode="generic", patches=None, compare_gt=False
    )

    MSE = np.mean(np.square(output_torch_pw - output_tf_pw))
    print(f"MSE: {MSE}")
    assert MSE < 1e-9
    np.testing.assert_almost_equal(output_torch_pw, output_tf_pw, decimal=2)
    np.testing.assert_almost_equal(output_torch_gen, output_tf_gen, decimal=2)
    np.testing.assert_almost_equal(output_torch_pw, output_torch_gen, decimal=2)
    np.testing.assert_almost_equal(output_tf_pw, output_tf_gen, decimal=2)


if __name__ == "__main__":
    test_compare_beamformers()
