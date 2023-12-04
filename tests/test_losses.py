""" Module for testing loss functions """

import numpy as np

from usbmd.tensorflow_ultrasound.losses import SMSLE


def test_smsle():
    """Test SMSLE loss function"""
    # Create random y_true and y_pred data
    y_true = np.random.rand(1, 11, 128, 512, 2).astype(np.float32)
    y_pred = np.random.rand(1, 11, 128, 512, 2).astype(np.float32)

    # Calculate SMSLE loss
    smsle = SMSLE()
    loss = smsle(y_true, y_pred)

    # Check if loss is a scalar
    assert loss.shape == ()


if __name__ == "__main__":
    test_smsle()
