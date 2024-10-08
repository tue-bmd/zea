"""Container for custom loss functions.

- **Author(s)**     : Ben Luijten
- **Date**          : -
"""

import keras
from keras import ops


class SMSLE(keras.losses.Loss):
    """Loss function for calculating the Signed-Mean-Squared-Logarithmic-Error. This loss function
    calculates the the mean squared error on log-scaled data, and then takes the sign of the
    difference between the predicted and ground truth values into account.
    See https://doi.org/10.1109/TMI.2020.3008537 for more information.
    """

    def __init__(self, dynamic_range=60, name="smsle", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dynamic_range = dynamic_range

    def call(self, y_true, y_pred):
        """
        Args:
            y_true (tensor): Ground truth values.
            y_pred (tensor): The predicted values.
        returns:
            loss (tensor): SMSLE loss value.
        """

        y_pred_max = ops.maximum(ops.max(ops.abs(y_pred)), keras.config.epsilon())
        y_true_max = ops.maximum(ops.max(ops.abs(y_true)), keras.config.epsilon())

        first_log_pos = ops.clip(
            20
            * ops.log(ops.clip(y_pred / y_pred_max, keras.config.epsilon(), 1) + 0.0)
            / ops.log(10),
            -self.dynamic_range,
            0,
        )
        secon_log_pos = ops.clip(
            20
            * ops.log(ops.clip(y_true / y_true_max, keras.config.epsilon(), 1) + 0.0)
            / ops.log(10),
            -self.dynamic_range,
            0,
        )

        first_log_neg = ops.clip(
            20
            * ops.log(ops.clip(-y_pred / y_pred_max, keras.config.epsilon(), 1) + 0.0)
            / ops.log(10),
            -self.dynamic_range,
            0,
        )
        secon_log_neg = ops.clip(
            20
            * ops.log(ops.clip(-y_true / y_true_max, keras.config.epsilon(), 1) + 0.0)
            / ops.log(10),
            -self.dynamic_range,
            0,
        )

        loss = 0.5 * ops.mean(
            ops.square(first_log_pos - secon_log_pos)
        ) + 0.5 * ops.mean(ops.square(first_log_neg - secon_log_neg))

        return loss
