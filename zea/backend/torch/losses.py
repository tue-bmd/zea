"""Container for custom loss functions."""

import torch
from torch import nn


class SMSLE(nn.Module):
    """Loss function for calculating the Signed-Mean-Squared-Logarithmic-Error. This loss function
    calculates the the mean squared error on log-scaled data, and then takes the sign of the
    difference between the predicted and ground truth values into account.
    See https://doi.org/10.1109/TMI.2020.3008537 for more information.
    """

    def __init__(self, dynamic_range=60):
        super().__init__()
        self.dynamic_range = dynamic_range

    def forward(self, y_true, y_pred):
        """
        Args:
            y_true (tensor): Ground truth values.
            y_pred (tensor): The predicted values.
        returns:
            loss (tensor): SMSLE loss value.
        """

        y_pred_max = torch.max(torch.abs(y_pred))
        y_true_max = torch.max(torch.abs(y_true))

        first_log_pos = torch.clamp(
            20
            * torch.log(
                torch.clamp(y_pred / y_pred_max, min=torch.finfo(torch.float32).eps)
                + 0.0
            )
            / torch.log(torch.tensor(10.0)),
            -self.dynamic_range,
            0,
        )
        secon_log_pos = torch.clamp(
            20
            * torch.log(
                torch.clamp(y_true / y_true_max, min=torch.finfo(torch.float32).eps)
                + 0.0
            )
            / torch.log(torch.tensor(10.0)),
            -self.dynamic_range,
            0,
        )

        first_log_neg = torch.clamp(
            20
            * torch.log(
                torch.clamp(-y_pred / y_pred_max, min=torch.finfo(torch.float32).eps)
                + 0.0
            )
            / torch.log(torch.tensor(10.0)),
            -self.dynamic_range,
            0,
        )
        secon_log_neg = torch.clamp(
            20
            * torch.log(
                torch.clamp(-y_true / y_true_max, min=torch.finfo(torch.float32).eps)
                + 0.0
            )
            / torch.log(torch.tensor(10.0)),
            -self.dynamic_range,
            0,
        )

        loss = 0.5 * torch.mean(
            torch.square(first_log_pos - secon_log_pos)
        ) + 0.5 * torch.mean(torch.square(first_log_neg - secon_log_neg))

        return loss
