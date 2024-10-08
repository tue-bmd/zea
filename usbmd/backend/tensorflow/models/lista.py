"""Model and constructor for unfolded convolutional ISTA

Author(s)     : Ruud van Sloun
Modified by   : Nishith Chennakeshava
Date          : 03/02/2022
"""

# pylint: disable=abstract-method
import keras
from keras import Input, ops
from keras.layers import Add, Conv2D, UpSampling2D
from tensorflow.python.keras.layers import Layer

from usbmd.backend.tensorflow.layers.utils import get_activation
from usbmd.registry import post_processing_registry


@post_processing_registry(name="lista", framework="tensorflow")
def UnfoldingModel(
    input_dim,
    activation=None,
    folds=5,
    upsampling=1,
    P0_dim=None,
    P1_dim=None,
):
    """Unfolding Iterative Shrinking and Thresholding Model

    Args:
        input_dim (tuple): input dimensions
        activation (str, optional): final activation function.
            Defaults to None.
        folds (int, optional): number of folds. Defaults to 5.
        upsampling (int, optional): upsampling of output. Defaults to 1.
        P0_dim (list, optional): dimensions of conv kernels.
            Defaults to [1, 5].
        P1_dim (list, optional): dimensions of conv kernels.
            Defaults to [1, 5].

    Returns:
        tf model: unfolded LISTA model
    """
    if P0_dim is None:
        P0_dim = [1, 5]
    if P1_dim is None:
        P1_dim = [1, 5]

    inp = Input(shape=(input_dim[0], input_dim[1], 1))

    input_up = UpSampling2D(
        size=(upsampling, upsampling), interpolation="nearest", name="Upsample"
    )(inp)

    xP0 = Conv2D(
        P0_dim[0],
        (P0_dim[1], P0_dim[1]),
        activation=None,
        padding="same",
        strides=[1, 1],
        name="x_P0",
    )(input_up)

    x = xP0
    for k in range(1, folds):
        x_thresh = Prox(name=f"x_thresh_{k}")(x)
        x_thresh_P1 = Conv2D(
            P1_dim[0],
            (P1_dim[1], P1_dim[1]),
            activation=None,
            padding="same",
            strides=[1, 1],
            name=f"x_thresh_P1_{k}",
        )(x_thresh)

        xP0 = Conv2D(
            P0_dim[0],
            (P0_dim[1], P0_dim[1]),
            activation=None,
            padding="same",
            strides=[1, 1],
            name=f"x_P0_{k}",
        )(input_up)
        x = Add(name=f"x_{k}")([x_thresh_P1, xP0])

    x_thresh = Prox(name="S_out")(x)

    out = Conv2D(
        1, (1, 1), activation=None, padding="same", strides=[1, 1], name="SP1_out"
    )(x_thresh)
    out = get_activation(activation)(out)

    return keras.Model(inputs=inp, outputs=out)


class Prox(Layer):
    """Proximal layer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = None

    def build(self, input_shape):
        # pylint: disable=missing-function-docstring
        self.alpha = self.add_weight(
            shape=(1, 1), initializer="random_normal", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        # pylint: disable=missing-function-docstring,arguments-differ
        return ops.sign(inputs) * ops.relu(ops.abs(inputs) - ops.softplus(self.alpha))

    def compute_output_shape(self, input_shape):
        # pylint: disable=missing-function-docstring
        return input_shape
