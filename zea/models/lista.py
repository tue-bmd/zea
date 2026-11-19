"""Unfolded convolutional ISTA (LISTA).

LISTA (`Gregor and LeCun, 2010 <https://dl.acm.org/doi/10.5555/3104322.3104374>`_)
unrolls a fixed number of iterations of the Iterative Shrinkage and Thresholding
Algorithm (ISTA) into a neural network, where the measurement and reconstruction
operators of every iteration (fold) are learned convolutions and the
soft-thresholding step is a learned proximal operator.

.. doctest::

    >>> import numpy as np
    >>> from zea.models.lista import LISTA

    >>> model = LISTA(input_shape=(32, 32, 1), folds=3)
    >>> model(np.zeros((1, 32, 32, 1))).shape
    (1, 32, 32, 1)

"""

import keras
from keras import layers, ops

from zea.internal.registry import model_registry
from zea.models.base import BaseModel


@keras.saving.register_keras_serializable(package="zea")
class Prox(layers.Layer):
    """Proximal operator of the L1 norm with a learned threshold.

    Applies soft-thresholding, ``sign(x) * relu(|x| - threshold)``, where the
    threshold is ``softplus(alpha)`` with ``alpha`` a single learned weight. The
    softplus keeps the threshold positive without constraining the weight itself.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = None

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(1, 1),
            initializer="random_normal",
            trainable=True,
            name="alpha",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Apply the proximal operator.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Soft-thresholded tensor, with the same shape as `inputs`.
        """
        return ops.sign(inputs) * ops.relu(ops.abs(inputs) - ops.softplus(self.alpha))

    def compute_output_shape(self, input_shape):
        return input_shape


@model_registry(name="lista")
class LISTA(BaseModel):
    """Unfolded Iterative Shrinkage and Thresholding model."""

    def __init__(
        self,
        input_shape,
        folds=5,
        upsampling=1,
        filters=1,
        kernel_size=5,
        activation=None,
        name="lista",
        **kwargs,
    ):
        """Initialize a LISTA model.

        Args:
            input_shape (tuple): Input shape ``(height, width, channels)``.
            folds (int, optional): Number of unfolded ISTA iterations. Defaults to 5.
            upsampling (int, optional): Upsampling factor of the output relative to
                the input. Defaults to 1 (no upsampling).
            filters (int, optional): Number of filters in the unfolded convolutions.
                Defaults to 1.
            kernel_size (int, optional): Kernel size of the unfolded convolutions.
                Defaults to 5.
            activation (str, optional): Final activation function, resolved with
                :func:`keras.activations.get`. Defaults to None (linear).
            name (str, optional): Model name. Defaults to ``"lista"``.
        """
        super().__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.folds = folds
        self.upsampling = upsampling
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

        self.network = get_lista_network(
            input_shape=input_shape,
            folds=folds,
            upsampling=upsampling,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "folds": self.folds,
                "upsampling": self.upsampling,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
            }
        )
        return config

    def call(self, *args, **kwargs):
        return self.network(*args, **kwargs)


def get_lista_network(
    input_shape,
    folds=5,
    upsampling=1,
    filters=1,
    kernel_size=5,
    activation=None,
):
    """Build the unfolded ISTA architecture.

    Args:
        input_shape (tuple): Input shape ``(height, width, channels)``.
        folds (int, optional): Number of unfolded ISTA iterations. Defaults to 5.
        upsampling (int, optional): Upsampling factor of the output. Defaults to 1.
        filters (int, optional): Number of filters in the unfolded convolutions.
            Defaults to 1.
        kernel_size (int, optional): Kernel size of the unfolded convolutions.
            Defaults to 5.
        activation (str, optional): Final activation function. Defaults to None.

    Returns:
        keras.Model: The unfolded LISTA model.
    """
    assert len(input_shape) == 3, "input_shape must be a tuple of (height, width, channels)"
    assert folds >= 1, "folds must be at least 1"

    def conv(name):
        return layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            padding="same",
            name=name,
        )

    inputs = keras.Input(shape=input_shape)

    measurements = layers.UpSampling2D(
        size=(upsampling, upsampling), interpolation="nearest", name="upsample"
    )(inputs)

    # First fold: no previous estimate to threshold yet.
    x = conv(name="x_p0_0")(measurements)

    for fold in range(1, folds):
        x_thresh = Prox(name=f"x_thresh_{fold}")(x)
        x_thresh = conv(name=f"x_thresh_p1_{fold}")(x_thresh)
        x = layers.Add(name=f"x_{fold}")([x_thresh, conv(name=f"x_p0_{fold}")(measurements)])

    x = Prox(name="s_out")(x)
    outputs = layers.Conv2D(
        input_shape[-1],
        (1, 1),
        activation=activation,
        padding="same",
        name="sp1_out",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="lista_network")
