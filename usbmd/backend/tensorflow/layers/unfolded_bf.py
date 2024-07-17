"""

- **Author(s)**     : Ben Luijten
- **Date**          : Thu Feb 2 2021
"""

# It is TF convention to define layers in the build method
# pylint: disable=attribute-defined-outside-init, arguments-differ, unused-argument

# Either we add **kwargs to the methods, which then trigger an ignored-argument warning. Or we,
# do not add them, and we get an arguments-differ warning. Disabling the latter seemed to be the
# better option.

import numpy as np
import tensorflow as tf
import tf_keras as keras
import tf_keras.backend as K
from tf_keras.layers import Conv2D, Input, Lambda
from tf_keras.models import Model
from tf_keras.regularizers import l2

from usbmd.backend.tensorflow.layers.beamformers import BeamSumming
from usbmd.backend.tensorflow.losses import SMSLE
from usbmd.registry import tf_beamformer_registry


def NormalizeLayer():
    """Function that normalizes the input tensor based on its max value"""
    return Lambda(lambda x: x / K.max(K.abs(x) + K.epsilon(), axis=(1, 2, 3)))


def regularizer_unity(activity):
    """Regularizer that promotes unity gain (sum to 1) across the last dimension"""
    return 1 * tf.reduce_mean(tf.square(tf.reduce_sum(activity, axis=-1) - 1))


def antirect(x):
    """Antirectifier activation"""
    mean, _ = tf.nn.moments(x, axes=-1, keepdims=True)
    x = tf.math.l2_normalize(x - mean, axis=-1)
    return tf.nn.crelu(x)


class Prox(keras.layers.Layer):
    """Proximal operator layer"""

    def __init__(self, proxtype=0, **kwargs):
        """proxtype -> 0: soft thresholding
        1: smooth sigmoid-based soft thresholding
        2: positive soft threshold
        3: double tanh
        4: complex soft thresholding
        """
        super().__init__(**kwargs)
        self.proxtype = proxtype

    def build(self, input_shape):
        """Builds the model"""
        self.alpha = self.add_weight(
            name="kernel", shape=(1, 1), initializer="glorot_uniform", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        """Function call"""
        x = inputs

        if self.proxtype in (0, "softthres"):
            return K.sign(x) * K.relu(K.abs(x) - K.softplus(self.alpha))
        elif self.proxtype == 1:
            tau = 1
            return x / (1 + K.exp(-tau * (K.abs(x) - self.alpha)))
        elif self.proxtype == 2:
            return K.relu(x - K.softplus(self.alpha))
        elif self.proxtype == 3:
            return K.tanh(x + self.alpha) + K.tanh(x - self.alpha)
        elif self.proxtype == 4:
            x = tf.cast(x, "complex64")
            return K.exp(tf.complex(0.0, tf.math.angle(x))) * tf.cast(
                K.relu(K.abs(x) - K.softplus(self.alpha)), "complex64"
            )

    def get_config(self):
        """Returns the config, needed for serializing the model to JSON (during saving)"""
        config = super().get_config().copy()
        config.update(
            {
                "proxtype": self.proxtype,
            }
        )
        return config


def fmodel(input_shape, channels, kernel_size, activation):
    """Function that creates a model for predicting apodization weights"""

    inputs = Input(shape=(input_shape[1:]), batch_size=input_shape[0])

    # Merge channel dimensions (RF/IQ) with elements dimension
    inputs_merged = tf.concat(tf.unstack(inputs, axis=-1), -1)

    lay0 = Conv2D(channels, kernel_size, activation=activation, padding="same")(
        inputs_merged
    )

    lay1 = Conv2D(channels / 4, kernel_size, activation=activation, padding="same")(
        lay0
    )

    lay2 = Conv2D(channels / 4, kernel_size, activation=activation, padding="same")(
        lay1
    )

    lay3 = Conv2D(
        channels,
        kernel_size,
        activation=None,
        activity_regularizer=regularizer_unity,
        padding="same",
    )(lay2)

    outputs_merged = lay3

    # Unmerge channel dimension
    outputs = tf.stack(tf.split(outputs_merged, input_shape[-1], -1), axis=-1)

    return Model(inputs=inputs, outputs=outputs)


class simpleUnet(keras.layers.Layer):
    """Simple U-net model"""

    def __init__(self, n_layers=4):
        super().__init__()
        self.layers = []

        for _ in range(n_layers):
            self.layers.append(
                Conv2D(
                    2,
                    (11, 11),
                    padding="same",
                    activation=antirect,
                    kernel_regularizer=l2(1e-3),
                )
            )

        self.layers.append(Conv2D(1, (1, 1), padding="same", activation=None))

    def call(self, inputs):
        """Forward pass"""

        x = inputs
        x = tf.squeeze(x, axis=-2)  # Squeeze the elements dimension

        for layer in self.layers:
            x = layer(x)

        x = tf.expand_dims(x, axis=-2)  # Add channel dimension back

        return x


class WaveletProx(keras.layers.Layer):
    """Wavelet proximal operator layer"""

    def __init__(self, fold_nr, **kwargs):
        super().__init__(**kwargs)

        self.fold_nr = fold_nr
        self.f = simpleUnet(n_layers=2)
        self.f_inv = simpleUnet(n_layers=2)
        self.prox = Prox(proxtype=1)

        self.loss_rate = 0.1

    def call(self, inputs):
        """Forward pass"""
        x = inputs
        z1 = self.f(x)
        z2 = self.prox(z1)
        x_out = self.f_inv(z2)

        x_hat = self.f_inv(z1)

        # Add loss function to force x ~= f(f_inv(x))
        symmetry_loss = tf.reduce_sum(tf.square(inputs - x_hat))
        tf.summary.scalar(f"sym_loss_{self.fold_nr}", symmetry_loss)
        self.add_loss(self.loss_rate * symmetry_loss)

        return x_out


class FourierProx(keras.layers.Layer):
    """Fourier domain proximal operator layer"""

    def __init__(self):
        super().__init__()
        self.prox = Prox(proxtype=4)

    def call(self, inputs):
        """Forward pass"""
        x = inputs
        x = tf.transpose(x, perm=(0, 3, 1, 2))
        x = tf.signal.rfft2d(x)
        x = self.prox(x)
        x = tf.signal.irfft2d(x)
        x = tf.transpose(x, perm=(0, 2, 3, 1))
        x = tf.math.real(x)
        return x


@tf_beamformer_registry(name="neural_map", framework="tensorflow")
class NeuralMAP(BeamSumming):
    """Layer that implements a deep learning based MAP beamformer by unfolding"""

    def __init__(
        self,
        folds=4,
        kernel_size=3,
        proxtype="softthres",
        IQ=False,
        env=False,
        intermediate_outputs=True,
        shared_prox=False,
        end_with_prox=False,
        x_init=None,
        embodiment=2,
        **kwargs,
    ):
        """Initializes the unfolded beamformer layer

        Args:
            folds (int, optional): Number of DC + Prox steps. Defaults to 4.
            kernel_size (int, optional): Kernel size of the convolutional layers. Defaults to 3.
            proxtype (str, optional): Type of proximal operator. Defaults to 'softthres'.
            IQ (bool, optional): True if processing IQ data. Defaults to False.
            env (bool, optional): True if working with envelope detected targets. Defaults to False.
            intermediate_outputs (bool, optional): If true, the intermediate outputs of the unfolded
            beamformer are returned. Defaults to True.
            shared_prox (bool, optional): If True, the same proximal operator weights are shared
            across all folds. Defaults to False.
            end_with_prox (bool, optional): If True, the unfolded beamformer ends with a proximal
            step
            x_init (str, optional): Initial guess for x0. If set to 'DAS', the DAS beamformed output
            is used as initial guess. Otherwise, the initial guess is set to zero. Defaults to None.
            embodiment (int, optional): Sets a specific embodiment of the unfolded beamformer.
            Defaults to 2.
        """
        super().__init__(**kwargs)
        self.folds = folds
        self.kernel_size = kernel_size
        self.proxtype = proxtype

        self.intermediate_outputs = intermediate_outputs
        self.shared_prox = shared_prox
        self.end_with_prox = end_with_prox
        self.x_init = x_init
        self.embodiment = embodiment

        self.IQ = IQ  # True if processing IQ data
        self.env = env  # True if working with envelope detected targets

    def get_prox(self, fold_nr):
        """Select proximal operator based on proxtype"""
        if self.proxtype == "neural":
            return simpleUnet()
        elif self.proxtype == "fourier":
            return FourierProx()
        elif self.proxtype == "wavelet":
            return WaveletProx(fold_nr)
        elif isinstance(self.proxtype, int):
            return Prox(proxtype=self.proxtype)
        else:  # p(x)=1
            return Lambda(lambda x: x, name="identityProx")

    def build(self, input_shape):
        """Build the layer"""

        self.f = []
        self.prox = []

        for fold in range(self.folds):
            nodes = input_shape[-1] * input_shape[-2]
            # newshape = (input_shape[0],
            #             input_shape[1],
            #             input_shape[2],
            #             input_shape[3],
            #             nodes)
            _ = fmodel(input_shape, nodes, self.kernel_size, activation=antirect)
            self.f.append(_)

            if self.shared_prox and self.prox:
                self.prox.append(self.prox[-1])  # Reuse previous prox
            else:
                self.prox.append(self.get_prox(fold_nr=fold))

        self.output_layer = Conv2D(
            input_shape[-1], (1, 1), activation=None, padding="same"
        )

    def call(self, inputs, **kwargs):
        """Forward pass"""

        y = inputs

        outputs = {}

        # Initial guess for x0
        if self.x_init == "DAS":
            # Coherent compounding (axis 1) and element summing (axis -2)
            x = tf.reduce_mean(y, axis=[1, -2], keepdims=True)
        else:
            x = 0 * tf.reduce_mean(y, axis=[1, -2], keepdims=True)

        for i in range(0, self.folds):
            if self.intermediate_outputs and i > 0:
                # For env targets + IQ
                if self.env and self.IQ:
                    x_inter = tf.norm(x, axis=-1, keepdims=True)
                    x_inter = tf.squeeze(x_inter, axis=[1, -2, -1])
                outputs[f"x_{i}"] = x_inter

            # DC step-
            residual = tf.add(y, -x)  # Assuming a = 1 here

            if self.embodiment == 0:  # f() is neural network
                x_tilde = x + self.f[i](tf.concat(y, x, -1))

            elif self.embodiment == 1:
                x_tilde = x + self.f[i](tf.concat(y, residual), axis=-1)

            elif self.embodiment == 2:
                w = self.f[i](y)
                x_tilde = x + tf.reduce_sum(w * residual, axis=[1, -2], keepdims=True)

            elif self.embodiment == 3:
                w = self.f[i](residual)
                x_tilde = x + tf.reduce_sum(w * residual, axis=[1, -2], keepdims=True)

            else:
                raise NotImplementedError

            ##############################################################

            # Prox step
            if not self.end_with_prox and (i == self.folds - 1):
                x_hat = x_tilde
            else:
                x_hat = self.prox[i](x_tilde)

            # End of iteration
            x = x_hat

        x = tf.squeeze(x, axis=[1, -2])

        outputs["beamformed"] = x

        return outputs


@tf_beamformer_registry(name="able", framework="tensorflow")
class ABLE(NeuralMAP):
    """Adaptive Beamforming by Learning (ABLE)
    Since ABLE is a special case of NeuralMAP, we simply inherit from NeuralMAP"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            folds=1,
            kernel_size=1,
            proxtype=None,  # p(x)=1
            end_with_prox=False,  # No prox step
            **kwargs,
        )


# Test the functions
if __name__ == "__main__":
    input_shape = (1, 256, 256, 128, 2)  # (tx, X,Z,elements, RF/IQ)
    output_shape = (256, 256, 2)  # (X,Z, RF/IQ)

    # Create some dummy data
    y = np.random.rand(*input_shape)
    y = np.expand_dims(y, axis=0)  # Add batch dim
    y = np.nan_to_num(y)
    y = y / 2**15

    x = np.random.rand(*output_shape)
    x = np.expand_dims(x, axis=0)  # Add batch dim
    x = np.nan_to_num(x)
    x = x / 2**15

    # repeat x, and y N times
    N = 100
    y = np.repeat(y, N, axis=0)
    x = np.repeat(x, N, axis=0)

    inputs = keras.layers.Input(input_shape)
    outputs = NeuralMAP(4, (1, 1), intermediate_outputs=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    x_pred = model(np.expand_dims(y[0], 0))

    model.compile(loss=SMSLE(), optimizer="Adam", jit_compile=True)

    # Training test (y is input, x is output)
    _ = model.fit(y, x, batch_size=1, epochs=1000)
