"""
WARNNING: This code is not fully tested yet. It is only a draft.

- **Author(s)**     : Ben Luijten
- **Date**          : Thu Apr 20 2023
"""

# It is TF convention to define layers in the build method
# pylint: disable=attribute-defined-outside-init

# Either we add **kwargs to the methods, which then trigger an ignored-argument warning. Or we,
# do not add them, and we get an arguments-differ warning. Disabling the latter seemed to be the
# better option.
# pylint: disable=arguments-differ
# pylint: disable=unused-variable

import tensorflow as tf

from usbmd.backend.tensorflow.layers.beamformers import BeamSumming
from usbmd.registry import tf_beamformer_registry


@tf_beamformer_registry(name="mv", framework="tensorflow")
class MinimumVariance(BeamSumming):
    """Minimum variance beamforming layer"""

    def __init__(self, name="MV_layer", **kwargs):
        """Implementation of minimum variance beamforming in tensorflow

        Args:
            probe (Probe): A probe object
            config (utils.config.Config): A config object containing all parameters
            name (str, optional): Name of the layer. Defaults to 'MV_layer'.
        """

        # The MV beamformer is not yet fully implemented in tensorflow, lets raise a warning to let
        # the user know.
        tf.print(
            """WARNING: The MV beamformer is not yet fully implemented in Tensorflow. For
                 experimental use only!!"""
        )

        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        """Performs MV beamforming on tof-corrected input.

        Args:
            inputs (tensor): The TOF corrected input of shape
            (batch_size, n_tx, N_x, N_z, 1 if RF/2 if IQ)

        Returns:
            dict: Output dict with keys ('beamformed') with shape
            (batch_size, N_x, N_z, 1 if RF/2 if IQ)
        """
        x = inputs
        outputs = {}

        batch_size, n_tx, N_x, N_z, n_el, N_ch = x.get_shape()
        batches = []
        for b in range(batch_size):
            compounded = tf.zeros((N_x, N_z, N_ch))
            for tx in range(n_tx):
                columns = []
                for n_x in range(N_x):
                    column = self.mv_beamforming(x[b, tx, n_x, :, :])
                    columns.append(column)

                frame = tf.stack(columns, axis=0)
                compounded += frame

            batches.append(compounded)

        outputs["beamformed"] = tf.stack(batches, axis=0)
        return outputs

    def mv_beamforming(self, x, D=0.01):
        """Computation of the minimum variance beamformer

        Args:
            x (Tensor): Input data of shape (Nz, n_el, N_ch)
            D (float, optional): Diagonal loading factor. Defaults to 0.01.

        Returns:
            Tensor: MV beamformed output
        """

        in0 = x

        Nz, n_el, N_ch = x.get_shape()

        x = tf.transpose(x, perm=[0, 2, 1])  # (Nz, N_ch, n_el)
        x = tf.expand_dims(x, axis=-1)
        x_t = tf.transpose(x, perm=[0, 1, 3, 2])
        R = tf.matmul(x, x_t)

        trace = tf.linalg.trace(R)
        trace = tf.expand_dims(trace, axis=-1)
        trace = tf.expand_dims(trace, axis=-1)

        I = tf.eye(n_el, n_el, batch_shape=(Nz, N_ch))

        # Diagonal loading
        R = R + D * trace * I

        R_inv = tf.linalg.pinv(R)

        # MV simplifies to summing for a = 1 (ones vector)
        nom = tf.reduce_sum(R_inv, axis=-1, keepdims=True)

        den = tf.reduce_sum(nom, axis=-2, keepdims=True)

        w = tf.squeeze(nom / den, axis=-1)
        w = tf.transpose(w, perm=[0, 2, 1])  # (Nz, n_el, N_ch)

        x = in0 * w
        x = tf.reduce_sum(x, axis=-2)

        return x
