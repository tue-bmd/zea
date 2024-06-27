""" Coherence factor beamformer implementation in tensorflow
Based on the following paper: https://doi.org/10.1109/TUFFC.2010.1553
"""

import tensorflow as tf

from usbmd.backend.tensorflow.layers.beamformers import BeamSumming
from usbmd.registry import tf_beamformer_registry

# pylint: disable=arguments-differ


@tf_beamformer_registry(name="cf", framework="tensorflow")
class CoherenceFactor(BeamSumming):
    """Layer that implements the coherence factor beamformer."""

    def __init__(self, name="CF_layer", **kwargs):
        """Implementation of coherence factor beamforming in tensorflow

        Args:
            name (str, optional): Name of the layer. Defaults to 'CF_layer'.
        """
        super().__init__(name=name, **kwargs)
        self.epsilon = 1e-10

    def call(self, inputs):
        """Performs coherence factor beamforming on tof-corrected input.
        args:
            inputs (tensor): The TOF corrected input of shape
            (batch_size, n_tx, N_x, N_z, n_el, 1 if RF/2 if IQ)

        returns:
            dict: Output dict with keys ('beamformed') with shape
            (batch_size, N_x, N_z, 1 if RF/2 if IQ)
        """

        outputs = {}

        coherent_sum = tf.abs(tf.reduce_sum(inputs, axis=-2)) ** 2
        incoherent_sum = tf.reduce_sum(tf.abs(inputs) ** 2, axis=-2)

        cf = coherent_sum / (incoherent_sum + self.epsilon)
        beamformed = cf * tf.reduce_sum(inputs, axis=-2)
        compounded = tf.reduce_sum(beamformed, axis=1)

        outputs["beamformed"] = compounded
        return outputs
