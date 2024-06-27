"""Experimental implementation of random minimum beamforming in tensorflow.

This layer is based on the patent:
https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20200138412

The patent describes a minimum variance like beamformer, but instead of
calculating the covariance matrix, it uses a random set of apodization vectors
and selects the one with the lowest output power.

Author(s): Ben Luijten
"""

# It is TF convention to define layers in the build method
# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ

import time

import tensorflow as tf
import tqdm

from usbmd.backend.tensorflow.layers.beamformers import BeamSumming
from usbmd.backend.tensorflow.utils.gpu_config import get_device
from usbmd.registry import tf_beamformer_registry
from usbmd.utils import log


@tf_beamformer_registry(name="random_minimum", framework="tensorflow")
class RandomMinimum(BeamSumming):
    """Random minimum beamforming layer."""

    def __init__(self, N=100, **kwargs):
        """Initialise the random minimum beamforming layer.

        Args:
            N (int): Number of random apodization vectors to use.
        """
        super().__init__(**kwargs)
        self.N = N  # Number of random apodization vectors to use

        log.warning("RandomMinimum layer is experimental and may not work as expected.")

    def build(self, input_shape):
        """Build the random minimum beamforming layer."""
        n_el = input_shape[-2]

        self.apo_vectors = tf.random.uniform((self.N, n_el), 0, 1, tf.float32)
        self.apo_vectors = self.apo_vectors / tf.reduce_sum(
            self.apo_vectors, axis=-1, keepdims=True
        )

    def call(self, inputs):
        """Performs random minimum beamforming on tof-corrected input.

        Args:
            inputs (tensor): The TOF corrected input of shape
            (batch_size, n_tx, N_x, N_z, n_el, 1 if RF/2 if IQ)

        Returns:
            dict: Output dict with keys ('beamformed') with shape
            (batch_size, N_x, N_z, 1 if RF/2 if IQ)
        """

        x = inputs  # (batch_size, n_tx, N_x, N_z, n_el, 2)
        a = self.apo_vectors  # (N, n_el)

        output = []

        # Processing per batch, tx and x (column) to avoid memory issues
        for b in range(x.shape[0]):
            for t in range(x.shape[1]):
                for c in range(x.shape[2]):
                    y = tf.einsum("zei,ne->zeni", x[b, t, c, :, :, :], a)

                    # Beamsum over n_el
                    y = tf.reduce_mean(y, axis=-3, keepdims=False)
                    # append to output
                    output.append(y)

        output = tf.stack(output, axis=0)
        output = tf.reshape(
            output, (x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.N, x.shape[5])
        )

        # take the absolute value
        output = tf.abs(output)

        # find minimum over N apo dimension
        output = tf.reduce_min(output, axis=-2, keepdims=False)

        # apply tx compounding
        output = tf.reduce_mean(output, axis=1, keepdims=False)

        outputs = {}
        outputs["beamformed"] = output

        return outputs


if __name__ == "__main__":
    # Test the layer

    get_device()
    random_minimum = RandomMinimum(100)
    random_minimum = tf.function(random_minimum, jit_compile=True)

    N_batch = 1
    n_tx = 1
    N_x = 512
    N_z = 512
    n_el = 128
    N_ch = 2

    x = tf.random.uniform((N_batch, n_tx, N_x, N_z, n_el, N_ch), 0, 1, tf.float32)

    # Run, build and compile the function
    start = time.perf_counter()
    y = random_minimum(x)
    end = time.perf_counter()
    print("Time to Run, build, and compile: ", end - start)

    # time the function M times
    M = 100
    start = time.perf_counter()

    for i in tqdm.tqdm(range(M)):
        y = random_minimum(x)

    end = time.perf_counter()

    print("Time taken: ", (end - start) / M)
    print("Done!")
