"""Module that holds 1D filtering layers for tensorflow.

- **Author(s)**     : Ben Luijten
- **Date**          : Thu Jun 1st 2023
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers

# It is TF convention to define layers in the build method
# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ


class Filter1DLayer(keras.layers.Layer):
    """Base class for 1D filtering layers"""

    def __init__(self, filter_weights, axis=-1, trainable=False, **kwargs):
        """
        args:
            filter_weights: 1D numpy array of filter weights
            axis: axis over which the filter is applied
            trainable: whether the filter weights are trainable or not
        """
        super().__init__(**kwargs)
        self.filter_weights = filter_weights
        self.trainable = trainable
        self.axis = axis

    def build(self, input_shape):
        """Build the filter layer"""

        self.filter_layer = layers.Conv1D(
            filters=1,
            kernel_size=len(self.filter_weights),
            padding="same",
            use_bias=False,
            kernel_initializer=keras.initializers.Constant(self.filter_weights),
            trainable=self.trainable,
            name="filter_layer",
        )

        super().build(input_shape)

    def call(self, inputs):
        """Apply the filter layer"""

        # transpose such that the axis to be filtered, set in self.axis, is the last axis.
        perm = list(range(len(inputs.shape)))
        perm[-1], perm[self.axis] = perm[self.axis], perm[-1]
        inputs = tf.transpose(inputs, perm)

        # Add a filter dimension
        inputs = tf.expand_dims(inputs, axis=-1)

        filtered_inputs = self.filter_layer(inputs)

        # Remove the filter dimension
        filtered_inputs = tf.squeeze(filtered_inputs, axis=-1)

        # transpose back to original shape
        perm = list(range(len(filtered_inputs.shape)))
        perm[-1], perm[self.axis] = perm[self.axis], perm[-1]
        filtered_inputs = tf.transpose(filtered_inputs, perm)

        return filtered_inputs

    def get_config(self):
        """Return the layer configuration"""

        config = super().get_config()
        config.update({"filter_weights": self.filter_weights})
        return config


class Bandpass(Filter1DLayer):
    """Bandpass filter layer"""

    def __init__(self, bandwidth, sampling_frequency, center_frequency, N, **kwargs):
        """Initialize the bandpass filter
        args:
            bandwidth: bandwidth of the filter in Hz
            sampling_frequency: sampling frequency in Hz
            center_frequency: center frequency in Hz
            N: filter length
        """
        self.bandwidth = bandwidth
        self.sampling_frequency = sampling_frequency
        self.center_frequency = center_frequency
        self.N = N
        filter_weights = self._calculate_filter_coefficients(
            bandwidth, sampling_frequency, center_frequency, N
        )
        super().__init__(filter_weights, **kwargs)

    @staticmethod
    def _calculate_filter_coefficients(
        bandwidth, sampling_frequency, center_frequency, N
    ):
        """Calculate the bandpass filter coefficients"""
        taps = np.arange(-N / 2, N / 2)
        filter_weights = (
            2
            * np.cos(2 * np.pi * center_frequency / sampling_frequency * taps)
            * np.sinc(2 * bandwidth / sampling_frequency * taps)
        )
        return filter_weights


class Lowpass(Filter1DLayer):
    """Lowpass filter layer"""

    def __init__(self, cutoff, sampling_frequency, center_frequency, N, **kwargs):
        """Initialize the lowpass filter
        args:
            cutoff: cutoff frequency in Hz
            sampling_frequency: sampling frequency in Hz
            center_frequency: center frequency in Hz
            N: filter length
        """
        self.cutoff = cutoff
        self.sampling_frequency = sampling_frequency
        self.center_frequency = center_frequency
        self.N = N
        filter_weights = self._calculate_filter_coefficients(
            cutoff, sampling_frequency, N
        )
        super().__init__(filter_weights, **kwargs)

    @staticmethod
    def _calculate_filter_coefficients(cutoff, sampling_frequency, N):
        """Calculate the low-pass filter coefficients"""
        taps = np.arange(-N / 2, N / 2)
        filter_weights = np.sinc(2 * cutoff / sampling_frequency * taps) * np.blackman(
            N
        )
        return filter_weights


class Highpass(Filter1DLayer):
    """Highpass filter layer"""

    def __init__(self, cutoff, sampling_frequency, center_frequency, N, **kwargs):
        """Initialize the highpass filter
        args:
            cutoff: cutoff frequency in Hz
            sampling_frequency: sampling frequency in Hz
            center_frequency: center frequency in Hz
            N: filter length
        """
        self.cutoff = cutoff
        self.sampling_frequency = sampling_frequency
        self.center_frequency = center_frequency
        self.N = N
        filter_weights = self._calculate_filter_coefficients(
            cutoff, sampling_frequency, N
        )
        super().__init__(filter_weights, **kwargs)

    @staticmethod
    def _calculate_filter_coefficients(cutoff, sampling_frequency, N):
        """Calculate the high-pass filter coefficients"""
        taps = np.arange(-N / 2, N / 2)
        filter_weights = -np.sinc(2 * cutoff / sampling_frequency * taps) * np.blackman(
            N
        )
        return filter_weights
