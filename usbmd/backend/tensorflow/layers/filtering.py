""" Module that holds 1D filtering layers for tensorflow.

- **Author(s)**     : Ben Luijten
- **Date**          : Thu Jun 1st 2023
"""
import numpy as np
import tensorflow as tf

# It is TF convention to define layers in the build method
# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ


class Filter1DLayer(tf.keras.layers.Layer):
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

        self.filter_layer = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=len(self.filter_weights),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(self.filter_weights),
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

    def __init__(self, bandwidth, fs, fc, N, **kwargs):
        """Initialize the bandpass filter
        args:
            bandwidth: bandwidth of the filter in Hz
            fs: sampling frequency in Hz
            fc: center frequency in Hz
            N: filter length
        """
        self.bandwidth = bandwidth
        self.fs = fs
        self.fc = fc
        self.N = N
        filter_weights = self._calculate_filter_coefficients(bandwidth, fs, fc, N)
        super().__init__(filter_weights, **kwargs)

    @staticmethod
    def _calculate_filter_coefficients(bandwidth, fs, fc, N):
        """Calculate the bandpass filter coefficients"""
        taps = np.arange(-N / 2, N / 2)
        filter_weights = (
            2 * np.cos(2 * np.pi * fc / fs * taps) * np.sinc(2 * bandwidth / fs * taps)
        )
        return filter_weights


class Lowpass(Filter1DLayer):
    """Lowpass filter layer"""

    def __init__(self, cutoff, fs, fc, N, **kwargs):
        """Initialize the lowpass filter
        args:
            cutoff: cutoff frequency in Hz
            fs: sampling frequency in Hz
            fc: center frequency in Hz
            N: filter length
        """
        self.cutoff = cutoff
        self.fs = fs
        self.fc = fc
        self.N = N
        filter_weights = self._calculate_filter_coefficients(cutoff, fs, N)
        super().__init__(filter_weights, **kwargs)

    @staticmethod
    def _calculate_filter_coefficients(cutoff, fs, N):
        """Calculate the low-pass filter coefficients"""
        taps = np.arange(-N / 2, N / 2)
        filter_weights = np.sinc(2 * cutoff / fs * taps) * np.blackman(N)
        return filter_weights


class Highpass(Filter1DLayer):
    """Highpass filter layer"""

    def __init__(self, cutoff, fs, fc, N, **kwargs):
        """Initialize the highpass filter
        args:
            cutoff: cutoff frequency in Hz
            fs: sampling frequency in Hz
            fc: center frequency in Hz
            N: filter length
        """
        self.cutoff = cutoff
        self.fs = fs
        self.fc = fc
        self.N = N
        filter_weights = self._calculate_filter_coefficients(cutoff, fs, N)
        super().__init__(filter_weights, **kwargs)

    @staticmethod
    def _calculate_filter_coefficients(cutoff, fs, N):
        """Calculate the high-pass filter coefficients"""
        taps = np.arange(-N / 2, N / 2)
        filter_weights = -np.sinc(2 * cutoff / fs * taps) * np.blackman(N)
        return filter_weights
