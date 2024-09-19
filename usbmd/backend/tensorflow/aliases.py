"""Convert Tensorflow ops to numpy ops syntax used in main ops module."""

import tensorflow as tf

tf.arange = tf.range
tf.take_along_axis = tf.experimental.numpy.take_along_axis
tf.min = tf.reduce_min
tf.max = tf.reduce_max
tf.mean = tf.reduce_mean
tf.sum = tf.reduce_sum
tf.clip = tf.clip_by_value
tf.log10 = tf.experimental.numpy.log10
tf.log = tf.experimental.numpy.log
tf.permute = tf.transpose
tf.concatenate = tf.concat
tf.fft = tf.signal
tf.imag = tf.math.imag
tf.real = tf.math.real
tf.iscomplex = tf.experimental.numpy.iscomplex
tf.conj = tf.math.conj
tf.hstack = tf.experimental.numpy.hstack


def isinf(x):
    """Taken from keras.ops"""
    x = tf.convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.zeros(x.shape, tf.bool)
    return tf.math.is_inf(x)


tf.isinf = isinf
