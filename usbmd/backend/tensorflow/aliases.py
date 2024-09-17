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
