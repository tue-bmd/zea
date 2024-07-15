"""Module for defining aliases for ops module."""

# pylint: skip-file
import importlib.util

"""Convert Tensorflow ops to numpy ops syntax used in main ops module."""
if importlib.util.find_spec("tensorflow"):
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


"""Convert Torch ops to numpy ops syntax used in main ops module."""
if importlib.util.find_spec("torch"):
    import torch

    torch.expand_dims = torch.unsqueeze
    torch.convert_to_tensor = torch.tensor
    torch.shape = lambda x: x.shape
    torch.take_along_axis = torch.take_along_dim
    torch.cast = lambda x, dtype: x.type(dtype)
    torch.concatenate = torch.cat
    torch.iscomplex = torch.is_complex

"""Extent numpy ops a bit for main ops module."""
if importlib.util.find_spec("numpy"):
    import numpy as np

    np.cast = lambda x, dtype: x.astype(dtype)
    np.convert_to_tensor = lambda x: x
    np.complex = lambda x, y: x + 1j * y
    np.permute = np.transpose
