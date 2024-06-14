"""
Tensorflow utilities
"""
import numpy as np
import tensorflow as tf
from keras.layers import Lambda, LeakyReLU, ReLU

PI = tf.experimental.numpy.pi


def antirect(x):
    """Function that implements the antirectifier activation"""
    mean, _ = tf.nn.moments(x, axes=-1, keepdims=True)
    x = tf.math.l2_normalize(x - mean, axis=-1)
    return tf.nn.crelu(x)


def get_activation(activation: str = None):
    """Get activation function given string.

    Args:
        activation (str, optional): name of activation
            function. Defaults to None.

    Raises:
        ValueError: Cannot find activation function

    Returns:
        Tensorflow activation function
    """
    if activation.lower() == "relu":
        return ReLU()
    elif activation.lower() == "leakyrelu":
        return LeakyReLU()
    elif activation.lower() == "swish":
        return Lambda(lambda x: tf.keras.activations.swish(x))
    elif activation.lower() == "sigmoid":
        return Lambda(lambda x: tf.keras.activations.sigmoid(x))
    elif activation is None:
        return Lambda(lambda x: x)
    else:
        raise ValueError("Unknown activation function.")


def tf_cropping_and_padding(input_shape, target_shape):
    """Crop or pad a tensor to the specified shape.

    Args:
        input_shape: A list or tuple of integers representing the input shape.
        target_shape: A list or tuple of integers representing the desired shape.

    Returns:
        A tensorflow cropping layer (2D) for 4D tensors.
    """
    assert len(input_shape) == len(target_shape) == 2, "can only do 2D cropping"

    # Calculate the amount of cropping needed for each dimension
    diff = np.array(input_shape) - np.array(target_shape)
    cropping = ((0, diff[0]), (0, diff[1]))

    # replace negative values with zero and make padding tuple of tuples
    padding = tuple(tuple(0 if x > 0 else -x for x in row) for row in cropping)
    cropping = tuple(tuple(0 if x < 0 else x for x in row) for row in cropping)

    # Create a Cropping2D layer with the calculated crop amounts
    cropping_layer = tf.keras.layers.Cropping2D(cropping=cropping)

    # Create a ZeroPadding2D layer with the calculated padding amounts
    padding_layer = tf.keras.layers.ZeroPadding2D(padding=padding)

    return cropping_layer, padding_layer


def tf_complex_resize(tensor, image_size):
    """Resize / interpolate complex tensor"""
    magnitude = tf.math.abs(tensor)
    phase = tf.math.angle(tensor)

    # unwrap phase
    phase = tf_unwrap(phase)

    magnitude = tf.image.resize(magnitude, image_size)
    phase = tf.image.resize(phase, image_size)

    phase = tf.complex(tf.cast(0.0, dtype=phase.dtype), phase)
    magnitude = tf.cast(magnitude, dtype=tf.complex64)

    # complex = magnitude * exp(1j * phase)
    return tf.math.multiply(magnitude, tf.math.exp(phase))


def tf_unwrap(tensor, axis=0):
    """Tensorflow phase unwrapping function"""
    pi_tf = tf.cast(PI, dtype=tensor.dtype)
    dphi = tf.experimental.numpy.diff(tensor, axis=axis)
    dphi_pad = tf.gather(dphi, 0, axis=axis)
    dphi_pad = tf.expand_dims(dphi_pad, axis=axis)
    dphi = tf.concat([dphi_pad, dphi], axis=axis)
    dphi_m = ((dphi + pi_tf) % (2.0 * pi_tf)) - pi_tf
    dphi_m = tf.where((dphi_m == -PI) & (dphi > 0.0), pi_tf, dphi_m)
    phi_adj = dphi_m - dphi
    phi_adj = tf.where(tf.abs(dphi) < pi_tf, 0.0, phi_adj)
    return tensor + tf.math.cumsum(phi_adj, axis=axis)
