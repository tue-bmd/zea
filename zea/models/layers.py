"""Layers used in zea.models"""

import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x, embedding_min_frequency, embedding_max_frequency, embedding_dims):
    """Sinusoidal embedding layer."""
    frequencies = ops.exp(
        ops.linspace(
            ops.log(embedding_min_frequency),
            ops.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = ops.cast(2.0 * math.pi * frequencies, x.dtype)
    embeddings = ops.concatenate(
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=-1
    )
    return embeddings


def ResidualBlock(width, normalization="batch_norm"):
    """Residual block with swish activation.

    Args:
        width: Number of filters.
        normalization: Normalization type. One of ``"batch_norm"`` or
            ``"group_norm"``. Defaults to ``"batch_norm"``.
    """

    def apply(x):
        input_width = ops.shape(x)[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        if normalization == "group_norm":
            x = layers.GroupNormalization(groups=min(32, width))(x)
        else:
            x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth, normalization="batch_norm"):
    """Downsampling block with residual connections."""

    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width, normalization=normalization)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth, normalization="batch_norm"):
    """Upsampling block with residual connections."""

    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width, normalization=normalization)(x)
        return x

    return apply
