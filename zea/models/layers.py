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


@keras.saving.register_keras_serializable()
class ResidualBlock(layers.Layer):
    """Residual block with swish activation.

    If the input channel dimension differs from ``width``, a 1×1 convolution
    is used to project the residual connection.
    """

    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.width = width

    def build(self, input_shape):
        input_width = input_shape[-1]
        self.needs_projection = input_width != self.width
        if self.needs_projection:
            self.proj = layers.Conv2D(self.width, kernel_size=1, name="residual_proj")
        self.norm = layers.BatchNormalization(center=False, scale=False)
        self.conv1 = layers.Conv2D(self.width, kernel_size=3, padding="same", activation="swish")
        self.conv2 = layers.Conv2D(self.width, kernel_size=3, padding="same")
        self.add = layers.Add()
        super().build(input_shape)

    def call(self, x):
        residual = self.proj(x) if self.needs_projection else x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.add([x, residual])

    def get_config(self):
        config = super().get_config()
        config.update({"width": self.width})
        return config


@keras.saving.register_keras_serializable()
class DownBlock(layers.Layer):
    """Downsampling block with residual connections.

    Expects a tuple ``(x, skips)`` where ``skips`` is a **mutable** list.
    Appends intermediate activations to ``skips`` for the decoder.
    """

    def __init__(self, width, block_depth, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.block_depth = block_depth

    def build(self, input_shape):
        # input_shape is a list of shapes: [x_shape, skips_shape (ignored)]
        self.res_blocks = [
            ResidualBlock(self.width, name=f"res_{i}") for i in range(self.block_depth)
        ]
        self.pool = layers.AveragePooling2D(pool_size=2)
        super().build(input_shape)

    def call(self, inputs):
        x, skips = inputs
        for block in self.res_blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x, skips

    def get_config(self):
        config = super().get_config()
        config.update({"width": self.width, "block_depth": self.block_depth})
        return config


@keras.saving.register_keras_serializable()
class UpBlock(layers.Layer):
    """Upsampling block with residual connections.

    Expects a tuple ``(x, skips)`` where ``skips`` is a **mutable** list.
    Pops skip connections and concatenates them before each residual block.
    """

    def __init__(self, width, block_depth, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.block_depth = block_depth

    def build(self, input_shape):
        self.upsample = layers.UpSampling2D(size=2, interpolation="bilinear")
        self.concat_layers = [layers.Concatenate(name=f"cat_{i}") for i in range(self.block_depth)]
        self.res_blocks = [
            ResidualBlock(self.width, name=f"res_{i}") for i in range(self.block_depth)
        ]
        super().build(input_shape)

    def call(self, inputs):
        x, skips = inputs
        x = self.upsample(x)
        for cat, block in zip(self.concat_layers, self.res_blocks):
            x = cat([x, skips.pop()])
            x = block(x)
        return x, skips

    def get_config(self):
        config = super().get_config()
        config.update({"width": self.width, "block_depth": self.block_depth})
        return config
