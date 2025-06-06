"""UNet models and architectures"""

import keras
from keras import layers

from zea import log
from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.layers import DownBlock, ResidualBlock, UpBlock, sinusoidal_embedding
from zea.models.preset_utils import register_presets
from zea.models.presets import unet_presets


@model_registry(name="unet")
class UNet(BaseModel):
    """UNet model"""

    def __init__(
        self,
        image_shape,
        widths,
        block_depth,
        image_range,
        name="unet",
        **kwargs,
    ):
        """Initializes a UNet model"""

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.image_range = image_range
        self.widths = widths
        self.block_depth = block_depth

        self.network = get_unetwork(self.image_shape, self.widths, self.block_depth)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "image_range": self.image_range,
                "widths": self.widths,
                "block_depth": self.block_depth,
            }
        )
        return config

    def call(self, *args, **kwargs):
        return self.network(*args, **kwargs)


def get_unetwork(
    image_shape,
    widths,
    block_depth,
):
    """Get a basic UNet architecture

    Args:
        image_shape: tuple, (height, width, channels)
        widths: list, number of filters in each layer
        block_depth: int, number of residual blocks in each down/up block

    Returns:
        keras.Model
    """
    assert (
        len(image_shape) == 3
    ), "image_shape must be a tuple of (height, width, channels)"

    image_height, image_width, n_channels = image_shape
    noisy_images = keras.Input(shape=(image_height, image_width, n_channels))

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(n_channels, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model(noisy_images, x, name="residual_unet")


@model_registry(name="unet_time_conditional")
class UNetTimeConditional(BaseModel):
    """UNet model with time-conditional sinusoidal embedding"""

    def __init__(
        self,
        image_shape,
        widths,
        block_depth,
        image_range,
        embedding_min_frequency=1.0,
        embedding_max_frequency=1000.0,
        embedding_dims=32,
        name="unet_time_conditional",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.image_shape = image_shape
        self.image_range = image_range
        self.widths = widths
        self.block_depth = block_depth
        self.embedding_min_frequency = embedding_min_frequency
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_dims = embedding_dims
        self.network = get_time_conditional_unetwork(
            self.image_shape,
            self.widths,
            self.block_depth,
            self.embedding_min_frequency,
            self.embedding_max_frequency,
            self.embedding_dims,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "image_range": self.image_range,
                "widths": self.widths,
                "block_depth": self.block_depth,
                "embedding_min_frequency": self.embedding_min_frequency,
                "embedding_max_frequency": self.embedding_max_frequency,
                "embedding_dims": self.embedding_dims,
            }
        )
        return config

    def call(self, *args, **kwargs):
        return self.network(*args, **kwargs)


def get_time_conditional_unetwork(
    image_shape,
    widths=None,
    block_depth=None,
    embedding_min_frequency=1.0,
    embedding_max_frequency=1000.0,
    embedding_dims=32,
):
    """Get a basic UNet architecture with time-conditional sinusoidal embeddings

    Used in Diffusion Models.

    Args:
        image_shape: tuple, (height, width, channels)
        widths: list, number of filters in each layer
        block_depth: int, number of residual blocks in each down/up block
        embedding_min_frequency: float, minimum frequency for sinusoidal embeddings
        embedding_max_frequency: float, maximum frequency for sinusoidal embeddings
        embedding_dims: int, number of dimensions for sinusoidal embeddings

    Returns:
        keras.Model
    """
    assert (
        len(image_shape) == 3
    ), "image_shape must be a tuple of (height, width, channels)"

    if widths is None:
        log.warning("No widths provided, using default widths [32, 64, 96, 128]")
        widths = [32, 64, 96, 128]
    if block_depth is None:
        log.warning("No block_depth provided, using default block_depth 2")
        block_depth = 2

    image_height, image_width, n_channels = image_shape
    noisy_images = keras.Input(shape=(image_height, image_width, n_channels))
    noise_variances = keras.Input(shape=(1, 1, 1))

    @keras.saving.register_keras_serializable()
    def _sinusoidal_embedding(x):
        return sinusoidal_embedding(
            x, embedding_min_frequency, embedding_max_frequency, embedding_dims
        )

    e = layers.Lambda(_sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=(image_height, image_width), interpolation="nearest")(
        e
    )

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(n_channels, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


register_presets(unet_presets, UNet)
register_presets(unet_presets, UNetTimeConditional)
