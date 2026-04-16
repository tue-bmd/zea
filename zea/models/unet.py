"""UNet models and architectures.

To try this model, simply load one of the available presets:

.. doctest::

    >>> from zea.models.unet import UNet

    >>> model = UNet.from_preset("unet-echonet-inpainter")

.. seealso::
    A tutorial notebook where this model is used:
    :doc:`../notebooks/models/unet_example`.

"""

import keras
from keras import layers

from zea import log

# ---------------------------------------------------------------------------
# Registry for network architecture builders (not full Model classes).
# This maps string names → callable(input_shape, **kwargs) → keras.Model
# ---------------------------------------------------------------------------
from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.layers import (
    DownBlock,
    ResidualBlock,
    UpBlock,
    sinusoidal_embedding,
)
from zea.models.preset_utils import register_presets
from zea.models.presets import unet_presets


@model_registry(name="unet")
class UNet(BaseModel):
    """UNet model"""

    def __init__(
        self,
        input_shape,
        widths,
        block_depth,
        input_range,
        name="unet",
        **kwargs,
    ):
        """Initializes a UNet model"""

        super().__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.input_range = input_range
        self.widths = widths
        self.block_depth = block_depth

        self.network = get_unetwork(self.input_shape, self.widths, self.block_depth)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "input_range": self.input_range,
                "widths": self.widths,
                "block_depth": self.block_depth,
            }
        )
        return config

    def call(self, *args, **kwargs):
        return self.network(*args, **kwargs)


def get_unetwork(
    input_shape,
    widths,
    block_depth,
):
    """Get a basic UNet architecture

    Args:
        input_shape: tuple, (height, width, channels)
        widths: list, number of filters in each layer
        block_depth: int, number of residual blocks in each down/up block

    Returns:
        keras.Model
    """
    assert len(input_shape) == 3, "input_shape must be a tuple of (height, width, channels)"

    image_height, image_width, n_channels = input_shape
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
    """UNet model with time-conditional sinusoidal embedding.

    Optionally includes temporal self-attention at the bottleneck for video
    diffusion, where frames are packed as channels
    (``image_shape = (H, W, T*C)``).

    Args:
        image_shape: ``(H, W, C)`` or ``(H, W, T*C)`` for video.
        widths: Filter counts per resolution level.
        block_depth: Number of residual blocks per down/up stage.
        image_range: Value range of input images.
        embedding_min_frequency: Min frequency for sinusoidal time embedding.
        embedding_max_frequency: Max frequency for sinusoidal time embedding.
        embedding_dims: Dimensionality of time embedding.
        embedding_conditioning: How to condition on the time embedding.
            ``"concat"`` (default) spatially tiles and concatenates;
            ``"add"`` projects and adds.
        temporal_attention_bottleneck: If ``True``, add temporal
            self-attention at the bottleneck.
        temporal_attention_heads: Number of attention heads for temporal
            attention.
    """

    def __init__(
        self,
        image_shape,
        widths,
        block_depth,
        image_range,
        embedding_min_frequency=1.0,
        embedding_max_frequency=1000.0,
        embedding_dims=32,
        embedding_conditioning="concat",
        temporal_attention_bottleneck=False,
        temporal_attention_heads=4,
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
        self.embedding_conditioning = embedding_conditioning
        self.temporal_attention_bottleneck = temporal_attention_bottleneck
        self.temporal_attention_heads = temporal_attention_heads
        self.network = get_time_conditional_unetwork(
            self.image_shape,
            self.widths,
            self.block_depth,
            self.embedding_min_frequency,
            self.embedding_max_frequency,
            self.embedding_dims,
            self.embedding_conditioning,
            self.temporal_attention_bottleneck,
            self.temporal_attention_heads,
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
                "embedding_conditioning": self.embedding_conditioning,
                "temporal_attention_bottleneck": self.temporal_attention_bottleneck,
                "temporal_attention_heads": self.temporal_attention_heads,
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
    embedding_conditioning="concat",
    temporal_attention_bottleneck=False,
    temporal_attention_heads=4,
):
    """Get a UNet architecture with time-conditional sinusoidal embeddings.

    Used in Diffusion Models.  Optionally adds temporal self-attention at the
    bottleneck for video diffusion with frames packed as channels.

    Args:
        image_shape: tuple, ``(height, width, channels)``.  For video data,
            channels = ``T * C`` where ``T`` is the number of frames.
        widths: list, number of filters in each layer.
        block_depth: int, number of residual blocks in each down/up block
            (defaults to 2 if ``None``).
        embedding_min_frequency: float, minimum frequency for sinusoidal
            embeddings.
        embedding_max_frequency: float, maximum frequency for sinusoidal
            embeddings.
        embedding_dims: int, number of dimensions for sinusoidal embeddings
            (must be even).
        embedding_conditioning: str, how to condition on the time embedding.
            ``"concat"`` spatially tiles and concatenates (original behavior);
            ``"add"`` projects and adds.
        temporal_attention_bottleneck: bool, if ``True`` add temporal
            self-attention at the bottleneck (useful for video diffusion).
        temporal_attention_heads: int, number of attention heads for temporal
            self-attention.

    Returns:
        ``keras.Model`` with inputs ``[noisy_images, noise_variances]``.
    """
    assert len(image_shape) == 3, "image_shape must be a tuple of (height, width, channels)"
    assert embedding_dims % 2 == 0, "embedding_dims must be even! (sin + cos)"

    if widths is None:
        log.warning("No widths provided, using default widths [32, 64, 96, 128]")
        widths = [32, 64, 96, 128]
    if block_depth is None:
        block_depth = 2

    image_height, image_width, n_channels = image_shape
    noisy_images = keras.Input(shape=(image_height, image_width, n_channels))
    noise_variances = keras.Input(shape=(1, 1, 1))

    @keras.saving.register_keras_serializable()
    def _sinusoidal_embedding(x):
        return sinusoidal_embedding(
            x, embedding_min_frequency, embedding_max_frequency, embedding_dims
        )

    e = layers.Lambda(_sinusoidal_embedding, output_shape=(1, 1, embedding_dims))(noise_variances)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)

    if embedding_conditioning == "concat":
        e = layers.UpSampling2D(size=(image_height, image_width), interpolation="nearest")(e)
        x = layers.Concatenate()([x, e])
    elif embedding_conditioning == "add":
        e_proj = layers.Conv2D(widths[0], kernel_size=1, padding="same")(e)
        x = layers.Add()([x, e_proj])
    else:
        raise ValueError(
            f"Invalid embedding_conditioning '{embedding_conditioning}', expected 'add' or 'concat'"
        )

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)
    if temporal_attention_bottleneck:
        x = x + TemporalAttention(num_heads=temporal_attention_heads)(x)
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(n_channels, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


# ===========================================================================
# Temporal attention
# ===========================================================================


@keras.saving.register_keras_serializable()
class TemporalAttention(layers.Layer):
    """Lightweight self-attention over the temporal axis.

    For grayscale video data, expects ``(B, H, W, T)`` where ``T`` is the
    number of frames. Reshapes to ``(B, H*W, T)`` (treating each spatial
    position as an independent sequence), applies multi-head attention over
    the temporal dimension, and reshapes back. Designed to be used only at
    the bottleneck where spatial resolution is small.

    Args:
        num_heads: Number of attention heads.
        key_dim: Dimension of each attention head (defaults to T // num_heads).
    """

    def __init__(self, num_heads=4, key_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self._attn = None

    def build(self, input_shape):
        # For grayscale: channels dimension IS the temporal dimension
        t = input_shape[-1]
        key_dim = self.key_dim or max(t // self.num_heads, 1)
        self._attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
        )
        super().build(input_shape)

    def call(self, x):
        shape = keras.ops.shape(x)
        b, h, w, t = shape[0], shape[1], shape[2], shape[3]

        # Reshape (B, H, W, T) -> (B, H*W, T)
        # Each spatial position is treated as an independent sequence of T timesteps
        x = keras.ops.reshape(x, (b, h * w, t))
        # Transpose so that the time axis is the sequence dimension for attention
        x = keras.ops.transpose(x, axes=(0, 2, 1))
        x = self._attn(x, x)
        # Un-transpose back to (B, H*W, T)
        x = keras.ops.transpose(x, axes=(0, 2, 1))
        # Reshape back (B, H*W, T) -> (B, H, W, T)
        return keras.ops.reshape(x, (b, h, w, t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
            }
        )
        return config


register_presets(unet_presets, UNet)
register_presets(unet_presets, UNetTimeConditional)
