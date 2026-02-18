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
# This maps string names → callable(image_shape, **kwargs) → keras.Model
# ---------------------------------------------------------------------------
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
    assert len(image_shape) == 3, "image_shape must be a tuple of (height, width, channels)"

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
        block_depth: int, number of residual blocks in each down/up block (defaults to 2 if None)
        embedding_min_frequency: float, minimum frequency for sinusoidal embeddings
        embedding_max_frequency: float, maximum frequency for sinusoidal embeddings
        embedding_dims: int, number of dimensions for sinusoidal embeddings (must be even)

    Returns:
        keras.Model
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
    e = layers.UpSampling2D(size=(image_height, image_width), interpolation="nearest")(e)

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


# ===========================================================================
# Temporal layers
# ===========================================================================

VALID_TEMPORAL_SCOPES = ("none", "bottleneck", "all")


@keras.saving.register_keras_serializable()
class TemporalConv(layers.Layer):
    """Lightweight temporal convolution operating on frames packed as channels.

    For grayscale video data, the input shape is ``(B, H, W, T)`` where ``T``
    is the number of frames. Internally reshapes to 5-D ``(B, H, W, T, 1)``,
    applies a ``(1, 1, k)`` Conv3D, and reshapes back. Output shape matches
    input (shape-preserving).

    Args:
        n_frames: Number of temporal frames ``T`` (equals the channel dimension).
        temporal_kernel_size: Kernel size along the temporal axis.
        depthwise: If ``True``, use depthwise convolution (minimal params).
    """

    def __init__(
        self,
        temporal_kernel_size=3,
        depthwise=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temporal_kernel_size = temporal_kernel_size
        self.depthwise = depthwise
        self._temporal_conv = None

    def build(self, input_shape):
        # Since C=1, filters=1 and groups=1 (depthwise with 1 channel is just regular conv)
        self._temporal_conv = layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, self.temporal_kernel_size),
            padding="same",
        )
        super().build(input_shape)

    def call(self, x):
        shape = keras.ops.shape(x)
        b, h, w, t = shape[0], shape[1], shape[2], shape[3]

        # Reshape (B, H, W, T) -> (B, H, W, T, 1)
        x = keras.ops.reshape(x, (b, h, w, t, 1))
        x = self._temporal_conv(x)
        # Reshape back (B, H, W, T, 1) -> (B, H, W, T)
        return keras.ops.reshape(x, (b, h, w, t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "temporal_kernel_size": self.temporal_kernel_size,
                "depthwise": self.depthwise,  # kept for compatibility but unused
            }
        )
        return config


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
        x = self._attn(x, x)
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


# ===========================================================================
# Temporal UNet — unified model and network builder
# ===========================================================================


def _temporal_residual_block(x, width, n_frames, temporal_kernel_size, temporal_depthwise):
    """Apply a spatial ResidualBlock followed by a residual TemporalConv (inline)."""
    x = ResidualBlock(width)(x)
    x = x + TemporalConv(
        n_frames=n_frames,
        temporal_kernel_size=temporal_kernel_size,
        depthwise=temporal_depthwise,
    )(x)
    return x


def _temporal_down(
    x, skips, width, block_depth, n_frames, temporal_kernel_size, temporal_depthwise
):
    """Temporal encoder stage: temporal residual blocks + average pooling."""
    for _ in range(block_depth):
        x = _temporal_residual_block(x, width, n_frames, temporal_kernel_size, temporal_depthwise)
        skips.append(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    return x


def _temporal_up(x, skips, width, block_depth, n_frames, temporal_kernel_size, temporal_depthwise):
    """Temporal decoder stage: upsample + concat skips + temporal residual blocks."""
    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
    for _ in range(block_depth):
        x = layers.Concatenate()([x, skips.pop()])
        x = _temporal_residual_block(x, width, n_frames, temporal_kernel_size, temporal_depthwise)
    return x


@model_registry(name="unet_temporal_time_conditional")
class UNetTemporalTimeConditional(BaseModel):
    """Temporal UNet with (2+1)D factorized convolutions and time-conditional
    sinusoidal embeddings for video diffusion.

    Frames are packed as channels: input shape ``(H, W, T*C)`` where ``T`` is
    the number of frames and ``C`` is the per-frame channel count.

    The ``temporal_scope`` parameter controls where temporal convolutions are
    injected:

    - ``"bottleneck"``: Only the bottleneck uses temporal convolutions and
      optional temporal self-attention.  Encoder and decoder use standard
      spatial ``DownBlock`` / ``UpBlock``.  Cheapest option.
    - ``"all"``: Every residual block throughout the encoder, bottleneck, and
      decoder is augmented with a temporal convolution.  Maximum temporal
      modelling capacity.
    - ``"none"``: No temporal convolutions at all (equivalent to a standard
      time-conditional UNet operating on packed channels).

    Args:
        image_shape: ``(H, W, T*C)`` — height, width, frames×channels.
        n_frames: Number of temporal frames ``T``.
        widths: Filter counts per resolution level.
        block_depth: Number of residual blocks per down/up stage.
        image_range: Value range of input images.
        temporal_scope: Where to inject temporal convolutions.
            One of ``"bottleneck"``, ``"all"``, or ``"none"``.
        temporal_kernel_size: Kernel size for temporal convolutions.
        temporal_depthwise: Use depthwise temporal convs (cheaper).
        temporal_attention_bottleneck: Add temporal self-attention at bottleneck.
        temporal_attention_heads: Number of attention heads.
        embedding_min_frequency: Min frequency for sinusoidal time embedding.
        embedding_max_frequency: Max frequency for sinusoidal time embedding.
        embedding_dims: Dimensionality of time embedding.
    """

    def __init__(
        self,
        image_shape,
        widths,
        block_depth,
        image_range,
        temporal_scope="bottleneck",
        temporal_kernel_size=3,
        temporal_depthwise=False,
        temporal_attention_bottleneck=True,
        temporal_attention_heads=4,
        embedding_min_frequency=1.0,
        embedding_max_frequency=1000.0,
        embedding_dims=32,
        embedding_conditioning="add",  # | "concat"
        name="unet_temporal_time_conditional",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert temporal_scope in VALID_TEMPORAL_SCOPES, (
            f"temporal_scope must be one of {VALID_TEMPORAL_SCOPES}, got '{temporal_scope}'"
        )

        self.image_shape = image_shape
        self.image_range = image_range
        self.widths = widths
        self.block_depth = block_depth
        self.temporal_scope = temporal_scope
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_depthwise = temporal_depthwise
        self.temporal_attention_bottleneck = temporal_attention_bottleneck
        self.temporal_attention_heads = temporal_attention_heads
        self.embedding_min_frequency = embedding_min_frequency
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_dims = embedding_dims
        self.embedding_conditioning = embedding_conditioning

        self.network = get_temporal_time_conditional_unetwork(
            image_shape=self.image_shape,
            widths=self.widths,
            block_depth=self.block_depth,
            temporal_scope=self.temporal_scope,
            temporal_kernel_size=self.temporal_kernel_size,
            temporal_depthwise=self.temporal_depthwise,
            temporal_attention_bottleneck=self.temporal_attention_bottleneck,
            temporal_attention_heads=self.temporal_attention_heads,
            embedding_min_frequency=self.embedding_min_frequency,
            embedding_max_frequency=self.embedding_max_frequency,
            embedding_dims=self.embedding_dims,
            embedding_conditioning=self.embedding_conditioning,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "n_frames": self.n_frames,
                "image_range": self.image_range,
                "widths": self.widths,
                "block_depth": self.block_depth,
                "temporal_scope": self.temporal_scope,
                "temporal_kernel_size": self.temporal_kernel_size,
                "temporal_depthwise": self.temporal_depthwise,
                "temporal_attention_bottleneck": self.temporal_attention_bottleneck,
                "temporal_attention_heads": self.temporal_attention_heads,
                "embedding_min_frequency": self.embedding_min_frequency,
                "embedding_max_frequency": self.embedding_max_frequency,
                "embedding_dims": self.embedding_dims,
            }
        )
        return config

    def summary(self):
        return self.network.summary()

    def call(self, *args, **kwargs):
        return self.network(*args, **kwargs)


def _maybe_temporal_conv(x, temporal_conv_kwargs):
    """Optionally apply a residual TemporalConv. Returns x unchanged if None."""
    if temporal_conv_kwargs is None:
        return x
    return x + TemporalConv(**temporal_conv_kwargs)(x)


def get_temporal_time_conditional_unetwork(
    image_shape,
    widths=None,
    block_depth=None,
    temporal_scope="bottleneck",
    temporal_kernel_size=3,
    temporal_depthwise=True,
    temporal_attention_bottleneck=True,
    temporal_attention_heads=4,
    embedding_min_frequency=1.0,
    embedding_max_frequency=1000.0,
    embedding_dims=32,
    embedding_conditioning="add",  # | "concat"
):
    """Build a temporal UNet with time-conditional sinusoidal embeddings.

    Uses standard 2D ``DownBlock`` / ``UpBlock`` / ``ResidualBlock`` throughout.
    Temporal mixing is introduced by optionally appending a shape-preserving
    depthwise ``(1, 1, k)`` Conv3D (as a residual addition) after each block.

    The ``temporal_scope`` parameter determines where these are inserted:

    - ``"none"``: Purely spatial — no temporal convolutions.
    - ``"bottleneck"``: Temporal convs only after bottleneck residual blocks
      (+ optional temporal self-attention).
    - ``"all"``: Temporal convs after every residual block in encoder,
      bottleneck, and decoder.

    Args:
        image_shape: ``(H, W, T*C)`` — spatial dims + packed frame channels.
        widths: Filter counts per resolution level.
        block_depth: Residual blocks per down/up stage.
        temporal_scope: One of ``"none"``, ``"bottleneck"``, ``"all"``.
        temporal_kernel_size: Temporal conv kernel size.
        temporal_depthwise: Use depthwise temporal convs (default ``True``).
        temporal_attention_bottleneck: Add temporal attention at bottleneck.
        temporal_attention_heads: Number of attention heads.
        embedding_min_frequency: Min freq for sinusoidal embedding.
        embedding_max_frequency: Max freq for sinusoidal embedding.
        embedding_dims: Embedding dimensionality (must be even).

    Returns:
        ``keras.Model`` with inputs ``[noisy_images, noise_variances]``.
    """
    assert len(image_shape) == 3, "image_shape must be (height, width, channels)"
    assert embedding_dims % 2 == 0, "embedding_dims must be even! (sin + cos)"
    assert temporal_scope in VALID_TEMPORAL_SCOPES, (
        f"temporal_scope must be one of {VALID_TEMPORAL_SCOPES}, got '{temporal_scope}'"
    )

    if widths is None:
        log.warning("No widths provided, using default widths [32, 64, 96, 128]")
        widths = [32, 64, 96, 128]
    if block_depth is None:
        block_depth = 2

    image_height, image_width, n_channels = image_shape
    # Build temporal conv kwargs (or None when not needed)
    _tc_kwargs = dict(
        temporal_kernel_size=temporal_kernel_size,
        depthwise=temporal_depthwise,
    )
    enc_dec_tc = _tc_kwargs if temporal_scope == "all" else None
    bottleneck_tc = _tc_kwargs if temporal_scope in ("bottleneck", "all") else None

    noisy_images = keras.Input(shape=(image_height, image_width, n_channels))
    noise_variances = keras.Input(shape=(1, 1, 1))

    # ---- Time embedding ----
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

    # ---- Encoder ----
    skips = []
    for width in widths[:-1]:
        x, skips = DownBlock(width, block_depth)([x, skips])
        x = _maybe_temporal_conv(x, enc_dec_tc)

    # ---- Bottleneck ----
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)
        x = _maybe_temporal_conv(x, bottleneck_tc)

    if bottleneck_tc is not None and temporal_attention_bottleneck:
        x = x + TemporalAttention(
            num_heads=temporal_attention_heads,
        )(x)

    # ---- Decoder ----
    for width in reversed(widths[:-1]):
        x, skips = UpBlock(width, block_depth)([x, skips])
        x = _maybe_temporal_conv(x, enc_dec_tc)

    x = layers.Conv2D(n_channels, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="temporal_residual_unet")


register_presets(unet_presets, UNet)
register_presets(unet_presets, UNetTimeConditional)
register_presets(unet_presets, UNetTemporalTimeConditional)
