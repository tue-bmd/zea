"""Diffusion Transformer (DiT) backend.

This module provides a `Diffusion Transformer
<https://arxiv.org/abs/2212.09748>`_ (Peebles & Xie, 2023) network that can be
used as a drop-in backend for :class:`~zea.models.diffusion.DiffusionModel` and
:class:`~zea.models.flow_matching.FlowMatchingModel`.

The network exposes exactly the same call signature as the time-conditional
UNet backend (:func:`~zea.models.unet.get_time_conditional_unetwork`): it takes
a list ``[noisy_images, time_scalar]`` where ``noisy_images`` has shape
``(batch, height, width, channels)`` and ``time_scalar`` has shape
``(batch, 1, 1, 1)``, and returns a tensor with the same shape as
``noisy_images``.  This makes it interchangeable with the UNet backend without
any changes to the sampling, training, or guidance machinery.

The architecture follows the original DiT with **adaLN-Zero** conditioning:

1. The image is split into non-overlapping patches and linearly embedded into a
   sequence of tokens (patch embedding via a strided convolution).
2. Learnable positional embeddings are added to the tokens.
3. The (scalar) diffusion time is embedded with a sinusoidal embedding followed
   by an MLP to produce a conditioning vector ``c``.
4. A stack of transformer blocks processes the tokens.  Each block modulates its
   layer-normalised activations with shift/scale/gate parameters regressed from
   ``c`` (adaptive layer norm, zero-initialised so the block starts as the
   identity).
5. A final adaLN-modulated linear layer projects each token back to its pixel
   patch, and the patches are reassembled (unpatchified) into an image.

.. seealso::

    Peebles & Xie, *Scalable Diffusion Models with Transformers*, 2023.
    https://arxiv.org/abs/2212.09748
"""

from __future__ import annotations

import keras
from keras import layers, ops

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.layers import sinusoidal_embedding


def modulate(x, shift, scale):
    """Apply adaptive layer-norm modulation.

    Args:
        x: Token tensor of shape ``(batch, num_tokens, hidden_size)``.
        shift: Shift tensor of shape ``(batch, hidden_size)``.
        scale: Scale tensor of shape ``(batch, hidden_size)``.

    Returns:
        Modulated tensor ``x * (1 + scale) + shift`` of the same shape as ``x``.
    """
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


@keras.saving.register_keras_serializable(package="zea")
class AddPositionEmbedding(layers.Layer):
    """Add learnable positional embeddings to a sequence of tokens."""

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name="pos_embed",
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_embed


@keras.saving.register_keras_serializable(package="zea")
class DiTBlock(layers.Layer):
    """A single DiT transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_fc1 = layers.Dense(mlp_hidden, activation="gelu")
        self.mlp_fc2 = layers.Dense(hidden_size)
        # adaLN-Zero: regress the 6 modulation parameters from the conditioning
        # vector. Zero-initialised so the block is the identity at init.
        self.ada_modulation = layers.Dense(
            6 * hidden_size,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(self, x, c):
        modulation = self.ada_modulation(keras.activations.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.split(
            modulation, 6, axis=-1
        )

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa[:, None, :] * self.attn(h, h)

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp_fc2(self.mlp_fc1(h))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zea")
class FinalLayer(layers.Layer):
    """Final adaLN-modulated projection from tokens back to pixel patches."""

    def __init__(self, patch_size, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm = layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)
        self.linear = layers.Dense(
            patch_size * patch_size * out_channels,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )
        # adaLN-Zero modulation producing (shift, scale) for the final norm.
        # Created in build() once the hidden size is known from the input shape.
        self.ada_modulation = None

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.ada_modulation = layers.Dense(
            2 * hidden_size,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )
        super().build(input_shape)

    def call(self, x, c):
        shift, scale = ops.split(self.ada_modulation(keras.activations.silu(c)), 2, axis=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "out_channels": self.out_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zea")
class Unpatchify(layers.Layer):
    """Reassemble a sequence of pixel patches into an image."""

    def __init__(self, grid_height, grid_width, patch_size, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.patch_size = patch_size
        self.out_channels = out_channels

    def call(self, x):
        batch_size = ops.shape(x)[0]
        p = self.patch_size
        x = ops.reshape(
            x,
            (batch_size, self.grid_height, self.grid_width, p, p, self.out_channels),
        )
        # (B, gh, p, gw, p, C)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return ops.reshape(
            x,
            (
                batch_size,
                self.grid_height * p,
                self.grid_width * p,
                self.out_channels,
            ),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_height": self.grid_height,
                "grid_width": self.grid_width,
                "patch_size": self.patch_size,
                "out_channels": self.out_channels,
            }
        )
        return config


def get_time_conditional_dit_network(
    image_shape,
    patch_size=8,
    hidden_size=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    embedding_min_frequency=1.0,
    embedding_max_frequency=1000.0,
    embedding_dims=256,
):
    """Build a time-conditional Diffusion Transformer (DiT) network.

    The returned model has the same input/output contract as
    :func:`~zea.models.unet.get_time_conditional_unetwork`, so it can be used
    interchangeably as a backend for diffusion / flow-matching models.

    Args:
        image_shape: Tuple ``(height, width, channels)``.  Both ``height`` and
            ``width`` must be divisible by ``patch_size``.
        patch_size: Side length of the (square) image patches.
        hidden_size: Token embedding dimension. Must be divisible by
            ``num_heads``.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden-dimension expansion ratio of the per-token MLP.
        embedding_min_frequency: Minimum frequency for the sinusoidal time
            embedding.
        embedding_max_frequency: Maximum frequency for the sinusoidal time
            embedding.
        embedding_dims: Dimensionality of the sinusoidal time embedding
            (must be even).

    Returns:
        keras.Model: A functional model mapping ``[noisy_images, time_scalar]``
        to a tensor with the same shape as ``noisy_images``.
    """
    assert len(image_shape) == 3, "image_shape must be a tuple of (height, width, channels)"
    image_height, image_width, n_channels = image_shape
    assert image_height % patch_size == 0 and image_width % patch_size == 0, (
        f"image height/width ({image_height}, {image_width}) must be divisible by "
        f"patch_size ({patch_size})."
    )
    assert hidden_size % num_heads == 0, (
        f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
    )
    assert embedding_dims % 2 == 0, "embedding_dims must be even! (sin + cos)"

    grid_height = image_height // patch_size
    grid_width = image_width // patch_size

    noisy_images = keras.Input(shape=(image_height, image_width, n_channels))
    noise_variances = keras.Input(shape=(1, 1, 1))

    # --- Patch embedding: (B, H, W, C) -> (B, num_patches, hidden_size) ---
    x = layers.Conv2D(hidden_size, kernel_size=patch_size, strides=patch_size)(noisy_images)
    x = layers.Reshape((grid_height * grid_width, hidden_size))(x)
    x = AddPositionEmbedding()(x)

    # --- Time conditioning vector c ---
    @keras.saving.register_keras_serializable(package="zea")
    def _sinusoidal_embedding(t):
        return sinusoidal_embedding(
            t, embedding_min_frequency, embedding_max_frequency, embedding_dims
        )

    t = layers.Reshape((1,))(noise_variances)
    c = layers.Lambda(_sinusoidal_embedding, output_shape=(embedding_dims,))(t)
    c = layers.Dense(hidden_size, activation="swish")(c)
    c = layers.Dense(hidden_size)(c)

    # --- Transformer blocks ---
    for _ in range(depth):
        x = DiTBlock(hidden_size, num_heads, mlp_ratio)(x, c)

    # --- Final projection + unpatchify ---
    x = FinalLayer(patch_size, n_channels)(x, c)
    x = Unpatchify(grid_height, grid_width, patch_size, n_channels)(x)

    return keras.Model([noisy_images, noise_variances], x, name="diffusion_transformer")


@model_registry(name="dit_time_conditional")
class DiTTimeConditional(BaseModel):
    """Diffusion Transformer with time-conditional (adaLN-Zero) embedding."""

    def __init__(
        self,
        image_shape,
        image_range=(0, 1),
        patch_size=8,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        embedding_min_frequency=1.0,
        embedding_max_frequency=1000.0,
        embedding_dims=256,
        name="dit_time_conditional",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.image_shape = image_shape
        self.image_range = image_range
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.embedding_min_frequency = embedding_min_frequency
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_dims = embedding_dims
        self.network = get_time_conditional_dit_network(
            image_shape=self.image_shape,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            embedding_min_frequency=self.embedding_min_frequency,
            embedding_max_frequency=self.embedding_max_frequency,
            embedding_dims=self.embedding_dims,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "image_range": self.image_range,
                "patch_size": self.patch_size,
                "hidden_size": self.hidden_size,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "embedding_min_frequency": self.embedding_min_frequency,
                "embedding_max_frequency": self.embedding_max_frequency,
                "embedding_dims": self.embedding_dims,
            }
        )
        return config

    def call(self, *args, **kwargs):
        return self.network(*args, **kwargs)
