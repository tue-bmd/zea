"""UltraPIPS: perceptual similarity for B-mode ultrasound.

UltraPIPS is an LPIPS-style perceptual metric whose backbone is an ultrasound
foundation model instead of a network pretrained on natural images. The
features of both images are unit-normalized along the channel axis, compared
with a squared difference, averaged over space, summed over channels and
summed over all feature levels. Unlike LPIPS there is no learned linear head:
every channel counts equally.

To try this model, simply load one of the available presets:

.. code-block:: python

    from zea.models.ultrapips import UltraPIPS

    model = UltraPIPS.from_preset("ultrapips-tusa")
    distance = model([image1, image2])  # images in [-1, 1], shape (B, H, W, C)

Backbone
--------
The default (and currently only) backbone is the Swin transformer encoder of
TUSA (Grutman et al., 2026), a self-supervised texture model trained on open
B-mode datasets from many organs. It is the ``swinViT`` of a 2D MONAI
``SwinUNETR`` (v2) with ``feature_size=36``, ``depths=(2, 3, 4, 5)`` and
``num_heads=(3, 6, 12, 24)``, run on ``128 x 128`` single-channel images. The
five hidden states it returns (patch embedding plus the four stages) are the
feature levels UltraPIPS compares.

.. important::
    This is a ``zea`` implementation of the model.
    For the original code, see `here <https://github.com/talg2324/UltraPIPS>`_
    and `here <https://github.com/talg2324/tusa>`_ for the TUSA backbone.

.. citation:: grutman2026ultrapips

"""

import itertools

import keras
import numpy as np
from keras import layers, ops

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import ultrapips_presets

# Constants of the PyTorch layers the TUSA weights were trained with.
_LAYER_NORM_EPS = 1e-5
_INSTANCE_NORM_EPS = 1e-5
_ATTENTION_MASK_VALUE = -100.0
# ITU-R 601-2 luma weights, as used by ``torchvision`` for grayscale conversion.
_GRAYSCALE_WEIGHTS = (0.2989, 0.587, 0.114)


def _layer_norm(x, epsilon=_LAYER_NORM_EPS):
    """Layer norm over the last axis, without an affine transform."""
    mean = ops.mean(x, axis=-1, keepdims=True)
    var = ops.var(x, axis=-1, keepdims=True)
    return (x - mean) / ops.sqrt(var + epsilon)


def _instance_norm(x, epsilon=_INSTANCE_NORM_EPS):
    """Instance norm of a ``(B, H, W, C)`` tensor, without an affine transform."""
    mean = ops.mean(x, axis=(1, 2), keepdims=True)
    var = ops.var(x, axis=(1, 2), keepdims=True)
    return (x - mean) / ops.sqrt(var + epsilon)


def _window_and_shift_size(spatial_size, window_size, shift_size):
    """Shrink the window to the feature map when the map is not larger than it.

    A window that covers the whole feature map cannot be shifted, so the shift is
    dropped along that axis too.
    """
    window_size, shift_size = list(window_size), list(shift_size)
    for i, size in enumerate(spatial_size):
        if size <= window_size[i]:
            window_size[i] = size
            shift_size[i] = 0
    return tuple(window_size), tuple(shift_size)


def window_partition(x, window_size):
    """Split ``(B, H, W, C)`` into windows of shape ``(B * num_windows, wh * ww, C)``."""
    _, h, w, c = x.shape
    wh, ww = window_size
    x = ops.reshape(x, (-1, h // wh, wh, w // ww, ww, c))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, wh * ww, c))


def window_reverse(windows, window_size, height, width):
    """Inverse of :func:`window_partition`, back to ``(B, H, W, C)``."""
    wh, ww = window_size
    c = windows.shape[-1]
    x = ops.reshape(windows, (-1, height // wh, width // ww, wh, ww, c))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, height, width, c))


def relative_position_index(window_size):
    """Index into the relative position bias table for every pair of window tokens.

    Returns:
        np.ndarray: Integer array of shape ``(wh * ww, wh * ww)``.
    """
    wh, ww = window_size
    coords = np.stack(np.meshgrid(np.arange(wh), np.arange(ww), indexing="ij"))
    coords = coords.reshape(2, -1)
    relative = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0)
    relative[:, :, 0] += wh - 1
    relative[:, :, 1] += ww - 1
    relative[:, :, 0] *= 2 * ww - 1
    return relative.sum(-1)


def shifted_window_mask(height, width, window_size, shift_size):
    """Attention mask that stops shifted windows from mixing wrapped-around regions.

    Args:
        height (int): Padded feature map height.
        width (int): Padded feature map width.
        window_size (tuple[int, int]): Window size.
        shift_size (tuple[int, int]): Cyclic shift.

    Returns:
        np.ndarray: ``(num_windows, N, N)`` float32 mask with ``0`` where tokens may
        attend to each other and ``-100`` where they may not.
    """
    img_mask = np.zeros((1, height, width, 1), dtype="float32")
    cnt = 0
    (wh, ww), (sh, sw) = window_size, shift_size
    for hs in (slice(-wh), slice(-wh, -sh), slice(-sh, None)):
        for ws in (slice(-ww), slice(-ww, -sw), slice(-sw, None)):
            img_mask[:, hs, ws, :] = cnt
            cnt += 1
    windows = img_mask.reshape(1, height // wh, wh, width // ww, ww, 1)
    windows = windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, wh * ww)
    mask = windows[:, None, :] - windows[:, :, None]
    return np.where(mask != 0, _ATTENTION_MASK_VALUE, 0.0).astype("float32")


class WindowAttention(layers.Layer):
    """Window-based multi-head self-attention with a relative position bias."""

    def __init__(self, dim, num_heads, window_size, qkv_bias=True, **kwargs):
        """Initialize the attention layer.

        Args:
            dim (int): Number of feature channels.
            num_heads (int): Number of attention heads.
            window_size (tuple[int, int]): Window size the bias table is sized for.
            qkv_bias (bool): Whether the query, key and value projection has a bias.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = tuple(window_size)
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.proj = layers.Dense(dim, name="proj")
        self._relative_position_index = relative_position_index(self.window_size)

    def build(self, input_shape):
        wh, ww = self.window_size
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * wh - 1) * (2 * ww - 1), self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="relative_position_bias_table",
        )
        self.qkv.build(input_shape)
        self.proj.build((*input_shape[:-1], self.dim))

    def call(self, x, attn_mask=None):
        """Attend within each window.

        Args:
            x (Tensor): Windows of shape ``(B * num_windows, N, C)``.
            attn_mask (np.ndarray, optional): ``(num_windows, N, N)`` additive mask
                for shifted windows.

        Returns:
            Tensor: Same shape as ``x``.
        """
        n, c = x.shape[1], x.shape[2]
        head_dim = c // self.num_heads

        qkv = ops.reshape(self.qkv(x), (-1, n, 3, self.num_heads, head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        index = self._relative_position_index[:n, :n].reshape(-1)
        bias = ops.take(self.relative_position_bias_table, index, axis=0)
        bias = ops.transpose(ops.reshape(bias, (n, n, self.num_heads)), (2, 0, 1))

        attn = ops.matmul(q * self.scale, ops.swapaxes(k, -2, -1))
        attn = attn + bias[None]
        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn = ops.reshape(attn, (-1, num_windows, self.num_heads, n, n))
            attn = attn + ops.convert_to_tensor(attn_mask, dtype=attn.dtype)[None, :, None]
            attn = ops.reshape(attn, (-1, self.num_heads, n, n))
        attn = ops.softmax(attn, axis=-1)

        x = ops.transpose(ops.matmul(attn, v), (0, 2, 1, 3))
        return self.proj(ops.reshape(x, (-1, n, c)))


class SwinTransformerBlock(layers.Layer):
    """Swin transformer block: (shifted) window attention followed by an MLP."""

    def __init__(
        self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, qkv_bias=True, **kwargs
    ):
        """Initialize the block.

        Args:
            dim (int): Number of feature channels.
            num_heads (int): Number of attention heads.
            window_size (tuple[int, int]): Window size.
            shift_size (tuple[int, int]): Cyclic shift, ``(0, 0)`` for regular windows.
            mlp_ratio (float): Hidden size of the MLP relative to ``dim``.
            qkv_bias (bool): Whether the query, key and value projection has a bias.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.norm1 = layers.LayerNormalization(epsilon=_LAYER_NORM_EPS, name="norm1")
        self.attn = WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias, name="attn")
        self.norm2 = layers.LayerNormalization(epsilon=_LAYER_NORM_EPS, name="norm2")
        self.fc1 = layers.Dense(int(dim * mlp_ratio), name="fc1")
        self.fc2 = layers.Dense(dim, name="fc2")

    def build(self, input_shape):
        wh, ww = self.window_size
        self.norm1.build(input_shape)
        self.attn.build((None, wh * ww, self.dim))
        self.norm2.build(input_shape)
        self.fc1.build(input_shape)
        self.fc2.build((*input_shape[:-1], self.fc1.units))

    def _attention(self, x, attn_mask):
        _, h, w, c = x.shape
        window_size, shift_size = _window_and_shift_size((h, w), self.window_size, self.shift_size)
        x = self.norm1(x)

        # Pad to a whole number of windows. The (zero) padding takes part in the
        # attention of the regular windows, exactly as in the reference.
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        if pad_b or pad_r:
            x = ops.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
        hp, wp = h + pad_b, w + pad_r

        shifted = any(s > 0 for s in shift_size)
        if shifted:
            x = ops.roll(x, shift=(-shift_size[0], -shift_size[1]), axis=(1, 2))
        else:
            attn_mask = None

        windows = self.attn(window_partition(x, window_size), attn_mask=attn_mask)
        x = window_reverse(windows, window_size, hp, wp)

        if shifted:
            x = ops.roll(x, shift=shift_size, axis=(1, 2))
        return x[:, :h, :w, :]

    def call(self, x, attn_mask=None):
        """Apply the block to a ``(B, H, W, C)`` feature map.

        Args:
            x (Tensor): Feature map.
            attn_mask (np.ndarray, optional): Shifted window mask, ignored for
                blocks without a shift.
        """
        x = x + self._attention(x, attn_mask)
        hidden = keras.activations.gelu(self.fc1(self.norm2(x)), approximate=False)
        return x + self.fc2(hidden)


class PatchMerging(layers.Layer):
    """Halve the resolution and double the channels by merging 2x2 neighborhoods."""

    def __init__(self, dim, **kwargs):
        """Initialize the layer.

        Args:
            dim (int): Number of input channels.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.norm = layers.LayerNormalization(epsilon=_LAYER_NORM_EPS, name="norm")
        self.reduction = layers.Dense(2 * dim, use_bias=False, name="reduction")

    def build(self, input_shape):
        merged = (*input_shape[:-1], 4 * self.dim)
        self.norm.build(merged)
        self.reduction.build(merged)

    def call(self, x):
        """Merge a ``(B, H, W, C)`` map into ``(B, ceil(H / 2), ceil(W / 2), 2C)``."""
        _, h, w, _ = x.shape
        if h % 2 or w % 2:
            x = ops.pad(x, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)))
        # Same neighbor order as MONAI: (0, 0), (1, 0), (0, 1), (1, 1) as (row, col).
        x = ops.concatenate(
            [x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], axis=-1
        )
        return self.reduction(self.norm(x))


class UnetResBlock(layers.Layer):
    """Residual block of two 3x3 convolutions with instance norm and leaky ReLU."""

    def __init__(self, filters, negative_slope=0.01, **kwargs):
        """Initialize the block.

        Args:
            filters (int): Number of input and output channels.
            negative_slope (float): Slope of the leaky ReLU.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.negative_slope = negative_slope
        self.conv1 = layers.Conv2D(filters, 3, padding="same", use_bias=False, name="conv1")
        self.conv2 = layers.Conv2D(filters, 3, padding="same", use_bias=False, name="conv2")

    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.conv2.build((*input_shape[:-1], self.filters))

    def call(self, x):
        """Apply the block to a ``(B, H, W, C)`` feature map."""
        out = _instance_norm(self.conv1(x))
        out = ops.leaky_relu(out, self.negative_slope)
        out = _instance_norm(self.conv2(out))
        return ops.leaky_relu(out + x, self.negative_slope)


class SwinStage(layers.Layer):
    """One stage of the Swin encoder: alternating regular and shifted blocks, then merging."""

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, **kwargs):
        """Initialize the stage.

        Args:
            dim (int): Number of input channels.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            window_size (tuple[int, int]): Window size.
            mlp_ratio (float): Hidden size of the MLP relative to ``dim``.
            qkv_bias (bool): Whether the query, key and value projection has a bias.
        """
        super().__init__(**kwargs)
        self.window_size = tuple(window_size)
        self.shift_size = tuple(s // 2 for s in self.window_size)
        self.blocks = [
            SwinTransformerBlock(
                dim,
                num_heads,
                window_size,
                shift_size=(0, 0) if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                name=f"block{i}",
            )
            for i in range(depth)
        ]
        self.downsample = PatchMerging(dim, name="downsample")

    def build(self, input_shape):
        for block in self.blocks:
            block.build(input_shape)
        self.downsample.build(input_shape)

    def call(self, x):
        """Apply the stage to a ``(B, H, W, C)`` feature map."""
        _, h, w, _ = x.shape
        window_size, shift_size = _window_and_shift_size((h, w), self.window_size, self.shift_size)
        hp = -(-h // window_size[0]) * window_size[0]
        wp = -(-w // window_size[1]) * window_size[1]
        attn_mask = None
        if any(s > 0 for s in shift_size):
            attn_mask = shifted_window_mask(hp, wp, window_size, shift_size)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.downsample(x)


class SwinViT(layers.Layer):
    """2D Swin transformer encoder (MONAI ``SwinUNETR.swinViT``, ``use_v2=True``).

    Returns the five hidden states of the encoder: the patch embedding and the
    output of each of the four stages, each optionally layer-normalized.
    """

    def __init__(
        self,
        in_channels=1,
        feature_size=36,
        depths=(2, 3, 4, 5),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7),
        patch_size=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs,
    ):
        """Initialize the encoder.

        Args:
            in_channels (int): Number of input image channels.
            feature_size (int): Embedding dimension; doubles at every stage.
            depths (tuple[int]): Number of transformer blocks per stage.
            num_heads (tuple[int]): Number of attention heads per stage.
            window_size (tuple[int, int]): Attention window size.
            patch_size (int): Patch size of the embedding.
            mlp_ratio (float): Hidden size of the MLPs relative to their input.
            qkv_bias (bool): Whether the query, key and value projections have a bias.
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)
        self.window_size = tuple(window_size)
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias

        self.patch_embed = layers.Conv2D(
            feature_size, patch_size, strides=patch_size, padding="valid", name="patch_embed"
        )
        self.res_blocks = []
        self.stages = []
        for i, (depth, heads) in enumerate(zip(self.depths, self.num_heads)):
            dim = feature_size * 2**i
            self.res_blocks.append(UnetResBlock(dim, name=f"res_block{i}"))
            self.stages.append(
                SwinStage(
                    dim, depth, heads, self.window_size, mlp_ratio, qkv_bias, name=f"stage{i}"
                )
            )

    def build(self, input_shape):
        b, h, w, _ = input_shape
        self.patch_embed.build(input_shape)
        h, w = -(-h // self.patch_size), -(-w // self.patch_size)
        for i, (res_block, stage) in enumerate(zip(self.res_blocks, self.stages)):
            shape = (b, h, w, self.feature_size * 2**i)
            res_block.build(shape)
            stage.build(shape)
            h, w = -(-h // 2), -(-w // 2)

    def call(self, x, normalize=True):
        """Encode a ``(B, H, W, C)`` image.

        Args:
            x (Tensor): Input image, normalized the way the weights expect.
            normalize (bool): Whether to layer-normalize (without affine) each
                returned hidden state over its channels.

        Returns:
            list[Tensor]: Five ``(B, H_i, W_i, C_i)`` feature maps.
        """
        _, h, w, _ = x.shape
        pad_h, pad_w = -h % self.patch_size, -w % self.patch_size
        if pad_h or pad_w:
            x = ops.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

        x = self.patch_embed(x)
        outputs = [x]
        for res_block, stage in zip(self.res_blocks, self.stages):
            x = stage(res_block(x))
            outputs.append(x)
        if normalize:
            outputs = [_layer_norm(out) for out in outputs]
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "feature_size": self.feature_size,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "patch_size": self.patch_size,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
            }
        )
        return config


@model_registry(name="ultrapips")
class UltraPIPS(BaseModel):
    """Perceptual image patch similarity for B-mode ultrasound (UltraPIPS)."""

    BACKBONES = ("tusa",)

    def __init__(self, backbone="tusa", image_size=128, disable_checks=False, **kwargs):
        """Initialize the UltraPIPS model.

        Args:
            backbone (str, optional): Feature extractor. Only ``"tusa"`` (the default
                of the reference implementation) is ported. Defaults to ``"tusa"``.
            image_size (int, optional): Side length the images are resized to
                before they are encoded. The TUSA backbone was trained at 128.
                Defaults to 128.
            disable_checks (bool, optional): Disable the input value checks, e.g. to
                trace the metric in a TensorFlow graph. Defaults to False.
        """
        super().__init__(**kwargs)
        if backbone not in self.BACKBONES:
            raise ValueError(
                f"Unknown UltraPIPS backbone '{backbone}', choose from {self.BACKBONES}."
            )
        self.backbone = backbone
        self.image_size = image_size
        self.disable_checks = disable_checks
        self.net = SwinViT(name="tusa")
        self.trainable = False

    def build(self, input_shape=None):
        """Build the backbone (its input shape is fixed by ``image_size``)."""
        self.net.build((None, self.image_size, self.image_size, 1))
        self.built = True

    def preprocess_input(self, image):
        """Convert images to the input the backbone expects.

        Converts RGB to grayscale, resizes to ``image_size`` (bilinear with
        antialiasing, like ``torchvision``) and keeps the ``[-1, 1]`` range, which
        is the ``(x - 0.5) / 0.5`` normalization TUSA was trained with.

        Args:
            image (Tensor): ``(B, H, W, C)`` image in ``[-1, 1]`` with 1 or 3 channels.

        Returns:
            Tensor: ``(B, image_size, image_size, 1)`` tensor.
        """
        image = ops.convert_to_tensor(image, dtype="float32")
        if image.shape[-1] == 3:
            # Convert in [0, 1] where the luma weights are defined (they sum to 0.9999).
            weights = ops.convert_to_tensor(_GRAYSCALE_WEIGHTS, dtype=image.dtype)
            image = ops.sum((image + 1.0) / 2.0 * weights, axis=-1, keepdims=True) * 2.0 - 1.0
        if tuple(image.shape[1:3]) != (self.image_size, self.image_size):
            image = ops.image.resize(
                image,
                (self.image_size, self.image_size),
                interpolation="bilinear",
                antialias=True,
            )
        return image

    def features(self, image):
        """Encode an image into its (layer-normalized) backbone feature maps.

        Args:
            image (Tensor): ``(B, H, W, C)`` image in ``[-1, 1]`` with 1 or 3 channels.

        Returns:
            list[Tensor]: Five ``(B, H_i, W_i, C_i)`` feature maps.
        """
        return self.net(self.preprocess_input(image))

    @staticmethod
    def _normalize_tensor(in_feat, eps=1e-10):
        """Unit-normalize features along the channel axis."""
        return in_feat / ops.sqrt(ops.sum(in_feat**2, axis=-1, keepdims=True) + eps)

    def call(self, inputs):
        """Compute the UltraPIPS distance.

        Args:
            inputs (list): Two images of shape ``(B, H, W, C)`` or ``(H, W, C)`` with
                1 or 3 channels and values in ``[-1, 1]``.

        Returns:
            Tensor: Distance of shape ``(B,)``, or a scalar without a batch dimension.
        """
        input1, input2 = inputs
        if not self.disable_checks and not (self._valid_img(input1) and self._valid_img(input2)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors with shape [B, H, W, C]"
                f" or [H, W, C]. Got input with shape {input1.shape} and {input2.shape} and values"
                f" in range {[ops.min(input1), ops.max(input1)]} and"
                f" {[ops.min(input2), ops.max(input2)]} when all values are expected to be in"
                " the [-1, 1] range."
            )

        has_batch_dim = ops.ndim(input1) == 4
        if not has_batch_dim:
            input1, input2 = input1[None], input2[None]

        # Encode both images in one pass; every normalization in the backbone is
        # per sample, so batching them together does not change the features.
        batch_size = ops.shape(input1)[0]
        feats = self.features(ops.concatenate([input1, input2], axis=0))

        distance = 0.0
        for feat in feats:
            feat = self._normalize_tensor(feat)
            diff = ops.square(feat[:batch_size] - feat[batch_size:])
            distance = distance + ops.sum(ops.mean(diff, axis=(1, 2)), axis=-1)

        if not has_batch_dim:
            distance = ops.squeeze(distance, axis=0)
        return distance

    @staticmethod
    def _valid_img(img) -> bool:
        """Check that input is a valid image to the network."""
        value_check = ops.max(img) <= 1.0 and ops.min(img) >= -1
        shape_check = ops.ndim(img) in [3, 4] and ops.shape(img)[-1] in [1, 3]
        return shape_check and value_check

    def get_config(self):
        """Serialize the model arguments."""
        config = super().get_config()
        config.update(
            {
                "backbone": self.backbone,
                "image_size": self.image_size,
                "disable_checks": self.disable_checks,
            }
        )
        return config

    def custom_load_weights(self, preset, backend="keras", **kwargs):
        """Load weights from a preset (Hugging Face handle or local directory).

        Args:
            preset (str): Preset identifier passed from :meth:`from_preset`.
            backend (str): ``"keras"`` loads ``model.weights.h5``; ``"torch"``
                converts the original TUSA ``unet.pt`` checkpoint and needs PyTorch.
        """
        loader = get_preset_loader(preset)
        self.build()
        if backend == "keras":
            self.load_weights(loader.get_file("model.weights.h5"), **kwargs)
        elif backend == "torch":  # pragma: no cover
            self._load_from_pth(loader.get_file("unet.pt"))
        else:
            raise ValueError(f"Unsupported backend '{backend}' for UltraPIPS preset")

    def _load_from_pth(self, pth_path):  # pragma: no cover
        """Load the original TUSA PyTorch checkpoint into this Keras model.

        Args:
            pth_path (str): Path to the TUSA ``unet.pt`` state dict.
        """
        import torch  # only needed for weight conversion

        self.build()
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
        load_torch_state_dict(self.net, state_dict)

    @classmethod
    def from_pth(cls, pth_path, **kwargs):  # pragma: no cover
        """Create an :class:`UltraPIPS` model from the original TUSA PyTorch checkpoint.

        Args:
            pth_path (str): Path to the TUSA ``unet.pt`` state dict.
            **kwargs: Passed to the constructor.

        Returns:
            UltraPIPS: Model with the converted weights.
        """
        model = cls(**kwargs)
        model._load_from_pth(pth_path)
        return model


def load_torch_state_dict(net, state_dict, prefix="swinViT."):
    """Copy the ``swinViT`` part of a TUSA (MONAI ``SwinUNETR``) state dict into a :class:`SwinViT`.

    Kernels are permuted from PyTorch's ``(out, in, h, w)`` to Keras'
    ``(h, w, in, out)`` and dense weights are transposed. The relative position
    index buffers are not copied: they are recomputed from the window size.

    Args:
        net (SwinViT): Built encoder to load the weights into.
        state_dict (dict): ``{key: tensor}`` mapping from PyTorch, e.g. the full
            ``SwinUNETR`` state dict.
        prefix (str): Prefix of the encoder keys in ``state_dict``.

    Raises:
        KeyError: If the state dict does not match the encoder's architecture.
    """
    arrays = {
        k[len(prefix) :]: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    used = set()

    def get(key):
        used.add(key)
        return arrays[key]

    def conv(key):
        return np.transpose(get(key), (2, 3, 1, 0))

    def dense(key):
        return get(key).T

    def layer_norm(layer, key):
        layer.set_weights([get(f"{key}.weight"), get(f"{key}.bias")])

    net.patch_embed.set_weights([conv("patch_embed.proj.weight"), get("patch_embed.proj.bias")])
    for i, (res_block, stage) in enumerate(zip(net.res_blocks, net.stages)):
        res = f"layers{i + 1}c.0.layer"
        res_block.conv1.set_weights([conv(f"{res}.conv1.conv.weight")])
        res_block.conv2.set_weights([conv(f"{res}.conv2.conv.weight")])

        layer = f"layers{i + 1}.0"
        for j, block in enumerate(stage.blocks):
            key = f"{layer}.blocks.{j}"
            used.add(f"{key}.attn.relative_position_index")
            layer_norm(block.norm1, f"{key}.norm1")
            block.attn.relative_position_bias_table.assign(
                get(f"{key}.attn.relative_position_bias_table")
            )
            block.attn.qkv.set_weights(
                [dense(f"{key}.attn.qkv.weight"), get(f"{key}.attn.qkv.bias")]
            )
            block.attn.proj.set_weights(
                [dense(f"{key}.attn.proj.weight"), get(f"{key}.attn.proj.bias")]
            )
            layer_norm(block.norm2, f"{key}.norm2")
            block.fc1.set_weights(
                [dense(f"{key}.mlp.linear1.weight"), get(f"{key}.mlp.linear1.bias")]
            )
            block.fc2.set_weights(
                [dense(f"{key}.mlp.linear2.weight"), get(f"{key}.mlp.linear2.bias")]
            )
        layer_norm(stage.downsample.norm, f"{layer}.downsample.norm")
        stage.downsample.reduction.set_weights([dense(f"{layer}.downsample.reduction.weight")])

    unused = sorted(set(arrays) - used)
    if unused:
        raise KeyError(f"Unused keys in the TUSA state dict: {unused}")


register_presets(ultrapips_presets, UltraPIPS)
