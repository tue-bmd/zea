"""Speckle2Self self-supervised speckle reduction model for ultrasound images.

Implements the SpeckleReductionNet architecture from the Speckle2Self paper
(Li et al., Medical Image Analysis, 2025) as a `zea.Model`.

Usage
-----

.. code-block:: python

    from zea.models.speckle2self import Speckle2Self

    model = Speckle2Self.from_preset("speckle2self-invivo")
    despeckled = model(bmode_frames)

Architecture notes
------------------
- 3 independent encoders (highRes, midRes, lowRes) + 1 shared decoder.
- Encoder: 4 x [Conv2d(stride=2) + InstanceNorm + ReLU + ResidualBlock].
- Decoder: ResidualBlock + 4 x [ConvTranspose2d(stride=2) + InstanceNorm + ReLU].
- ``fuse=False`` means the decoder does not use inter-encoder skip connections;
  ``I_clean_hr = decoder(encoder_highRes(hr))`` depends only on the
  high-resolution input.
- InstanceNorm is implemented as ``GroupNormalization(groups=C, scale=False,
  center=False)`` which is mathematically equivalent.

.. important::

    This is a ``zea`` implementation of the model.
    For the original `paper <https://arxiv.org/abs/2507.06828>`_ and `
    code <https://github.com/noseefood/speckle2self>`_.

.. note::

    The Keras implementation (``_SpeckleReductionNetKeras``) is the **primary**
    inference backend.  The PyTorch classes in :func:`_build_torch_classes` are
    kept for ONNX conversion only and are never imported at runtime.  An ONNX
    fallback is also provided via :meth:`Speckle2Self.from_onnx` for environments
    that have ``onnxruntime`` but not ``torch``.

.. note::

    For the ONNX fallback, ``onnxruntime`` must be installed::

        pip install onnxruntime

"""

import keras
import numpy as np
from keras import ops

import zea
from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import speckle2self_presets

INFERENCE_SIZE = 512


def _build_torch_classes():  # pragma: no cover
    """Build and return the PyTorch model classes for SpeckleReductionNet.

    PyTorch is imported lazily so it is *only* required when converting a
    ``.pth`` checkpoint to ONNX — not for inference.

    Architecture verified against ``model_2833.pth`` state-dict keys and the
    exported ONNX op graph:

    * Each ``ConvBlock``: ``Conv2d(stride) → InstanceNorm2d(affine=False) → ReLU``
    * Each ``ConvTransposeBlock``: ``ConvTranspose2d(stride=2) → InstanceNorm2d → ReLU``
    * Each ``ResidualBlock``: ``[ConvBlock+ReLU, ConvBlock(no ReLU)] + skip Add``
    * Encoder: 4 levels, each ``conv_block_N(stride=2) → residual_block_N``
    * Decoder: ``residual_block_start`` + 4 ``conv_block_N(ConvTranspose)`` + residuals

    Returns:
        dict: ``{"SpeckleReductionNet": cls, "_SRNSingleInputWrapper": cls}``
    """
    import torch.nn as nn  # noqa: F401 (torch required for ONNX export)

    class ConvBlock(nn.Module):  # pragma: no cover
        """Conv2d(stride) + InstanceNorm2d(affine=False) + optional ReLU."""

        def __init__(self, in_ch, out_ch, stride=1, activation=True):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
            self.norm = nn.InstanceNorm2d(out_ch, affine=False)
            self.act = nn.ReLU() if activation else None

        def forward(self, x):
            x = self.norm(self.conv(x))
            if self.act is not None:
                x = self.act(x)
            return x

    class ConvTransposeBlock(nn.Module):  # pragma: no cover
        """ConvTranspose2d(stride=2) + InstanceNorm2d(affine=False) + optional ReLU."""

        def __init__(self, in_ch, out_ch, activation=True):
            super().__init__()
            self.conv = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            self.norm = nn.InstanceNorm2d(out_ch, affine=False)
            self.act = nn.ReLU() if activation else None

        def forward(self, x):
            x = self.norm(self.conv(x))
            if self.act is not None:
                x = self.act(x)
            return x

    class ResidualBlock(nn.Module):  # pragma: no cover
        """Two-layer residual block.

        First sub-block: ConvBlock + ReLU.
        Second sub-block: ConvBlock (no ReLU before skip Add).
        """

        def __init__(self, ch):
            super().__init__()
            self.model = nn.Sequential(
                ConvBlock(ch, ch, activation=True),  # .model.0.*
                ConvBlock(ch, ch, activation=False),  # .model.1.*
            )

        def forward(self, x):
            return x + self.model(x)

    class Encoder(nn.Module):  # pragma: no cover
        """4-level encoder: each level has stride-2 ConvBlock then ResidualBlock."""

        def __init__(self):
            super().__init__()
            self.conv_block_1 = ConvBlock(1, 32, stride=2)
            self.residual_block_1 = ResidualBlock(32)
            self.conv_block_2 = ConvBlock(32, 64, stride=2)
            self.residual_block_2 = ResidualBlock(64)
            self.conv_block_3 = ConvBlock(64, 128, stride=2)
            self.residual_block_3 = ResidualBlock(128)
            self.conv_block_4 = ConvBlock(128, 256, stride=2)
            self.residual_block_end = ResidualBlock(256)

        def forward(self, x):
            x = self.residual_block_1(self.conv_block_1(x))
            x = self.residual_block_2(self.conv_block_2(x))
            x = self.residual_block_3(self.conv_block_3(x))
            x = self.residual_block_end(self.conv_block_4(x))
            return x  # (B, 256, H/16, W/16)

    class Decoder(nn.Module):  # pragma: no cover
        """4-level decoder without skip connections (fuse=False).

        Bottleneck residual then 4 × ConvTranspose + ResidualBlock.
        """

        def __init__(self):
            super().__init__()
            self.residual_block_start = ResidualBlock(256)
            self.conv_block_1 = ConvTransposeBlock(256, 128)
            self.residual_block_1 = ResidualBlock(128)
            self.conv_block_2 = ConvTransposeBlock(128, 64)
            self.residual_block_2 = ResidualBlock(64)
            self.conv_block_3 = ConvTransposeBlock(64, 32)
            self.residual_block_3 = ResidualBlock(32)
            self.conv_block_4 = ConvTransposeBlock(32, 1)

        def forward(self, x):
            x = self.residual_block_start(x)
            x = self.residual_block_1(self.conv_block_1(x))
            x = self.residual_block_2(self.conv_block_2(x))
            x = self.residual_block_3(self.conv_block_3(x))
            x = self.conv_block_4(x)
            return x  # (B, 1, H, W)

    class SpeckleReductionNet(nn.Module):  # pragma: no cover
        """Multi-scale speckle reduction network with three encoders.

        For I_clean_hr the decoder only uses encoder_highRes output because
        the decoder has fuse=False. The lr/mid paths are used during training
        for the self-supervised consistency loss only.
        """

        def __init__(self):
            super().__init__()
            self.encoder_highRes = Encoder()
            self.encoder_lowRes = Encoder()
            self.encoder_midRes = Encoder()
            self.decoder = Decoder()

        def forward(self, hr, lr, mid):
            return (
                self.decoder(self.encoder_highRes(hr)),
                self.decoder(self.encoder_lowRes(lr)),
                self.decoder(self.encoder_midRes(mid)),
            )

    class _SRNSingleInputWrapper(nn.Module):  # pragma: no cover
        """Single-input ONNX wrapper — only runs the high-res path."""

        def __init__(self, srn):
            super().__init__()
            self.srn = srn

        def forward(self, x):
            return self.srn.decoder(self.srn.encoder_highRes(x))

    return {
        "SpeckleReductionNet": SpeckleReductionNet,
        "_SRNSingleInputWrapper": _SRNSingleInputWrapper,
    }


def convert_to_onnx(pth_path, onnx_path, input_size=(1, 1, 512, 512)):  # pragma: no cover
    """Convert a SpeckleReductionNet ``.pth`` checkpoint to ONNX.

    Uses a single-input wrapper (encoder_highRes + decoder) since the
    I_clean_hr output does not depend on the lr/mid paths (fuse=False).

    Args:
        pth_path (str): Path to the ``.pth`` checkpoint file.
        onnx_path (str): Destination path for the ONNX file.
        input_size (tuple): Dummy input shape ``(B, C, H, W)``.
            H and W must be multiples of 16.

    Raises:
        ImportError: If ``torch`` is not installed.
    """
    import torch

    classes = _build_torch_classes()
    srn = classes["SpeckleReductionNet"]()
    state = torch.load(pth_path, map_location="cpu")
    srn.load_state_dict(state, strict=False)
    srn.eval()

    wrapper = classes["_SRNSingleInputWrapper"](srn)
    wrapper.eval()

    dummy = torch.zeros(*input_size)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        do_constant_folding=True,
    )
    print(f"[✓] Exported ONNX model → {onnx_path}")


class _ConvBlock(keras.layers.Layer):
    """Conv2D(3x3, stride) + InstanceNorm + optional ReLU (channels_last).

    For ``stride > 1``, uses ``ZeroPadding2D(1) + Conv2D(valid)`` to
    reproduce PyTorch's symmetric ``padding=1`` exactly.  For ``stride == 1``,
    ``padding='same'`` is equivalent (symmetric by design when kernel%2==1).
    """

    def __init__(self, out_ch, stride=1, activation=True, **kwargs):
        super().__init__(**kwargs)
        if stride > 1:
            self.pad = keras.layers.ZeroPadding2D(padding=1)
            self.conv = keras.layers.Conv2D(
                out_ch, 3, strides=stride, padding="valid", use_bias=True
            )
        else:
            self.pad = None
            self.conv = keras.layers.Conv2D(
                out_ch, 3, strides=stride, padding="same", use_bias=True
            )
        # GroupNorm(groups=out_ch, eps=1e-5) ≡ InstanceNorm2D(affine=False, eps=1e-5)
        self.norm = keras.layers.GroupNormalization(
            groups=out_ch, scale=False, center=False, epsilon=1e-5
        )
        self.act = keras.layers.ReLU() if activation else None

    def call(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class _ConvTransposeBlock(keras.layers.Layer):
    """Conv2DTranspose(3x3, stride=2) + InstanceNorm + optional ReLU (channels_last).

    Uses ``padding='valid'`` and crops the first row/column of the full output
    to reproduce PyTorch's ``ConvTranspose2d(padding=1, output_padding=1)``
    alignment exactly.
    """

    def __init__(self, out_ch, activation=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2DTranspose(
            out_ch, 3, strides=2, padding="valid", use_bias=True
        )
        # GroupNorm(groups=out_ch, eps=1e-5) ≡ InstanceNorm2D(affine=False, eps=1e-5)
        self.norm = keras.layers.GroupNormalization(
            groups=out_ch, scale=False, center=False, epsilon=1e-5
        )
        self.act = keras.layers.ReLU() if activation else None

    def call(self, x):
        x = self.conv(x)
        # Crop to match PyTorch ConvTranspose2d(padding=1, output_padding=1):
        # full output is (2H+1)×(2W+1); keep rows/cols [1:] → 2H×2W
        x = x[:, 1:, 1:, :]
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class _ResidualBlock(keras.layers.Layer):
    """Two-layer residual: ConvBlock(+ReLU) → ConvBlock(no ReLU) + skip Add."""

    def __init__(self, ch, **kwargs):
        super().__init__(**kwargs)
        self.block0 = _ConvBlock(ch, stride=1, activation=True)
        self.block1 = _ConvBlock(ch, stride=1, activation=False)
        self.add = keras.layers.Add()

    def call(self, x):
        return self.add([x, self.block1(self.block0(x))])


class _Encoder(keras.layers.Layer):
    """4-level encoder: stride-2 ConvBlock + ResidualBlock per level."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_block_1 = _ConvBlock(32, stride=2)
        self.residual_block_1 = _ResidualBlock(32)
        self.conv_block_2 = _ConvBlock(64, stride=2)
        self.residual_block_2 = _ResidualBlock(64)
        self.conv_block_3 = _ConvBlock(128, stride=2)
        self.residual_block_3 = _ResidualBlock(128)
        self.conv_block_4 = _ConvBlock(256, stride=2)
        self.residual_block_end = _ResidualBlock(256)

    def call(self, x):
        x = self.residual_block_1(self.conv_block_1(x))
        x = self.residual_block_2(self.conv_block_2(x))
        x = self.residual_block_3(self.conv_block_3(x))
        x = self.residual_block_end(self.conv_block_4(x))
        return x  # (B, H/16, W/16, 256)


class _Decoder(keras.layers.Layer):
    """4-level decoder without skip connections (fuse=False)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.residual_block_start = _ResidualBlock(256)
        self.conv_block_1 = _ConvTransposeBlock(128)
        self.residual_block_1 = _ResidualBlock(128)
        self.conv_block_2 = _ConvTransposeBlock(64)
        self.residual_block_2 = _ResidualBlock(64)
        self.conv_block_3 = _ConvTransposeBlock(32)
        self.residual_block_3 = _ResidualBlock(32)
        self.conv_block_4 = _ConvTransposeBlock(1)  # final output layer

    def call(self, x):
        x = self.residual_block_start(x)
        x = self.residual_block_1(self.conv_block_1(x))
        x = self.residual_block_2(self.conv_block_2(x))
        x = self.residual_block_3(self.conv_block_3(x))
        x = self.conv_block_4(x)
        return x  # (B, H, W, 1)


class _SpeckleReductionNetKeras(keras.layers.Layer):
    """Keras encoder_highRes + decoder path (channels_last, NHWC)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    def call(self, x):
        return self.decoder(self.encoder(x))


def _load_pth_into_keras_net(keras_net, state_dict):  # pragma: no cover
    """Transfer weights from a PyTorch state dict into a Keras SRN model.

    Handles the axis permutation required between PyTorch (NCHW) and Keras
    (NHWC) weight layouts:

    * Conv2D:      PT ``[O, I, kH, kW]`` → Keras ``[kH, kW, I, O]``
    * Conv2DTranspose: PT ``[I, O, kH, kW]`` → Keras ``[kH, kW, O, I]``

    The same ``(2, 3, 1, 0)`` permutation applies to both.

    Args:
        keras_net: :class:`_SpeckleReductionNetKeras` instance (must be built).
        state_dict: dict of ``{key: tensor_or_ndarray}`` from PyTorch.
    """
    sd = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in state_dict.items()}

    def _set(keras_conv, pt_key_base):
        w = np.transpose(sd[f"{pt_key_base}.weight"], (2, 3, 1, 0))
        b = sd[f"{pt_key_base}.bias"]
        keras_conv.set_weights([w, b])

    def _set_res(keras_res, pt_prefix):
        _set(keras_res.block0.conv, f"{pt_prefix}.model.0.conv")
        _set(keras_res.block1.conv, f"{pt_prefix}.model.1.conv")

    # Encoder (uses encoder_highRes weights)
    enc = keras_net.encoder
    _set(enc.conv_block_1.conv, "encoder_highRes.conv_block_1.conv")
    _set_res(enc.residual_block_1, "encoder_highRes.residual_block_1")
    _set(enc.conv_block_2.conv, "encoder_highRes.conv_block_2.conv")
    _set_res(enc.residual_block_2, "encoder_highRes.residual_block_2")
    _set(enc.conv_block_3.conv, "encoder_highRes.conv_block_3.conv")
    _set_res(enc.residual_block_3, "encoder_highRes.residual_block_3")
    _set(enc.conv_block_4.conv, "encoder_highRes.conv_block_4.conv")
    _set_res(enc.residual_block_end, "encoder_highRes.residual_block_end")

    # Decoder
    dec = keras_net.decoder
    _set_res(dec.residual_block_start, "decoder.residual_block_start")
    _set(dec.conv_block_1.conv, "decoder.conv_block_1.conv")
    _set_res(dec.residual_block_1, "decoder.residual_block_1")
    _set(dec.conv_block_2.conv, "decoder.conv_block_2.conv")
    _set_res(dec.residual_block_2, "decoder.residual_block_2")
    _set(dec.conv_block_3.conv, "decoder.conv_block_3.conv")
    _set_res(dec.residual_block_3, "decoder.residual_block_3")
    _set(dec.conv_block_4.conv, "decoder.conv_block_4.conv")


@model_registry(name="speckle2self")
class Speckle2Self(BaseModel):
    """Self-supervised speckle reduction model for ultrasound images.

    Native Keras 3 implementation of the Speckle2Self architecture
    (Li et al., Medical Image Analysis, 2025).

    .. note::

        The model applies per-image linear normalisation
        before the network and clips outputs to ``[0, 1]``.

    Example:
        .. code-block:: python

            import numpy as np
            from zea.models.speckle2self import Speckle2Self

            model = Speckle2Self.from_preset("speckle2self-invivo")
            env = np.random.rand(2, 512, 512, 1).astype("float32")
            out = model(env)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = _SpeckleReductionNetKeras()
        self._onnx_sess = None  # set by from_onnx(); overrides Keras inference

    def call(self, inputs):
        """Run speckle reduction on a batch of B-Mode images.

        Pads ``height`` and ``width`` to multiples of 16 (required by the 4-level
        encoder), runs inference, then crops back and clips to ``[0, 1]``.

        Args:
            inputs (array-like): B-Mode images. Auto-normalization is applied
                per image inside this method. Shape: ``(N, H, W, 1)``.

        Returns:
            np.ndarray: Despeckled images, same shape as input, values in
            ``[0, 1]``.
        """
        assert inputs.ndim == 4, (
            f"Input should have 4 dimensions (B, H, W, C), but has {inputs.ndim}."
        )

        assert inputs.shape[-1] == 1, f"Input should have 1 channel, but has {inputs.shape[-1]}."

        original_size = ops.shape(inputs)[1:3]
        inputs = ops.image.resize(inputs, [INFERENCE_SIZE, INFERENCE_SIZE])
        inputs = zea.func.normalize(inputs, output_range=(0, 1))

        if self._onnx_sess is not None:
            outputs = self._call_onnx(inputs)
        else:
            outputs = self._call_keras(inputs)

        outputs = ops.image.resize(outputs, original_size)

        outputs = ops.clip(outputs, 0, 1)

        return outputs

    def _call_keras(self, inputs):
        num_frames, height, width, num_channels = ops.shape(inputs)

        output = self.net(inputs, training=False)

        return output

    def _call_onnx(self, inputs):  # pragma: no cover
        inputs = ops.convert_to_numpy(inputs).astype(np.float32)

        inputs = np.transpose(inputs, (0, 3, 1, 2))  # (N, 1, H, W) for ONNX input

        num_frames, num_channels, height, width = inputs.shape

        in_name = self._onnx_sess.get_inputs()[0].name
        out_name = self._onnx_sess.get_outputs()[0].name
        output = self._onnx_sess.run([out_name], {in_name: inputs})[0]

        output = np.transpose(output, (0, 2, 3, 1))  # back to (N, H, W, 1)

        return output

    def _load_from_pth(self, pth_path):  # pragma: no cover
        """Load weights from a PyTorch ``.pth`` checkpoint into the Keras net.

        Args:
            pth_path (str): Path to the ``.pth`` file produced by PyTorch.
        """
        import torch  # only needed for weight loading

        # Build the full model (outer Speckle2Self) so save_weights works later
        if not self.built:
            self(np.zeros((1, 16, 16, 1), dtype=np.float32))

        state_dict = torch.load(pth_path, map_location="cpu")
        _load_pth_into_keras_net(self.net, state_dict)

    def custom_load_weights(self, preset, backend="keras", **kwargs):
        """Load weights from a preset (Hugging Face or local directory).

        Args:
            preset: Preset identifier passed from :meth:`from_preset`.
                Accepts Hugging Face handles (``hf://...``) or local directory
                paths.
            backend: Which backend to use for loading weights. Options:

                - ``"keras"``: Load native Keras weights from ``model.weights.h5``.
                - ``"torch"``: Load PyTorch checkpoint from ``model.pth``, original
                    source for the weights.
                - ``"onnx"``: Load ONNX file from ``model.onnx`` using ONNX Runtime.

        """
        loader = get_preset_loader(preset)

        if backend == "keras":
            self._onnx_sess = None
            filename = loader.get_file("model.weights.h5")
            if not self.built:
                self(ops.zeros((1, 16, 16, 1), dtype="float32"))
            self.load_weights(filename)
        elif backend == "onnx":  # pragma: no cover
            try:
                import onnxruntime
            except ImportError as e:
                raise ImportError(
                    "Install onnxruntime or provide a .weights.h5 file for Speckle2Self"
                ) from e
            filename = loader.get_file("model.onnx")
            self._onnx_sess = onnxruntime.InferenceSession(filename)
        elif backend == "torch":  # pragma: no cover
            self._onnx_sess = None
            filename = loader.get_file("model.pth")
            self._load_from_pth(filename)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported backend '{backend}' for Speckle2Self preset")

    @classmethod
    def from_pth(cls, pth_path):  # pragma: no cover
        """Create a Speckle2Self model from a local PyTorch ``.pth`` file.

        Args:
            pth_path (str): Path to the ``.pth`` checkpoint.

        Returns:
            Speckle2Self: Fully initialised Keras model.

        Raises:
            ImportError: If ``torch`` is not installed.
        """
        model = cls()
        model._load_from_pth(pth_path)
        return model

    @classmethod
    def from_onnx(cls, onnx_path):  # pragma: no cover
        """Create a Speckle2Self model from a local ONNX file (legacy fallback).

        Use :meth:`from_pth` or :meth:`from_preset` when possible.

        Args:
            onnx_path (str): Path to the ``.onnx`` file.

        Returns:
            Speckle2Self: Model instance using ONNX Runtime for inference.

        Raises:
            ImportError: If ``onnxruntime`` is not installed.
        """
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError(
                "onnxruntime is not installed. "
                "Please run `pip install onnxruntime` to use this model."
            ) from e
        model = cls()
        model._onnx_sess = onnxruntime.InferenceSession(onnx_path)
        return model


register_presets(speckle2self_presets, Speckle2Self)
