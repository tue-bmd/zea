"""InversionNet: full-waveform inversion of ultrasound computed tomography data.

InversionNet is an encoder-decoder network that maps raw multi-source waveform
data straight to a speed-of-sound (SOS) map, without an iterative solver. It was
introduced for seismic FWI (Wu & Lin, 2020), and is the reference baseline of the
`OpenPros <https://open-pros.github.io/>`_ limited-view prostate USCT benchmark.

Usage
-----

.. code-block:: python

    from zea.models.inversionnet import InversionNet

    model = InversionNet.from_preset("inversionnet-openpros")
    sos = model(waveforms)  # (B, 1000, 161, 40) -> (B, 401, 161, 1) in [-1, 1]

Architecture notes
------------------
- Encoder: a stride-2 stack that collapses the ``(time, receiver)`` plane to
  ``1 x 1`` at 512 channels. The first levels stride over time only, because the
  waveform axis (1000 samples) is much longer than the receiver axis (161).
- Decoder: transposed convolutions back up to ``448 x 192``, cropped to the
  ``401 x 161`` image grid, followed by a ``tanh`` output block.
- The output is in ``[-1, 1]``; map it to physical units with
  :class:`~zea.ops.Normalize` (OpenPros uses ``1300-3600 m/s``).

.. admonition:: References

   Y. Wu and Y. Lin. *InversionNet: An Efficient and Accurate Data-Driven Full
   Waveform Inversion.* IEEE Transactions on Computational Imaging, 6:419-433, 2020.
   `DOI: 10.1109/TCI.2019.2956866 <https://doi.org/10.1109/TCI.2019.2956866>`_
   (`arXiv:1811.07875 <https://arxiv.org/abs/1811.07875>`_)

   H. Wang, Y. Wu, Y. Feng, P. Jin, L. Zhang, S. Feng, J. Wiskin, B. Turkbey,
   P. A. Pinto, B. J. Wood, S. Luo, Y. Chen, E. Boctor and Y. Lin.
   *OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed
   Tomography.* 2025. `arXiv:2505.12261 <https://arxiv.org/abs/2505.12261>`_

.. important::

    This is a ``zea`` implementation of the model. The ``inversionnet-openpros``
    weights are the pretrained baseline released with the OpenPros benchmark
    (`dataset <https://open-pros.github.io/>`_,
    `code <https://github.com/hanchenwang/OpenPros>`_, CC-BY-4.0). Please cite both
    papers above when you use them.

.. note::

    Because the encoder ends in a fixed ``8 x 6`` convolution and the decoder in a
    fixed crop, the input size is part of the architecture. The OpenPros preset
    expects exactly :data:`OPENPROS_INPUT_SHAPE`.
"""

import keras
import numpy as np

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import inversionnet_presets

__all__ = ["InversionNet", "OPENPROS_INPUT_SHAPE"]

#: Input shape ``(time, receivers, sources)`` the OpenPros preset was trained on.
OPENPROS_INPUT_SHAPE = (1000, 161, 40)

# PyTorch BatchNorm2d defaults, which Keras does not share. Keras' ``momentum`` weights
# the *existing* running statistic where PyTorch's weights the *incoming* batch, so
# PyTorch's default of 0.1 is Keras' 0.9. Only matters when training or fine-tuning.
_BN_EPSILON = 1e-5
_BN_MOMENTUM = 0.9


class ConvBlock(keras.layers.Layer):
    """Conv2D + BatchNorm + LeakyReLU (or ``tanh``), channels-last.

    PyTorch pads symmetrically, which Keras' ``padding="same"`` does not do for
    ``stride > 1``, so the padding is always made explicit with
    :class:`~keras.layers.ZeroPadding2D` + ``padding="valid"``.
    """

    def __init__(self, out_ch, kernel_size=3, strides=1, padding=1, activation="leaky_relu", **kw):
        super().__init__(**kw)
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
        self.pad = (
            keras.layers.ZeroPadding2D(((pad_h, pad_h), (pad_w, pad_w)))
            if (pad_h or pad_w)
            else None
        )
        self.conv = keras.layers.Conv2D(out_ch, kernel_size, strides=strides, padding="valid")
        self.norm = keras.layers.BatchNormalization(epsilon=_BN_EPSILON, momentum=_BN_MOMENTUM)
        self.act = (
            keras.layers.Activation("tanh")
            if activation == "tanh"
            else keras.layers.LeakyReLU(negative_slope=0.2)
        )

    def call(self, x):
        if self.pad is not None:
            x = self.pad(x)
        return self.act(self.norm(self.conv(x)))


class DeconvBlock(keras.layers.Layer):
    """Conv2DTranspose + BatchNorm + LeakyReLU, channels-last.

    Keras' ``padding="valid"`` returns the full transposed-convolution output;
    PyTorch's ``padding=p`` crops ``p`` from each side of it, which is what the
    :class:`~keras.layers.Cropping2D` here reproduces.
    """

    def __init__(self, out_ch, kernel_size, strides, padding=0, **kw):
        super().__init__(**kw)
        self.conv = keras.layers.Conv2DTranspose(
            out_ch, kernel_size, strides=strides, padding="valid"
        )
        self.crop = keras.layers.Cropping2D(padding) if padding else None
        self.norm = keras.layers.BatchNormalization(epsilon=_BN_EPSILON, momentum=_BN_MOMENTUM)
        self.act = keras.layers.LeakyReLU(negative_slope=0.2)

    def call(self, x):
        x = self.conv(x)
        if self.crop is not None:
            x = self.crop(x)
        return self.act(self.norm(x))


@model_registry(name="inversionnet")
class InversionNet(BaseModel):
    """Encoder-decoder network mapping USCT waveforms to a speed-of-sound map.

    Args:
        waveform_shape (tuple): Input shape ``(time, receivers, channels)``, without
            the batch axis. The architecture is tied to it — see the module note —
            so it is part of the config. Defaults to :data:`OPENPROS_INPUT_SHAPE`.
        enc_ch (tuple): Base-2 exponents of the encoder channel widths. The first
            and last entry are the input and bottleneck convolution; the entries
            in between each become a stride-2 block plus a stride-1 block.
        enc_side (tuple): Per intermediate encoder level, whether to stride over
            the time axis only (``1``) instead of both axes (``0``). Must have
            ``len(enc_ch) - 2`` entries.
        bottle_conv (tuple): Kernel of the final, ``"valid"`` encoder convolution.
            It collapses the feature map to ``1 x 1``, so it must equal that map's
            spatial size.
        bottle_deconv (tuple): Kernel of the first decoder transposed convolution,
            which expands ``1 x 1`` back to ``bottle_deconv``.
        dec_ch (tuple): Base-2 exponents of the decoder channel widths. Every
            entry past the first doubles the spatial size.
        crop (tuple): ``(top, bottom, left, right)`` pixels to remove from the
            decoder output to reach the image grid.

    Example:
        .. code-block:: python

            import numpy as np
            from zea.models.inversionnet import InversionNet, OPENPROS_INPUT_SHAPE

            model = InversionNet.from_preset("inversionnet-openpros")
            waveforms = np.zeros((1, *OPENPROS_INPUT_SHAPE), dtype="float32")
            sos = model(waveforms)  # (1, 401, 161, 1), in [-1, 1]
    """

    def __init__(
        self,
        waveform_shape=OPENPROS_INPUT_SHAPE,
        enc_ch=(6, 6, 6, 7, 7, 8, 8, 9),
        enc_side=(1, 0, 0, 0, 0, 0),
        bottle_conv=(8, 6),
        bottle_deconv=(7, 3),
        dec_ch=(9, 8, 7, 6, 5, 4, 3),
        crop=(23, 24, 15, 16),
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(enc_ch) != len(enc_side) + 2:
            raise ValueError(
                f"enc_ch must have two more entries than enc_side (the input and bottleneck "
                f"convolutions), got {len(enc_ch)} and {len(enc_side)}."
            )
        self.waveform_shape = tuple(waveform_shape)
        self.enc_ch = tuple(enc_ch)
        self.enc_side = tuple(enc_side)
        self.bottle_conv = tuple(bottle_conv)
        self.bottle_deconv = tuple(bottle_deconv)
        self.dec_ch = tuple(dec_ch)
        self.crop = tuple(crop)

        widths = [2**c for c in self.enc_ch]
        encoder = [ConvBlock(widths[0], kernel_size=(7, 1), strides=(2, 1), padding=(3, 0))]
        for level, side in enumerate(self.enc_side, start=1):
            width = widths[level]
            if side:
                encoder.append(ConvBlock(width, kernel_size=(3, 1), strides=(2, 1), padding=(1, 0)))
                encoder.append(ConvBlock(width, kernel_size=(3, 1), padding=(1, 0)))
            else:
                encoder.append(ConvBlock(width, strides=2))
                encoder.append(ConvBlock(width))
        encoder.append(ConvBlock(widths[-1], kernel_size=self.bottle_conv, padding=0))
        self.encoder = encoder

        decoder = []
        for i, exponent in enumerate(self.dec_ch):
            width = 2**exponent
            if i == 0:
                decoder.append(DeconvBlock(width, self.bottle_deconv, strides=2))
            else:
                decoder.append(DeconvBlock(width, 4, strides=2, padding=1))
            decoder.append(ConvBlock(width))
        self.decoder = decoder

        top, bottom, left, right = self.crop
        self.output_crop = keras.layers.Cropping2D(((top, bottom), (left, right)))
        self.output_block = ConvBlock(1, activation="tanh")

    def call(self, inputs):
        """Reconstruct a speed-of-sound map from waveform data.

        Args:
            inputs (array-like): Waveforms of shape ``(B, time, receivers, channels)``,
                preprocessed exactly as during training. For the OpenPros preset that
                is a sign-preserving ``log1p`` compression followed by a min-max
                normalization to ``[-1, 1]``.

        Returns:
            Tensor: Speed-of-sound maps of shape ``(B, height, width, 1)``, in
            ``[-1, 1]``.
        """
        if len(inputs.shape) != 4:
            raise ValueError(
                f"Input should have 4 dimensions (batch, time, receivers, channels), "
                f"but has {len(inputs.shape)}."
            )
        if tuple(inputs.shape[1:]) != self.waveform_shape:
            raise ValueError(
                f"Input should have shape (batch, {', '.join(map(str, self.waveform_shape))}), "
                f"but has {tuple(inputs.shape)}."
            )
        x = inputs
        for block in self.encoder:
            x = block(x)
        for block in self.decoder:
            x = block(x)
        return self.output_block(self.output_crop(x))

    def get_config(self):
        """Serialize the architecture arguments."""
        config = super().get_config()
        config.update(
            {
                "waveform_shape": self.waveform_shape,
                "enc_ch": self.enc_ch,
                "enc_side": self.enc_side,
                "bottle_conv": self.bottle_conv,
                "bottle_deconv": self.bottle_deconv,
                "dec_ch": self.dec_ch,
                "crop": self.crop,
            }
        )
        return config

    def _build(self):
        """Materialize the weights (symbolically, so no data is pushed through)."""
        if not self.built:
            self(keras.Input(batch_shape=(None, *self.waveform_shape)))

    def _load_from_pth(self, pth_path):  # pragma: no cover
        """Load an original PyTorch checkpoint into this Keras model.

        Args:
            pth_path (str): Path to the ``.pth`` state dict.
        """
        import torch  # only needed for weight conversion

        self._build()
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
        load_torch_state_dict(self, state_dict)

    def custom_load_weights(self, preset, backend="keras", **kwargs):
        """Load weights from a preset (Hugging Face handle or local directory).

        Args:
            preset (str): Preset identifier passed from :meth:`from_preset`.
            backend (str): ``"keras"`` loads ``model.weights.h5``; ``"torch"``
                converts the original ``model.pth`` checkpoint and needs PyTorch.
        """
        loader = get_preset_loader(preset)
        if backend == "keras":
            filename = loader.get_file("model.weights.h5")
            self._build()
            self.load_weights(filename)
        elif backend == "torch":  # pragma: no cover
            self._load_from_pth(loader.get_file("model.pth"))
        else:
            raise ValueError(f"Unsupported backend '{backend}' for InversionNet preset")

    @classmethod
    def from_pth(cls, pth_path, **kwargs):  # pragma: no cover
        """Create an :class:`InversionNet` from an original PyTorch checkpoint.

        Args:
            pth_path (str): Path to the ``.pth`` state dict.
            **kwargs: Passed to the constructor.

        Returns:
            InversionNet: Model with the converted weights.
        """
        model = cls(**kwargs)
        model._load_from_pth(pth_path)
        return model


def load_torch_state_dict(model, state_dict):  # pragma: no cover
    """Copy an original InversionNet PyTorch state dict into a Keras model.

    The PyTorch model is a flat ``nn.Sequential`` encoder and decoder, so its keys
    are ``{encoder,decoder}.{block}.layers.{0,1}.*`` (convolution, batch norm) and
    map one-to-one onto this model's block lists. Kernels are permuted from
    PyTorch's ``(out, in, h, w)`` to Keras' ``(h, w, in, out)``; the same
    permutation applies to transposed convolutions, whose PyTorch layout is
    ``(in, out, h, w)``.

    Args:
        model (InversionNet): Built model to load the weights into.
        state_dict (dict): ``{key: tensor}`` mapping from PyTorch.

    Raises:
        KeyError: If the state dict does not match the model's architecture.
    """
    arrays = {
        k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in state_dict.items()
    }

    def _set_block(block, prefix):
        block.conv.set_weights(
            [
                np.transpose(arrays[f"{prefix}.layers.0.weight"], (2, 3, 1, 0)),
                arrays[f"{prefix}.layers.0.bias"],
            ]
        )
        block.norm.set_weights(
            [
                arrays[f"{prefix}.layers.1.{name}"]
                for name in ("weight", "bias", "running_mean", "running_var")
            ]
        )

    for i, block in enumerate(model.encoder):
        _set_block(block, f"encoder.{i}")
    for i, block in enumerate(model.decoder):
        _set_block(block, f"decoder.{i}")
    _set_block(model.output_block, "output")


register_presets(inversionnet_presets, InversionNet)
