"""
Tiny Autoencoder (TAESD) model converted to Tensorflow.

Source: https://github.com/madebyollin/taesd
See example usage in [examples/taesd](examples/taesd).

- **Author(s)**     : Wessel van Nierop
"""

from pathlib import Path

import keras
from keras import ops

from usbmd.models.base import BaseModel
from usbmd.models.preset_utils import register_presets
from usbmd.models.presets import taesdxl_presets
from usbmd.registry import model_registry
from usbmd.tools.hf import load_model_from_hf


@model_registry(name="taesd")
class TinyAutoencoder(BaseModel):
    """[TAESD](https://github.com/madebyollin/taesd) model in TensorFlow."""

    def __init__(self, pretrained_path=None, grayscale=True, **kwargs):
        """
        Initializes the TAESD model with the given parameters.

        Args:
            pretrained_path (str): Path to the pretrained model. Default is None which
                will load from huggingface.
            grayscale (bool): Whether to use grayscale images. Default is True.
            **kwargs: Additional keyword arguments to pass to the superclass initializer.
        """
        super().__init__(**kwargs)
        self.pretrained_path = pretrained_path
        self.grayscale = grayscale

        self.encoder = TinyEncoder(self.pretrained_path)
        self.decoder = TinyDecoder(self.pretrained_path)

    def encode(self, inputs):
        """Encode the input images.

        Args:
            inputs (tensor): Input images of shape (batch_size, height, width, channels).
        """
        if self.grayscale:
            inputs = ops.concatenate(
                [inputs, inputs, inputs], axis=-1
            )  # grayscale to RGB
        return self.encoder(inputs)

    def decode(self, inputs):
        """Decode the encoded images.

        Args:
            inputs (tensor): Input images of shape (batch_size, height, width, 4).
        """
        decoded = self.decoder(inputs)
        if self.grayscale:
            decoded = ops.image.rgb_to_grayscale(decoded, data_format="channels_last")
        return decoded

    def call(self, inputs):  # pylint: disable=arguments-differ
        """Applies the full autoencoder to the input."""
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        return decoded

    def load_weights(self, filepath, skip_mismatch=False, **kwargs):
        """TFSM layer does not support loading weights."""
        pass


def _load_layer(path, layer_name):
    assert layer_name in ["encoder", "decoder"]
    if path is None:
        path = load_model_from_hf("usbmd/taesdxl")  # will download encoder and decoder
    path = Path(path)
    layer = keras.layers.TFSMLayer(
        path / layer_name,
        call_endpoint="serving_default",
    )
    return layer


@model_registry(name="taesd_encoder")
class TinyEncoder(keras.models.Model):
    """Encoder from TAESD model."""

    def __init__(self, pretrained_path=None, **kwargs):
        """
        Initializes the TAESD encoder.

        Args:
            pretrained_path (str): Path to the pretrained model directory. Default is None which
                will load from huggingface.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init__(**kwargs)
        self.pretrained_path = pretrained_path

        self.encoder = _load_layer(self.pretrained_path, "encoder")

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        Applies the encoder to the input.
        """
        encoded = self.encoder(inputs)
        return encoded[next(iter(encoded))]  # because encoded is dict, take first key

    def load_weights(self, filepath, skip_mismatch=False, **kwargs):
        """TFSM layer does not support loading weights."""
        pass


@model_registry(name="taesd_decoder")
class TinyDecoder(keras.models.Model):
    """Decoder from TAESD model."""

    def __init__(self, pretrained_path=None, **kwargs):
        """
        Initializes the TAESD decoder.

        Args:
            pretrained_path (str): Path to the pretrained model directory. Default is None which
                will load from huggingface.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """

        super().__init__(**kwargs)
        self.pretrained_path = pretrained_path

        self.decoder = _load_layer(self.pretrained_path, "decoder")

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        Applies the decoder to the input.
        """

        decoded = self.decoder(inputs)
        return decoded[next(iter(decoded))]  # because decoded is dict, take first key

    def load_weights(self, filepath, skip_mismatch=False, **kwargs):
        """TFSM layer does not support loading weights."""
        pass


register_presets(taesdxl_presets, TinyAutoencoder)
