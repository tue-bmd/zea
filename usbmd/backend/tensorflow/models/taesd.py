"""
Tiny Autoencoder (TAESD) model converted to Tensorflow.

Source: https://github.com/madebyollin/taesd
See example usage in [examples/taesd](examples/taesd).

- **Author(s)**     : Wessel van Nierop
"""

from pathlib import Path

import keras
from huggingface_hub import snapshot_download
from keras import ops


class TinyAutoencoder(keras.models.Model):
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
        if self.grayscale:
            inputs = ops.concatenate(
                [inputs, inputs, inputs], axis=-1
            )  # grayscale to RGB
        return self.encoder(inputs)

    def decode(self, inputs):
        decoded = self.decoder(inputs)
        if self.grayscale:
            decoded = ops.image.rgb_to_grayscale(decoded, data_format="channels_last")
        return decoded

    def call(self, inputs):
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        return decoded


def _load_layer(path, layer_name):
    assert layer_name in ["encoder", "decoder"]
    if path is None:
        path = snapshot_download("usbmd/taesdxl")  # will download encoder and decoder
    path = Path(path)
    layer = keras.layers.TFSMLayer(
        path / layer_name,
        call_endpoint="serving_default",
    )
    return layer


class TinyEncoder(keras.models.Model):
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

    def call(self, inputs):
        """
        Applies the encoder to the input.
        """
        encoded = self.encoder(inputs)
        return encoded[next(iter(encoded))]  # because encoded is dict, take first key


class TinyDecoder(keras.models.Model):
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

    def call(self, inputs):
        """
        Applies the decoder to the input.
        """

        decoded = self.decoder(inputs)
        return decoded[next(iter(decoded))]  # because decoded is dict, take first key
