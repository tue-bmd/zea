"""
Tiny Autoencoder (TAESD) model converted to Tensorflow.

Source: https://github.com/madebyollin/taesd
See example usage in [examples/taesd](examples/taesd).

- **Author(s)**     : Wessel van Nierop
"""

from pathlib import Path

import keras
from keras import ops
from regex import P

from usbmd.models.base import BaseModel
from usbmd.models.preset_utils import get_preset_loader, register_presets
from usbmd.models.presets import (
    taesdxl_decoder_presets,
    taesdxl_encoder_presets,
    taesdxl_presets,
)
from usbmd.registry import model_registry


@model_registry(name="taesdxl")
class TinyAutoencoder(BaseModel):
    """[TAESD](https://github.com/madebyollin/taesd) model in TensorFlow.

    custom_load_weights is implemen
    """

    def __init__(self, **kwargs):
        """
        Initializes the TAESD model with the given parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the superclass initializer.
        """
        super().__init__(**kwargs)

        self.encoder = TinyEncoder()
        self.decoder = TinyDecoder()

        self._grayscale = False

    def encode(self, inputs):
        """Encode the input images.

        Args:
            inputs (tensor): Input images of shape (batch_size, height, width, channels).
        """
        if self.encoder.network is None or self.decoder.network is None:
            raise ValueError(
                "Please load model using `TinyAutoencoder.from_preset()` before calling."
            )

        if ops.shape(inputs)[-1] == 1:
            self._grayscale = True
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
        if self._grayscale:
            decoded = ops.image.rgb_to_grayscale(decoded, data_format="channels_last")
        return decoded

    def call(self, inputs):  # pylint: disable=arguments-differ
        """Applies the full autoencoder to the input."""
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        return decoded

    def custom_load_weights(self, preset, skip_mismatch=False, **kwargs):
        """TFSM layer does not support loading weights."""
        self.encoder.custom_load_weights(preset)
        self.decoder.custom_load_weights(preset)


@model_registry(name="taesdxl_encoder")
class TinyEncoder(BaseModel):
    """Encoder from TAESD model."""

    def __init__(self, **kwargs):
        """
        Initializes the TAESD encoder.

        Args:
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init__(**kwargs)

        self.download_files = [
            "encoder/variables/variables.data-00000-of-00001",
            "encoder/variables/variables.index",
            "encoder/saved_model.pb",
            "encoder/fingerprint.pb",
        ]
        self.network = None

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        Applies the encoder to the input.
        """
        if self.network is None:
            raise ValueError(
                "Please load model using `TinyEncoder.from_preset()` before calling."
            )
        encoded = self.network(inputs)
        return encoded[next(iter(encoded))]  # because encoded is dict, take first key

    def custom_load_weights(self, preset, **kwargs):
        """TFSM layer does not support loading weights."""
        loader = get_preset_loader(preset)

        for file in self.download_files:
            filename = loader.get_file(file)

        base_path = Path(filename)
        base_path = str(base_path).split("encoder")[0]

        self.network = _load_layer(base_path, "encoder")


@model_registry(name="taesdxl_decoder")
class TinyDecoder(BaseModel):
    """Decoder from TAESD model."""

    def __init__(self, **kwargs):
        """
        Initializes the TAESD decoder.

        Args:
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """

        super().__init__(**kwargs)

        self.download_files = [
            "decoder/variables/variables.data-00000-of-00001",
            "decoder/variables/variables.index",
            "decoder/saved_model.pb",
            "decoder/fingerprint.pb",
        ]
        self.network = None

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        Applies the decoder to the input.
        """
        if self.network is None:
            raise ValueError(
                "Please load model using `TinyDecoder.from_preset()` before calling."
            )
        decoded = self.network(inputs)
        return decoded[next(iter(decoded))]  # because decoded is dict, take first key

    def custom_load_weights(self, preset, **kwargs):
        """TFSM layer does not support loading weights."""
        loader = get_preset_loader(preset)
        for file in self.download_files:
            filename = loader.get_file(file)

        base_path = Path(filename)
        base_path = str(base_path).split("decoder")[0]

        self.network = _load_layer(base_path, "decoder")


def _load_layer(path, layer_name):
    assert layer_name in ["encoder", "decoder"]
    path = Path(path)
    layer = keras.layers.TFSMLayer(
        path / layer_name,
        call_endpoint="serving_default",
    )
    return layer


register_presets(taesdxl_presets, TinyAutoencoder)
register_presets(taesdxl_encoder_presets, TinyEncoder)
register_presets(taesdxl_decoder_presets, TinyDecoder)
