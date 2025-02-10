"""
Tiny Autoencoder (TAESD) model converted to Tensorflow.

Source: https://github.com/madebyollin/taesd
See example usage in [examples/taesd](examples/taesd).

- **Author(s)**     : Wessel van Nierop
"""

from pathlib import Path

import keras
import tensorflow as tf
from keras import backend, ops
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
    """[TAESD](https://github.com/madebyollin/taesd) model in TensorFlow."""

    def __init__(self, **kwargs):
        """
        Initializes the TAESD model with the given parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the superclass initializer.
        """
        if backend.backend() not in ["tensorflow", "jax"]:
            raise NotImplementedError(
                "TinyDecoder is only currently supported with the "
                "TensorFlow or Jax backend."
            )

        _fix_tf_to_jax_resize_nearest_neighbor()

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
        # NOTE: Here you can compress the encoding a little bit more by going
        # to uint8 like in the original model
        # https://github.com/huggingface/diffusers/blob/cd30820/src/diffusers/models/autoencoders/autoencoder_tiny.py?plain=1#L336-L342 # pylint: disable=line-too-long
        decoded = self.decode(encoded)
        return decoded

    def custom_load_weights(self, preset, **kwargs):  # pylint: disable=unused-argument
        """Load the weights for the encoder and decoder."""
        self.encoder.custom_load_weights(preset)
        self.decoder.custom_load_weights(preset)


class TinyBase(BaseModel):
    """Base class for TAESD encoder and decoder."""

    def __init__(self, tiny_type=None, **kwargs):
        # Assertions
        assert tiny_type in [
            "encoder",
            "decoder",
        ], "Type must be either 'encoder' or 'decoder'."
        if backend.backend() not in ["tensorflow", "jax"]:
            raise NotImplementedError(
                f"{self.__class__.__name__} is only currently supported with the "
                "TensorFlow or Jax backend."
            )

        super().__init__(**kwargs)
        self.network = None

        self.download_files = [
            f"{tiny_type}/variables/variables.data-00000-of-00001",
            f"{tiny_type}/variables/variables.index",
            f"{tiny_type}/saved_model.pb",
            f"{tiny_type}/fingerprint.pb",
        ]

    def build(self, input_shape):
        """Builds the network."""
        self.maybe_convert_to_jax(input_shape)

    def maybe_convert_to_jax(self, input_shape):
        """Converts the network to Jax if backend is Jax."""
        if backend.backend() == "jax":
            inputs = ops.zeros(input_shape)
            jax_func, jax_params = tf2jax.convert(tf.function(self.network), inputs)

            def call_fn(
                params, state, rng, inputs, training
            ):  # pylint: disable=unused-argument
                return jax_func(state, inputs)

            self.network = keras.layers.JaxLayer(call_fn, state=jax_params)

    def _load_layer(self, path: Path | str):
        if backend.backend() == "tensorflow":
            return keras.layers.TFSMLayer(path, call_endpoint="serving_default")
        elif backend.backend() == "jax":
            return tf.saved_model.load(path)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} is only currently supported with the "
                f"TensorFlow or Jax backend. You are using {backend.backend()}."
            )

    def custom_load_weights(self, preset, **kwargs):  # pylint: disable=unused-argument
        """Load the weights for the encoder or decoder."""
        loader = get_preset_loader(preset)

        for file in self.download_files:
            filename = loader.get_file(file)

        base_path = Path(filename).parent
        self.network = self._load_layer(base_path)

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        Applies the network to the input.
        """
        if self.network is None:
            raise ValueError(
                f"Please load model using `{self.__class__.__name__}.from_preset()` before calling."
            )

        out = self.network(inputs)
        if backend.backend() == "tensorflow":
            # because decoded is dict, take first key
            out = out[next(iter(out))]
        return out


@model_registry(name="taesdxl_encoder")
class TinyEncoder(TinyBase):
    """Encoder from TAESD model."""

    def __init__(self, **kwargs):
        """
        Initializes the TAESD encoder.

        Args:
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init__(tiny_type="encoder", **kwargs)


@model_registry(name="taesdxl_decoder")
class TinyDecoder(TinyBase):
    """Decoder from TAESD model."""

    def __init__(self, **kwargs):
        """
        Initializes the TAESD decoder.

        Args:
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init__(tiny_type="decoder", **kwargs)


def _fix_tf_to_jax_resize_nearest_neighbor():
    # This block of code is used to allow the Jax backend to work with TAESD
    # It overrides the ResizeNearestNeighbor op to allow align_corners=True and half_pixel_centers=True
    # This means outputs of the jax model might not be a 100% match to the tensorflow model
    if backend.backend() == "jax":
        try:
            import jax  # pylint: disable=import-outside-toplevel
            import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
            import tf2jax  # pylint: disable=import-outside-toplevel

            def _resize_nearest_neighbor(proto):
                """Parse a ResizeNearestNeighbor op."""
                tf2jax._src.ops._check_attrs(
                    proto, {"T", "align_corners", "half_pixel_centers"}
                )

                def _func(images: jnp.ndarray, size: jnp.ndarray) -> jnp.ndarray:
                    if len(images.shape) != 4:
                        raise ValueError(
                            "Expected A 4D tensor with shape [batch, height, width, channels], "
                            f"found {images.shape}"
                        )

                    inp_batch, _, _, inp_channels = images.shape
                    out_height, out_width = size.tolist()

                    return jax.image.resize(
                        images,
                        shape=(inp_batch, out_height, out_width, inp_channels),
                        method=jax.image.ResizeMethod.NEAREST,
                    )

                return _func

            # hack to allow align_corners=True and half_pixel_centers=True
            tf2jax._src.ops._jax_ops["ResizeNearestNeighbor"] = _resize_nearest_neighbor
        except ImportError as exc:
            raise ImportError(
                "To use the Jax backend with TAESD, please install the `tf2jax` package."
            ) from exc


register_presets(taesdxl_presets, TinyAutoencoder)
register_presets(taesdxl_encoder_presets, TinyEncoder)
register_presets(taesdxl_decoder_presets, TinyDecoder)
