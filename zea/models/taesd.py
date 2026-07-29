"""
Tiny Autoencoder (TAESD) model.

.. doctest::

    >>> from zea.models.taesd import TinyAutoencoder

    >>> model = TinyAutoencoder.from_preset("taesdxl")  # doctest: +SKIP

.. important::
    This is a ``zea`` implementation of the model.
    For the original code, see `here <https://github.com/madebyollin/taesd>`_.

.. seealso::
    A tutorial notebook where this model is used:
    :doc:`../notebooks/models/taesd_autoencoder_example`.

"""

from pathlib import Path

import keras
from keras import backend, ops

from zea import log
from zea.backend import _import_tf
from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import taesdxl_decoder_presets, taesdxl_encoder_presets, taesdxl_presets


@model_registry(name="taesdxl")
class TinyAutoencoder(BaseModel):
    """Tiny Autoencoder model.

    .. note::

        This model currently only supports TensorFlow and Jax backends.

    """

    def __init__(self, **kwargs):
        """
        Initializes the TAESD model with the given parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the superclass initializer.
        """
        if backend.backend() not in ["tensorflow", "jax"]:
            raise NotImplementedError(
                "TinyDecoder is only currently supported with the TensorFlow or Jax backend."
            )

        tf = _import_tf(force=True)
        assert tf is not None, (
            "TensorFlow is not installed. Please install TensorFlow to use TinyAutoencoder. "
            "This is required even if you are using the Jax backend, the model is built "
            "using TensorFlow."
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
            inputs = ops.concatenate([inputs, inputs, inputs], axis=-1)  # grayscale to RGB
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

    def call(self, inputs):
        """Applies the full autoencoder to the input."""
        encoded = self.encode(inputs)
        # NOTE: Here you can compress the encoding a little bit more by going
        # to uint8 like in the original model
        # https://github.com/huggingface/diffusers/blob/cd30820/src/diffusers/models/autoencoders/autoencoder_tiny.py?plain=1#L336-L342 # noqa: E501
        decoded = self.decode(encoded)
        return decoded

    def custom_load_weights(self, preset, **kwargs):
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

    def maybe_convert_to_jax(self, input_shape):  # pragma: no cover
        """Converts the network to Jax if backend is Jax."""
        if backend.backend() == "jax":
            inputs = ops.zeros(input_shape)
            from zea.backend import tf2jax

            tf = _import_tf(force=True)

            jax_func, jax_params = tf2jax.convert(  # ty: ignore[unresolved-attribute]
                tf.function(self.network), inputs
            )

            def call_fn(params, state, rng, inputs, training):
                return jax_func(state, inputs)

            self.network = keras.layers.JaxLayer(call_fn, state=jax_params)

    def _load_layer(self, path: Path | str):  # pragma: no cover
        if backend.backend() == "tensorflow":
            return keras.layers.TFSMLayer(path, call_endpoint="serving_default")
        elif backend.backend() == "jax":
            tf = _import_tf(force=True)
            return tf.saved_model.load(path)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} is only currently supported with the "
                f"TensorFlow or Jax backend. You are using {backend.backend()}."
            )

    def custom_load_weights(self, preset, **kwargs):
        """Load the weights for the encoder or decoder."""
        loader = get_preset_loader(preset)

        for file in self.download_files:
            filename = loader.get_file(file)

        base_path = Path(filename).parent
        self.network = self._load_layer(base_path)

    def call(self, inputs):
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
    # It overrides the ResizeNearestNeighbor op to allow align_corners=True
    # and half_pixel_centers=True. This means outputs of the jax model might
    # not be a 100% match to the tensorflow model
    if backend.backend() != "jax":
        return

    import jax
    import jax.numpy as jnp

    from zea.backend import tf2jax

    def _resize_nearest_neighbor(proto):
        """Parse a ResizeNearestNeighbor op."""
        tf2jax._src.ops._check_attrs(proto, {"T", "align_corners", "half_pixel_centers"})  # ty: ignore[unresolved-attribute]  # fmt: skip

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
    tf2jax._src.ops._jax_ops["ResizeNearestNeighbor"] = _resize_nearest_neighbor  # ty: ignore[unresolved-attribute]  # fmt: skip # noqa: E501


register_presets(taesdxl_presets, TinyAutoencoder)
register_presets(taesdxl_encoder_presets, TinyEncoder)
register_presets(taesdxl_decoder_presets, TinyDecoder)


def convert_original_weights(model_name="madebyollin/taesdxl", output_dir=None):  # pragma: no cover
    """Convert the original PyTorch TAESD weights to TensorFlow / Keras v3 models.

    This is how the ``taesdxl`` presets on the Hugging Face Hub were created; it is
    kept here for reproducibility and is not needed to *use* the model.

    The conversion goes PyTorch -> ONNX -> TensorFlow and therefore needs a few
    extra packages that are not part of the ``zea`` dependencies::

        pip install torch diffusers[torch] onnx==1.16.1 onnxruntime==1.18.1 onnx2tf \\
            onnx-graphsurgeon onnxsim==0.4.33 sne4onnx sng4onnx tf-keras

    Args:
        model_name (str, optional): Hugging Face model id of the original PyTorch
            autoencoder. Defaults to ``"madebyollin/taesdxl"``.
        output_dir (str | Path, optional): Folder to write the converted models to.
            Defaults to a timestamped folder under ``./temp/zea``.

    Returns:
        Path: Folder containing the converted ``encoder`` and ``decoder`` models.
    """
    import time

    # Imported here (not at module level) so that merely importing this module never
    # pulls in torch or the onnx toolchain: they are only needed for the conversion.
    import torch
    from diffusers import AutoencoderTiny
    from onnx2tf import convert

    vae = AutoencoderTiny.from_pretrained(model_name, torch_dtype=torch.float32)
    vae.eval()

    if output_dir is None:
        output_dir = Path(
            f"./temp/zea/{model_name.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Both halves are exported separately: zea loads them as TinyEncoder / TinyDecoder.
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    }
    for name, submodel, example_input in [
        ("encoder", vae.encoder, torch.rand((1, 3, 256, 256), dtype=torch.float32)),
        ("decoder", vae.decoder, torch.rand((1, 4, 32, 32), dtype=torch.float32)),
    ]:
        onnx_path = str(output_dir / f"{name}.onnx")
        torch.onnx.export(
            submodel,
            (example_input,),
            onnx_path,
            input_names=["input"],
            dynamic_axes=dynamic_axes,
        )
        convert(onnx_path, output_folder_path=str(output_dir / name), output_keras_v3=True)

    log.success(f"Models saved to {log.yellow(str(output_dir))}")
    return output_dir
