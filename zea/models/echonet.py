"""
Echonet-Dynamic segmentation model for cardiac ultrasound segmentation.

To try this model, simply load one of the available presets:

.. doctest::

    >>> from zea.models.echonet import EchoNetDynamic

    >>> model = EchoNetDynamic.from_preset("echonet-dynamic")  # doctest: +SKIP

.. important::
    This is a ``zea`` implementation of the model.
    For the original paper and code, see `here <https://echonet.github.io/dynamic/>`_.

    Ouyang, David, et al. "Video-based AI for beat-to-beat assessment of cardiac function."
    *Nature 580.7802 (2020): 252-256*

.. seealso::
    A tutorial notebook where this model is used:
    :doc:`../notebooks/models/left_ventricle_segmentation_example`.

"""

from pathlib import Path

import keras
import wget
from keras import backend, ops

from zea.backend import _import_tf
from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import echonet_dynamic_presets

INFERENCE_SIZE = 112

SEGMENTATION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/deeplabv3_resnet50_random.pt"
)
EJECTION_FRACTION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt"
)

tf = _import_tf()


@model_registry(name="echonet-dynamic")
class EchoNetDynamic(BaseModel):
    """EchoNet-Dynamic segmentation model for cardiac ultrasound segmentation.

    Preprocessing should normalize the input images with mean and standard deviation.

    """

    def __init__(self, **kwargs):
        if backend.backend() not in ["tensorflow", "jax"]:
            raise NotImplementedError(
                "EchoNetDynamic is only currently supported with the TensorFlow or Jax backend."
            )
        assert tf is not None, (
            "TensorFlow is not installed. Please install TensorFlow to use EchoNetDynamic. This is "
            "required even if you are using the Jax backend, the model is built using TensorFlow."
        )

        super().__init__(**kwargs)

        self.download_files = [
            "variables/variables.data-00000-of-00001",
            "variables/variables.index",
            "saved_model.pb",
            "fingerprint.pb",
        ]
        self.network = None

    def build(self, input_shape):
        """Builds the network."""
        self.maybe_convert_to_jax(input_shape)

    def maybe_convert_to_jax(self, input_shape):
        """Converts the network to Jax if backend is Jax."""
        if backend.backend() == "jax":
            inputs = ops.zeros(input_shape)
            from zea.backend import tf2jax

            jax_func, jax_params = tf2jax.convert(tf.function(self.network), inputs)

            def call_fn(params, state, rng, inputs, training):
                return jax_func(state, inputs)

            self.network = keras.layers.JaxLayer(call_fn, state=jax_params)

    def call(self, inputs):
        """Segment the input image."""
        if self.network is None:
            raise ValueError(
                "Please load model using `EchoNetDynamic.from_preset()` before calling."
            )

        assert inputs.ndim == 4, (
            f"Input should have 4 dimensions (B, H, W, C), but has {inputs.ndim}."
        )

        assert inputs.shape[-1] == 1 or inputs.shape[-1] == 3, (
            f"Input should have 1 or 3 channels, but has {inputs.shape[-1]}."
        )

        # resize image to 112x112
        original_size = ops.shape(inputs)[1:3]
        inputs = ops.image.resize(inputs, [INFERENCE_SIZE, INFERENCE_SIZE])

        if inputs.shape[-1] != 3:
            inputs = ops.tile(inputs, [1, 1, 1, 3])

        if backend.backend() == "tensorflow":
            output = self.network(inputs)["segmentation"]
        elif backend.backend() == "jax":
            output = self.network(inputs)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} is only currently supported with the "
                f"TensorFlow or Jax backend. You are using {backend.backend()}."
            )

        # resize output to original size
        output = ops.image.resize(output, original_size)

        return output

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

    def custom_load_weights(self, preset, **kwargs):
        """Load the weights for the segmentation model."""
        loader = get_preset_loader(preset)
        for file in self.download_files:
            filename = loader.get_file(file)

        base_path = Path(filename).parent

        self.network = self._load_layer(base_path)

    def _download_original_weights(self, weights_folder=None):
        """Download the originals weights from the EchoNet Github repository."""
        if weights_folder is None:
            weights_folder = "./echonet_weights"

        weights_folder = Path(weights_folder)
        url = SEGMENTATION_WEIGHTS_URL

        if not Path(weights_folder).exists():
            print(f"Creating folder at {weights_folder} to store weights")
            Path(weights_folder).mkdir()

        assert weights_folder.is_dir(), (
            f"weights_folder {weights_folder} is not a directory. "
            "Please specify the path to the folder containing the weights"
        )

        file_path = weights_folder / Path(url).name
        if not file_path.is_file():
            print(
                "Downloading Segmentation Weights, ",
                url,
                " to ",
                file_path,
            )
            filename = wget.download(url, out=str(weights_folder))

            assert Path(filename).name == Path(url).name, (
                f"Downloaded file {Path(filename).name} does not match expected filename "
                f"{Path(url).name}"
            )
            assert len(list(weights_folder.glob("*.pt"))) != 0, (
                f"No .pt files found in {weights_folder}. "
                "Please make sure the correct weights are downloaded."
            )

        else:
            print(f"EchoNet weights found in {file_path}")
        return file_path


register_presets(echonet_dynamic_presets, EchoNetDynamic)
