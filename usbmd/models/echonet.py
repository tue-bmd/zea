"""Echonet-Dynamic segmentation model for cardiac ultrasound segmentation.

- **Author(s)**     : Tristan Stevens, adapted from https://echonet.github.io/dynamic/
- **Date**          : 20/11/2023
"""

from pathlib import Path

import keras
import wget
from keras import backend, ops

from usbmd.models.base import BaseModel
from usbmd.models.preset_utils import get_preset_loader, register_presets
from usbmd.models.presets import echonet_dynamic_presets
from usbmd.registry import model_registry

INFERENCE_SIZE = 112

SEGMENTATION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/deeplabv3_resnet50_random.pt"
)
EJECTION_FRACTION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt"
)


@model_registry(name="echonet-dynamic")
class EchoNetDynamic(BaseModel):
    """EchoNet-Dynamic segmentation model for cardiac ultrasound segmentation.

    Original paper and code: https://echonet.github.io/dynamic/

    This class extracts useful parts of the original code and wraps it in a
    easy to use class.

    Preprocessing should normalize the input images with mean and standard deviation.

    """

    def __init__(self, **kwargs):
        if backend.backend() != "tensorflow":
            raise NotImplementedError(
                "EchoNetDynamic is only currently supported with the "
                "TensorFlow backend."
            )

        super().__init__(**kwargs)

        self.download_files = [
            "variables/variables.data-00000-of-00001",
            "variables/variables.index",
            "saved_model.pb",
            "fingerprint.pb",
        ]
        self.network = None

    def call(self, inputs):
        """Segment the input image."""
        if self.network is None:
            raise ValueError(
                "Please load model using `EchoNetDynamic.from_preset()` before calling."
            )

        assert (
            inputs.ndim == 4
        ), f"Input should have 4 dimensions (B, H, W, C), but has {inputs.ndim}."

        assert (
            inputs.shape[-1] == 1 or inputs.shape[-1] == 3
        ), f"Input should have 1 or 3 channels, but has {inputs.shape[-1]}."

        # resize image to 112x112
        original_size = ops.shape(inputs)[1:3]
        inputs = ops.image.resize(inputs, [INFERENCE_SIZE, INFERENCE_SIZE])

        if inputs.shape[-1] != 3:
            inputs = ops.tile(inputs, [1, 1, 1, 3])

        output = self.network(inputs)["segmentation"]

        # resize output to original size
        output = ops.image.resize(output, original_size)

        return output

    def custom_load_weights(self, preset, **kwargs):
        """TFSM layer does not support loading weights."""
        loader = get_preset_loader(preset)
        for file in self.download_files:
            filename = loader.get_file(file)

        base_path = Path(filename).parent

        self.network = keras.layers.TFSMLayer(
            base_path,
            call_endpoint="serving_default",
        )

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
