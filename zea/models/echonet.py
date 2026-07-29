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

.. note::
    This model is only currently supported with the TensorFlow or JAX :ref:`backend-installation`.
    When using TensorFlow as backend, the model will work out of the box. When using JAX as backend,
    the model is built using TensorFlow and then converted to JAX. This requires both
    TensorFlow and JAX to be installed, which can be tricky regarding compatible CUDA versions.
    One option is to run in our :ref:`docker-information` container, which has been tested
    to work with both backends.

"""

from pathlib import Path

import keras
import wget
from keras import backend, ops

from zea import log
from zea.backend import _import_tf
from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import echonet_dynamic_presets
from zea.models.utils import onnx2tf_saved_model_kwargs

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

    Preprocessing should normalize the input images with mean and standard deviation.

    """

    def __init__(self, **kwargs):
        if backend.backend() not in ["tensorflow", "jax"]:
            raise NotImplementedError(
                "EchoNetDynamic is only currently supported with the TensorFlow or JAX backend."
            )
        tf = _import_tf(force=True)
        assert tf is not None, (
            "TensorFlow is not installed. Please install TensorFlow to use EchoNetDynamic. This is "
            "required even if you are using the JAX backend, the model is built using TensorFlow. "
            "Installing JAX and TensorFlow together is tricky, regarding compatible CUDA versions. "
            "One option is to run in our Docker container, which has been tested to work with "
            "both backends. See https://zea.readthedocs.io/en/latest/installation.html#docker "
            "for more details."
        )

        super().__init__(**kwargs)

        self.download_files = [
            "variables/variables.data-00000-of-00001",
            "variables/variables.index",
            "saved_model.pb",
            "fingerprint.pb",
        ]
        self.network = None

    def build(self, input_shape):  # pragma: no cover
        """Builds the network."""
        self.maybe_convert_to_jax()

    def maybe_convert_to_jax(self):  # pragma: no cover
        """Converts the network to JAX if backend is JAX.

        JAX conversion traces the SavedModel using an example input of shape
        ``(1, INFERENCE_SIZE, INFERENCE_SIZE, 3)``. At runtime, ``call()`` may pass
        ``(B, INFERENCE_SIZE, INFERENCE_SIZE, 3)`` after resize/tile preprocessing.
        """
        if backend.backend() == "jax":
            from zea.backend import tf2jax

            tf = _import_tf(force=True)

            inputs = ops.zeros([1, INFERENCE_SIZE, INFERENCE_SIZE, 3])

            jax_func, jax_params = tf2jax.convert(  # ty: ignore[unresolved-attribute]
                tf.function(self.network), inputs
            )

            def call_fn(params, state, rng, inputs, training):
                with tf2jax.override_config(  # ty: ignore[unresolved-attribute]
                    "strict_shape_check", False
                ):
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
        """Load the weights for the segmentation model."""
        loader = get_preset_loader(preset)
        for file in self.download_files:
            filename = loader.get_file(file)

        base_path = Path(filename).parent

        self.network = self._load_layer(base_path)


register_presets(echonet_dynamic_presets, EchoNetDynamic)


def download_original_weights(weights_folder=None):
    """Download the original weights from the EchoNet Github repository.

    Args:
        weights_folder (str | Path, optional): Folder to download the weights into.
            Defaults to ``./echonet_weights``.

    Returns:
        Path: Path to the downloaded checkpoint.
    """
    if weights_folder is None:
        weights_folder = "./echonet_weights"

    weights_folder = Path(weights_folder)
    url = SEGMENTATION_WEIGHTS_URL

    if not weights_folder.exists():
        log.info(f"Creating folder at {weights_folder} to store weights")
        weights_folder.mkdir(parents=True)

    assert weights_folder.is_dir(), (
        f"weights_folder {weights_folder} is not a directory. "
        "Please specify the path to the folder containing the weights"
    )

    file_path = weights_folder / Path(url).name
    if not file_path.is_file():
        log.info(f"Downloading segmentation weights from {url} to {file_path}")
        filename = wget.download(url, out=str(weights_folder))

        assert Path(filename).name == Path(url).name, (
            f"Downloaded file {Path(filename).name} does not match expected filename "
            f"{Path(url).name}"
        )
    else:
        log.info(f"EchoNet weights found in {file_path}")
    return file_path


def convert_original_weights(output_dir=None, weights_folder=None):  # pragma: no cover
    """Convert the original PyTorch EchoNet-Dynamic weights to a TensorFlow SavedModel.

    This is how the ``echonet-dynamic`` preset on the Hugging Face Hub was created;
    it is kept here for reproducibility and is not needed to *use* the model.

    The conversion goes PyTorch -> ONNX -> TensorFlow and therefore needs a few
    extra packages that are not part of the ``zea`` dependencies::

        pip install torch torchvision onnx==1.16.1 onnxruntime==1.18.1 onnx2tf \\
            onnx-graphsurgeon onnxsim sne4onnx sng4onnx tf-keras

    onnxsim is not optional despite onnx2tf only warning when it is missing: without
    it the ASPP pooling branch keeps a dynamically sized Resize that the TensorFlow
    exporter rejects. It is left unpinned because 0.4.x has no wheel for recent
    Python versions.

    .. note::
        torch and TensorFlow have to coexist in one process here, which not every
        combination of wheels survives. Prefer the ``zeahub/all`` container, and
        import torch before this module if you hit a crash while building the
        torch model.

    Args:
        output_dir (str | Path, optional): Folder to write the converted model to.
            Defaults to a timestamped folder under ``./temp/zea``.
        weights_folder (str | Path, optional): Folder to download the original
            weights into. See :func:`download_original_weights`.

    Returns:
        Path: Folder containing the converted TensorFlow SavedModel.
    """
    import time

    # Imported here (not at module level) so that merely importing this module never
    # pulls in torch or the onnx toolchain: they are only needed for the conversion.
    import torch
    import torchvision
    from onnx2tf import convert

    checkpoint_path = download_original_weights(weights_folder)

    # No pretrained weights at all: the checkpoint below overwrites both the
    # backbone and the classifier, so downloading ImageNet weights is wasted work.
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None, weights_backbone=None, aux_loss=False
    )
    model.classifier[-1] = torch.nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    )

    # The checkpoint was saved from a DataParallel model, so its keys are prefixed.
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"])
    model.eval()

    if output_dir is None:
        output_dir = Path(f"./temp/zea/echonet-dynamic-{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = str(output_dir / "echonet-dynamic.onnx")
    torch.onnx.export(
        model.module,  # unwrap DataParallel
        (torch.rand((1, 3, INFERENCE_SIZE, INFERENCE_SIZE), dtype=torch.float32),),
        onnx_path,
        input_names=["input"],
        output_names=["segmentation"],
        # Only the batch axis is dynamic: ResizeBilinear fixes the spatial dimensions.
        # Keys must match input_names / output_names, or they are silently ignored.
        dynamic_axes={"input": {0: "batch_size"}, "segmentation": {0: "batch_size"}},
        # Legacy TorchScript exporter: the dynamo exporter (torch >= 2.9 default)
        # emits a dynamically sized upsample that onnx2tf cannot convert.
        dynamo=False,
    )

    saved_model_dir = output_dir / "tensorflow"
    convert(
        onnx_path,
        output_folder_path=str(saved_model_dir),
        output_keras_v3=False,
        output_signaturedefs=False,
        **onnx2tf_saved_model_kwargs(),
    )
    if not (saved_model_dir / "saved_model.pb").is_file():
        raise RuntimeError(
            f"onnx2tf reported success but wrote no SavedModel to {saved_model_dir}. "
            "See zea.models.utils.onnx2tf_saved_model_kwargs."
        )

    log.success(f"Model saved to {log.yellow(str(output_dir))}")
    return output_dir
