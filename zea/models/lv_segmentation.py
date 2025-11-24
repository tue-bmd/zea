"""
nnU-Net segmentation model trained on the augmented CAMUS dataset.

To try this model, simply load one of the available presets:

.. doctest::

    >>> from zea.models.lv_segmentation import AugmentedCamusSeg

    >>> model = AugmentedCamusSeg.from_preset("augmented_camus_seg")

The model segments both the left ventricle and myocardium.

At the time of writing (17 September 2025) and to the best of our knowledge,
it is the state-of-the-art model for left ventricle segmentation on the CAMUS dataset.

.. important::
    This is a ``zea`` implementation of the model.
    For the original paper and code, see `here <https://github.com/GillesVanDeVyver/EchoGAINS>`_.

    Van De Vyver, Gilles, et al.
    "Generative augmentations for improved cardiac ultrasound segmentation using diffusion models."
    *https://arxiv.org/abs/2502.20100*

.. seealso::
    A tutorial notebook where this model is used:
    :doc:`../notebooks/models/left_ventricle_segmentation_example`.

.. note::
    The model is originally a PyTorch model converted to ONNX. To use this model, you must have `onnxruntime` installed. This is required for ONNX model inference.

    You can install it using pip:

    .. code-block:: bash

        pip install onnxruntime

"""  # noqa: E501

from keras import ops

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import augmented_camus_seg_presets

INFERENCE_SIZE = 256


@model_registry(name="augmented_camus_seg")
class AugmentedCamusSeg(BaseModel):
    """
    nnU-Net based left ventricle and myocardium segmentation model.

    - Trained on the augmented CAMUS dataset.
    - This class loads an ONNX model and provides inference for cardiac ultrasound segmentation tasks.

    """  # noqa: E501

    def call(self, inputs):
        """
        Run inference on the input data using the loaded ONNX model.

        Args:
            inputs (np.ndarray): Input image or batch of images for segmentation.
                Shape: [batch, 1, 256, 256]
                Range: Any numeric range; normalized internally.

        Returns:
            np.ndarray: Segmentation mask(s) for left ventricle and myocardium.
                Shape: [batch, 3, 256, 256]  (logits for background, LV, myocardium)

        Raises:
            ValueError: If model weights are not loaded.
        """
        if not hasattr(self, "onnx_sess"):
            raise ValueError("Model weights not loaded. Please call custom_load_weights() first.")
        input_name = self.onnx_sess.get_inputs()[0].name
        output_name = self.onnx_sess.get_outputs()[0].name
        inputs = ops.convert_to_numpy(inputs).astype("float32")
        output = self.onnx_sess.run([output_name], {input_name: inputs})[0]
        return output

    def custom_load_weights(self, preset, **kwargs):
        """Load the ONNX weights for the segmentation model."""
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "onnxruntime is not installed. Please run "
                "`pip install onnxruntime` to use this model."
            )
        loader = get_preset_loader(preset)
        filename = loader.get_file("model.onnx")
        self.onnx_sess = onnxruntime.InferenceSession(filename)


register_presets(augmented_camus_seg_presets, AugmentedCamusSeg)
