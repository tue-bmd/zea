"""
The model is the nnU-Net model trained on the augmented CAMUS dataset from the following publication:
Van De Vyver, Gilles, et al.
"Generative augmentations for improved cardiac ultrasound segmentation using diffusion models."
arXiv preprint arXiv:2502.20100 (2025).

GitHub original repo: https://github.com/GillesVanDeVyver/EchoGAINS

At the time of writing (17 September 2025) and to the best of our knowledge,
it is the state-of-the-art model for left ventricle segmentation on the CAMUS dataset.

The model is originally a PyTorch model converted to ONNX. The model segments the left ventricle and myocardium.

Note:
-----
To use this model, you must install the `onnxruntime` Python package:

    pip install onnxruntime

This is required for ONNX model inference.
"""  # noqa: E501

from keras import ops

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import augmented_camus_seg_presets


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
