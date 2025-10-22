"""
The model is the movilenetV2 based image quality model from:
Van De Vyver, et al. "Regional Image Quality Scoring for 2-D Echocardiography Using Deep Learning."
Ultrasound in Medicine & Biology 51.4 (2025): 638-649.

GitHub original repo: https://github.com/GillesVanDeVyver/arqee

The model is originally a PyTorch model converted to ONNX. The model predicts the regional image quality of
the myocardial regions in apical views. It can also be used to get the overall image quality by averaging the
regional scores.

Note:
-----
To use this model, you must install the `onnxruntime` Python package:

    pip install onnxruntime

This is required for ONNX model inference.
"""  # noqa: E501

import numpy as np
from keras import ops

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import regional_quality_presets

# Visualization colors and helper for regional quality (arqee-inspired)
QUALITY_COLORS = np.array(
    [
        [0.929, 0.106, 0.141],  # not visible, red
        [0.957, 0.396, 0.137],  # poor, orange
        [1, 0.984, 0.090],  # ok, yellow
        [0.553, 0.776, 0.098],  # good, light green
        [0.09, 0.407, 0.216],  # excellent, dark green
    ]
)
REGION_LABELS = [
    "basal_left",
    "mid_left",
    "apical_left",
    "apical_right",
    "mid_right",
    "basal_right",
    "annulus_left",
    "annulus_right",
]
QUALITY_CLASSES = ["not visible", "poor", "ok", "good", "excellent"]


@model_registry(name="mobilenetv2_regional_quality")
class MobileNetv2RegionalQuality(BaseModel):
    """
    MobileNetV2 based regional image quality scoring model for myocardial regions in apical views.

    This class loads an ONNX model and provides inference for regional image quality scoring tasks.
    """

    def preprocess_input(self, inputs):
        """
        Normalize input image(s) to [0, 255] range.

        Args:
            inputs (np.ndarray): Input image(s), any numeric range.

        Returns:
            np.ndarray: Normalized image(s) in [0, 255] range.
        """
        inputs = ops.convert_to_numpy(inputs).astype("float32")
        max_val = np.max(inputs)
        min_val = np.min(inputs)
        denom = max_val - min_val
        if denom > 0.0:
            inputs = (inputs - min_val) / denom * 255.0
        else:
            inputs = np.zeros_like(inputs, dtype=np.float32)
        return inputs

    def call(self, inputs):
        """
        Predict regional image quality scores for input image(s).

        Args:
            inputs (np.ndarray): Input image or batch of images.
            Shape: [batch, 1, 256, 256]

        Returns:
            np.ndarray: Regional quality scores.
                Shape is [batch, 8] with regions in order:
                basal_left, mid_left, apical_left, apical_right,
                mid_right, basal_right, annulus_left, annulus_right
        """
        if not hasattr(self, "onnx_sess"):
            raise ValueError("Model weights not loaded. Please call custom_load_weights() first.")
        input_name = self.onnx_sess.get_inputs()[0].name
        output_name = self.onnx_sess.get_outputs()[0].name
        inputs = self.preprocess_input(inputs)

        output = self.onnx_sess.run([output_name], {input_name: inputs})[0]
        slope = self.slope_intercept[0]
        intercept = self.slope_intercept[1]
        output_debiased = (output - intercept) / slope
        return output_debiased

    def custom_load_weights(self, preset, **kwargs):
        """Load ONNX model weights and bias correction for regional image quality scoring."""
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
        filename = loader.get_file("slope_intercept_bias_correction.npy")
        self.slope_intercept = np.load(filename)


register_presets(regional_quality_presets, MobileNetv2RegionalQuality)
