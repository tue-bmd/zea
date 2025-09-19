"""
The model is the movilenetV2 based image quality model from:
Van De Vyver, et al. "Regional Image Quality Scoring for 2-D Echocardiography Using Deep Learning."
Ultrasound in Medicine & Biology 51.4 (2025): 638-649.

The model is originally a PyTorch model converted to ONNX. The model predicts the regional image quality of
the myocardial regions in apical views. It can also be used to get the overall image quality by averaging the
regional scores.
"""
from zea.internal.registry import model_registry
from zea.models.base import BaseModel
import onnxruntime
import numpy as np
import os
from zea.models.preset_utils import register_presets
from zea.models.presets import camus_presets
from huggingface_hub import hf_hub_download

REPO_ID = 'gillesvdv/mobilenetv2_regional_quality'
FILE_NAME = "mobilenetv2_regional_quality.zip"

@model_registry(name="myocardial_img_quality")
class myocardial_img_quality(BaseModel):

    def __init__(self, ):
        super().__init__()

    def call(self, input):
        if not hasattr(self, 'onnx_sess'):
            raise ValueError("Model weights not loaded. Please call custom_load_weights() first.")
        input_name = self.onnx_sess.get_inputs()[0].name
        output_name = self.onnx_sess.get_outputs()[0].name
        # scale input to [0, 255]
        max_val = np.max(input)
        min_val = np.min(input)
        input = (input - min_val) / (max_val - min_val) * 255.0

        output = self.onnx_sess.run([output_name], {input_name: input})[0]
        slope = self.slope_intercept[0]
        intercept = self.slope_intercept[1]
        output_debiased = (output - intercept) / slope
        return output_debiased

    def custom_load_weights(self, model_dir="./"):
        """
        Load the weights for the segmentation model.
        Downloads the model if it does not exist locally from MODEL_WEIGHTS_URL.
        :param model_dir: Local path to folder where to save the model
        """
        onnx_model_path = os.path.join(model_dir, "mobilenetv2_regional_quality","model.onnx")
        slope_intercept_path = os.path.join(model_dir, "mobilenetv2_regional_quality","slope_intercept_bias_correction.npy")


        if not os.path.exists(onnx_model_path) or not os.path.exists(slope_intercept_path):
            downloaded_file_path = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, cache_dir=model_dir)
            os.system(f"unzip -o {downloaded_file_path} -d {model_dir}")

        self.model_path = onnx_model_path
        self.onnx_sess = onnxruntime.InferenceSession(onnx_model_path)
        self.slope_intercept = np.load(slope_intercept_path)


register_presets(camus_presets, myocardial_img_quality)