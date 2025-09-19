"""
The model is the nnU-Net model trained on the augmented CAMUS dataset from the following publication:
Van De Vyver, Gilles, et al.
"Generative augmentations for improved cardiac ultrasound segmentation using diffusion models."
arXiv preprint arXiv:2502.20100 (2025).

At the time of writing (17 September 2025) and to the best of my knowledge, it is the state-of-the-art model for
left ventricle segmentation on the CAMUS dataset.

The model is originally a PyTorch model converted to ONNX. The model segments the left ventricle and myocardium.
"""
from zea.internal.registry import model_registry
from zea.models.base import BaseModel
import onnxruntime
import numpy as np
import os
import requests
from zea.models.preset_utils import register_presets
from zea.models.presets import camus_presets



SEGMENTATION_WEIGHTS_URL = "https://huggingface.co/gillesvdv/augmented_camus_seg/resolve/main/augmented_camus_seg.onnx"

@model_registry(name="augmented_camus_seg")
class augmented_camus_seg(BaseModel):

    def __init__(self,):
        super().__init__()

    def call(self, input):
        if not hasattr(self, 'onnx_sess'):
            raise ValueError("Model weights not loaded. Please call custom_load_weights() first.")
        input_name = self.onnx_sess.get_inputs()[0].name
        output_name = self.onnx_sess.get_outputs()[0].name
        output = self.onnx_sess.run([output_name], {input_name: input})[0]
        return output

    def custom_load_weights(self,model_path="./augmented_camus_seg.onnx"):
        """
        Load the weights for the segmentation model.
        Downloads the model if it does not exist locally from SEGMENTATION_WEIGHTS_URL.
        :param model_path: Local path to save the model
        """
        if not os.path.exists(model_path):
            r = requests.get(SEGMENTATION_WEIGHTS_URL)
            with open(model_path, "wb") as f:
                f.write(r.content)
            print(f"Downloaded model to {model_path}")
        self.model_path = model_path
        self.onnx_sess = onnxruntime.InferenceSession(model_path)


register_presets(camus_presets, augmented_camus_seg)
    