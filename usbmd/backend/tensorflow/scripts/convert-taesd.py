"""
Converts the taesdxl model to ONNX and then to TensorFlow.

Run a usbmd/keras3:latest container and install:
```bash
sudo pip install diffusers["torch"] \
onnx==1.16.1 \
nvidia-pyindex \
onnx-graphsurgeon \
onnxruntime==1.18.1 \
onnxsim==0.4.33 \
simple_onnx_processing_tools \
sne4onnx>=1.0.13 \
sng4onnx>=1.0.4 \
tensorflow>=2.17.0 \
protobuf==3.20.3 \
onnx2tf \
h5py>=3.11.0 \
psutil==5.9.5 \
ml_dtypes==0.3.2 \
tf-keras~=2.16 \
flatbuffers>=23.5.26
```
"""

import time
from pathlib import Path

import torch
from diffusers import AutoencoderTiny
from onnx2tf import convert

# Load torch model
model_name = "madebyollin/taesdxl"
# model_name = "madebyollin/taesd"
vae = AutoencoderTiny.from_pretrained(model_name, torch_dtype=torch.float32)
vae.eval()

# Set up input tensors
input_encoder_tensor = torch.rand((1, 3, 256, 256), dtype=torch.float32)
input_decoder_tensor = torch.rand((1, 4, 32, 32), dtype=torch.float32)

# Where to save the models
save_to_path = Path(
    "/mnt/z/Ultrasound-BMd/pretrained/taesdxl/wessel-{timestamp}".format(
        timestamp=time.strftime("%Y%m%d-%H%M%S")
    )
)
save_to_path.mkdir()

# Go to ONNX
torch.onnx.export(
    vae.encoder,  # model to export
    (input_encoder_tensor,),  # inputs of the model,
    str(save_to_path / "taesdxl-encoder.onnx"),  # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamic_axes={  # Allow dynamic axes for the spatial dimensions
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    },
)

torch.onnx.export(
    vae.decoder,  # model to export
    (input_decoder_tensor,),  # inputs of the model,
    str(save_to_path / "taesdxl-decoder.onnx"),  # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamic_axes={  # Allow dynamic axes for the spatial dimensions
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    },
)

# Convert to TF
convert(
    str(save_to_path / "taesdxl-encoder.onnx"),
    output_folder_path=str(save_to_path / "encoder"),
    output_keras_v3=True,
)
convert(
    str(save_to_path / "taesdxl-decoder.onnx"),
    output_folder_path=str(save_to_path / "decoder"),
    output_keras_v3=True,
)
