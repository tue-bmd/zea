"""
Converts the echonet-dynamic model to ONNX and then to TensorFlow.

For more info see https://echonet.github.io/dynamic/

Run a usbmd/keras3:v2.1.1 (or try with latest) container and install:

```bash
wget \
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
- **Author(s)**     : Tristan Stevens
- **Date**          : 20/13/2024
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

SEGMENTATION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/deeplabv3_resnet50_random.pt"
)
EJECTION_FRACTION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt"
)


import time
from pathlib import Path

import torch
import torchvision
import wget

from usbmd import log

INFERENCE_SIZE = 112

SEGMENTATION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/deeplabv3_resnet50_random.pt"
)
EJECTION_FRACTION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt"
)


def download_weights(weights_folder):
    """Download the weights for the EchoNet segmentation model."""
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
    return file_path


file_path = download_weights("./echonet_weights")

model = torchvision.models.segmentation.deeplabv3_resnet50(
    pretrained=False, aux_loss=False
)
model.classifier[-1] = torch.nn.Conv2d(
    model.classifier[-1].in_channels,
    1,
    kernel_size=model.classifier[-1].kernel_size,
)

device = torch.device("cuda")
model = torch.nn.DataParallel(model)
model.to(device)
checkpoint = torch.load(file_path)
model.load_state_dict(checkpoint["state_dict"])

model.eval()

# Set up input tensors
input_tensor = torch.rand((1, 3, INFERENCE_SIZE, INFERENCE_SIZE), dtype=torch.float32)
input_tensor = input_tensor.to(device)

# Where to save the models
timestamp = time.strftime("%Y%m%d-%H%M%S")
save_to_path = Path(
    f"/mnt/z/Ultrasound-BMd/pretrained/echonet-dynamic/tristan-{timestamp}"
)
save_to_path.mkdir(parents=True, exist_ok=True)

output_onnx_path = str(save_to_path / "echonet-dynamic.onnx")
# Go to ONNX
torch.onnx.export(
    model.module,  # model to export, calling .module instead of model because of DataParallel
    (input_tensor,),  # inputs of the model,
    output_onnx_path,  # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    output_names=["segmentation"],  # Rename outputs for the ONNX model
    # couldn't do all spatial dimensions because of ResizeBilinear
    dynamic_axes={  # Allow dynamic axes for the spatial dimensions
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

# Convert to TF
model = convert(
    output_onnx_path,
    output_folder_path=save_to_path / "tensorflow",
    output_keras_v3=False,  # we do manually
    output_signaturedefs=False,
)

log.success(f"Model saved to {log.yellow(save_to_path)}")
