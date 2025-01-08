
<!-- This is the readme for the github page (more complete readme for pdocs can be found in usmbd/README.md) -->
# usbmd <img src="docs/static/usbmd_logo_v3.svg" style="float: right; width: 20%; height: 20%;" align="right" alt="usbmd Logo" />

The ultrasound toolbox (usbmd) is a collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts. Check out the full documentation [here](http://131.155.125.142:6001/) (only available within the TU/e network).

The idea of this toolbox is that it is self-sustained, meaning ultrasound researchers can use the tools to create new models / algorithms and after completed, can add them to the toolbox. This repository is being maintained by researchers from the [BM/d lab](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab/) at Eindhoven University of Technology. Currently for [internal](LICENSE) use only.

In case of any questions, feel free to [contact](mailto:t.s.w.stevens@tue.nl).

Currently usbmd offers:

- Complete ultrasound signal processing and image reconstruction pipeline.
- A collection of models for ultrasound image and signal processing.
- Multi-Backend Support via [Keras3](https://keras.io/keras_3/): You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax)


## Installation

### Editable install

This package can be installed like any open-source python package from PyPI.
Make sure you are in the root folder (`ultrasound-toolbox`) where the [`pyproject.toml`](pyproject.toml) file is located and run the following command from terminal:

```shell
pip install -e .[opencv-python-headless]
```

Other install options can be found in the [Install.md](Install.md) file.


## Example usage
After installation, you can use the package as follows in your own project. `usbmd` is written in Python on top of [Keras 3](https://keras.io/about/). This means that under the hood we use the Keras framework to implement the pipeline and models. Keras allows you to set a backend ("jax", "tensorflow", "torch" or "numpy"), which means you can use `usbmd` alongside all your projects that are implemented in their respective frameworks. To get started you first have to specify your preferred backend. This can be done by setting the `KERAS_BACKEND` environment variable, either in your code or in your terminal. The default backend used by `usbmd` is "numpy", if no backend is specified before importing `usbmd`. This will not allow you to use the GPU for processing.


```shell
# set the backend in your terminal
export KERAS_BACKEND="jax"
```

```python
# or set the backend in your code at the top of your script
import os
os.environ["KERAS_BACKEND"] = "jax"
```

After setting the backend you can simply import `usbmd`
```python
import usbmd
```

> **Note:** You should make sure to install the requirements for your chosen backend as these are not included by default in a plain usbmd install. For example, if you choose "jax" as your backend, make sure to install the necessary JAX libraries. The provided docker image (see [Install.md](Install.md)) has all the necessary libraries pre-installed. Alternatively, you can install the necessary libraries by running `pip install usbmd[jax]` although this is not extensively tested (yet).

More complete examples can be found in the [examples](examples) folder.

The easiest way to get started is to use the Interface class
```python
import matplotlib.pyplot as plt

from usbmd import Interface, setup

# choose your config file
# all necessary settings should be in the config file
config_path = "configs/config_picmus_rf.yaml"

# setup function handles local data paths, default config settings and GPU usage
# make sure to create your own users.yaml using usbmd/datapaths.py
users_paths = "users.yaml"
config = setup(config_path, users_paths, create_user=True)

# initialize the Interface class with your config
interface = Interface(config)
image = interface.run()
# interface.plot()

# plot the image
plt.figure()
plt.imshow(image, cmap="gray")
plt.show()
```

### Loading a single file
The `Interface` class is a convenient way to load and inspect your data. However for more custom use cases, you might want to load and process the data yourself.
We do this by manually loading a single usbmd file with `load_usbmd_file` and processing it with the `Process` class.
```python
import matplotlib.pyplot as plt

from usbmd import setup
from usbmd.data import load_usbmd_file
from usbmd.processing import Process

# choose your config file
# all necessary settings should be in the config file
config_path = "configs/config_picmus_rf.yaml"

# setup function handles local data paths, default config settings and GPU usage
# make sure to create your own users.yaml using usbmd/datapaths.py
users_paths = "users.yaml"
config = setup(config_path, users_paths, create_user=True)

# we now manually point to our data
data_path = "Z:/Ultrasound-BMd/data/USBMD_datasets/PICMUS/database/simulation/contrast_speckle/contrast_speckle_simu_dataset_rf/contrast_speckle_simu_dataset_rf.hdf5"

# only 1 frame in PICMUS to be selected
selected_frames = [0]

# loading a file manually using `load_usbmd_file`
data, scan, probe = load_usbmd_file(
    data_path, frames=selected_frames, config=config, data_type="raw_data"
)

# initialize the Process class
process = Process(config=config, scan=scan, probe=probe)

# initialize the processing pipeline so it know what kind
# of data it is processing and what it should output
process.set_pipeline(dtype="raw_data", to_dtype="image")

# index the first frame
data_frame = data[0]

# processing the data from raw_data to image
image = process.run(data_frame)

plt.figure()
plt.imshow(image, cmap="gray")

# we can also process a single plane wave angle by
# setting the `selected_transmits` parameter in the scan object
process.scan.selected_transmits = 1

image = process.run(data_frame)

plt.figure()
plt.imshow(image, cmap="gray")

# lastly instead of setting the pipeline with `dtype` and `to_dtype`
# we can also opt for passing a custom operation chain as follows

# initialize the processing pipeline
process.set_pipeline(
    operation_chain=[
        {"name": "tof_correction"},
        {"name": "delay_and_sum"},
        {"name": "demodulate"},
        {"name": "envelope_detect"},
        {"name": "downsample"},
        {"name": "normalize"},
        # we now only set log_compress parameters to show how it can be done
        # if you don't pass any parameters it will use default or
        # params from config / scan / probe
        {"name": "log_compress", "params": {"dynamic_range": (-40, 0)}},
    ],
)

image = process.run(data_frame)

plt.figure()
plt.imshow(image, cmap="gray")

```

### Handling multiple files (i.e. datasets)

You can also make use of the `USBMDDataSet` class to load and process multiple files at once.
We will have to manually initialize the `Scan` and `Probe` classes and pass them to the `Process` class. This was done automatically in the `Interface` in the first example.

```python
import matplotlib.pyplot as plt

import usbmd
from usbmd import setup
from usbmd.data import USBMDDataSet
from usbmd.probes import Probe
from usbmd.processing import Process
from usbmd.scan import Scan
from usbmd.utils import update_dictionary, safe_initialize_class
from usbmd.utils.device import init_device

device = init_device("torch", "auto:1")

# let's check if your usbmd version is up to date
assert usbmd.__version__ >= "2.0", "Please update usbmd to version 2.0 or higher"

# choose your config file with all your settings
config_path = "configs/config_picmus_rf.yaml"

# setup function handles local data paths, default config settings and GPU usage
# make sure to create your own users.yaml using usbmd/datapaths.py
users_paths = "users.yaml"
config = setup(config_path, users_paths, create_user=True)

# intialize the dataset
dataset = USBMDDataSet(config.data)

# get scan and probe parameters from the dataset and config
file_scan_params = dataset.get_scan_parameters_from_file()
file_probe_params = dataset.get_probe_parameters_from_file()
config_scan_params = config.scan

# merging of manual config and dataset scan parameters
scan_params = update_dictionary(file_scan_params, config_scan_params)
scan = safe_initialize_class(Scan, **scan_params)
probe = Probe(**file_probe_params)
process = Process(config=config, scan=scan, probe=probe)

# initialize the processing pipeline
process.set_pipeline(
    operation_chain=[
        {"name": "tof_correction"},
        {"name": "delay_and_sum"},
        {"name": "demodulate"},
        {"name": "envelope_detect"},
        {"name": "downsample"},
        {"name": "normalize"},
        {"name": "log_compress"},
    ],
    device=device,
)

# pick a frame from the dataset
file_idx = 0
frame_idx = 10
data = dataset[(file_idx, frame_idx)]

# process the data
image = process.run(data)

# plot the image
plt.figure()
plt.imshow(image, cmap="gray")
plt.show()
```

### Batch processing
For batch processing you can request multiple frames from the `USBMDDataSet` class. For the `Process` we need to set a pipeline `with_batch_dim` processing set to True.

```python
file_idx = 0

# the following are now all valid `frame_idx` examples
frame_idx = 1 # just asking for a single frame
frame_idx = (0, 1, 2, 3) # asking for multiple frames
frame_idx = 'all' # return all frames of the file specified with `file_idx` in the dataset
data = dataset[(file_idx, frame_idx)]

# now it is wise to do inform the process class that we are processing a batch with `with_batch_dim=True`
# unless you picked a single frame with `frame_idx` then you can set it to False
process.set_pipeline(operation_chain=operation_chain, with_batch_dim=True)

images = process.run(data)
```

## Models

`usbmd` also contains a collection of models that can be used for various tasks. An example of how to use the `EchoNetDynamic` model is shown below. Simply use the `from_preset` method to load a model with a specific preset. All models can be found in the `usbmd.models` module.

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras import ops
import matplotlib.pyplot as plt

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.echonet import EchoNetDynamic
from usbmd.utils.selection_tool import add_shape_from_mask
from usbmd.utils.visualize import plot_image_grid, set_mpl_style

data_paths = set_data_paths()
init_device("tensorflow")

val_dataset = h5_dataset_from_directory(
    data_paths.data_root / "USBMD_datasets/CAMUS/val",
    key="data/image",
    batch_size=16,
    shuffle=True,
    image_size=[256, 256],
    resize_type="resize",
    image_range=[-60, 0],
    normalization_range=[-1, 1],
    seed=42,
)

presets = list(EchoNetDynamic.presets.keys())
log.info(f"Available built-in usbmd presets for EchoNet: {presets}")

model = EchoNetDynamic.from_preset("echonet-dynamic")

batch = next(iter(val_dataset))

masks = model(batch)

masks = ops.squeeze(masks, axis=-1)
masks = ops.convert_to_numpy(masks)

set_mpl_style()

# create figure of images in batch
fig, _ = plot_image_grid(batch)
axes = fig.axes[:batch.shape[0]]
for ax, mask in zip(axes, masks):
    # add segmentation on top of image in figure
    add_shape_from_mask(ax, mask, color="red", alpha=0.5)
plt.show()
```