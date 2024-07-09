
<!-- This is the readme for the github page (more complete readme for pdocs can be found in usmbd/README.md) -->
# Ultrasound toolbox

The ultrasound toolbox (usbmd) is a collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts. Check out the full documentation by opening [index.html](docs/usbmd/index.html) locally in your browser.

The idea of this toolbox is that it is self-sustained, meaning ultrasound researchers can use the tools to create new models / algorithms and after completed, can add them to the toolbox. This repository is being maintained by researchers from the [BM/d lab](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab/) at Eindhoven University of Technology. Currently for [internal](LICENSE) use only.

In case of any questions, feel free to [contact](mailto:t.s.w.stevens@tue.nl).

## Installation

### Editable install

This package can be installed like any open-source python package from PyPI.
Make sure you are in the root folder (`ultrasound-toolbox`) where the [`setup.py`](setup.py) file is located and run the following command from terminal:

```bash
python -m pip install -e .
```

Other install options can be found in the [Install.md](Install.md) file.


## Example usage
After installation, you can use the package as follows in your own project:

```python
# import usbmd package
import usbmd
# or if you want to use the Tensorflow tools
import usbmd.backend.tensorflow as usbmd_tf
# or if you want to use the Pytorch tools
import usbmd.backend.pytorch as usbmd_torch
```

More complete examples can be found in the [examples](examples) folder.

The easiest way to get started is to use the DataloaderUI class
```python
import matplotlib.pyplot as plt

from usbmd.setup_usbmd import setup
from usbmd.ui import DataLoaderUI

# choose your config file
# all necessary settings should be in the config file
config_path = "configs/config_picmus_rf.yaml"

# setup function handles local data paths, default config settings and GPU usage
# make sure to create your own users.yaml using usbmd/datapaths.py
users_paths = "users.yaml"
config = setup(config_path, users_paths, create_user=True)

# initialize the DataloaderUI class with your config
ui = DataLoaderUI(config)
image = ui.run()
# ui.plot()

# plot the image
plt.figure()
plt.imshow(image, cmap="gray")
plt.show()
```

The `DataloaderUI` class is a convenient way to load and inspect your data. However for more custom use cases, you might want to load and process the data yourself.
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
        {"name": "beamform"},
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

You can also make use of the `USBMDDataSet` class to load and process multiple files at once.
We will have to manually initialize the `Scan` and `Probe` classes and pass them to the `Process` class. This was done automatically in the `DataloaderUI` in the first example.

```python
import matplotlib.pyplot as plt

import usbmd
from usbmd import setup
from usbmd.data import USBMDDataSet
from usbmd.probes import Probe
from usbmd.processing import Process
from usbmd.scan import Scan
from usbmd.utils import update_dictionary

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
scan = Scan(**scan_params)
probe = Probe(**file_probe_params)
process = Process(config=config, scan=scan, probe=probe)

# initialize the processing pipeline
process.set_pipeline(
    operation_chain=[
        {"name": "beamform"},
        {"name": "demodulate"},
        {"name": "envelope_detect"},
        {"name": "downsample"},
        {"name": "normalize"},
        {"name": "log_compress"},
    ],
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
