
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
from usbmd import tensorflow_ultrasound as usmbd_tf
# or if you want to use the Pytorch tools
from usbmd import pytorch_ultrasound as usbmd_torch
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
# make sure to create your own users.yaml using usbmd/common.py
config = setup(config_path, "users.yaml")

# initialize the DataloaderUI class with your config
ui = DataLoaderUI(config)
image = ui.run()
# ui.plot()

# plot the image
plt.figure()
plt.imshow(image, cmap="gray")
plt.show()
```

The DataloaderUI class is a convenient way to load and inspect your data. However for more custom use cases, you might want to load and process the data yourself.

```python
import matplotlib.pyplot as plt

from usbmd.data_format.usbmd_data_format import load_usbmd_file
from usbmd.processing import Process
from usbmd.setup_usbmd import setup_config

# choose your config file
# all necessary settings should be in the config file
config_path = "configs/config_picmus_rf.yaml"
# setup_config only loads, validates and sets defauls in the config file
config = setup_config(config_path)

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

# index the first frame
data_frame = data[0]

# processing the data from raw_data to image
image = process.run(data_frame, dtype="raw_data", to_dtype="image")

plt.figure()
plt.imshow(image, cmap="gray")

# we can also process a single plane wave angle by
# setting the `selected_transmits` parameter in the scan object
scan.selected_transmits = 1
process = Process(config=config, scan=scan, probe=probe)

image = process.run(data_frame, dtype="raw_data", to_dtype="image")

plt.figure()
plt.imshow(image, cmap="gray")
```