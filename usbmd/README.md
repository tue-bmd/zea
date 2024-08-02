<!-- This is the readme for the pdoc documentation (used as header in index.html) -->
# Ultrasound toolbox

The ultrasound toolbox (usbmd) is a collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts.
The idea of this toolbox is that it is self-sustained, meaning ultrasound researchers can use the tools to create new models / algorithms and after completed, can add them to the toolbox. This repository is being maintained by researchers from the [BM/d lab](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab/) at Eindhoven University of Technology. Currently for [internal](LICENSE) use only.

In case of any questions, feel free to [contact](mailto:t.s.w.stevens@tue.nl).

## Table of contents

* [Quick setup](#quick-setup)
* [Data](#data)
* [How to use with Verasonics](#how-to-use-with-verasonics)
* [Detailed installation guide](#detailed-installation-guide)
* [How to contribute](#how-to-contribute)

## Quick setup

### Installation

This package can be installed like any open-source python package from PyPI.
Make sure you are in the root folder (`ultrasound-toolbox`) where the [`pyproject.toml`](../pyproject.toml) file is located and run the following command from terminal:

```shell
python -m pip install -e .
```

 For more detailed info on the installation check out the [detailed installation guide](#detailed-installation-guide).
 Alternatively, you can [run this code via Docker](#running-the-code-with-docker) using included [Dockerfile](../../Dockerfile).

### Getting started

#### Importing

After installation, you can use the package as follows in your own project:

```python
# import usbmd package
import usbmd
# or if you want to use the Tensorflow tools
import usbmd.backend.tensorflow as usbmd_tf
# or if you want to use the Pytorch tools
import usbmd.backend.pytorch as usbmd_torch
```

#### User interface

In order to get started with usbmd stand-alone, you can run [`ui.py`](ui.html), which runs the "user interface" tool for inspecting datasets. First, it will ask for a config file for which you can choose one of your own configs or one of the defaults in the [`configs`](../../configs) folder. Second, you can navigate to the appropriate datafile (make sure it is in the dataset you specified in the config). Depending on the settings, it will render and show the image. There are already some example configs:

```shell
python ui.py --config configs/config_picmus.yaml
```

If you make your own config, make sure it can be validated using the [config validation](utils/config_validation.html) schema. This ensures it has the correct structure and all required parameters are present.

#### GPU support

Make sure that before using any GPU enabled functionality (importing torch / tensorflow) the following code is run:

```python
# import the init_device function
from usbmd.utils.device import init_device

# initialize device manually
device = init_device("torch", "auto:1", hide_devices=None)

# or using your config
device = init_device(config.ml_library, config.device, hide_devices=config.hide_devices)
```

Alternatively, you can use the `setup` function using a config file, which will initialize the device and setup the data paths:

```python
# import the setup function
from usbmd.setup_usbmd import setup
config = setup("configs/config_picmus_rf.yaml")
```

## Data

### Data paths

In order to use this repository and point to the correct data paths, you'll need to create a user profile. We have a script to guide you through the setup and create your userprofile; start by running `python usbmd/datapaths.py` (see [`datapaths.py`](datapaths.html)). When you run this script, you will be prompted to provide a path to your data directory -- the default location is `Z:\Ultrasound-BMd\data` which is the path to the data on the NAS. Your user profile will then be created at `users.yaml`. Once it's created, you can edit your profile to add multiple devices or data paths -- see the example below.

```yaml
MY_USER_NAME:
  MY_DEVICE_NAME:
    system: windows
    data_root:
      local: D:\Datasets
      remote: Z:\Ultrasound-BMd\data
```

#### Datastructure

This repository can support custom datastructures by implementing your own [Dataset](datasets.html) class, but the preferred way makes use of the `.hdf5` file format. For more information on dataset format, see [usbmd/data_format/README.md](data_format/index.html). The datasets are structured as follows:

```c
data_file.hdf5                  // [unit], [array shape], [type]
├── data
│    │  (see data types)
│    └── `dtype`                // [-], [n_frames, n_tx, n_el, n_ax, n_ch], [float32]
│
│  (all settings go here)
├── scan
│    │── center_frequency       // [Hz], [-], [float32]
│    │── sampling_frequency     // [Hz], [-], [float32]
│    │── n_tx                   // [-], [-], [int16]
│    │── n_el                   // [-], [-], [int16]
│    │── n_ax                   // [-], [-], [int16]
│    │── angles                 // [rad], [n_tx, 2], [float32]
│    │── virtual_sources        // [m], [n_tx, 3], [float32]
│    │── transmit_apodization   // [-], [n_tx, n_el], [float32]
│    │── tzero                  // [-], [n_tx, n_el], [float32]
│    │── probe_geometry         // [m], [n_el, 3], [float32]
│    │── sound_speed            // [m/s], [-], [float32]
│    │── initial_times          // [s], [-], [float32]
│    └── ... (other optional parameters)
```

#### Data Flow Diagram

![Data flow](../docs/usbmd/diagrams_dataflow.png)
![Data flow](diagrams_dataflow.png)

#### Data types

The following terminology is used in the code when referring to different data types.

* `raw_data` --> The raw channel data, storing the time-samples from each distinct ultrasound transducer.
  - [n_frames, n_tx, n_el, n_ax, n_ch]
* `aligned_data` --> Time-of-flight (TOF) corrected data. This is the data that is time aligned with respect to the array geometry.
  - [n_frames, n_tx, n_el, n_ax, n_ch]
* `beamformed_data` --> Beamformed or also known as beamsummed data. Aligned data is coherently summed together along the elements. The data has now been transformed from the aperture domain to the spatial domain.
  - [n_frames, n_z, n_x]
* `envelope_data` --> The envelope of the signal is here detected and the center frequency is removed from the signal.
  - [n_frames, n_z, n_x]
* `image` --> After log compression of the envelope data, the image is formed.
  - [n_frames, n_z, n_x]
* `image_sc` --> The scan converted image is transformed cartesian (`x, y`) format to account for possible curved arrays. Possibly interpolation is performed to obtain the preferred pixel resolution.
  - [n_frames, output_size_z, output_size_x]

## How to use with Verasonics

Record plane wave data using the Verasonics system, for instance using your favorite flash angles example script. Then save the data using the save RF button which saves the matlab workspace to disk along with all acquisition parameters needed for reconstruction. You can use [`matlab.py`](data/convert/matlab.html) to convert those workspace files to usbmd format. One way to quickly read those generated `.hdf5` files is though the [`ui.py`](ui.html) script. Adapt one of the configs in template configs folder and point to your dataset. Then when running the [`ui.py`](ui.html) you can select that config and start visualizing your newly generated datafile.


## How to contribute

Please see [`CONTRIBUTING.md`](../CONTRIBUTING.md) on guidelines to contribute to this repository.
Make sure your code complies with the style formatting of this repo. To do that, check if pylint runs succesfully (10/10) by running the following in the root directory:

```shell
pip install pylint
pylint usbmd
```

Also make sure all the pytest tests are running succesfully (100%) by running the following command in the root directory:

```shell
pytest ./tests
```

Currently this is only required for the develop / main branch.
