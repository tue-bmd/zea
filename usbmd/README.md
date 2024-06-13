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
Make sure you are in the root folder (`ultrasound-toolbox`) where the [`setup.py`](../../setup.py) file is located and run the following command from terminal:

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

Record plane wave data using the Verasonics system, for instance using your favorite flash angles example script. Then save the data using the provided [`save_to_usbmd_format.m`](../../usbmd/verasonics/save_to_usbmd_format.m) script. Which will save the raw rf data, along with all acquisition parameters needed for reconstruction, to disk in `.hdf5` format. You can create your own dataset and inherite a sepate [Dataset](datasets.html), or simply copy the `.hdf5` datafile to the `Z:\Ultrasound-BMd\data\USBMD_Verasonics\raw_data` directory. This way, the default Verasonics dataset in the toolbox is used to load the data. Run the [`ui.py`](ui.html) script and select your newly generated datafile to visualize the data.

## Detailed installation guide

Recommended is to run in an anaconda environment.
Install anaconda from [here](https://docs.conda.io/en/latest/miniconda.html).

To reproduce the environment on your own machine run the following commands:

```shell
conda create -n usbmd python=3.9
conda activate usbmd
python -m pip install --upgrade pip

# Install usbmd
cd "<repo_root>" # e.g. cd "C:\Users\Projects\ultrasound-toolbox"
python -m pip install -e .
# which runs the following under the hood as well:
# pip install -r requirements.txt
```

The -e option stands for editable, which is important because it allows you to change the source code of the package without reinstalling it. You can leave it out, but changing the code in the repository won't change the installation (which is OK if you do not need to change it). Furthermore it installs all required dependencies, except for the Tensorflow and Pytorch libaries. This allows people to have a quick install of usbmd, if they do not need the ML tools. Also, often these installations for ML libraries are more involved and differ from system to system.

## Running the code with Docker

This package also includes a [Dockerfile](../../Dockerfile). that you can use to run the code in a containerized environment.

1. Install Docker on your machine. You can download [Docker from the official website](https://www.docker.com/get-started).

    If you are using VSCode, you can use the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension. This extension will automatically build and run the Docker container for you. You can start the docker container via [ctrl+shift+p] --> [Dev Containers: Reopen in container]. Similarly, you can close the session via [ctrl+shift+p] --> [Dev Containers: Reopen folder locally].

    If you are not using VSCode, you can follow the steps below to run the code in a Docker container:

2. Build the Docker image by running the following command from the root folder

    ```shell
    docker build -t ultrasound-bmd .
    ```

    This command will build a Docker image named ultrasound-bmd.

3. Run the Docker container by running the following command:

    ```shell
    docker run -it --rm -v /path/to/your/data:/data ultrasound-bmd
    ```

    This command will start a Docker container from the `ultrasound-bmd image`. The `-v` flag mounts the `/path/to/your/data` folder on your machine to the `/data` folder inside the container. This way, you can access the data from inside the container, and open an interactive shell inside the container (-it option). The --rm option ensures that the container is automatically removed when it stops.
    Note that you need to replace `<local_path_to_data>` with the absolute path to the directory on your local machine that contains the input data that you want to process.

### ML libraries installation

To install Tensorflow >= 2.8 ([installation guide](https://www.tensorflow.org/install/pip))

```shell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
# Verify install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

To install Pytorch >= 1.13 ([installation guide](https://pytorch.org/get-started/locally/))

```shell
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
conda install cudatoolkit
# Verify install:
python -c "import torch; print(torch.cuda.is_available())"
```

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
