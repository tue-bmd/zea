# Ultrasound-BMd
Collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts.

## Table of contents
* [Setup](#setup)
* [Data](#data)
* [Documentation](#documentation)

## Setup
### usbmd installation
This package can be installed like any open-source python package from PyPI.
Make sure you are in the root folder (`Ultrasound-BMd`) where the `setup.py` file is located and run the following command from terminal:
```bash
python -m pip install -e .
```
The -e option stands for editable, which is important because it allows you to change the source code of the package without reinstalling it. You can leave it out, but changing the code in the repository won't change the installation (which is OK if you do not need to change it).

You can use the package as follows:
```Python
# Local imports
from usbmd import tensorflow_ultrasound as usmbd_tf
```

### Conda environment
Install anaconda from [here](https://www.anaconda.com/products/individual#windows).

To reproduce the environment on your own machine perform:
```bash
conda env create -f conda/tf26_usbmd.yaml
```

The conda environment file is created with:
```bash
conda env export --from-history > conda/tf26_usbmd.yaml
```
the use of the `--from-history` flag leaves out dependencies and creates a cleaner export file.  Also, the environment file will work across different platforms as a result of this.

## Data

### Data paths
In order to use this repository and point to the correct data paths, you can enter the location of your dataroot in the [`common.py`](common.py) file. It is possible to add multiple devices / platforms per user by means of if statements.
The default location is `Z:\Ultrasound-BMd\data` which is the path to the data on the NAS.

### Datastructure
This repository can support multiply datastructures [TODO: insert which], but the preferred way makes use of the `hdf5` file format and is structured as follows:
```
data_file.hdf5                [unit], [array shape] 
└── US
    ├── data
    │   ├── real              [-], [n_angles, n_elem, n_ax]
    │   └── imag              [-], [n_angles, n_elem, n_ax] 
    │	
    ├── angles                [m], [n_angles]
    ├── initial_time          [s]
    ├── modulation_frequency  [Hz] 
    ├── probe_geometry        [m], [n_elem, 3]
    ├── sampling_frequency    [Hz]
    ├── sound_speed           [m/s]
    ├── PRF (optional)        [Hz]
    └── ... (other optional parameters)
```
### Data flow
The following terminology is used in the code when referring to different data types.
- `raw_data` --> The raw channel data, storing the time-samples from each distinct ultrasound transducer.
- `aligned_data` --> Time-of-flight (TOF) corrected data. This is the data that is time aligned with respect to the array geometry.
- `beamformed_data` --> Beamformed or also known as beamsummed data. Aligned data is coherently summed together along the elements. The data has now been transformed from the aperture domain to the spatial domain.
- `envelope_data` --> The envelope of the signal is here detected and the center frequency is removed from the signal.
- `image` --> After log compression of the envelope data, the image is formed.
- `image_sc` --> The scan converted image is transformed cartesian (`x, y`) format to account for possible curved arrays. Possibly interpolation is performed to obtain the preferred pixel resolution.

## Documentation
In order to document the code properly, please follow [these](docs/example_google_docstrings.py) docstring style guides when adding code to the repository.

In case of any questions, feel free to [contact](mailto:t.s.w.stevens@tue.nl).