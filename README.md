test2

# Ultrasound-BMd
Collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts.

## Datastructure
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

## Data paths
In order to use this repository and point to the correct data paths, you can enter the location of your dataset in the [`common.py`](common.py) file. It is possible to add multiple devices / platforms per user by means of if statements.
The default location is `Z:\Ultrasound-BMd\data` which is the path to the data on the NAS.

## Environment
Install anaconda from [here](https://www.anaconda.com/products/individual#windows).

To reproduce the environment on your own machine perform:
```bash
conda env create -f conda/tf26_usbmd.yml
```

The conda environment file is created with:
```bash
conda env export --from-history > conda/tf26_usbmd.yml
```
the use of the `--from-history` flag leaves out dependencies and creates a cleaner export file.  Also, the environment file will work across different platforms as a result of this.

## Documentation
In order to document the code properly, please follow [these](docs/example_google_docstrings.py) docstring style guides when adding code to the repository.

In case of any questions, feel free to [contact](mailto:t.s.w.stevens@tue.nl).
