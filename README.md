# Ultrasound-BMd
Collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts.


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
