

<!-- This is the readme for the github page (more complete readme for pdocs can be found in usmbd/README.md) -->
# Ultrasound-BMd
Ultrasound-BMd (usbmd) is a collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts. Check out the full documentation by opening [index.html](docs/usbmd/index.html) locally in your browser.

The idea of this toolbox is that it is self-sustained, meaning ultrasound researchers can use the tools to create new models / algorithms and after completed, can add them to the toolbox. This repository is being maintained by researchers from the [BM/d lab](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab/) at Eindhoven University of Technology. Currently for [internal](LICENSE) use only.

In case of any questions, feel free to [contact](mailto:t.s.w.stevens@tue.nl).
## Quick setup
#### usbmd installation
This package can be installed like any open-source python package from PyPI.
Make sure you are in the root folder (`Ultrasound-BMd`) where the [`setup.py`](setup.py) file is located and run the following command from terminal:
```bash
python -m pip install -e .
```
For more detailed info on the installation check out the documentation.

#### usbmd import
After installation, you can use the package as follows in your own project:
```Python
# import usbmd package
import usbmd
# or if you want to use the Tensorflow tools
from usbmd import tensorflow_ultrasound as usmbd_tf
# or if you want to use the Pytorch tools
from usbmd import pytorch_ultrasound as usbmd_torch
```
