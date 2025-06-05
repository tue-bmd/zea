<!-- This is the readme for the github page (more complete readme for pdocs can be found in usmbd/README.md) -->
# usbmd <img src="docs/_static/usbmd_logo_v3.svg" style="float: right; width: 20%; height: 20%;" align="right" alt="usbmd Logo" />

<!-- https://raw.githubusercontent.com/tue-bmd/usbmd/main/docs/_static/usbmd_logo_v3.png -->

<!-- [![Continuous integration](https://github.com/tue-bmd/usbmd/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/tue-bmd/usbmd/actions/workflows/ci-pipeline.yaml)
[![PyPI version](https://img.shields.io/pypi/v/usbmd)](https://pypi.org/project/usbmd/)
[![Documentation Status](https://readthedocs.org/projects/usbmd/badge/?version=latest)](https://usbmd.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/tue-bmd/usbmd)](https://github.com/tue-bmd/usbmd/blob/main/LICENSE) -->

The ultrasound toolbox (``usbmd``) is a collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts. Check out the full documentation [here](https://usbmd-toolbox.web.app/).

The idea of this toolbox is that it is self-sustained, meaning ultrasound researchers can use the tools to create new models / algorithms and after completed, can add them to the toolbox. This repository is being maintained by researchers from the [BM/d lab](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab/) at Eindhoven University of Technology. Currently for [internal](LICENSE) use only.

Currently ``usbmd`` offers:

- Complete ultrasound signal processing and image reconstruction [pipeline](usbmd/ops.py).
- A collection of [models](usbmd/models) for ultrasound image and signal processing.
- Multi-Backend Support via [Keras3](https://keras.io/keras_3/): You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax)


### 📖 Citation

If you use **usbmd** in your research, please cite our work.

You can find the citation information by clicking the **"Cite this repository"** button on the top right of this page.
