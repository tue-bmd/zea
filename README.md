<!-- This is the readme for the github page (more complete readme for pdocs can be found in usmbd/README.md) -->
# zea <img src="docs/_static/zea-log.svg" style="float: right; width: 20%; height: 20%;" align="right" alt="zea Logo" />

<!-- https://raw.githubusercontent.com/tue-bmd/zea/main/docs/_static/zea-logo.png -->

<!-- [![Continuous integration](https://github.com/tue-bmd/zea/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/tue-bmd/zea/actions/workflows/ci-pipeline.yaml)
[![PyPI version](https://img.shields.io/pypi/v/zea)](https://pypi.org/project/zea/)
[![Documentation Status](https://readthedocs.org/projects/zea/badge/?version=latest)](https://zea.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/tue-bmd/zea)](https://github.com/tue-bmd/zea/blob/main/LICENSE) -->

The ultrasound toolbox (``zea``) is a collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts. Check out the full documentation [here](https://usbmd-toolbox.web.app/).

The idea of this toolbox is that it is self-sustained, meaning ultrasound researchers can use the tools to create new models / algorithms and after completed, can add them to the toolbox. This repository is being maintained by researchers from the [BM/d lab](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab/) at Eindhoven University of Technology. Currently for [internal](LICENSE) use only.

Currently ``zea`` offers:

- Complete ultrasound signal processing and image reconstruction [pipeline](zea/ops.py).
- A collection of [models](zea/models) for ultrasound image and signal processing.
- Multi-Backend Support via [Keras3](https://keras.io/keras_3/): You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax)


### 📖 Citation

If you use **zea** in your research, please cite our work.

You can find the citation information by clicking the **"Cite this repository"** button on the top right of this page.
