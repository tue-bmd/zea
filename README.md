<!-- This is the readme for the github page (more complete readme for pdocs can be found in usmbd/README.md) -->
# zea <img src="docs/_static/zea-logo.svg" style="float: right; width: 20%; height: 20%;" align="right" alt="zea Logo" />

<!-- https://raw.githubusercontent.com/tue-bmd/zea/main/docs/_static/zea-logo.png -->

<!-- [![Continuous integration](https://github.com/tue-bmd/zea/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/tue-bmd/zea/actions/workflows/ci-pipeline.yaml)
[![PyPI version](https://img.shields.io/pypi/v/zea)](https://pypi.org/project/zea/)
[![Documentation Status](https://readthedocs.org/projects/zea/badge/?version=latest)](https://zea.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/tue-bmd/zea)](https://github.com/tue-bmd/zea/blob/main/LICENSE) -->

Welcome to the documentation for the `zea` package: *A Toolbox for Cognitive Ultrasound Imaging.*

`zea` is a Python library that offers ultrasound signal processing, image reconstruction, and deep learning. Currently, `zea` offers:

- A flexible ultrasound signal processing and image reconstruction pipeline written in your favorite deep learning framework.
- A complete set of data acquisition loading tools for ultrasound data and acquisition parameters, designed for deep learning workflows.
- A collection of pretrained models for ultrasound image and signal processing.
- **Multi-Backend Support via [Keras3](https://keras.io/keras_3/):** You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax).

> [!WARNING]
> **Beta!**
> This package is highly experimental and under active development. It is mainly used to support [our research](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab) and as a basis for our publications. That being said, we are happy to share it with the ultrasound community and hope it will be useful for your research as well.

> [!NOTE]
> 📖 If you use `zea` in your research, please cite our work.
> You can find the citation information by clicking the **"Cite this repository"** button on the top right of this page.
