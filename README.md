# zea <img src="https://raw.githubusercontent.com/tue-bmd/zea/main/docs/_static/zea-logo.png" width="120" height="120" align="right" alt="zea Logo" />


[![PyPI version](https://img.shields.io/pypi/v/zea)](https://pypi.org/project/zea/)
[![Continuous integration](https://github.com/tue-bmd/zea/actions/workflows/tests.yaml/badge.svg)](https://github.com/tue-bmd/zea/actions/workflows/tests.yaml)
[![Documentation Status](https://readthedocs.org/projects/zea/badge/?version=latest)](https://zea.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/tue-bmd/zea)](https://github.com/tue-bmd/zea/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/tue-bmd/zea/branch/main/graph/badge.svg)](https://codecov.io/gh/tue-bmd/zea)
[![status](https://joss.theoj.org/papers/fa923917ca41761fe0623ca6c350017d/status.svg)](https://joss.theoj.org/papers/fa923917ca41761fe0623ca6c350017d)
[![arXiv](https://img.shields.io/badge/arXiv-B31B1B?style=flat&logo=arXiv&logoColor=white)](https://arxiv.org/abs/2512.01433)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/zeahub)
[![GitHub stars](https://img.shields.io/github/stars/tue-bmd/zea?style=social)](https://github.com/tue-bmd/zea/stargazers)

Welcome to the `zea` package: *A Toolbox for Cognitive Ultrasound Imaging.*

- 📚 Full documentation: [zea.readthedocs.io](https://zea.readthedocs.io)
- 🔬 Try hands-on examples (with Colab): [Examples & Tutorials](https://zea.readthedocs.io/en/latest/examples.html)
- ⚙️ Installation guide: [Installation](https://zea.readthedocs.io/en/latest/installation.html)

`zea` is a Python library that offers ultrasound signal processing, image reconstruction, and deep learning. Currently, `zea` offers:

- A flexible ultrasound signal processing and image reconstruction [Pipeline](https://zea.readthedocs.io/en/latest/pipeline.html) written in your favorite deep learning framework.
- A complete set of [Data](https://zea.readthedocs.io/en/latest/data-acquisition.html) loading tools for ultrasound data and acquisition parameters, designed for deep learning workflows.
- A collection of pretrained [Models](https://zea.readthedocs.io/en/latest/models.html) for ultrasound image and signal processing.
- A set of action selection functions for cognitive ultrasound in the [Agent](https://zea.readthedocs.io/en/latest/agent.html) module.
- **Multi-Backend Support via [Keras3](https://keras.io/keras_3/):** You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax).

Check out the [About](https://zea.readthedocs.io/en/latest/about.html) page for more information and the motivation behind `zea`. For any questions or suggestions, please feel free to open an [issue on GitHub](https://github.com/tue-bmd/zea/issues). If you want to contribute, check out the [Contributing](https://zea.readthedocs.io/en/latest/contributing.html) guide.

> [!WARNING]
> **Beta!**
> This package is highly experimental and under active development. It is mainly used to support [our research](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab) and as a basis for our publications. That being said, we are happy to share it with the ultrasound community and hope it will be useful for your research as well.

> [!NOTE]
> 📖 Please cite `zea` in your publications if it helps your research. You can find citation info [here](https://zea.readthedocs.io/en/latest/getting-started.html#citation).
