<p align="center">
  <img src="https://raw.githubusercontent.com/tue-bmd/zea/main/docs/_static/zea-logo.png" width="140" alt="zea logo">
</p>

<h1 align="center">zea</h1>
<p align="center"><em>A Toolbox for Cognitive Ultrasound Imaging</em></p>

<p align="center">
  <a href="https://pypi.org/project/zea/"><img src="https://img.shields.io/pypi/v/zea" alt="PyPI version"></a>
  <a href="https://github.com/tue-bmd/zea/actions/workflows/tests.yaml"><img src="https://github.com/tue-bmd/zea/actions/workflows/tests.yaml/badge.svg" alt="Continuous integration"></a>
  <a href="https://zea.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/zea/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://github.com/tue-bmd/zea/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tue-bmd/zea" alt="License"></a>
  <a href="https://codecov.io/gh/tue-bmd/zea"><img src="https://codecov.io/gh/tue-bmd/zea/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://joss.theoj.org/papers/fa923917ca41761fe0623ca6c350017d"><img src="https://joss.theoj.org/papers/fa923917ca41761fe0623ca6c350017d/status.svg" alt="status"></a>
  <a href="https://arxiv.org/abs/2512.01433"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=flat&logo=arXiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/zeahub"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://github.com/tue-bmd/zea/stargazers"><img src="https://img.shields.io/github/stars/tue-bmd/zea?style=social" alt="GitHub stars"></a>
</p>

Welcome to the `zea` package.

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
> This package is under active development. See a list of ongoing [research](https://zea.readthedocs.io/en/latest/about.html#papers) supported by `zea`. We are happy to share it with the ultrasound community and hope it will be useful for your research as well.

> [!NOTE]
> 📖 Please cite `zea` in your publications if it helps your research. You can find citation info [here](https://zea.readthedocs.io/en/latest/getting-started.html#citation).
