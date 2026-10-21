<p align="center">
  <a href="https://zea.readthedocs.io/en/latest/getting-started.html"><img src="https://raw.githubusercontent.com/tue-bmd/zea/main/docs/_static/zea-logo.png" width="140" alt="zea logo"></a>
</p>

<h1 align="center">zea</h1>
<p align="center"><em>A Toolbox for Cognitive Ultrasound Imaging</em></p>

<p align="center">
  <a href="https://pypi.org/project/zea/"><img src="https://img.shields.io/pypi/v/zea" alt="PyPI version"></a>
  <a href="https://github.com/tue-bmd/zea/actions/workflows/tests.yaml"><img src="https://github.com/tue-bmd/zea/actions/workflows/tests.yaml/badge.svg" alt="Continuous integration"></a>
  <a href="https://zea.readthedocs.io/en/latest/?badge=latest"><img src="https://img.shields.io/readthedocs/zea" alt="Documentation Status"></a>
  <a href="https://codecov.io/gh/tue-bmd/zea"><img src="https://codecov.io/gh/tue-bmd/zea/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/tue-bmd/zea/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tue-bmd/zea" alt="License"></a>
  <br>
  <a href="https://joss.theoj.org/papers/fa923917ca41761fe0623ca6c350017d"><img src="https://joss.theoj.org/papers/fa923917ca41761fe0623ca6c350017d/status.svg" alt="JOSS"></a>
  <a href="https://arxiv.org/abs/2512.01433"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=flat&logo=arXiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/zeahub"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://github.com/tue-bmd/zea/stargazers"><img src="https://img.shields.io/github/stars/tue-bmd/zea?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="https://zea.readthedocs.io"><b>Documentation</b></a> ·
  <a href="https://zea.readthedocs.io/en/latest/getting-started.html"><b>Getting Started</b></a> ·
  <a href="https://zea.readthedocs.io/en/latest/examples.html"><b>Examples</b></a> ·
  <a href="https://zea.readthedocs.io/en/latest/models.html"><b>Models</b></a> ·
  <a href="https://huggingface.co/zeahub"><b>Datasets</b></a>
</p>

`zea` is a Python library for ultrasound signal processing, image reconstruction, and deep learning. It is built on [Keras 3](https://keras.io/keras_3/), so everything runs on [JAX](https://github.com/google/jax), [PyTorch](https://github.com/pytorch/pytorch), or [TensorFlow](https://github.com/tensorflow/tensorflow).

<!-- Showcase animation: rendered and hosted in https://gist.github.com/tristan-deep/6d0bb210f95fc530ca63fd8721402c30 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/tristan-deep/6d0bb210f95fc530ca63fd8721402c30/raw/zea_showcase_dark.webp">
    <img src="https://gist.githubusercontent.com/tristan-deep/6d0bb210f95fc530ca63fd8721402c30/raw/zea_showcase_light.webp" width="100%" alt="Animated tour of the zea modules: Pipeline, models, Parameters and Probe, and File, Dataset and Dataloader">
  </picture>
</p>

## Installation

```shell
pip install zea
```

Additionally, `zea` needs one of the Keras backends installed: [JAX](https://jax.readthedocs.io/en/latest/installation.html) (recommended), [PyTorch](https://pytorch.org/get-started/locally/), or [TensorFlow](https://www.tensorflow.org/install). See the [installation guide](https://zea.readthedocs.io/en/latest/installation.html) for details regarding more advanced installation options, such as GPU support.

## Quick start

`zea` offers many models and tooling for advanced ultrasound processing. But in the most basic sense it offers a fast and flexible ultrasound processing pipeline implemented in a deep learning framework for easy integration with state-of-the-art models. The following example streams an in-vivo recording from [Hugging Face](https://huggingface.co/zeahub), beamforms it, and saves the B-mode image:

```python
import zea

zea.init_device()

config = zea.Config.from_path("hf://zeahub/picmus/config_iq.yaml")
path = (
    "hf://zeahub/picmus/in_vivo/carotid_cross/"
    "carotid_cross_expe_dataset_iq/carotid_cross_expe_dataset_iq.hdf5"
)

with zea.File(path) as file:
    data = file.data.raw_data[0]
    parameters = file.load_parameters()

parameters.update(**config.parameters)

pipeline = zea.Pipeline.from_config(config, with_batch_dim=False)
inputs = pipeline.prepare_parameters(parameters)
image = pipeline(data=data, **inputs, return_numpy=True)["data"]

image = zea.display.to_8bit(image)
image.save("bmode.png")
```

The [Getting Started](https://zea.readthedocs.io/en/latest/getting-started.html) guide walks through this example step by step, and the [Examples](https://zea.readthedocs.io/en/latest/examples.html) page has many more notebooks you can open in Colab.

## Core modules

| | |
|---|---|
| 🔧 **[Pipeline](https://zea.readthedocs.io/en/latest/pipeline.html)** | Flexible, differentiable signal processing and image reconstruction, written in your favorite deep learning framework. |
| 💾 **[Data](https://zea.readthedocs.io/en/latest/data-acquisition.html)** | Loading tools for ultrasound data and acquisition parameters, designed for deep learning workflows. Datasets on Hugging Face are streamed, so only the frames you read are fetched. |
| 🧠 **[Models](https://zea.readthedocs.io/en/latest/models.html)** | Pretrained models for ultrasound image and signal processing, hosted on [Hugging Face](https://huggingface.co/zeahub). |
| 🎯 **[Agent](https://zea.readthedocs.io/en/latest/agent.html)** | Action selection for cognitive ultrasound, with flexible transmit and acquisition design: an action decides what to acquire next, such as which scan lines to fire. |

Read the [About](https://zea.readthedocs.io/en/latest/about.html) page for the motivation behind `zea` and the [research](https://zea.readthedocs.io/en/latest/about.html#papers) it supports.

## Contributing

Questions and suggestions are welcome as an [issue on GitHub](https://github.com/tue-bmd/zea/issues). If you want to contribute, see the [Contributing](https://zea.readthedocs.io/en/latest/contributing.html) guide.

## Citation

If `zea` helps your research, please cite our [JOSS paper](https://doi.org/10.21105/joss.09881). More citation info is in the [docs](https://zea.readthedocs.io/en/latest/getting-started.html#citation).

<details>
<summary>BibTeX</summary>

```bibtex
@article{stevens2026zea,
    author       = {Stevens, Tristan S.W. and van Nierop, Wessel L. and Luijten, Ben and van de Schaft, Vincent and Nolan, Oisín and Federici, Beatrice and van Harten, Louis D. and Penninga, Simon W. and Schueler, Noortje I.P. and van Sloun, Ruud J.G.},
    title        = {{zea}: A Toolbox for Cognitive Ultrasound Imaging},
    year         = 2026,
    journal      = {Journal of Open Source Software},
    publisher    = {The Open Journal},
    volume       = 11,
    number       = 121,
    pages        = 9881,
    doi          = {10.21105/joss.09881},
}
```

</details>
