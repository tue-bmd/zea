---
title: 'zea: A Toolbox for Cognitive Ultrasound Imaging'
tags:
  - Python
  - ultrasound
  - Keras
  - machine learning
  - signal processing
  - deep generative models
authors:
  - name: Tristan S.W. Stevens
    orcid: 0000-0002-8563-5931
  - name: Wessel L. van Nierop
    orcid: 0009-0003-3141-3369
  - name: Ben Luijten
    orcid: 0000-0002-1797-8721
  - name: Vincent van de Schaft
    orcid: 0000-0002-8515-5372
  - name: Oisín Nolan
    orcid: 0009-0002-6939-7627
  - name: Beatrice Federici
    orcid: 0009-0003-2496-8825
  - name: Louis D. van Harten
    orcid: 0000-0002-0943-2825
  - name: Simon W. Penninga
    orcid: 0009-0003-4095-8168
  - name: Noortje I.P. Schueler
    orcid: 0009-0003-7134-6850
  - name: Ruud J.G. van Sloun
    orcid: 0000-0003-2845-0495
affiliations:
 - name: Eindhoven University of Technology, the Netherlands
date: 20 June 2025
bibliography: paper.bib

---

# Summary
Ultrasound imaging is a powerful medical imaging modality that is widely used in clinical settings for various applications, including obstetrics, cardiology, and abdominal imaging. While ultrasound imaging is non-invasive, real-time, and relatively low-cost compared to other imaging modalities such as MRI or CT, it still faces challenges in terms of image quality, and interpretation. Many signal processing steps are required to extract useful information from the raw ultrasound data, such as filtering, beamforming, and image reconstruction. Traditional ultrasound imaging techniques often suffer from reduced image quality as naive assumptions are made in these processing steps which do not account for the complex nature of ultrasound signals. Furthermore, acquisition (action) and reconstruction (perception) of ultrasound is often performed disjointly. Cognitive ultrasound imaging [@van2024active], see \autoref{fig:diagram}, is a novel approach that aims to address these challenges by leveraging more powerful generative models, enabled by advances in deep learning, to close the action-perception loop. This approach requires a redesign of current common ultrasound imaging pipeline, where parameters are expected to be changed dynamically based on past and current observations. Furthermore, the high-dimensional nature of ultrasound data requires powerful deep generative models to learn the structured distribution of ultrasound signals. This necessitates a flexible and efficient toolbox that can handle the complexities of cognitive ultrasound imaging, including a real-time ultrasound reconstruction pipeline, dynamic parameter adjustment, and advanced generative modeling.

We present `zea` (pronounced *ze-yah*), a Python package for cognitive ultrasound imaging that provides a flexible, modular and differentiable pipeline for ultrasound data processing, as well as a collection of pre-defined models for ultrasound image and signal processing. The toolbox is designed to be easy to use, with a high-level interface that allows users to define their own ultrasound reconstruction pipelines, and to integrate deep learning models into the pipeline. The toolbox is built on top of Keras 3 [@chollet2015keras], which provides a framework for building and training deep learning models with the three major deep learning frameworks as backend: TensorFlow [@abadi2016tensorflow], PyTorch [@NEURIPS2019_9015] and JAX [@jax2018github]. This means that it is easy to integrate a custom ultrasound reconstruction pipeline in a machine learning workflow. In the past few years, several works have used and contributed to `zea`, including @van2024off, @stevens2024dehazing, @nolan2024active, @federici2024active, @stevens2025sequential, @penninga2025deep and @stevens2025high.

![Schematic overview of the action-perception loop in ultrasound imaging.\label{fig:diagram}](zea_perception_action-Light.svg){ width=100% }

# Statement of need
The ultrasound research community has advanced significantly due to a variety of high-quality software, including simulation tools such as `Field II` [@jensen2004simulation] and `k-wave` [@treeby2010k], as well as reconstruction and real-time processing libraries like `USTB` [@rodriguez2017ultrasound], `MUST` [@garcia2021make], `ARRUS` [@jarosik2020arrus], `FAST` [@smistad2021fast], `QUPS` [@brevett2024qups], and `vbeam` [@magnus2023vbeam]. However, most existing solutions are not designed for cognitive ultrasound imaging, where the integration of deep learning and dynamic, closed-loop ultrasound reconstruction pipelines is essential. Our aim with `zea` is to provide a complementary, highly flexible and differentiable pipeline written in a modern deep learning framework, as well as offer a convenient platform to provide several pretrained models. This addresses the need for a modular and extensible library that supports cognitive ultrasound workflows and seamless integration with state-of-the-art machine learning models. While the full realization of cognitive ultrasound imaging remains an ongoing effort, we hope this toolbox will help spur further research and development in the field.

# Overview of functionality
`zea` is an open-source Python package, available at [http://github.com/tue-bmd/zea](http://github.com/tue-bmd/zea), that consists of the following core components:

- **Data**: A set of utility classes and functions such as `zea.data.File`, `zea.data.Dataset` and `make_dataloader()`, to handle data for machine learning workflows. `zea` works with HDF5 files, with data and acquisition parameters stored together in a single file. Finally, we provide some examples on popular ultrasound datasets, such as CAMUS [@leclerc2019deep], PICMUS [@liebgott2016plane], and EchoNet-dynamic [@ouyang2019echonet].
- **Pipeline**: A modular and differentiable pipeline class that allows users to define a sequence of operations to process ultrasound data. The pipeline is stateless and supports *Just in Time* (JIT) compilation for maximum performance. Ultimately this allows for dynamic parameter adjustment, as well as real-time integration of deep learning models inside the ultrasound reconstruction pipeline.
- **Models**: A collection of pre-defined models for ultrasound image and signal processing. Similar to the data, these models can be loaded locally or from the Hugging Face Hub. Besides more commonly supervised models, `zea` also provides a set of (deep) generative models, with an interface to solve inverse problems in ultrasound imaging within a probabilistic machine learning framework.
- **Agents**: A set of tools to interact with the pipeline and models. These agents can be used to alter the pipeline parameters, or select a subset of acquired data. The agent module closes the action-perception loop, tying together acquisition and reconstruction of ultrasound data.

# Example usage
Below, we will show a brief overview of how to use the main components of `zea`, including the data handling, pipeline, models, and agents. For more detailed examples and use cases, please refer to the example notebooks available on the documentation: [https://zea.readthedocs.io/](https://zea.readthedocs.io/).

## Data
`zea` stores data as well as acquisition parameters together in HDF5 files, which can be easily loaded and saved through the `zea.data` API.

```python
import zea
# path to a local or remote HDF5 file in zea format
path = "hf://zeahub/..."

# read data and acquisition parameters an HDF5 file
with zea.File(path, mode="r") as file:
    file.summary()

    data = file.load_data("raw_data", indices=[0])
    scan = file.scan()
    probe = file.probe()
```

While loading individual data files with `zea.File` or managing multiple files using `zea.Dataset` is convenient for rapid prototyping and exploration, efficient data-driven workflows, such as training deep learning models, require robust data loading utilities. To address this, `zea` offers the `make_dataloader()` function, which is fully compatible with `zea` formatted HDF5 files. This utility streamlines the preparation of data for training by supporting essential features like batching, shuffling, caching, and preprocessing.

```python
from zea.backend.tensorflow import make_dataloader

dataset_path = "hf://zeahub/camus-sample/val"
dataloader = make_dataloader(
    dataset_path,
    key="data/image_sc",
    batch_size=4,
    shuffle=True,
    clip_image_range=[-60, 0],
    image_range=[-60, 0],
    normalization_range=[0, 1],
    image_size=(256, 256),
    resize_type="resize", # or "center_crop or "random_crop"
    seed=4,
)

for batch in dataloader:
    ... # your training loop here

```

## Pipeline
The core of `zea` is a modular and differentiable pipeline class designed for ultrasound data processing. Built on modern deep learning frameworks, this pipeline enables users to compose both built-in and custom operations, including the integration of deep learning models within the processing workflow. The pipeline is stateless, meaning it does not retain information between operations, which facilitates dynamic parameter adjustment and supports real-time reconstruction scenarios. Additionally, the pipeline offers *Just-in-Time* (JIT) compilation, which can significantly enhance performance by optimizing the execution of operations at runtime.

```python
import zea
from zea.ops import *

pipeline = zea.Pipeline(
    operations=[
        Demodulate(),
        PatchedGrid(
            operations=[
                TOFCorrection(),
                PfieldWeighting(),
                DelayAndSum(),
            ],
            num_patches=100,
        ),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
)

# local or remote Hugging Face path to hdf5 file
path = (
    "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    "contrast_speckle_expe_dataset_rf/contrast_speckle_expe_dataset_rf.hdf5"
)
data, scan, probe = zea.load_file(
    path=path,
    data_type="raw_data",
    scan_kwargs={"xlims": (-20e-3, 20e-3), "zlims": (0e-3, 80e-3)},
)

parameters = pipeline.prepare_parameters(probe, scan)

inputs = {pipeline.key: data}

# parameters can be dynamically passed here as keyword arguments
outputs = pipeline(**inputs, **parameters)

image = outputs[pipeline.output_key]
```

## Models
One contribution of `zea` is to extend conventional ultrasound imaging pipelines with data-driven models, such as deep generative models, to learn the structured distribution of ultrasound signals. This allows for more powerful reconstruction and denoising capabilities, as well as the ability to perform inverse problems in a probabilistic machine learning framework. The `zea.models` subpackage provides a collection of pre-defined models for ultrasound image and signal processing, which can be easily integrated into the reconstruction pipeline.


```python
import keras
import zea
from zea.models.diffusion import DiffusionModel

# use a built-in preset or a local / remote HF path to your model
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")

# sample from the model's prior distribution
prior_samples = model.sample(n_samples=16, n_steps=90, verbose=True)
prior_samples = keras.ops.squeeze(prior_samples, axis=-1)

# set up a pipeline to process the prior samples into images
pipeline = zea.Pipeline([zea.ops.ScanConvert(order=2)])

parameters = {
    "theta_range": [-0.78, 0.78],  # [-45, 45] in radians
    "rho_range": [0, 1],
}
parameters = pipeline.prepare_parameters(**parameters)

# process the prior samples through the pipeline
images = pipeline(data=prior_samples, **parameters)["data"]

# plot
zea.visualize.set_mpl_style()
fig, _ = zea.visualize.plot_image_grid(
    images, vmin=-1, vmax=1,
)
```

Which will generate the samples as seen in \autoref{fig:samples}.

![Diffusion posterior samples.\label{fig:samples}](diffusion_prior_samples.png){ width=80% }

## Agent
The `agent` subpackage provides tools and utilities for agent-based algorithms within the ``zea`` framework. They provide tools to alter pipeline or model parameters, select a subset of acquired data, or perform other actions that are necessary to close the action-perception loop in cognitive ultrasound imaging. Currently, the current functions support intelligent focused transmit scheme design via _active perception_ [@van2024active], with implementations of key algorithms such as _Greedy Entropy Minimization_, and mask generation functions to create measurement models mapping from fully-observed to subsampled data.

```python
import zea
import numpy as np

agent = zea.agent.selection.GreedyEntropy(
    n_actions=7,
    n_possible_actions=112,
    img_width=112,
    img_height=112,
)

# (batch, samples, height, width)
particles = np.random.rand(1, 10, 112, 112)
lines, mask = agent.sample(particles)
```

# Availability, Development, and Documentation
`zea` is available through PyPI via `pip install zea`, the development version is available via GitHub. GitHub Actions manage continuous integration through automated code testing (PyTest), code linting and formatting (Ruff), and documentation generation (Sphinx). The documentation is hosted on ReadTheDocs. At the time of writing, 15 example notebooks are available, covering the various discussed components of the toolbox. The package is licensed under the Apache License 2.0, which allows for both academic and commercial use.

# References
