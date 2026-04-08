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
    affiliation: 1
    corresponding: true
  - name: Wessel L. van Nierop
    orcid: 0009-0003-3141-3369
    affiliation: 1
  - name: Ben Luijten
    orcid: 0000-0002-1797-8721
    affiliation: 1
  - name: Vincent van de Schaft
    orcid: 0000-0002-8515-5372
    affiliation: 1
  - name: Oisín Nolan
    orcid: 0009-0002-6939-7627
    affiliation: 1
  - name: Beatrice Federici
    orcid: 0009-0003-2496-8825
    affiliation: 1
  - name: Louis D. van Harten
    orcid: 0000-0002-0943-2825
    affiliation: 1
  - name: Simon W. Penninga
    orcid: 0009-0003-4095-8168
    affiliation: 1
  - name: Noortje I.P. Schueler
    affiliation: 1
    orcid: 0009-0003-7134-6850
  - name: Ruud J.G. van Sloun
    orcid: 0000-0003-2845-0495
    affiliation: 1
affiliations:
  - index: 1
    name: Eindhoven University of Technology, the Netherlands
date: 20 June 2025
bibliography: paper.bib

---

# Summary
Ultrasound imaging is a powerful medical imaging modality that is widely used in clinical settings for various applications, including obstetrics, cardiology, and abdominal imaging. While ultrasound imaging is non-invasive, real-time, and relatively low-cost compared to other imaging modalities such as MRI or CT, it still faces challenges in terms of image quality and interpretation. Many signal processing steps are required to extract useful information from the raw ultrasound data, such as filtering, beamforming, and image reconstruction. Traditional ultrasound imaging techniques often suffer from reduced image quality as naive assumptions are made in these processing steps, which do not account for the complex nature of ultrasound signals. Furthermore, acquisition (action) and reconstruction (perception) of ultrasound is often performed disjointly. Cognitive ultrasound imaging [@van2024active], see \autoref{fig:diagram}, is a novel approach that aims to address these challenges by leveraging more powerful generative models, enabled by advances in deep learning, to close the action-perception loop. This approach requires a redesign of current common ultrasound imaging pipeline, where parameters are expected to be changed dynamically based on past and current observations. Furthermore, the high-dimensional nature of ultrasound data requires powerful deep generative models to learn the structured distribution of ultrasound signals and to effectively solve inverse problems that capture the challenges of ultrasound imaging [@stevens2025deep]. This necessitates a flexible and efficient toolbox that can handle the complexities of cognitive ultrasound imaging, including a real-time ultrasound reconstruction pipeline, dynamic parameter adjustment, and advanced generative modeling.

We present `zea` (pronounced *ze-yah*), a Python package for cognitive ultrasound imaging that provides a flexible, modular and differentiable pipeline for ultrasound data processing, as well as a collection of pre-defined models for ultrasound image and signal processing. The toolbox is designed to be easy to use, with a high-level interface that allows users to define their own ultrasound reconstruction pipelines, and to integrate deep learning models into the pipeline. The toolbox is built on top of Keras 3 [@chollet2015keras], which provides a framework for building and training deep learning models with the three major deep learning frameworks as backend: TensorFlow [@abadi2016tensorflow], PyTorch [@NEURIPS2019_9015] and JAX [@jax2018github]. This means that it is easy to integrate a custom ultrasound reconstruction pipeline in a machine learning workflow. In the past few years, several works have used and contributed to `zea`, including @luijten2020adaptive, @van2024off, @stevens2024dehazing, @nolan2024active, @federici2024active, @stevens2025sequential, @penninga2025deep and @stevens2025high.

![High-level overview of an ultrasound perception-action loop implemented in zea.\label{fig:diagram}](assets/zea_perception_action-Light.pdf){ width=100% }

# Statement of need
The ultrasound research community has advanced significantly due to publically available high-quality software, including simulation tools such as `Field II` [@jensen2004simulation] and `k-wave` [@treeby2010k], as well as reconstruction and real-time processing libraries like `USTB` [@rodriguez2017ultrasound], `MUST` [@garcia2021make], `ARRUS` [@jarosik2020arrus], `FAST` [@smistad2021fast], `QUPS` [@brevett2024qups], and `vbeam` [@magnus2023vbeam]. However, existing solutions are not well-equipped for cognitive ultrasound imaging, where the integration of deep learning and dynamic, closed-loop ultrasound reconstruction pipelines is essential. Our aim with `zea` is to provide a complementary, highly flexible and differentiable pipeline written in a modern deep learning framework, as well as offer a convenient platform for pretrained models. This addresses the need for a modular and extensible library that supports cognitive ultrasound workflows and seamless integration with state-of-the-art machine learning models. While the full realization of cognitive ultrasound imaging remains an ongoing effort, we hope this toolbox will help spur further research and development in the field.

# Overview of functionality
`zea` is an open-source Python package, available at [http://github.com/tue-bmd/zea](http://github.com/tue-bmd/zea), that consists of the following core components:

- **Data**: A set of data handling classes such as `zea.File`, `zea.Dataset` and `zea.Dataloader`, suited for machine learning workflows. `zea` works with HDF5 files, storing data and acquisition parameters together in a single file, which can be easily loaded and saved through the `zea` API. For more demanding workflows, such as training deep learning models, `zea` offers robust data loading utilities such as batching, shuffling, caching, and preprocessing. Additionally, we provide examples and conversion scripts for popular ultrasound datasets, such as CAMUS [@leclerc2019deep], PICMUS [@liebgott2016plane], and EchoNet [@ouyang2020video].
- **Pipeline**: A modular and differentiable pipeline class that allows users to define a sequence of operations (`zea.Operation`) to process ultrasound data. The pipeline is stateless and supports *Just in Time* (JIT) compilation. Ultimately, this allows for dynamic parameter adjustment, as well as real-time integration of deep learning models inside the ultrasound reconstruction pipeline.
- **Models**: A collection of pre-defined models for ultrasound image and signal processing. Similar to the data, these models can be loaded locally or from the [Hugging Face Hub](https://huggingface.co/zeahub). Besides supervised models, `zea` also provides a set of (deep) generative models, with an interface to solve inverse problems in ultrasound imaging within a probabilistic machine learning framework.
- **Agents**: A set of tools to interact with the pipeline and models. These agents can be used to alter the pipeline parameters, or select a subset of acquired data. The agent module closes the action-perception loop [@van2024active], tying together acquisition and reconstruction of ultrasound data.

For detailed examples and use cases, please refer to the example notebooks available on the documentation: [https://zea.readthedocs.io/](https://zea.readthedocs.io/).

# Availability, Development, and Documentation
`zea` is available through PyPI via `pip install zea`, and the development version is available via GitHub. GitHub Actions manage continuous integration through automated code testing (PyTest), code linting and formatting (Ruff), and documentation generation (Sphinx). The documentation is hosted on ReadTheDocs. At the time of writing, 20 example notebooks are available, covering the various discussed components of the toolbox. The package is licensed under the Apache License 2.0, which allows for both academic and commercial use.

# References
