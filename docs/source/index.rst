zea
===================

Welcome to the documentation for the ``zea`` package: *A Toolbox for Cognitive Ultrasound Imaging.*

``zea`` is a Python library for ultrasound signal processing, image reconstruction, and deep learning. It is built on `Keras 3 <https://keras.io/keras_3/>`_, so everything runs on `JAX <https://github.com/google/jax>`_, `PyTorch <https://github.com/pytorch/pytorch>`_, or `TensorFlow <https://github.com/tensorflow/tensorflow>`_.

..
   The showcase animation is rendered and hosted in a gist (it is too large for this repo):
   https://gist.github.com/tristan-deep/6d0bb210f95fc530ca63fd8721402c30

.. image:: https://gist.githubusercontent.com/tristan-deep/6d0bb210f95fc530ca63fd8721402c30/raw/zea_showcase_light.webp
   :class: only-light
   :width: 100%
   :alt: Animated tour of the zea modules: Pipeline, models, Parameters and Probe, and File, Dataset and Dataloader

.. image:: https://gist.githubusercontent.com/tristan-deep/6d0bb210f95fc530ca63fd8721402c30/raw/zea_showcase_dark.webp
   :class: only-dark
   :width: 100%
   :alt: Animated tour of the zea modules: Pipeline, models, Parameters and Probe, and File, Dataset and Dataloader

Installation
------------

.. code-block:: shell

   pip install zea

Additionally, ``zea`` needs one of the Keras backends installed: `JAX <https://jax.readthedocs.io/en/latest/installation.html>`__ (recommended), `PyTorch <https://pytorch.org/get-started/locally/>`__, or `TensorFlow <https://www.tensorflow.org/install>`__. See the :doc:`installation` guide for more advanced options, such as GPU support.

Quick start
-----------

``zea`` offers many models and tools for advanced ultrasound processing. But in the most basic sense it offers a fast and flexible ultrasound processing pipeline implemented in a deep learning framework, for easy integration with state-of-the-art models. The following example streams an in-vivo recording from `Hugging Face <https://huggingface.co/zeahub>`__, beamforms it, and saves the B-mode image:

.. The snippet is shared with README.md (and tested via tests/test_getting_started.py).

.. literalinclude:: ../../README.md
   :language: python
   :start-after: ```python
   :end-before: ```

The :doc:`getting-started` guide walks through this example step by step, and the :doc:`examples` page has many more notebooks you can open in Colab.

Core modules
------------

- :doc:`pipeline`: flexible, differentiable signal processing and image reconstruction, written in your favorite deep learning framework.
- :doc:`data-acquisition`: loading tools for ultrasound data and acquisition parameters, designed for deep learning workflows. Datasets on Hugging Face are streamed, so only the frames you read are fetched.
- :doc:`models`: pretrained models for ultrasound image and signal processing, hosted on `Hugging Face <https://huggingface.co/zeahub>`__.
- :doc:`agent`: action selection for cognitive ultrasound, with flexible transmit and acquisition design: an action decides what to acquire next, such as which scan lines to fire.

Check out the :doc:`about` page for the motivation behind ``zea`` and the :ref:`research <papers>` it supports. Questions and suggestions are welcome as an `issue on GitHub <https://github.com/tue-bmd/zea/issues>`_. If you want to contribute, see the :doc:`contributing` guide.

.. note::

   If ``zea`` helps your research, please cite our `JOSS paper <https://doi.org/10.21105/joss.09881>`_. See the citation details :ref:`here <citation>`.

.. toctree::
   :caption: User Guide
   :maxdepth: 2
   :hidden:

   getting-started
   installation
   examples

.. toctree::
   :caption: Development
   :hidden:

   contributing

.. toctree::
   :caption: Reference
   :maxdepth: 2
   :hidden:

   _autosummary/zea
   data-acquisition
   config
   pipeline
   models
   agent
   environment
   cli

.. toctree::
   :caption: Project
   :maxdepth: 1
   :hidden:

   about
   GitHub Project <https://github.com/tue-bmd/zea>
   HuggingFace Hub <https://huggingface.co/zeahub>
