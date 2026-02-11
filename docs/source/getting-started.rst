Getting Started
===============

``zea`` provides a framework for cognitive ultrasound imaging. At the heart of ``zea`` are :doc:`data-acquisition` (``zea.data``), :doc:`pipeline` (``zea.Pipeline``), and :doc:`models` (``zea.Models``) modules. These modules provide the necessary tools to load, process, and analyze ultrasound data.

.. tip::

   A more complete set of examples can be found on the :doc:`examples` page.

Let's take a quick look at how to use ``zea`` to load and process ultrasound data.

.. code-block:: python

   import zea
   # setting up cpu / gpu usage
   zea.init_device()

   # loading a config file from Hugging Face, but can also load a local config file
   config = zea.Config.from_hf(
      "zeahub/configs", "config_picmus_rf.yaml", repo_type="dataset",
   )

   path = config.data.dataset_folder + "/" + config.data.file_path
   with zea.File(path) as file:
      data = file.load_data("raw_data", indices=0)
      probe = file.probe()
      scan = file.scan(**config.scan)

   # using the pipeline as specified in the config file
   pipeline = zea.Pipeline.from_config(
      config.pipeline,
      with_batch_dim=False,
   )
   # preparing the parameters (converting to tensors)
   parameters = pipeline.prepare_parameters(probe, scan)

   # running the pipeline!
   image = pipeline(data=data, **parameters)["data"]

Similarly, we can easily load one of the pretrained models from the :mod:`zea.models` module and use it for inference.

.. code-block:: python

   import zea
   from zea.models.echonet import EchoNetDynamic

   zea.init_device()

   # presets can also paths to local checkpoints of the model
   model = EchoNetDynamic.from_preset("echonet-dynamic")

   # we'll load a single file from the dataset
   with zea.Dataset("hf://zeahub/camus-sample/", "image_sc") as dataset:
      file = dataset[0]
      image = file.load_data("image_sc", indices=0)

   image = zea.func.translate(image, config.data.dynamic_range, (-1, 1))
   masks = model(image[None, ..., None])


``zea`` also provides a simple command line interface (CLI) to quickly visualize a ``zea`` data file.

.. code-block:: shell

   zea --config configs/config_picmus_rf.yaml

Installation
------------

A simple pip command will install the latest version of ``zea`` from `PyPI <https://pypi.org/project/zea>`_. For more installation instructions, please refer to the :doc:`installation` page.

.. code-block:: shell

   pip install zea


Backend
-------

.. backend-installation-start

``zea`` is written in Python on top of `Keras 3 <https://keras.io/about/>`_. This means that under the hood we use the Keras framework to implement the pipeline and models. Keras allows you to set a backend, which means you can use ``zea`` alongside your project that uses any of your preferred machine learning framework.

To use ``zea``, you need to install one of the supported machine learning backends: JAX, PyTorch or TensorFlow ``zea`` **will not run** without a backend installed.

- `Install JAX <https://jax.readthedocs.io/en/latest/installation.html>`__
- `Install PyTorch <https://pytorch.org/get-started/locally/>`__
- `Install TensorFlow <https://www.tensorflow.org/install>`__

If you are unsure which backend to use, we recommend JAX as it is currently the fastest backend.

After installing a backend, set the ``KERAS_BACKEND`` environment variable to one of the following:

.. tab-set::

   .. tab-item:: JAX

      .. tab-set::

         .. tab-item:: Python

            .. code-block:: python

               # at the top of your script before other imports
               import os
               os.environ["KERAS_BACKEND"] = "jax"
               import zea

         .. tab-item:: Conda

            .. code-block:: shell

               conda env config vars set KERAS_BACKEND=jax

         .. tab-item:: Shell

            .. code-block:: shell

               export KERAS_BACKEND=jax

   .. tab-item:: PyTorch

      .. tab-set::

         .. tab-item:: Python

            .. code-block:: python

               # at the top of your script before other imports
               import os
               os.environ["KERAS_BACKEND"] = "torch"
               import zea

         .. tab-item:: Conda

            .. code-block:: shell

               conda env config vars set KERAS_BACKEND=torch

         .. tab-item:: Shell

            .. code-block:: shell

               export KERAS_BACKEND=torch

   .. tab-item:: TensorFlow

      .. tab-set::

         .. tab-item:: Python

            .. code-block:: python

               # at the top of your script before other imports
               import os
               os.environ["KERAS_BACKEND"] = "tensorflow"
               import zea

         .. tab-item:: Conda

            .. code-block:: shell

               conda env config vars set KERAS_BACKEND=tensorflow

         .. tab-item:: Shell

            .. code-block:: shell

               export KERAS_BACKEND=tensorflow

   .. tab-item:: NumPy

      .. tab-set::

         .. tab-item:: Python

            .. code-block:: python

               # at the top of your script before other imports
               # note NumPy backend has limited functionality
               import os
               os.environ["KERAS_BACKEND"] = "numpy"
               import zea

         .. tab-item:: Conda

            .. code-block:: shell

               # note NumPy backend has limited functionality
               conda env config vars set KERAS_BACKEND=numpy

         .. tab-item:: Shell

            .. code-block:: shell

               # note NumPy backend has limited functionality
               export KERAS_BACKEND=numpy

.. backend-installation-end

.. _citation:

Citation
--------

If you use ``zea`` in your research, please cite using :cite:p:`started-stevens2025zea` and :cite:p:`started-van2024active`. Our preprint paper can be found on `arXiv <https://arxiv.org/abs/2512.01433>`_. Also, in case you use them, don't forget to ensure proper attribution to authors of specific models and datasets that are supported by ``zea``.

.. bibliography:: ../../paper/paper.bib
   :style: unsrt
   :keyprefix: started-
   :labelprefix: B-

   stevens2025zea
   van2024active

Or you can use the following BibTeX entry:

.. literalinclude:: ../../paper/paper.bib
   :language: bibtex
   :lines: 1-7
