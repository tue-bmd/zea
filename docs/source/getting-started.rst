Getting Started
===============

``zea`` provides a framework for cognitive ultrasound imaging. At the heart of ``zea`` are :doc:`data-acquisition` (``zea.File``, ``zea.Dataset``, ``zea.Dataloader``), :doc:`pipeline` (``zea.Pipeline``), and :doc:`models` (``zea.Models``) classes. These provide the necessary tools to load, process, and analyze ultrasound data.

.. tip::

   A more complete set of examples can be found on the :doc:`examples` page.

Let's take a quick look at how to use ``zea`` to load and process ultrasound data.

The diagram and code snippet below illustrate the basic data flow in ``zea``: loading a file and
assembling parameters, building and applying the pipeline, and visualising results.

.. raw:: html

   <div style="display: flex; flex-direction: column; align-items: center; margin: 3em 0;">
     <!-- Dark mode image -->
     <img
       src="_static/zea_workflow_dark.svg"
       alt="zea workflow diagram"
       style="display: none; width: 80%; padding-bottom: 1em;"
       class="only-dark"
     />
     <!-- Light mode image -->
     <img
       src="_static/zea_workflow_light.svg"
       alt="zea workflow diagram"
       style="display: none; width: 80%; padding-bottom: 1em;"
       class="only-light"
     />
     <div style="text-align: center; font-style: italic; color: var(--color-foreground-secondary, #666);">
       Overview of the zea data processing workflow.
     </div>
   </div>
   <style>
     @media (prefers-color-scheme: dark) {
       .only-dark { display: block !important; }
     }
     @media (prefers-color-scheme: light), (prefers-color-scheme: no-preference) {
       .only-light { display: block !important; }
     }
   </style>

① Generate ``zea.Parameters`` from a zea file using ``File.load_parameters()`` — this combines
scan and probe information from the file to compute all parameters needed for beamforming.
② Optionally apply additional parameter overrides from a ``config.yaml``.
③ Initialise the pipeline, either from a ``config.yaml`` or manually in code.
④ Pass data and parameters to the pipeline for processing.
⑤ Visualise your outputs.

An example of this workflow in code is shown below. In this example, we load a config file and a data file from Hugging Face (PICMUS dataset), but you can also load local files.

.. code-block:: python

   import matplotlib.pyplot as plt

   import zea

   # setting up cpu / gpu usage
   zea.init_device()
   # plotting style
   zea.visualize.set_mpl_style()

   # loading a config file from Hugging Face, but can also load a local config file
   config = zea.Config.from_path("hf://zeahub/picmus/config_iq.yaml", revision="v0.1.0")

   path = (
      "hf://zeahub/picmus/in_vivo/carotid_cross/"
      "carotid_cross_expe_dataset_iq/carotid_cross_expe_dataset_iq.hdf5"
   )
   with zea.File(path, revision="v0.1.0") as file:
      data = file.data.raw_data[0]
      # load the merged probe + scan parameters
      parameters = file.load_parameters()

   # update parameters with manual settings from the config file
   parameters.update(**config.parameters)

   # or manually set some parameters
   parameters.zlims = (0, 0.04)
   parameters.grid_size_x = 500
   parameters.grid_size_z = 800
   parameters.dynamic_range = (-50, 0)

   # using the pipeline as specified in the config file
   pipeline = zea.Pipeline.from_config(
      config,
      with_batch_dim=False,
   )
   # prepare the inputs (converts the needed parameters to tensors)
   inputs = pipeline.prepare_parameters(parameters)

   # running the pipeline!
   image = pipeline(data=data, **inputs)["data"]

   xlims_mm = [v * 1e3 for v in parameters.xlims]
   zlims_mm = [v * 1e3 for v in parameters.zlims]
   extent = [xlims_mm[0], xlims_mm[1], zlims_mm[1], zlims_mm[0]]

   # plot figure
   fig = plt.figure()
   plt.imshow(image, cmap="gray", extent=extent)
   plt.title("B-Mode")
   plt.xlabel("X (mm)")
   plt.ylabel("Z (mm)")
   plt.show()

.. raw:: html

   <div style="display: flex; flex-direction: column; align-items: center; margin: 2em 0;">
     <img src="_static/carotid_dark.png" alt="B-mode carotid image" style="display: none; width: 60%;" class="only-dark" />
     <img src="_static/carotid_light.png" alt="B-mode carotid image" style="display: none; width: 60%;" class="only-light" />
     <div style="text-align: center; font-style: italic; color: var(--color-foreground-secondary, #666);">
       B-mode image of a carotid cross-section produced by the pipeline above.
     </div>
   </div>

Similarly, we can easily load one of the pretrained models from the :mod:`zea.models` module and use it for inference. Let's load a pretrained despeckling model and apply it to the B-mode image we just generated.

.. code-block:: python

   import keras
   import numpy as np

   import zea
   from zea.models.speckle2self import Speckle2Self

   model = Speckle2Self.from_preset("hf://zeahub/speckle2self-invivo")

   # Run despeckling (Speckle2Self) model inference
   despeckled = model(image[None, ..., None])
   despeckled = keras.ops.convert_to_numpy(despeckled)
   despeckled = despeckled.squeeze()

   # gamma correction for better visualization
   despeckled_viz = np.power(despeckled, 1.3)

   # plot figure
   fig = plt.figure()
   plt.imshow(despeckled_viz, cmap="gray", extent=extent)
   plt.title("Despeckled B-Mode (Speckle2Self)")
   plt.xlabel("X (mm)")
   plt.ylabel("Z (mm)")
   plt.show()

.. raw:: html

   <div style="display: flex; flex-direction: column; align-items: center; margin: 2em 0;">
     <img src="_static/carotid_despeckled_dark.png" alt="Despeckled B-mode carotid image" style="display: none; width: 60%;" class="only-dark" />
     <img src="_static/carotid_despeckled_light.png" alt="Despeckled B-mode carotid image" style="display: none; width: 60%;" class="only-light" />
     <div style="text-align: center; font-style: italic; color: var(--color-foreground-secondary, #666);">
       Despeckled B-mode carotid image after applying the Speckle2Self model.
     </div>
   </div>

A full list of available pretrained models can be found in the :doc:`models` page.

.. seealso::

   For a more detailed walkthrough of this example please refer to the :doc:`notebooks/models/speckle2self_despeckling_example` notebook.

``zea`` also provides a simple command line interface (CLI) to quickly visualize a ``zea`` data file. For more information on the CLI, please refer to the :doc:`cli` page or run ``zea --help`` in your terminal.

.. code-block:: shell

   zea process --dataset hf://zeahub/picmus/ --config configs/config_picmus_rf.yaml

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

If you use ``zea`` in your research, please cite using :cite:p:`started-stevens2026zea` and :cite:p:`started-van2024active`. Our preprint paper can be found on `arXiv <https://arxiv.org/abs/2512.01433>`_. Also, in case you use them, don't forget to ensure proper attribution to authors of specific models and datasets that are supported by ``zea``.

.. bibliography:: ../../paper/paper.bib
   :style: unsrt
   :keyprefix: started-
   :labelprefix: B-

   stevens2026zea
   van2024active

Or you can use the following BibTeX entry:

.. literalinclude:: ../../paper/paper.bib
   :language: bibtex
   :lines: 1-11