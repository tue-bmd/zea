zea
===================

Welcome to the documentation for the ``zea`` package: *A Toolbox for Cognitive Ultrasound Imaging.*

``zea`` is a Python library that offers ultrasound signal processing, image reconstruction, and deep learning. Currently ``zea`` offers:

- A flexible ultrasound signal processing and image reconstruction :doc:`pipeline` written in your favorite deep learning framework.
- A complete set of :doc:`data-acquisition` loading tools for ultrasound data and acquisition parameters, designed for deep learning workflows.
- A collection of pretrained :doc:`models` for ultrasound image and signal processing.
- Multi-Backend Support via `Keras3 <https://keras.io/keras_3/>`_: You can use `PyTorch <https://github.com/pytorch/pytorch>`_, `TensorFlow <https://github.com/tensorflow/tensorflow>`_, or `JAX <https://github.com/google/jax>`_.

Check out the :doc:`about` page for more information and the motivation behind ``zea``. For any questions or suggestions, please feel free to open an `issue on GitHub <https://github.com/tue-bmd/zea/issues>`_. If you want to contribute, check out the :doc:`contributing` guide.

.. admonition:: Beta!
   :class: warning

   This package is highly experimental and under active development. It is mainly used to support `our research <https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab>`_ and as a basis for our publications. That being said, we are happy to share it with the ultrasound community and hope it will be useful for your research as well.

.. note::

   If you use ``zea`` in your research, please consider the citation details :ref:`here <citation>`.

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
   parameters
   pipeline
   models
   environment
   cli

.. toctree::
   :caption: Project
   :maxdepth: 1
   :hidden:

   about
   GitHub Project <https://github.com/tue-bmd/zea>
   HuggingFace Hub <https://huggingface.co/zeahub>
