usbmd
===================

Welcome to the documentation for the **usbmd** package: a Python toolbox for ultrasound signal processing, image reconstruction, and deep learning.

Currently usbmd offers:

- Complete ultrasound signal processing and image reconstruction `pipeline <https://github.com/tue-bmd/ultrasound-toolbox/blob/main/usbmd/ops.py>`_.
- A collection of `models <https://github.com/tue-bmd/ultrasound-toolbox/tree/main/usbmd/models>`_ for ultrasound image and signal processing.
- Multi-Backend Support via `Keras3 <https://keras.io/keras_3/>`_: You can use `PyTorch <https://github.com/pytorch/pytorch>`_, `TensorFlow <https://github.com/tensorflow/tensorflow>`_, or `JAX <https://github.com/google/jax>`_.

.. admonition:: Beta!
   :class: warning

   This package is highly experimental and under active development. It is mainly used to support our research and as a basis for our publications. Please use at your own risk.


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

   _autosummary/usbmd

   .. autosummary::
      :toctree: _autosummary
      :recursive:
      usbmd

.. toctree::
   :caption: Project Links
   :maxdepth: 1
   :hidden:

   GitHub Project <https://github.com/tue-bmd/ultrasound-toolbox>
