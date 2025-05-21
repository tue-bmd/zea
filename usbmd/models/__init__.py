# pylint: disable=line-too-long
"""
usbmd models
============

usbmd contains a collection of models for various tasks, all located in the :mod:`usbmd.models` package.

Currently, the following models are available (all inherited from :class:`usbmd.models.base.Model`):

- :class:`usbmd.models.echonet.EchoNetDynamic`: A model for echocardiography segmentation.
- :class:`usbmd.models.carotid_segmenter.CarotidSegmenter`: A model for carotid artery segmentation.
- :class:`usbmd.models.unet.UNet`: A simple U-Net implementation.
- :class:`usbmd.models.lpips.LPIPS`: A model implementing the perceptual similarity metric.

Presets for these models can be found in :mod:`usbmd.models.presets`.

To use these models, you can import them directly from the :mod:`usbmd.models` module and load the pretrained weights using the :meth:`from_preset` method. For example:

.. code-block:: python

    from usbmd.models import UNet
    model = UNet.from_preset('unet-echonet-inpainter')

You can list all available presets using the :attr:`presets` attribute:

.. code-block:: python

    presets = list(UNet.presets.keys())
    print(f"Available built-in usbmd presets for UNet: {presets}")


usbmd generative models
========================

In addition to models, usbmd provides both classical and deep generative models for tasks such as image generation, inpainting, and denoising. These models inherit from :class:`usbmd.models.generative.GenerativeModel` or :class:`usbmd.models.deepgenerative.DeepGenerativeModel`.
Typically, these models have some additional methods, such as:

- :meth:`fit` for training the model on data
- :meth:`sample` for generating new samples from the learned distribution
- :meth:`posterior_sample` for drawing samples from the posterior given measurements
- :meth:`log_density` for computing the log-probability of data under the model

The following generative models are currently available:

- :class:`usbmd.models.diffusion.DiffusionModel`: A deep generative diffusion model for ultrasound image generation.
- :class:`usbmd.models.gmm.GMM`: A Gaussian Mixture Model.

An example of how to use the :class:`usbmd.models.diffusion.DiffusionModel` is shown below:

.. code-block:: python

    from usbmd.models import DiffusionModel
    model = DiffusionModel.from_preset('diffusion-echonet-dynamic')
    samples = model.sample(n_samples=4)


Contributing and adding new models
----------------------------------

Please follow the guidelines in the :ref:`contributing` page if you would like to contribute a new model to usbmd.

The following steps are recommended when adding a new model:

1. Create a new module in the :mod:`usbmd.models` package for your model: ``usbmd.models.mymodel``.
2. Add a model class that inherits from :class:`usbmd.models.base.Model`. For generative models, inherit from :class:`usbmd.models.generative.GenerativeModel` or :class:`usbmd.models.deepgenerative.DeepGenerativeModel` as appropriate. Make sure you implement the :meth:`call` method.
3. Upload the pretrained model weights to `our Hugging Face <https://huggingface.co/usbmd>`_. Should be a ``config.json`` and a ``model.weights.h5`` file. See `Keras documentation <https://keras.io/guides/serialization_and_saving/>`_ how those can be saved from your model. Simply drag and drop the files to the Hugging Face website to upload them.

   .. tip::
      It is recommended to use the mentioned saving procedure. However, alternate saving methods are also possible, see the :class:`usbmd.models.echonet.EchoNet` module for an example. You do now have to implement a :meth:`custom_load_weights` method in your model class.

4. Add a preset for the model in :mod:`usbmd.models.presets`. This basically allows you to have multiple weights presets for a given model architecture.
5. Make sure to register the presets in your model module by importing the presets module and calling ``register_presets`` with the model class as an argument.
"""
