"""Collection of (generative) models for ultrasound imaging.

``zea`` contains a collection of models for various tasks, all located in the :mod:`zea.models` package.

Currently, the following models are available (all inherited from :class:`zea.models.BaseModel`):

- :class:`zea.models.echonet.EchoNetDynamic`: A model for echocardiography segmentation.
- :class:`zea.models.carotid_segmenter.CarotidSegmenter`: A model for carotid artery segmentation.
- :class:`zea.models.unet.UNet`: A simple U-Net implementation.
- :class:`zea.models.lpips.LPIPS`: A model implementing the perceptual similarity metric.
- :class:`zea.models.taesd.TinyAutoencoder`: A tiny autoencoder model for image compression.

Presets for these models can be found in :mod:`zea.models.presets`.

To use these models, you can import them directly from the :mod:`zea.models` module and load the pretrained weights using the :meth:`from_preset` method. For example:

.. code-block:: python

    from zea.models import UNet

    model = UNet.from_preset("unet-echonet-inpainter")

You can list all available presets using the :attr:`presets` attribute:

.. code-block:: python

    presets = list(UNet.presets.keys())
    print(f"Available built-in zea presets for UNet: {presets}")


zea generative models
=======================

In addition to models, zea provides both classical and deep generative models for tasks such as image generation, inpainting, and denoising. These models inherit from :class:`zea.models.generative.GenerativeModel` or :class:`zea.models.deepgenerative.DeepGenerativeModel`.
Typically, these models have some additional methods, such as:

- :meth:`fit` for training the model on data
- :meth:`sample` for generating new samples from the learned distribution
- :meth:`posterior_sample` for drawing samples from the posterior given measurements
- :meth:`log_density` for computing the log-probability of data under the model

The following generative models are currently available:

- :class:`zea.models.diffusion.DiffusionModel`: A deep generative diffusion model for ultrasound image generation.
- :class:`zea.models.gmm.GaussianMixtureModel`: A Gaussian Mixture Model.

An example of how to use the :class:`zea.models.diffusion.DiffusionModel` is shown below:

.. code-block:: python

    from zea.models import DiffusionModel

    model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
    samples = model.sample(n_samples=4)


Contributing and adding new models
==================================

Please follow the guidelines in the :ref:`contributing` page if you would like to contribute a new model to zea.

The following steps are recommended when adding a new model:

1. Create a new module in the :mod:`zea.models` package for your model: ``zea.models.mymodel``.
2. Add a model class that inherits from :class:`zea.models.base.Model`. For generative models, inherit from :class:`zea.models.generative.GenerativeModel` or :class:`zea.models.deepgenerative.DeepGenerativeModel` as appropriate. Make sure you implement the :meth:`call` method.
3. Upload the pretrained model weights to `our Hugging Face <https://huggingface.co/zea>`_. Should be a ``config.json`` and a ``model.weights.h5`` file. See `Keras documentation <https://keras.io/guides/serialization_and_saving/>`_ how those can be saved from your model. Simply drag and drop the files to the Hugging Face website to upload them.

   .. tip::
      It is recommended to use the mentioned saving procedure. However, alternate saving methods are also possible, see the :class:`zea.models.echonet.EchoNet` module for an example. You do now have to implement a :meth:`custom_load_weights` method in your model class.

4. Add a preset for the model in :mod:`zea.models.presets`. This basically allows you to have multiple weights presets for a given model architecture.
5. Make sure to register the presets in your model module by importing the presets module and calling ``register_presets`` with the model class as an argument.
"""

from . import (
    carotid_segmenter,
    dense,
    diffusion,
    echonet,
    generative,
    gmm,
    layers,
    lpips,
    presets,
    taesd,
    unet,
    utils,
)
