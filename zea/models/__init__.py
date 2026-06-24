"""Collection of (generative) models for ultrasound imaging.

``zea`` contains a collection of models for various tasks, all located in the :mod:`zea.models` package.

See the following dropdown for a list of available models:

.. dropdown:: **Available models**

    - :class:`zea.models.echonet.EchoNetDynamic`: A model for left ventricle segmentation.
    - :class:`zea.models.carotid_segmenter.CarotidSegmenter`: A model for carotid artery segmentation.
    - :class:`zea.models.echonetlvh.EchoNetLVH`: A model for left ventricle hypertrophy segmentation.
    - :class:`zea.models.unet.UNet`: A simple U-Net implementation.
    - :class:`zea.models.lpips.LPIPS`: A model implementing the perceptual similarity metric.
    - :class:`zea.models.taesd.TinyAutoencoder`: A tiny autoencoder model for image compression.
    - :class:`zea.models.regional_quality.MobileNetv2RegionalQuality`: A scoring model for myocardial regions in apical views.
    - :class:`zea.models.lv_segmentation.AugmentedCamusSeg`: A nnU-Net based left ventricle and myocardium segmentation model.
    - :class:`zea.models.speckle2self.Speckle2Self`: A self-supervised speckle reduction model for ultrasound images.

Presets for these models can be found in :mod:`zea.models.presets`. Presets are pre-trained weights for the models, which can be used to initialize the models for inference or further training. Each model class has a :attr:`presets` attribute that lists the available presets for that model. We store the presets on `Hugging Face Hub <https://huggingface.co/zeahub/models>`__, and they are downloaded automatically when loading a model with a preset.

To use these models, you can import them directly from the :mod:`zea.models` module and load the pretrained weights using the :meth:`~zea.models.base.BaseModel.from_preset` method. For example:

.. doctest::

    >>> from zea.models.unet import UNet

    >>> model = UNet.from_preset("unet-echonet-inpainter")

You can list all available presets using the :attr:`presets` attribute:

.. doctest::

    >>> from zea.models.unet import UNet
    >>> presets = list(UNet.presets.keys())
    >>> print(f"Available built-in zea presets for UNet: {presets}")
    Available built-in zea presets for UNet: ['unet-echonet-inpainter']


Generative models
=======================

In addition to regular models, ``zea`` provides generative models for tasks such as image
generation, inpainting, and denoising. The key difference is that generative models have
sampling methods implemented. There are two base classes:
:class:`~zea.models.generative.GenerativeModel` for classical models (e.g. a Gaussian
mixture model) and :class:`~zea.models.generative.DeepGenerativeModel` for
neural-network-based models — the latter also inherits from
:class:`~zea.models.base.BaseModel`, adding Keras features like weight saving and preset
loading. Both expose the following methods:

- :meth:`~zea.models.generative.GenerativeModel.fit` for training the model on data
- :meth:`~zea.models.generative.GenerativeModel.sample` for generating new samples from the learned distribution
- :meth:`~zea.models.generative.GenerativeModel.posterior_sample` for drawing samples from the posterior given measurements
- :meth:`~zea.models.generative.GenerativeModel.log_density` for computing the log-probability of data under the model

See the following dropdown for a list of available *generative* models:

.. dropdown:: **Available models**

    - :class:`zea.models.diffusion.DiffusionModel`: A deep generative diffusion model for ultrasound image generation.
    - :class:`zea.models.flow_matching.FlowMatchingModel`: A flow matching generative model for ultrasound image generation.
    - :class:`zea.models.gmm.GaussianMixtureModel`: A Gaussian Mixture Model.
    - :class:`zea.models.hvae.HierarchicalVAE`: A hierarchical variational autoencoder for ultrasound image generation.

An example of how to use the :class:`zea.models.diffusion.DiffusionModel` is shown below:

.. doctest::

    >>> from zea.models.diffusion import DiffusionModel

    >>> model = DiffusionModel.from_preset("diffusion-echonet-dynamic")  # doctest: +SKIP
    >>> samples = model.sample(n_samples=4)  # doctest: +SKIP


"""

from . import (
    carotid_segmenter,
    deeplabv3,
    dense,
    diffusion,
    dit,
    echonet,
    echonetlvh,
    flow_matching,
    generative,
    gmm,
    hvae,
    layers,
    lpips,
    lv_segmentation,
    presets,
    regional_quality,
    speckle2self,
    taesd,
    unet,
    utils,
)
