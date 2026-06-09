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

Presets for these models can be found in :mod:`zea.models.presets`.

To use these models, you can import them directly from the :mod:`zea.models` module and load the pretrained weights using the :meth:`from_preset` method. For example:

.. doctest::

    >>> from zea.models.unet import UNet

    >>> model = UNet.from_preset("unet-echonet-inpainter")

You can list all available presets using the :attr:`presets` attribute:

.. doctest::

    >>> from zea.models.unet import UNet
    >>> presets = list(UNet.presets.keys())
    >>> print(f"Available built-in zea presets for UNet: {presets}")
    Available built-in zea presets for UNet: ['unet-echonet-inpainter']


zea generative models
=======================

In addition to models, zea provides both classical and deep generative models for tasks such as image generation, inpainting, and denoising. These models inherit from :class:`zea.models.generative.GenerativeModel` or :class:`zea.models.deepgenerative.DeepGenerativeModel`.
Typically, these models have some additional methods, such as:

- :meth:`fit` for training the model on data
- :meth:`sample` for generating new samples from the learned distribution
- :meth:`posterior_sample` for drawing samples from the posterior given measurements
- :meth:`log_density` for computing the log-probability of data under the model

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


Contributing and adding new models
==================================

Please follow the guidelines in the :ref:`contributing` page if you would like to contribute a new model to zea.

The following steps are recommended when adding a new model:

1. Create a new module in the :mod:`zea.models` package for your model: ``zea.models.mymodel``.
2. Add a model class that inherits from :class:`zea.models.base.Model`. For generative models, inherit from :class:`zea.models.generative.GenerativeModel` or :class:`zea.models.deepgenerative.DeepGenerativeModel` as appropriate. Make sure you implement the :meth:`call` method.
3. Upload the pretrained model weights to `our Hugging Face <https://huggingface.co/zeahub>`_. Should be a ``config.json`` and a ``model.weights.h5`` file. See `Keras documentation <https://keras.io/guides/serialization_and_saving/>`_ how those can be saved from your model. Simply drag and drop the files to the Hugging Face website to upload them.

   .. tip::
      It is recommended to use the mentioned saving procedure. However, alternate saving methods are also possible, see the :class:`zea.models.echonet.EchoNet` module for an example. You do now have to implement a :meth:`custom_load_weights` method in your model class.

4. Add a preset for the model in :mod:`zea.models.presets`. This basically allows you to have multiple weights presets for a given model architecture.
5. Make sure to register the presets in your model module by importing the presets module and calling ``register_presets`` with the model class as an argument.
6. Lastly, add the model to the :mod:`zea.models` package by importing it in the :file:`__init__.py` file.

Adding non-Keras (custom) models
----------------------------------

The recommended approach for any model is to implement it as a **native Keras 3
model**. This gives you backend-agnostic execution (JAX, TensorFlow, PyTorch)
and the full preset/weight-loading infrastructure for free.

For models originally trained in PyTorch, the typical workflow is:

1. **Vendor the architecture** — copy the PyTorch network code into your module
   (e.g. inside a ``_build_torch_classes()`` helper that imports ``torch``
   lazily so that PyTorch is only required for weight conversion, not inference).

2. **Implement the Keras architecture** — write ``keras.layers.Layer`` subclasses
   that replicate each block. Key API differences to handle:

   * **Padding for stride-2 Conv2D**: Keras ``padding='same'`` is asymmetric for
     ``stride > 1``; use ``ZeroPadding2D(1) + Conv2D(padding='valid')`` to match
     PyTorch's symmetric ``padding=1``.
   * **ConvTranspose**: Keras ``Conv2DTranspose(padding='valid')`` gives the full
     output; crop ``x[:, 1:, 1:, :]`` (NHWC) to reproduce PyTorch's
     ``ConvTranspose2d(padding=1, output_padding=1)`` alignment.
   * **InstanceNorm**: use ``GroupNormalization(groups=C, scale=False,
     center=False, epsilon=1e-5)`` for ``InstanceNorm2d(affine=False)``.
   * **Weight axes**: Conv2D — ``(2,3,1,0)``; Conv2DTranspose — ``(2,3,1,0)``
     (same permutation, different semantics).
   * **Input format**: Keras defaults to channels-last (NHWC); transpose
     NCHW → NHWC in ``call()`` and back before returning.

3. **Write a weight-loading helper** — a function that maps PyTorch state-dict
   keys to the Keras layer tree and calls ``layer.set_weights([...])``.

4. **Add** ``from_pth(path)`` **classmethod** — wraps the weight loader for
   convenient local testing.

5. Optionally **add an ONNX fallback** — for environments that have
   ``onnxruntime`` but not ``torch``, you can keep a ``from_onnx(path)``
   classmethod and an ``_onnx_sess`` attribute; override ``call()`` to
   dispatch to the ONNX path when the session is set.

6. Follow steps 3-6 from the standard guide above for HF upload, presets, and
   registration.

See :mod:`zea.models.speckle2self` for a complete worked example of this
pattern (native Keras + PyTorch weight loading + optional ONNX fallback).
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
