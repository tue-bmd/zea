.. _models:

Models
========

.. automodule:: zea.models
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :no-index:

.. _adding-models:

Adding a new model
------------------

New models are welcome! Please follow the :ref:`contributing` guide for the general
workflow (forking, branches, pull requests, etc.). The steps below walk you through
what is specific to adding a model.

1. Create a new file in ``zea/models/`` for your model, e.g. ``zea/models/mymodel.py``.
2. Add a model class that inherits from :class:`zea.models.base.BaseModel`. For generative
   models, use :class:`~zea.models.generative.GenerativeModel` or
   :class:`~zea.models.generative.DeepGenerativeModel` as the base class. Implement
   the ``call`` method.
3. Upload the pretrained weights to `our Hugging Face <https://huggingface.co/zeahub>`_.
   The expected files are a ``config.json`` and a ``model.weights.h5``. See the
   `Keras documentation <https://keras.io/guides/serialization_and_saving/>`_ for how
   to save these. You can drag and drop the files directly on the Hugging Face website.

   .. tip::
      Alternate saving methods are also possible. See :class:`zea.models.echonet.EchoNet`
      for an example — in that case you need to implement a ``custom_load_weights``
      method in your model class.

4. Add a preset for your model in :mod:`zea.models.presets`. Presets let you register
   multiple sets of weights for the same model architecture.
5. In your model file, import the presets module and call ``register_presets`` with your
   model class to activate the presets.
6. Import your model in ``zea/models/__init__.py`` to make it part of the package.

Adding non-Keras (custom) models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
