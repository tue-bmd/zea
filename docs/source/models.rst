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

1. **Vendor the architecture** — keep the PyTorch network code at hand while you
   port it. If you want it in the module long-term (e.g. for ONNX export), put it
   in a ``_build_torch_classes()`` helper that imports ``torch`` lazily, so that
   PyTorch is only required for weight conversion and never for inference.

2. **Implement the Keras architecture** — write ``keras.layers.Layer`` subclasses
   that replicate each block. Key API differences to handle:

   * **Padding for strided Conv2D**: Keras ``padding='same'`` is asymmetric for
     ``stride > 1``; use ``ZeroPadding2D(p) + Conv2D(padding='valid')`` to match
     PyTorch's symmetric ``padding=p``.
   * **ConvTranspose**: Keras ``Conv2DTranspose(padding='valid')`` gives the full
     output, from which PyTorch's ``padding=p`` crops ``p`` on every side — so
     follow it with ``Cropping2D(p)``. A PyTorch ``output_padding`` adds to the
     end only, so ``padding=1, output_padding=1`` becomes ``x[:, 1:, 1:, :]``.
   * **Normalization**: ``BatchNormalization`` needs ``epsilon=1e-5`` to match
     PyTorch's ``BatchNorm2d`` default, and takes its weights as ``[weight, bias,
     running_mean, running_var]``. For ``InstanceNorm2d(affine=False)`` use
     ``GroupNormalization(groups=C, scale=False, center=False, epsilon=1e-5)``.
   * **Weight axes**: Conv2D — ``(2,3,1,0)``; Conv2DTranspose — ``(2,3,1,0)``
     (same permutation, different semantics).
   * **Input format**: Keras defaults to channels-last (NHWC); transpose
     NCHW → NHWC in ``call()`` and back before returning.
   * **Layer names**: do not name custom layer *classes* with a leading
     underscore. Keras derives the layer name from the class name, and the
     TensorFlow backend rejects names starting with ``_``.

3. **Write a weight-loading helper** — a function that maps PyTorch state-dict
   keys to the Keras layer tree and calls ``layer.set_weights([...])``. The
   weights only exist once the model is built; building it symbolically with
   ``self(keras.Input(batch_shape=...))`` avoids pushing real data through it.

4. **Add** ``from_pth(path)`` **classmethod** — wraps the weight loader for
   convenient local testing.

5. **Verify against the reference implementation** — before uploading the
   converted weights, run both models on the same input and compare. A correct
   port agrees to ~1e-5 in float32. If it does not, compare block by block: the
   layer where the difference appears is the one that was ported wrong.

   .. warning::
      Run that comparison on CPU. On an NVIDIA GPU both JAX and PyTorch use TF32
      for convolutions by default, which costs ~3 decimal digits and makes a
      correct port look broken (differences of ~1e-2 instead of ~1e-5).

6. Optionally **add an ONNX fallback** — for environments that have
   ``onnxruntime`` but not ``torch``, you can keep a ``from_onnx(path)``
   classmethod and an ``_onnx_sess`` attribute; override ``call()`` to
   dispatch to the ONNX path when the session is set.

7. Follow steps 3-6 from the standard guide above for HF upload, presets, and
   registration.

Two worked examples of this pattern: :mod:`zea.models.inversionnet` (native Keras
plus a PyTorch state-dict converter) and :mod:`zea.models.speckle2self` (the same,
plus a vendored PyTorch architecture and an optional ONNX fallback).
