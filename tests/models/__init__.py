"""Tests for the model architectures in :mod:`zea.models`.

One module per model module, so that ``zea/models/foo.py`` is tested by
``tests/models/test_foo.py``. The preset loading/saving infrastructure that all
of these models share is tested in ``tests/test_preset_utils.py`` instead.

Everything here runs offline and on CPU: no preset is downloaded from the Hugging
Face hub. Models whose weights only exist as a downloaded artifact (ONNX sessions,
TensorFlow SavedModels) are exercised against a stub, and the models that zea
builds itself are instantiated at a small enough size to stay fast.
"""
