.. _pipeline:

Pipeline
========

.. automodule:: zea.ops
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :no-index:

.. _adding-ops:

Adding a new operation
----------------------

New operations are welcome! Below is a step-by-step guide to add one.
Before you start, take a look at the :ref:`contributing guide <contributing>` for
the general contribution workflow (forking, branches, pull requests, etc.).

1. Write the operation
~~~~~~~~~~~~~~~~~~~~~~

Add your operation to the right file:

- ``zea/ops/tensor.py`` — general-purpose operations (e.g. filtering, normalization)
- ``zea/ops/ultrasound.py`` — ultrasound-specific processing (e.g. beamforming, envelope detection)

An operation is a Python class that inherits from :class:`~zea.ops.Operation`. The
``@ops_registry`` decorator gives it a name so it can be used in YAML pipelines.
The only thing you need to implement is a ``call`` method, which takes the data, does
something with it, and returns it in a dictionary.

Here is a minimal example:

.. code-block:: python

    # in zea/ops/tensor.py

    import keras.ops as ops
    from zea.internal.registry import ops_registry
    from zea.ops.base import Operation


    @ops_registry("my_scale")
    class MyScale(Operation):
        """Scale the input data by a constant factor."""

        def __init__(self, factor: float = 2.0, **kwargs):
            """
            Args:
                factor (float): The scale factor. Defaults to 2.0.
            """
            super().__init__(**kwargs)
            self.factor = factor

        def call(self, **kwargs):
            """
            Args:
                data (tensor): Input data of any shape.

            Returns:
                dict: Scaled data under the key ``"data"``.
            """
            data = kwargs[self.key]
            return {self.output_key: data * self.factor}

A few things to keep in mind:

- Use ``keras.ops`` rather than library-specific functions (e.g. ``torch.*``
  or ``jax.numpy.*``) so the operation works regardless of which backend is installed.
- Settings that are fixed when you create the operation (like ``factor`` above) go in
  ``__init__``. Values that may change from call to call should be arguments of ``call``.
- Always pass ``**kwargs`` to both ``super().__init__(**kwargs)`` and keep ``**kwargs``
  in the ``call`` signature — this ensures all standard options (like ``jit_compile``)
  remain available and any extra pipeline parameters are passed through correctly.
- Add a docstring following the style described in the :ref:`contributing guide <contributing>`.

2. Expose the operation
~~~~~~~~~~~~~~~~~~~~~~~

Add your class to two places in ``zea/ops/__init__.py``:

**The import at the top** (add it alongside the other classes from the same file):

.. code-block:: python

    from .tensor import GaussianBlur, MyScale, Normalize, Pad, Threshold

**The** ``__all__`` **list at the bottom:**

.. code-block:: python

    __all__ = [
        ...
        "MyScale",
        ...
    ]

This makes the operation available as ``zea.ops.MyScale``.

3. Write a test
~~~~~~~~~~~~~~~

Add a test in the ``tests/`` directory (e.g. ``tests/ops/test_tensor.py``).
A minimal test creates the operation, runs it on some dummy data, and checks the result:

.. code-block:: python

    import keras
    import pytest
    from zea.ops import MyScale


    def test_my_scale_default():
        op = MyScale()
        data = keras.ops.ones((4, 4))
        out = op(data=data)["data"]
        assert keras.ops.convert_to_numpy(out).mean() == pytest.approx(2.0)


    def test_my_scale_factor():
        op = MyScale(factor=0.5)
        data = keras.ops.ones((4, 4))
        out = op(data=data)["data"]
        assert keras.ops.convert_to_numpy(out).mean() == pytest.approx(0.5)

Run the tests with ``pytest`` (or ``uv run pytest`` when using ``uv``). See the
:ref:`running tests <running-tests>` section in the contributing guide for more details.
