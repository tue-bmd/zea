"""Operations and Pipelines for ultrasound data processing.

The :mod:`zea.ops` module contains a collection of operations (:class:`Operation`) that can be applied to ultrasound data. These operations can be used on their own or as part of a pipeline. A :class:`Pipeline` is a sequence of operations that are applied to the data in a specific order.

We implement a range of common operations for ultrasound data processing, but also support
a variety of basic tensor operations. Lastly, all existing Keras operations (see `Keras Ops API <https://keras.io/api/ops/>`_) are available as `zea` operations as well (see :mod:`zea.ops.keras_ops`), and thus can be easily integrated in common ultrasound processing pipelines.

.. seealso::
    A tutorial notebook where the usage of operations and pipelines is demonstrated:
    :doc:`../notebooks/pipeline/zea_pipeline_example`.

Stand-alone usage of operations
-------------------------------

In many settings, it can be useful to apply an :class:`Operation` directly to the data, without using a :class:`Pipeline`. In that case, you can simply initialize the operation and call it with the data.

.. doctest::

    >>> import keras
    >>> from zea.ops import EnvelopeDetect
    >>> data = keras.random.uniform((2000, 128, 1))
    >>> # static arguments are passed in the constructor
    >>> envelope_detect = EnvelopeDetect(axis=-1)
    >>> # other (dynamic) parameters can be passed here along with the data
    >>> # the output is again a dictionary
    >>> envelope_data = envelope_detect(data=data)["data"]

.. note::

    Besides the :mod:`zea.ops` API, we also have a functional (:mod:`zea.func`) API which contains the functional building
    blocks that many of the :mod:`zea.ops` operations are built on. These can be used for more low-level processing, and can be found in the :mod:`zea.func` module. For instance, the :class:`EnvelopeDetect` operation is built on top of the :func:`zea.func.envelope_detect` function in :mod:`zea.func`. You can use these functions directly as well, if you prefer a more functional programming style. The advantage of using the :mod:`zea.ops` API is that these operations can be easily integrated into pipelines.

Using a pipeline
----------------

There are many ways to initialize a :class:`Pipeline`. In its essence, a :class:`Pipeline` is just a sequence of multiple :class:`Operation`.
A :class:`Pipeline` will chain these operations together, so that the output of one operation is the input of the next. All operations takes
a dictionary of tensors and parameters as inputs and passes these along to the next operation, only picking the parameters they need.

One of the more common pipelines you will encounter is a basic ultrasound raw channel data to B-mode image pipeline, which consists of a sequence of operations like demodulation, beamforming, envelope detection, normalization and log compression:

.. doctest::

    >>> from zea.ops import (
    ...     Beamform,
    ...     Demodulate,
    ...     Pipeline,
    ...     EnvelopeDetect,
    ...     Normalize,
    ...     LogCompress,
    ... )

    >>> operations = [
    ...     Demodulate(),
    ...     Beamform(beamformer="delay_and_sum"),
    ...     EnvelopeDetect(),
    ...     Normalize(),
    ...     LogCompress(),
    ... ]
    >>> pipeline = Pipeline(operations)

In fact this is so common that we created a handy utility function to create this pipeline with default parameters:

.. doctest::

    >>> pipeline = Pipeline.from_default()

Calling a pipeline
^^^^^^^^^^^^^^^^^^

A :class:`Operation` or :class:`Pipeline` is called with **keyword arguments only**. The primary input data
(often raw RF data) should be passed under the key given by :attr:`Pipeline.key` (``"data"`` by default), and
the result is a dictionary whose final output is stored under :attr:`Pipeline.output_key`. All other parameters
that the operations need — such as scan geometry, probe layout, and reconstruction settings — are passed
as additional keyword arguments alongside the data. In simple terms, a flat dictionary of tensors containing all
the necessary information is passed to the pipeline, and a dictionary of outputs is returned. This dictionary is
internally routed through each operation in the pipeline, which picks the parameters it needs and produces intermediate
outputs until the final output is produced.

Additionally, all these input arguments should be converted to tensors at the start, as the operations and pipelines are
implemented with the machine learning backend of choice (JAX, TensorFlow, or PyTorch). One can use the :meth:`Pipeline.prepare_parameters` method to convert commonly used objects such as :class:`~zea.Scan` and :class:`~zea.Probe` into a flat dictionary of tensors that can be directly passed to the pipeline.

See the tutorial notebook :doc:`../notebooks/pipeline/zea_pipeline_example` for a complete example including data loading, parameter preparation, and pipeline execution on real ultrasound data. Below a minimal stand-alone snippet is shown
to illustrate the calling convention:

.. doctest::

    >>> import keras
    >>> from zea.ops import Pipeline, Normalize, LogCompress

    >>> pipeline = Pipeline(
    ...     operations=[Normalize(), LogCompress()],
    ...     with_batch_dim=False,
    ... )

    >>> data = keras.ops.abs(keras.random.normal((64, 64)))

    >>> # Pass data under pipeline.key (default: "data") together with any needed parameters
    >>> parameters = {"dynamic_range": (-60, 0)}
    >>> inputs = {"data": data}
    >>> outputs = pipeline(**inputs, **parameters)
    >>> data_out = outputs[pipeline.output_key]
    >>> data_out = keras.ops.convert_to_numpy(data_out)
    >>> print(f"min: {data_out.min()}, max: {data_out.max()}")
    min: -60.0, max: 0.0


Saving and loading pipelines
----------------------------

It can be quite handy to share pipelines across machines, or accompany a dataset or publication
with a specific zea pipeline configuration. For this reason, we support saving and loading pipelines
in a human-readable YAML format. The preferred way to persist a pipeline is :meth:`Pipeline.to_yaml`
for saving and :meth:`Pipeline.from_path` for loading. Together they form a lossless round-trip: every
operation and its parameters are serialized to a plain YAML file that can be
version-controlled, shared, or reproduced on any machine.

.. doctest::

    >>> from zea import Pipeline
    >>> from zea.ops import Beamform, EnvelopeDetect, Normalize, LogCompress

    >>> pipeline = Pipeline(
    ...     operations=[
    ...         Demodulate(),
    ...         Beamform(beamformer="delay_and_sum"),
    ...         EnvelopeDetect(),
    ...         Normalize(),
    ...         LogCompress(),
    ...     ],
    ... )

    >>> # Save to YAML
    >>> pipeline.to_yaml("bmode_pipeline.yaml")
    >>> # Load back from YAML
    >>> loaded_pipeline = Pipeline.from_path("bmode_pipeline.yaml")

.. testcleanup::

    import os
    os.remove("bmode_pipeline.yaml")

Pipelines hosted on the `Hugging Face Hub <https://huggingface.co/zeahub>`_ can be loaded
directly using an ``hf://`` URI, without manually downloading any files:

.. doctest::

    >>> pipeline = Pipeline.from_path("hf://zeahub/picmus/config_iq.yaml")
    >>> print(pipeline)
    Beamform(PatchedGrid(TOFCorrection -> DelayAndSum) -> ReshapeGrid) -> EnvelopeDetect -> Normalize -> LogCompress

The YAML format is human-readable and straightforward to edit by hand. A typical B-mode
pipeline looks like this:

.. code-block:: yaml

    pipeline:
      operations:
        - name: demodulate
        - name: beamform
          params:
            beamformer: delay_and_sum
            num_patches: 100
        - name: envelope_detect
        - name: normalize
        - name: log_compress

Device selection
----------------

It can be handy to execute a :class:`Pipeline` on a specific device (GPU / CPU).
Call :func:`zea.init_device` at the start of a script to select a device.
It returns the selected device string — or a list of strings when
multiple GPUs are requested — which can be passed directly to the pipeline or
used with the :class:`~zea.device` context manager:

.. code-block:: python

    import zea

    # Single GPU — auto-selects the one with the most free memory
    device = zea.init_device("auto:1")  # e.g. "gpu:0"

    # Two GPUs — auto-selects by free memory, returns a list
    devices = zea.init_device("auto:2")  # e.g. ["gpu:0", "gpu:1"]

.. note::

    :func:`zea.init_device` should be called **before** importing heavy ML
    libraries (JAX, TensorFlow, PyTorch) so that ``CUDA_VISIBLE_DEVICES`` is
    configured before they initialise.

To run a pipeline on a specific device, use the :class:`~zea.device` context
manager or pass ``device=`` to the pipeline constructor. Whereas everything
created and executed inside the context manager will be placed on the specified device,
passing ``device=`` to the pipeline will ensure that tensors passed to the pipeline
are automatically moved to the specified device.

.. code-block:: python

    pipeline = zea.Pipeline([zea.ops.keras_ops.Abs()])

    # Option 1: context manager
    with zea.device("gpu:0"):
        data = np.random.randn(100, 100)
        # make sure data is created inside the context manager
        data = keras.ops.convert_to_tensor(data)
        output = pipeline(data=data)["data"]

    # Option 2: device argument on the pipeline itself
    data = np.random.randn(100, 100)
    data = keras.ops.convert_to_tensor(data)
    pipeline = zea.Pipeline([zea.ops.keras_ops.Abs()], device="gpu:0")
    # data will be automatically moved to the specified device when passed to the pipeline
    output = pipeline(data=data)["data"]

"""

from zea.internal.registry import beamformer_registry, ops_registry
from zea.ops import keras_ops

from .base import Identity, Lambda, Mean, Operation, get_ops
from .pipeline import (
    Beamform,
    CoherenceFactor,
    DelayAndSum,
    DelayMultiplyAndSum,
    GeneralizedCoherenceFactor,
    Map,
    PatchedGrid,
    Pipeline,
    Refocus,
)
from .tensor import GaussianBlur, Normalize, Pad, Threshold
from .ultrasound import (
    AnisotropicDiffusion,
    ApplyWindow,
    BandPassFilter,
    ChannelsToComplex,
    CommonMidpointPhaseError,
    Companding,
    ComplexToChannels,
    Demodulate,
    Downsample,
    EnvelopeDetect,
    FirFilter,
    LeeFilter,
    LogCompress,
    LowPassFilterIQ,
    PfieldWeighting,
    ReshapeGrid,
    ScanConvert,
    Simulate,
    TOFCorrection,
    UpMix,
)

__all__ = [
    # Registry
    "ops_registry",
    "beamformer_registry",
    # Base operations
    "Identity",
    "Lambda",
    "Mean",
    "Operation",
    "get_ops",
    # Pipeline
    "DelayAndSum",
    "DelayMultiplyAndSum",
    "CoherenceFactor",
    "GeneralizedCoherenceFactor",
    "Beamform",
    "Map",
    "PatchedGrid",
    "Pipeline",
    "Refocus",
    # Tensor operations
    "GaussianBlur",
    "Normalize",
    "Pad",
    "Threshold",
    # Ultrasound operations
    "AnisotropicDiffusion",
    "ApplyWindow",
    "BandPassFilter",
    "ChannelsToComplex",
    "Companding",
    "ComplexToChannels",
    "Demodulate",
    "Downsample",
    "EnvelopeDetect",
    "FirFilter",
    "LeeFilter",
    "LogCompress",
    "LowPassFilterIQ",
    "PfieldWeighting",
    "ReshapeGrid",
    "ScanConvert",
    "Simulate",
    "TOFCorrection",
    "UpMix",
    "CommonMidpointPhaseError",
    # Keras operations
    "keras_ops",
]
