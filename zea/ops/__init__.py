"""Operations and Pipelines for ultrasound data processing.

This module contains two important classes, :class:`Operation` and :class:`Pipeline`,
which are used to process ultrasound data. A pipeline is a sequence of operations
that are applied to the data in a specific order.

We implement a range of common
operations for ultrasound data processing (:mod:`zea.ops.ultrasound`), but also support
a variety of basic tensor operations (:mod:`zea.ops.tensor`). Lastly, all existing Keras
operations (see `Keras Ops API <https://keras.io/api/ops/>`_) are available as `zea`
operations as well (see :mod:`zea.ops.keras_ops`).

Stand-alone manual usage
------------------------

Operations can be run on their own:

Examples
^^^^^^^^
.. doctest::

    >>> import numpy as np
    >>> from zea.ops import EnvelopeDetect
    >>> data = np.random.randn(2000, 128, 1)
    >>> # static arguments are passed in the constructor
    >>> envelope_detect = EnvelopeDetect(axis=-1)
    >>> # other parameters can be passed here along with the data
    >>> envelope_data = envelope_detect(data=data)

Using a pipeline
----------------

You can initialize with a default pipeline or create your own custom pipeline.

.. doctest::

    >>> from zea.ops import Pipeline, EnvelopeDetect, Normalize, LogCompress
    >>> pipeline = Pipeline.from_default()

    >>> operations = [
    ...     EnvelopeDetect(),
    ...     Normalize(),
    ...     LogCompress(),
    ... ]
    >>> pipeline_custom = Pipeline(operations)

One can also load a pipeline from a config or yaml/json file:

.. doctest::

    >>> from zea import Pipeline

    >>> # From JSON string
    >>> json_string = '{"operations": ["identity"]}'
    >>> pipeline = Pipeline.from_json(json_string)

    >>> # from yaml file
    >>> import yaml
    >>> from zea import Config
    >>> # Create a sample pipeline YAML file
    >>> pipeline_dict = {
    ...     "operations": [
    ...         {"name": "identity"},
    ...     ]
    ... }
    >>> with open("pipeline.yaml", "w") as f:
    ...     yaml.dump(pipeline_dict, f)
    >>> yaml_file = "pipeline.yaml"
    >>> pipeline = Pipeline.from_yaml(yaml_file)

.. testcleanup::

    import os

    os.remove("pipeline.yaml")

Example of a yaml file:

.. code-block:: yaml

    pipeline:
      operations:
        - name: demodulate
        - name: beamform
          params:
            type: das
            pfield: false
            num_patches: 100
        - name: envelope_detect
        - name: normalize
        - name: log_compress

"""

from zea.internal.registry import ops_registry
from zea.ops import keras_ops

from .base import (
    Identity,
    Lambda,
    Mean,
    Merge,
    Operation,
    Stack,
    get_ops,
)
from .pipeline import (
    Beamform,
    BranchedPipeline,
    DelayAndSum,
    DelayMultiplyAndSum,
    Map,
    PatchedGrid,
    Pipeline,
)
from .tensor import (
    GaussianBlur,
    Normalize,
    Pad,
    Threshold,
)
from .ultrasound import (
    AnisotropicDiffusion,
    ApplyWindow,
    BandPassFilter,
    ChannelsToComplex,
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
    CommonMidpointPhaseError,
    UpMix,
)

__all__ = [
    # Registry
    "ops_registry",
    # Base operations
    "Identity",
    "Lambda",
    "Mean",
    "Merge",
    "Operation",
    "Stack",
    "get_ops",
    # Pipeline
    "DelayAndSum",
    "DelayMultiplyAndSum",
    "Beamform",
    "BranchedPipeline",
    "Map",
    "PatchedGrid",
    "Pipeline",
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
