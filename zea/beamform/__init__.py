"""Beamforming subpackage for ultrasound imaging.

The ``zea.beamform`` subpackage provides core algorithms and utilities for ultrasound beamforming,
including delay calculations, time-of-flight correction, lens correction, and pressure field computation.

Modules
-------

- :mod:`zea.beamform.beamformer` -- Main beamforming functions and time-of-flight correction.
- :mod:`zea.beamform.delays` -- Delay calculation routines for plane wave and focused transmissions.
- :mod:`zea.beamform.lens_correction` -- Lens-corrected delay computation.
- :mod:`zea.beamform.pfield` -- Pressure field computation for transmit compounding and adaptive beamforming.
- :mod:`zea.beamform.pixelgrid` -- Pixel grid generation for scan conversion and beamforming.

For practical usage and examples of integrating beamforming operations into a processing pipeline,
see the pipeline example notebook: :doc:`../notebooks/pipeline/zea_pipeline_example`

"""

from . import beamformer, delays, lens_correction, pfield, pixelgrid
