"""Inverse beamforming: recover channel data from beamformed images.

The ``zea.inverse`` subpackage inverts the delay-and-sum (DAS) beamformer: it
recovers **pre-beamformed** channel data from a **post-beamformed** image by
expressing the beamformer as a differentiable linear operator and solving a
least-squares problem — optionally regularized with a point-scatterer prior.

Modules
-------

- :mod:`zea.inverse.operators` -- The DAS beamformer as a differentiable
  linear operator (:class:`DASOperator`) and a time-domain point-scatterer
  simulator (:class:`ScattererSimulator`).
- :mod:`zea.inverse.solvers` -- Matrix-free solver primitives
  (:func:`cgls`, :func:`linear_adjoint`).
- :mod:`zea.inverse.seeding` -- Scatterer seeding from a beamformed image
  (:func:`seed_scatterers`).
- :mod:`zea.inverse.inversion` -- High-level inversion drivers
  (:func:`invert_direct`, :func:`invert_scatterers`).

The DAS beamformer sums on the order of ``n_el * n_tx`` samples into every
pixel, so inverting a single compounded image for the full channel-data cube
is well-posed for *data fit* but severely underdetermined for *recovery*.
:func:`invert_direct` makes this concrete by computing the minimum-norm
(pseudo-inverse) solution, while :func:`invert_scatterers` regularizes the
nullspace with a physical point-scatterer parameterization, which recovers
channel data far better on point-target scans.

The scatterer-prior inversion follows the off-grid stochastic-optimization
scatterer model of van de Schaft et al., *Off-Grid Ultrasound Imaging by
Stochastic Optimization* (`arXiv:2407.02285
<https://arxiv.org/abs/2407.02285>`_).

Example
-------

.. code-block:: python

    import zea
    from zea.inverse import DASOperator, invert_scatterers

    with zea.File("path/to/scan.hdf5") as file:
        parameters = file.load_parameters(
            xlims=(-0.018, 0.018), zlims=(0.003, 0.04), pixels_per_wavelength=2
        )
        raw_data = file.data.raw_data[0, ..., 0]  # first frame, RF

    operator = DASOperator(parameters)
    image = operator.forward(raw_data)
    result = invert_scatterers(operator, image, n_scatterers=15000, n_iter=70)
    # result.channel_data is the recovered pre-beamformed data cube

For a walkthrough see the notebook: :doc:`../notebooks/pipeline/inverse_beamforming_example`.
"""

from zea.inverse.inversion import InversionResult, invert_direct, invert_scatterers
from zea.inverse.operators import DASOperator, ScattererSimulator
from zea.inverse.seeding import seed_scatterers
from zea.inverse.solvers import cgls, linear_adjoint

__all__ = [
    "DASOperator",
    "InversionResult",
    "ScattererSimulator",
    "cgls",
    "invert_direct",
    "invert_scatterers",
    "linear_adjoint",
    "seed_scatterers",
]
