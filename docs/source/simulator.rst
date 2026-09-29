.. _simulator:

Simulator
=========

``zea`` simulates ultrasound RF data from a cloud of point scatterers: a phantom of positions
and magnitudes in front of a probe, for the transmit scheme of a :class:`zea.Parameters`. The
simulator is an operation, :class:`zea.ops.Simulate`, so a phantom goes into a
:class:`zea.Pipeline` and an image comes out, and it is differentiable on jax.

- **Tutorial.** The :doc:`simulation notebook <notebooks/simulation/zea_simulation_example>`
  builds a probe, a scan and a phantom, simulates, beamforms the result, and adds sound speed
  and attenuation maps.
- **In a pipeline.** :class:`zea.ops.Simulate` documents what the operation takes from the
  parameters, what is optional, and the two simulation methods.
- **As functions.** The :mod:`zea.simulator` page documents the simulators
  (:func:`~zea.simulator.simulate_rf` and :func:`~zea.simulator.simulate_rf_td`) with every
  argument, the transmit pulse models, and the helpers that size the record and the FFT.

.. _simulator-internals:

Internals
---------

How the simulators are built, module by module, for readers who want to follow the code or
extend it. This section renders the docstrings of the modules of :mod:`zea.simulator`, which
are not part of the public API.

Frequency domain
^^^^^^^^^^^^^^^^

.. automodule:: zea.simulator.frequency_domain
   :no-members:
   :no-inherited-members:
   :no-special-members:

Time domain
^^^^^^^^^^^

.. automodule:: zea.simulator.time_domain
   :no-members:
   :no-inherited-members:
   :no-special-members:

Transmit pulse
^^^^^^^^^^^^^^

.. automodule:: zea.simulator.pulse
   :no-members:
   :no-inherited-members:
   :no-special-members:

Record
^^^^^^

.. automodule:: zea.simulator.record
   :no-members:
   :no-inherited-members:
   :no-special-members:

Element response
^^^^^^^^^^^^^^^^

.. automodule:: zea.simulator.response
   :no-members:
   :no-inherited-members:
   :no-special-members:
