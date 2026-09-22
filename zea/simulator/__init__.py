"""Ultrasound RF simulators.

The simulators produce RF data as a superposition of scatterer responses. Every scatterer has a
location and a magnitude. :func:`simulate_rf` works in the frequency domain (RFFT domain) and
is the reference; :func:`simulate_rf_td` is its time-domain approximation.

To use them in your code, simply call :func:`simulate_rf` with the desired transmit scheme
parameters and scatterers. To simulate a sequence of multiple frames, you can call
:func:`simulate_rf` repeatedly with different scatterer positions and magnitudes and then stack
the results. :class:`zea.ops.Simulate` wraps both simulators for a :class:`zea.Pipeline`.

The package is layered: :mod:`~zea.simulator.pulse` builds the transmit pulse,
:mod:`~zea.simulator.element` the element model and its responses,
:mod:`~zea.simulator.record` the record grid and its gates, and
:mod:`~zea.simulator.frequency_domain` and :mod:`~zea.simulator.time_domain` are the two
simulators on top of them.

Example usage
^^^^^^^^^^^^^

A simple example of simulating RF data with a single scatterer at the center of the probe. For a
more in depth example see the notebook: :doc:`../notebooks/data/zea_simulation_example`.

.. doctest::

    >>> from zea.simulator import simulate_rf
    >>> import numpy as np

    >>> raw_data = simulate_rf(
    ...     scatterer_positions=np.array([[0, 0, 20e-3]]),
    ...     scatterer_magnitudes=np.array([1.0]),
    ...     probe_geometry=np.stack(
    ...         [np.linspace(-20e-3, 20e-3, 64), np.zeros(64), np.zeros(64)], axis=-1
    ...     ),
    ...     apply_lens_correction=True,
    ...     lens_thickness=1e-3,
    ...     lens_sound_speed=1000,
    ...     sound_speed=1540,
    ...     n_ax=1024,
    ...     center_frequency=5e6,
    ...     sampling_frequency=20e6,
    ...     t0_delays=np.zeros((1, 64)),
    ...     initial_times=np.zeros(1),
    ...     element_width=0.2e-3,
    ...     attenuation_coef=0.5,
    ...     tx_apodizations=np.ones((1, 64)),
    ...     t_peak=np.full(1, 1 / 5e6),
    ... )

"""

from zea.func.ultrasound import apply_receive_chain
from zea.simulator.element import (
    attenuate,
    elevation_slab_bucket,
    elevation_slab_mask,
    min_distance,
    obliquity_factor,
    select_elevation_slab,
    spread,
)
from zea.simulator.frequency_domain import simulate_rf
from zea.simulator.pulse import (
    PULSE_MODELS,
    Pulse,
    butterworth_transfer,
    gaussian_transfer,
    generalized_normal_transfer,
    hann_burst_spectrum,
    measured_pulse,
    rect_burst_spectrum,
    sampled_spectrum,
    square_burst_pulses,
    square_burst_spectrum,
    transmit_pulse,
    transmit_pulses,
)
from zea.simulator.record import delay2
from zea.simulator.time_domain import simulate_rf_td

__all__ = [
    "simulate_rf",
    "simulate_rf_td",
    "apply_receive_chain",
    # Pulses
    "PULSE_MODELS",
    "Pulse",
    "transmit_pulse",
    "measured_pulse",
    "transmit_pulses",
    "sampled_spectrum",
    "hann_burst_spectrum",
    "rect_burst_spectrum",
    "square_burst_spectrum",
    "square_burst_pulses",
    "gaussian_transfer",
    "generalized_normal_transfer",
    "butterworth_transfer",
    # Elements
    "attenuate",
    "spread",
    "delay2",
    "obliquity_factor",
    "min_distance",
    "elevation_slab_mask",
    "select_elevation_slab",
    "elevation_slab_bucket",
]
