"""Ultrasound RF simulators.

The simulators produce RF data as a superposition of scatterer responses. Every scatterer has a
location and a magnitude, and optionally its own backscatter coefficient: ``scatter_exponent``
is one value shared by the medium, or a vector of one exponent per scatterer. Warning: the
exponent is an amplitude exponent, not an intensity one. That means Rayleigh scattering is 2,
not 4. :func:`simulate_rf` works in the frequency domain and is the reference;
:func:`simulate_rf_td` is its time-domain approximation, less accurate but faster for 2D probes
with few transmits.

Sound speed is one value for the medium, or a map: ``sos_map`` with its grid ``map_grid_x``,
``map_grid_z`` (and ``map_grid_y`` for a 3D map) makes every element-scatterer path run at the mean
slowness along the straight ray between them (:func:`zea.func.ultrasound.straight_ray_slowness`),
with ``sound_speed`` outside the map. Straight rays keep the geometry, so the directivity and the
spreading are those of the homogeneous medium; only the travel times change. Attenuation likewise
is one coefficient, or ``attenuation_map`` on the same grid: each path is then attenuated by the
mean coefficient along its straight ray (:func:`zea.func.ultrasound.straight_ray_mean`), with
``attenuation_coef`` outside the map. Either map can be given on its own. The attenuation grows
as ``f**attenuation_power``, linearly by default.

To use it, you can call :func:`simulate_rf` with the desired transmit scheme parameters and
scatterers directly, but the recommended path is to use :class:`zea.ops.Simulate`, which wraps the
simulators for pipelines, derives the FFT length automatically, and applies the receive chain
(electronic noise and time gain compensation, :func:`zea.func.apply_receive_chain`) to the
noiseless RF the simulators return.

:func:`record_reach`, :func:`record_bounds` and :func:`in_record` show which scatterers are
in-record for ``n_ax`` samples; use these to pre-prune your scatterer cloud to avoid wasting compute
on scatterers that are out of view (the simulator doesn't prune them, as moving clouds would
re-trigger jit compilation every frame). On that same note: when using the simulator for dynamic
scenes with varying scatterer numbers, consider padding your scatterer clouds to the next (half)
power of two, so jit only triggers once or twice.

``two_dimensional`` simulates in the imaging plane, as a 1D probe behind an ideal elevation lens.

:func:`pressure_field` evaluates the transmit field of the simulator on a grid of points. Should
be a more accurate version of the pfield code used in the beamformer. It will likely be integrated
with the beamformer in the future, but currently only included for visualization purposes.

The package is layered: :mod:`~zea.simulator.pulse` builds the transmit pulse,
:mod:`~zea.simulator.element` the element model and its responses,
:mod:`~zea.simulator.record` the record with its gates and band, and
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
from zea.simulator.element import attenuate, min_distance, obliquity_factor, spread
from zea.simulator.frequency_domain import pressure_field, simulate_rf
from zea.simulator.pulse import (
    PULSE_MODELS,
    Pulse,
    butterworth_transfer,
    gaussian_transfer,
    generalized_normal_transfer,
    hann_burst_spectrum,
    measured_pulse,
    rect_burst_spectrum,
    rect_chirp_spectrum,
    sampled_spectrum,
    square_burst_pulses,
    square_burst_spectrum,
    transmit_pulse,
    transmit_pulses,
)
from zea.simulator.record import (
    band_bins,
    fft_length,
    in_record,
    record_bounds,
    record_reach,
    scatter_exponent_bounds,
    smooth_size,
)
from zea.simulator.time_domain import simulate_rf_td

__all__ = [
    "simulate_rf",
    "simulate_rf_td",
    "pressure_field",
    "apply_receive_chain",
    # Pulses
    "PULSE_MODELS",
    "Pulse",
    "transmit_pulse",
    "measured_pulse",
    "transmit_pulses",
    "sampled_spectrum",
    "hann_burst_spectrum",
    "rect_chirp_spectrum",
    "rect_burst_spectrum",
    "square_burst_spectrum",
    "square_burst_pulses",
    "gaussian_transfer",
    "generalized_normal_transfer",
    "butterworth_transfer",
    # Elements
    "attenuate",
    "spread",
    "obliquity_factor",
    "min_distance",
    # Record
    "record_reach",
    "record_bounds",
    "in_record",
    "fft_length",
    "smooth_size",
    "band_bins",
    "scatter_exponent_bounds",
]
