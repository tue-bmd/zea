"""Ultrasound RF simulators.

The simulators produce RF data as the superposition of the echoes of point scatterers: a cloud
of positions and magnitudes in front of a probe, for the transmit scheme of a
:class:`zea.Parameters`. The output is RF of shape (n_tx, n_ax, n_el, 1). :func:`simulate_rf`
synthesises it in the frequency domain and is the reference; :func:`simulate_rf_td` is its
time-domain approximation, less accurate but faster for 2D probes with few transmits. Use them
through :class:`zea.ops.Simulate`, which wraps both for pipelines, derives the FFT length and
applies the receive chain (electronic noise and time gain compensation,
:func:`zea.func.apply_receive_chain`) to the noiseless simulator output. How the synthesis
works, module by module, is in the :ref:`simulator internals <simulator-internals>`.

Inputs
^^^^^^

Both simulators take the same required arguments. A :class:`zea.Parameters` holds them all,
with ``attenuation_coef`` (0) and ``apply_lens_correction`` (False) defaulted and ``t_peak``
derived from the waveform:

- Scene: ``scatterer_positions`` [m] of shape (n_scat, 3) and ``scatterer_magnitudes`` of
  shape (n_scat,).
- Probe: ``probe_geometry`` [m] of shape (n_el, 3) and ``element_width`` [m].
- Transmit scheme: ``t0_delays`` [s] and ``tx_apodizations`` of shape (n_tx, n_el),
  ``initial_times`` [s] and ``t_peak`` [s] of shape (n_tx,).
- Record: ``n_ax``, ``sampling_frequency`` [Hz] and ``center_frequency`` [Hz].
- Medium: ``sound_speed`` [m/s] and ``attenuation_coef`` [dB/cm/MHz].
- Lens: ``apply_lens_correction``, with ``lens_thickness`` [m] and ``lens_sound_speed`` [m/s]
  when it is set.

Everything else has a default. Both simulators take:

- ``waveforms_two_way`` with ``waveform_sampling_frequency``: the transmit pulse, see below.
- ``scatter_exponent``: the frequency dependence of the scattering amplitude, one value for
  the medium or, in :func:`simulate_rf`, a vector of one exponent per scatterer. It is an
  amplitude exponent, not an intensity one: Rayleigh scattering is 2, the default, not 4.
- ``element_height``: an eighth of the width of a 1D probe when not given.
- ``two_dimensional``: simulate in the imaging plane, as a 1D probe behind an ideal elevation
  lens. Less realistic, but comparable with 2D-only simulators.
- ``max_chunk_gb``: the memory budget of one block of work.

The rest is :func:`simulate_rf` only:

- Probe model: ``element_normals``, ``baffle_impedance_ratio``, ``n_sub_elements``,
  ``elevation_focus``, ``lens_attenuation_coef``, ``simplified_directivity``.
- Heterogeneous media: a sound speed map ``sos_map`` [m/s] and an attenuation map
  ``attenuation_map`` [dB/cm/MHz], either on its own, on a uniform grid ``map_grid_x``,
  ``map_grid_z`` (and ``map_grid_y`` for 3D). Every element-scatterer path is timed and
  attenuated with the mean of the map along the straight ray between them, sampled at
  ``n_sos_ray_samples`` points, with the scalar values outside the map. Attenuation scales
  with ``f**attenuation_power``, linear by default.
- Multi-plane transmits: ``t0_delays`` and ``tx_apodizations`` of shape (n_tx, n_mpt, n_el).
- Spectrum and jit: ``band_db``, ``n_fft`` and ``scatter_exponent_range``, derived when not
  given.

The argument docstrings of :func:`simulate_rf` describe each one.

Transmit pulse
^^^^^^^^^^^^^^

The pulse is two-way (pulse-echo): the excitation through the transducer on transmit and on
receive. Without ``waveforms_two_way`` the simulators use the default of
:func:`transmit_pulse`, a one-cycle burst at ``center_frequency`` through a 70 % Butterworth
transducer. ``waveforms_two_way`` is any sampled two-way waveform: the system's own, as stored
in a zea file, or :meth:`Pulse.waveform` of a pulse built with :func:`transmit_pulse`, which
has the parametric models (a pulser's burst through a Butterworth transducer, a Hann tone or
chirp, and the SIMUS model). ``waveform_sampling_frequency`` is its sampling frequency, 250 MHz
by default as in zea files. :func:`measured_pulse` turns a sampled waveform back into a
:class:`Pulse`, for its band, support and time to peak.

Helpers
^^^^^^^

:func:`record_reach`, :func:`record_bounds` and :func:`in_record` tell which scatterers the
record of ``n_ax`` samples can hold. The simulators do not prune the cloud, as a changing cloud
would re-trigger jit compilation every frame, so use these to leave out scatterers that are out
of view before simulating. For dynamic scenes with a varying number of scatterers, pad the cloud
to the next (half) power of two so that jit compiles only once or twice.

:func:`fft_length` sizes the FFT so that no echo wraps into the record;
:class:`zea.ops.Simulate` and :attr:`zea.Parameters.n_fft` call it.

:func:`pressure_field` evaluates the transmit field of :func:`simulate_rf` on a grid, for
visualisation. The pressure-field weighting of the beamformer, :mod:`zea.beamform.pfield`, is a
separate, simpler model.

Example usage
^^^^^^^^^^^^^

A single plane wave on a single scatterer, through :class:`zea.ops.Simulate` in a
:class:`zea.Pipeline`. The :class:`zea.Parameters` takes the probe from a :class:`zea.Probe`
(its geometry, element size, band and lens, when recorded) and adds the transmit scheme and the
medium; what the probe does not record is inferred, see :class:`zea.ops.Simulate`. For a more
in depth example see the notebook: :doc:`../notebooks/simulation/zea_simulation_example`.

.. doctest::

    >>> import numpy as np
    >>> import zea

    >>> probe = zea.Probe.from_name("verasonics_l11_4v")
    >>> parameters = zea.Parameters(
    ...     **probe.get_parameters(),
    ...     n_tx=1,
    ...     n_ax=1024,
    ...     center_frequency=probe.probe_center_frequency,
    ...     sampling_frequency=4 * probe.probe_center_frequency,
    ...     sound_speed=1540,
    ...     t0_delays=np.zeros((1, probe.n_el)),
    ...     tx_apodizations=np.ones((1, probe.n_el)),
    ...     initial_times=np.zeros(1),
    ...     attenuation_coef=0.5,
    ... )
    >>> pipeline = zea.Pipeline([zea.ops.Simulate()], with_batch_dim=False)
    >>> outputs = pipeline(
    ...     scatterer_positions=np.array([[0, 0, 20e-3]]),
    ...     scatterer_magnitudes=np.array([1.0]),
    ...     **pipeline.prepare_parameters(parameters),
    ... )
    >>> outputs[pipeline.output_key].shape
    (1, 1024, 128, 1)

"""

from zea.func.ultrasound import apply_receive_chain
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
from zea.simulator.response import attenuate, min_distance, obliquity_factor, spread
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
    # Responses
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
