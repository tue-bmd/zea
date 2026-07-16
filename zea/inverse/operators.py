"""Differentiable forward operators for inverse beamforming.

This module provides the two forward maps used by :mod:`zea.inverse`:

* :class:`DASOperator` — the delay-and-sum (DAS) beamformer as a linear
  operator mapping pre-beamformed channel data to a beamformed image. Built on
  the :mod:`zea.beamform.beamformer` primitives (:func:`calculate_delays`,
  :func:`apply_delays`, :func:`fnumber_mask`), so it shares its delay model
  (including lens correction) with the rest of ``zea``.
* :class:`ScattererSimulator` — a time-domain point-scatterer simulator that
  maps scatterer positions and magnitudes to pre-beamformed channel data using
  the scan's own (two-way) transmit waveforms. It uses the same delay model as
  :class:`DASOperator`, which makes the composition
  ``DASOperator.forward(ScattererSimulator(...))`` self-consistent: simulated
  echoes are sampled at their waveform peak by the beamformer.

Both operators are written with ``keras.ops`` and are differentiable on every
Keras backend, which is what enables the optimization-based inversion in
:mod:`zea.inverse.inversion`. Internally they iterate over transmits (and
scatterer chunks) with ``keras.ops.scan`` and rematerialize each step with
``keras.remat``, so peak memory under automatic differentiation stays bounded
even for scans with hundreds of transmits.
"""

import keras
import numpy as np
from keras import ops

from zea import log
from zea.beamform.beamformer import (
    apply_delays,
    calculate_delays,
    complex_rotate,
    fnum_window_fn_tukey,
    fnumber_mask,
)
from zea.inverse.solvers import linear_adjoint


def _sinc(x):
    """Normalized sinc function ``sin(pi x) / (pi x)`` with ``sinc(0) = 1``."""
    x = ops.where(ops.abs(x) < 1e-8, 1e-8, x)
    return ops.sin(np.pi * x) / (np.pi * x)


def _element_directivity(theta, element_width_wavelengths):
    """Far-field directivity of a rectangular element of the given width.

    Follows the hard-baffle model ``sinc(w sin(theta)) * cos(theta)`` with the
    element width ``w`` expressed in wavelengths. Equals 1 at normal incidence
    and rolls off towards grazing angles; wider elements are more directive.

    Args:
        theta (Tensor): Angle from the element normal in radians.
        element_width_wavelengths (float): Element width in wavelengths.

    Returns:
        Tensor: Directivity weights with the same shape as ``theta``.
    """
    return _sinc(element_width_wavelengths * ops.sin(theta)) * ops.cos(theta)


def _pad_and_chunk(x, chunk_size, pad_value):
    """Pad the leading axis to a multiple of ``chunk_size`` and reshape.

    ``(n, ...)`` becomes ``(n_chunks, chunk_size, ...)``.
    """
    n = int(x.shape[0])
    pad = (-n) % chunk_size
    if pad:
        pad_width = [(0, pad)] + [(0, 0)] * (len(x.shape) - 1)
        x = ops.pad(x, pad_width, constant_values=pad_value)
    return ops.reshape(x, (-1, chunk_size) + tuple(x.shape[1:]))


class DASOperator:
    """The DAS beamformer as a differentiable linear operator.

    Maps pre-beamformed channel data of shape ``(n_tx, n_ax, n_el)`` (or
    ``(n_tx, n_ax, n_el, n_ch)``) to a flattened beamformed image of shape
    ``(n_pix,)`` (or ``(n_pix, n_ch)``) by time-of-flight correction, receive
    f-number masking and summation over elements and transmits. Transmits are
    accumulated with a rematerialized ``ops.scan``, so memory under automatic
    differentiation does not grow with the number of transmits.

    Because the map is linear and differentiable, its adjoint (transpose) is
    available through :meth:`adjoint`, which is all that is needed for
    matrix-free least-squares inversion with :func:`zea.inverse.cgls`.

    Args:
        parameters (zea.Parameters): Acquisition parameters. The imaging grid
            is taken from ``parameters.flatgrid`` (set ``xlims`` / ``zlims`` /
            ``grid_size_x`` / ``grid_size_z`` on the parameters to control it),
            the receive aperture from ``parameters.f_number``, and the lens
            model from ``parameters.apply_lens_correction``.
        flatgrid (Tensor, optional): Custom pixel positions of shape
            ``(n_pix, 3)`` overriding ``parameters.flatgrid``.
        fnum_window_fn (callable, optional): Window function for the receive
            f-number mask. Defaults to
            :func:`zea.beamform.beamformer.fnum_window_fn_tukey`.
    """

    def __init__(self, parameters, flatgrid=None, fnum_window_fn=fnum_window_fn_tukey):
        self.parameters = parameters
        self.fnum_window_fn = fnum_window_fn
        # Cast to float32: grids from numpy default to float64, which the jax
        # backend silently demotes but tensorflow/torch propagate into dtype
        # mismatches inside the delay computation.
        self.flatgrid = ops.cast(
            ops.convert_to_tensor(parameters.flatgrid if flatgrid is None else flatgrid), "float32"
        )
        self._adjoint_fn = None

    @property
    def n_pix(self):
        """Number of pixels in the imaging grid."""
        return int(self.flatgrid.shape[0])

    @property
    def input_shape(self):
        """Shape of the channel-data input, ``(n_tx, n_ax, n_el)``."""
        params = self.parameters
        return (params.n_tx, params.n_ax, params.n_el)

    def _lens_kwargs(self):
        params = self.parameters
        if not getattr(params, "apply_lens_correction", False):
            return {}
        return {
            "apply_lens_correction": True,
            "lens_thickness": params.lens_thickness,
            "lens_sound_speed": params.lens_sound_speed,
        }

    def forward(self, channel_data):
        """Beamform channel data into a flattened image.

        Args:
            channel_data (Tensor): Pre-beamformed data of shape
                ``(n_tx, n_ax, n_el)`` for RF or ``(n_tx, n_ax, n_el, n_ch)``
                (``n_ch=2`` for IQ).

        Returns:
            Tensor: Beamformed image of shape ``(n_pix,)`` when the input was
            3D, else ``(n_pix, n_ch)``.
        """
        params = self.parameters
        squeeze = len(channel_data.shape) == 3
        data = channel_data[..., None] if squeeze else channel_data
        n_ax = int(data.shape[1])
        n_ch = int(data.shape[-1])

        # Delays in samples: tx (n_pix, n_tx), rx (n_pix, n_el).
        tx_delays, rx_delays = calculate_delays(
            self.flatgrid,
            t0_delays=params.t0_delays,
            tx_apodizations=params.tx_apodizations,
            probe_geometry=params.probe_geometry,
            initial_times=params.initial_times,
            sampling_frequency=params.sampling_frequency,
            sound_speed=params.sound_speed,
            focus_distances=params.focus_distances,
            polar_angles=params.polar_angles,
            t_peak=params.t_peak,
            transmit_origins=params.transmit_origins,
            **self._lens_kwargs(),
        )

        if params.f_number == 0:
            mask = ops.ones((self.n_pix, params.n_el, 1))
        else:
            mask = fnumber_mask(
                self.flatgrid, params.probe_geometry, params.f_number, self.fnum_window_fn
            )

        demodulation_frequency = params.demodulation_frequency
        sampling_frequency = params.sampling_frequency

        def _beamform_tx(image, data_tx, txdel_tx):
            """TOF-correct one transmit and add its image contribution."""
            delays = rx_delays + txdel_tx[:, None]
            tof = apply_delays(data_tx, delays, clip_min=0, clip_max=n_ax - 1) * mask
            if n_ch == 2:
                theta = 2 * np.pi * demodulation_frequency * delays / sampling_frequency
                tof = complex_rotate(tof, theta)
            return image + ops.sum(tof, axis=1)

        accumulate = keras.remat(_beamform_tx)

        def _scan_body(image, xs):
            data_tx, txdel_tx = xs
            return accumulate(image, data_tx, txdel_tx), None

        # A static `length` keeps the scan XLA-compilable on the tensorflow
        # backend (its while_loop needs a fixed iteration count under jit).
        image, _ = ops.scan(
            _scan_body,
            ops.zeros((self.n_pix, n_ch), dtype="float32"),
            (data, ops.transpose(tx_delays)),
            length=int(data.shape[0]),
        )
        return image[:, 0] if squeeze else image

    def __call__(self, channel_data):
        """Alias for :meth:`forward`."""
        return self.forward(channel_data)

    def adjoint(self, image):
        """Apply the adjoint (transpose) of the beamforming operator.

        Computed with backend-agnostic automatic differentiation via
        :func:`zea.inverse.linear_adjoint`; the result is exact because the
        operator is linear.

        Args:
            image (Tensor): Flattened image of shape ``(n_pix,)``.

        Returns:
            Tensor: Channel data of shape ``(n_tx, n_ax, n_el)``.
        """
        if self._adjoint_fn is None:
            template = ops.zeros(self.input_shape, dtype="float32")
            self._adjoint_fn = linear_adjoint(self.forward, template)
        return self._adjoint_fn(image)

    def to_grid(self, image):
        """Reshape a flattened image to the 2D imaging grid.

        Args:
            image (Tensor): Flattened image of shape ``(n_pix,)``.

        Returns:
            Tensor: Image of shape ``(grid_size_z, grid_size_x)``.
        """
        grid_shape = self.parameters.grid.shape[:-1]
        return ops.reshape(image, grid_shape)


class ScattererSimulator:
    """Time-domain point-scatterer forward model producing channel data.

    Simulates pre-beamformed RF channel data of shape ``(n_tx, n_ax, n_el)``
    as a superposition of point-scatterer echoes:

    .. math::

        d_t(i, e) = \\sum_p a_p \\, c_{t,p} \\, D_{rx}(p, e) \\,
            w_t\\!\\left(i / f_s - \\tau_{t,p,e}\\right)

    where :math:`\\tau_{t,p,e}` is the transmit + receive travel time computed
    with :func:`zea.beamform.beamformer.calculate_delays` (the same delay model
    as the beamformer), :math:`w_t` is the scan's two-way waveform for transmit
    :math:`t`, :math:`c_{t,p}` combines spherical spreading and transmit
    directivity, and :math:`D_{rx}` is the receive element directivity.

    The travel times exclude the waveform peak offset ``t_peak``, so a
    beamformer using the same parameters samples each echo exactly at its
    waveform peak — the simulator and :class:`DASOperator` form a consistent
    forward model for inversion.

    Scatterers are processed in chunks inside a rematerialized ``ops.scan``
    (over transmits and chunks), so peak memory — also under automatic
    differentiation — is bounded by roughly
    ``chunk_size * n_ax * n_el * 4`` bytes regardless of the total number of
    scatterers or transmits.

    For a frequency-domain simulator with parametric pulses see
    :func:`zea.simulator.simulate_rf`; this class instead uses the measured
    waveforms stored with the scan, which matters when inverting real
    acquisitions.

    Args:
        parameters (zea.Parameters): Acquisition parameters. Must provide
            ``waveforms_two_way`` of shape ``(n_tx, n_samples)``.
        apply_directivity (bool, optional): Apply transmit/receive element
            directivity (requires ``parameters.element_width``). Defaults to
            ``True``.
        chunk_size (int, optional): Number of scatterers processed per chunk.
            Defaults to ``1024``.
        waveform_sampling_frequency (float, optional): Sampling frequency of
            the stored waveforms in Hz. Defaults to ``250e6``.
        reference_distance (float, optional): Distance in meters at which the
            spherical-spreading gain is 1 (closer scatterers are clipped to 1).
            Defaults to ``1e-3``.
    """

    def __init__(
        self,
        parameters,
        apply_directivity=True,
        chunk_size=1024,
        waveform_sampling_frequency=250e6,
        reference_distance=1e-3,
    ):
        self.parameters = parameters
        self.chunk_size = int(chunk_size)
        self.waveform_sampling_frequency = waveform_sampling_frequency
        self.reference_distance = reference_distance

        element_width = getattr(parameters, "element_width", None)
        if apply_directivity:
            if element_width is None:
                log.warning(
                    "ScattererSimulator: `parameters.element_width` is not set; "
                    "disabling element directivity."
                )
                apply_directivity = False
            else:
                wavelength = parameters.sound_speed / np.mean(
                    ops.convert_to_numpy(parameters.center_frequency)
                )
                self._element_width_wavelengths = float(element_width) / wavelength
        self.apply_directivity = apply_directivity

        waveforms = ops.cast(ops.convert_to_tensor(parameters.waveforms_two_way), "float32")
        # Zero-pad both ends so that out-of-range interpolation returns 0.
        self._waveforms = ops.pad(waveforms, ((0, 0), (1, 1)))
        self._n_waveform_samples = int(self._waveforms.shape[1])

    def _interp_waveform(self, waveform, t):
        """Linearly interpolate a (zero-padded) waveform at times ``t`` (s)."""
        n = self._n_waveform_samples
        index = ops.clip(t * self.waveform_sampling_frequency + 1.0, 0.0, n - 1)
        low = ops.cast(ops.floor(index), "int32")
        high = ops.minimum(low + 1, n - 1)
        frac = index - ops.cast(low, index.dtype)
        return ops.take(waveform, low) * (1.0 - frac) + ops.take(waveform, high) * frac

    def geometry(self, positions):
        """Precompute the position-dependent terms of the forward model.

        When solving for scatterer magnitudes with fixed positions (the linear
        subproblem), pass the result to :meth:`__call__` via ``geometry=`` to
        avoid recomputing travel times on every operator application.

        Args:
            positions (Tensor): Scatterer positions ``(x, y, z)`` of shape
                ``(n_scat, 3)`` in meters.

        Returns:
            dict: Travel times and directivity/spreading weights.
        """
        params = self.parameters
        positions = ops.cast(ops.convert_to_tensor(positions), "float32")
        n_tx = params.n_tx

        lens_kwargs = {}
        if getattr(params, "apply_lens_correction", False):
            lens_kwargs = {
                "apply_lens_correction": True,
                "lens_thickness": params.lens_thickness,
                "lens_sound_speed": params.lens_sound_speed,
            }
        tx_delays, rx_delays = calculate_delays(
            positions,
            t0_delays=params.t0_delays,
            tx_apodizations=params.tx_apodizations,
            probe_geometry=params.probe_geometry,
            initial_times=params.initial_times,
            sampling_frequency=params.sampling_frequency,
            sound_speed=params.sound_speed,
            focus_distances=params.focus_distances,
            polar_angles=params.polar_angles,
            t_peak=ops.zeros((n_tx,), dtype="float32"),
            transmit_origins=params.transmit_origins,
            **lens_kwargs,
        )
        # Back to seconds; tx_times: (n_scat, n_tx), rx_times: (n_scat, n_el)
        tx_times = tx_delays / params.sampling_frequency
        rx_times = rx_delays / params.sampling_frequency

        # Spherical spreading from the transmit travel distance, clipped so
        # that scatterers closer than the reference distance are not boosted.
        initial_times = ops.convert_to_tensor(params.initial_times)
        tx_distances = params.sound_speed * (tx_times + initial_times[None, :])
        tx_gain = ops.clip(self.reference_distance / (tx_distances + 1e-6), 0.0, 1.0)

        if self.apply_directivity:
            probe_geometry = ops.convert_to_tensor(params.probe_geometry)
            tx_apodizations = ops.convert_to_tensor(params.tx_apodizations)

            # Receive directivity per (scatterer, element).
            offsets = positions[:, None, :] - probe_geometry[None, :, :]
            lateral = ops.sqrt(offsets[..., 0] ** 2 + offsets[..., 1] ** 2)
            theta_rx = ops.arctan2(lateral, offsets[..., 2])
            rx_gain = _element_directivity(theta_rx, self._element_width_wavelengths)

            # Transmit directivity from the apodization-weighted aperture origin.
            weights = tx_apodizations / (ops.sum(tx_apodizations, axis=1, keepdims=True) + 1e-9)
            tx_origins = ops.matmul(weights, probe_geometry)  # (n_tx, 3)
            offsets = positions[:, None, :] - tx_origins[None, :, :]
            lateral = ops.sqrt(offsets[..., 0] ** 2 + offsets[..., 1] ** 2)
            theta_tx = ops.arctan2(lateral, offsets[..., 2])
            tx_gain = tx_gain * _element_directivity(theta_tx, self._element_width_wavelengths)
        else:
            rx_gain = ops.ones((int(positions.shape[0]), params.n_el), dtype="float32")

        return {"tx_times": tx_times, "rx_times": rx_times, "tx_gain": tx_gain, "rx_gain": rx_gain}

    def __call__(self, magnitudes, positions=None, geometry=None):
        """Simulate channel data for the given scatterers.

        Args:
            magnitudes (Tensor): Scatterer magnitudes of shape ``(n_scat,)``.
            positions (Tensor, optional): Scatterer positions of shape
                ``(n_scat, 3)``. Required when ``geometry`` is not given.
            geometry (dict, optional): Precomputed output of :meth:`geometry`.
                Pass this when repeatedly simulating with fixed positions.

        Returns:
            Tensor: Channel data of shape ``(n_tx, n_ax, n_el)``.
        """
        if geometry is None:
            if positions is None:
                raise ValueError("Provide either `positions` or a precomputed `geometry`.")
            geometry = self.geometry(positions)

        params = self.parameters
        magnitudes = ops.cast(ops.convert_to_tensor(magnitudes), "float32")
        n_scat = int(magnitudes.shape[0])
        chunk_size = min(self.chunk_size, n_scat)
        axial_times = ops.arange(params.n_ax, dtype="float32") / params.sampling_frequency

        # Chunk the per-scatterer quantities: (n_chunks, chunk, ...). Padded
        # scatterers get a huge arrival time (echo lands outside the waveform,
        # interpolating into the zero padding) and zero magnitude.
        tx_time_chunks = _pad_and_chunk(geometry["tx_times"], chunk_size, 1e3)
        rx_time_chunks = _pad_and_chunk(geometry["rx_times"], chunk_size, 1e3)
        tx_gain_chunks = _pad_and_chunk(geometry["tx_gain"], chunk_size, 0.0)
        rx_gain_chunks = _pad_and_chunk(geometry["rx_gain"], chunk_size, 0.0)
        magnitude_chunks = _pad_and_chunk(magnitudes, chunk_size, 0.0)

        def _chunk_contribution(channel, waveform, tx_time, tx_gain, rx_time, rx_gain, magnitude):
            """Add one scatterer chunk's echoes to a transmit's channel data."""
            tau = tx_time[:, None] + rx_time  # (chunk, n_el)
            echoes = self._interp_waveform(waveform, axial_times[None, :, None] - tau[:, None, :])
            weights = (magnitude * tx_gain)[:, None] * rx_gain  # (chunk, n_el)
            return channel + ops.einsum("cae,ce->ae", echoes, weights)

        accumulate = keras.remat(_chunk_contribution)

        def _tx_body(carry, xs):
            waveform, tx_time_tx, tx_gain_tx = xs

            def _chunk_body(channel, xs_chunk):
                tx_time, tx_gain, rx_time, rx_gain, magnitude = xs_chunk
                return accumulate(
                    channel, waveform, tx_time, tx_gain, rx_time, rx_gain, magnitude
                ), None

            channel, _ = ops.scan(
                _chunk_body,
                ops.zeros((params.n_ax, params.n_el), dtype="float32"),
                (tx_time_tx, tx_gain_tx, rx_time_chunks, rx_gain_chunks, magnitude_chunks),
                length=int(rx_time_chunks.shape[0]),
            )
            # The channel doubles as the carry: the tensorflow backend requires
            # the stacked per-step outputs to have the carry's shape and dtype.
            return channel, channel

        # Scan over transmits; per-transmit inputs are the waveform and the
        # transmit-dependent times/gains (moved to a leading transmit axis).
        _, channel_data = ops.scan(
            _tx_body,
            ops.zeros((params.n_ax, params.n_el), dtype="float32"),
            (
                self._waveforms,
                ops.moveaxis(tx_time_chunks, -1, 0),
                ops.moveaxis(tx_gain_chunks, -1, 0),
            ),
            length=params.n_tx,
        )
        return channel_data
