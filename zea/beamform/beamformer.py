"""Main beamforming functions for ultrasound imaging.

This module implements the core time-of-flight (TOF) correction pipeline used
by the :class:`~zea.ops.ultrasound.TOFCorrection` operation.  It also exposes
the lower-level building blocks (delay computation, f-number masking, phase
rotation, etc.) so that they can be used independently.
"""

import keras
import numpy as np
from keras import ops

from zea.beamform.lens_correction import compute_lens_corrected_travel_times
from zea.func.tensor import vmap
from zea.internal.checks import _check_raw_data


def fnum_window_fn_rect(normalized_angle):
    """Rectangular window function for f-number masking.

    Returns 1 when ``normalized_angle <= 1`` and 0 otherwise.

    Args:
        normalized_angle (Tensor): Normalized angle values (0 = on-axis, 1 = edge
            of the f-number cone).

    Returns:
        Tensor: Binary mask with the same shape as *normalized_angle*.
    """
    return ops.where(normalized_angle <= 1.0, 1.0, 0.0)


def fnum_window_fn_hann(normalized_angle):
    """Hann window function for f-number masking.

    Provides a smooth cosine roll-off from 1 at ``normalized_angle = 0``
    to 0 at ``normalized_angle = 1``.

    Args:
        normalized_angle (Tensor): Normalized angle values.

    Returns:
        Tensor: Apodization weights with the same shape as *normalized_angle*.
    """
    return ops.where(
        normalized_angle <= 1.0,
        0.5 * (1 + ops.cos(np.pi * normalized_angle)),
        0.0,
    )


def fnum_window_fn_tukey(normalized_angle, alpha=0.5):
    """Tukey window function for f-number masking.

    A Tukey window is flat in the center and tapers with a cosine lobe near
    the edges.  Setting ``alpha = 0`` produces a rectangular window;
    ``alpha = 1`` produces a Hann window.

    Args:
        normalized_angle (Tensor): Normalized angle values in [0, 1].
        alpha (float, optional): Shape parameter controlling the fraction of
            the window inside the cosine taper.  Defaults to ``0.5``.

    Returns:
        Tensor: Apodization weights with the same shape as *normalized_angle*.
    """
    normalized_angle = ops.clip(ops.abs(normalized_angle), 0.0, 1.0)

    beta = 1.0 - alpha

    return ops.where(
        normalized_angle < beta,
        1.0,
        ops.where(
            normalized_angle < 1.0,
            0.5 * (1 + ops.cos(np.pi * (normalized_angle - beta) / (ops.abs(alpha) + 1e-6))),
            0.0,
        ),
    )


def tof_correction(
    data,
    flatgrid,
    t0_delays,
    tx_apodizations,
    sound_speed,
    probe_geometry,
    initial_times,
    sampling_frequency,
    demodulation_frequency,
    f_number,
    polar_angles,
    focus_distances,
    t_peak,
    tx_waveform_indices,
    transmit_origins,
    apply_lens_correction=False,
    lens_thickness=1e-3,
    lens_sound_speed=1000,
    fnum_window_fn=fnum_window_fn_rect,
    sos_map=None,
    sos_grid_x=None,
    sos_grid_z=None,
):
    """Time-of-flight (TOF) correction for ultrasound data on a flat pixel grid.

    Corrects raw RF or IQ data for differences in propagation time from the
    transmitter through each pixel and back to every receiving element.  Two
    modes are supported:

    * **Homogeneous medium** (default) — a constant ``sound_speed`` is used
      to compute delays analytically via :func:`calculate_delays`.
    * **Heterogeneous medium** — a spatially-varying speed-of-sound map
      (``sos_map``) is provided and delays are computed numerically via
      :func:`calculate_delays_heterogeneous_medium`.

    .. important::

       The heterogeneous mode currently requires **multistatic** acquisitions
       (``n_tx == n_el``).

    After delay computation the data is interpolated to the requested pixel
    positions, masked with the receive f-number aperture, and — for IQ data —
    phase-rotated to compensate for the demodulation carrier (see
    :func:`complex_rotate`).

    Args:
        data (Tensor): Input RF or IQ data of shape ``(n_tx, n_ax, n_el, n_ch)``.
            Use ``n_ch=1`` for RF data and ``n_ch=2`` for IQ (in-phase /
            quadrature).
        flatgrid (Tensor): Pixel locations ``(x, y, z)`` of shape ``(n_pix, 3)``.
        t0_delays (Tensor): Per-element transmit fire times, shifted so that
            the first element fires at *t = 0*, of shape ``(n_tx, n_el)``.
        tx_apodizations (Tensor): Transmit apodization weights of shape
            ``(n_tx, n_el)``.
        sound_speed (float): Speed of sound in m/s.
        probe_geometry (Tensor): Element positions ``(x, y, z)`` of shape
            ``(n_el, 3)``.
        initial_times (Tensor): Per-transmit time offsets of shape ``(n_tx,)``.
        sampling_frequency (float): Sampling frequency in Hz.
        demodulation_frequency (float): Demodulation (carrier) frequency in Hz.
            Only used when ``n_ch=2`` (IQ data).
        f_number (float): Receive f-number.  Set to ``0`` to disable
            f-number masking.
        polar_angles (Tensor): Steering angles in radians of shape ``(n_tx,)``.
        focus_distances (Tensor): Focus distances in meters of shape
            ``(n_tx,)``.  Use ``0`` or ``np.inf`` for plane-wave transmission.
        t_peak (Tensor): Time of each waveform peak in seconds of shape
            ``(n_waveforms,)``.
        tx_waveform_indices (Tensor): Index into ``t_peak`` for each transmit
            of shape ``(n_tx,)``.
        transmit_origins (Tensor): Origin of each transmit beam of shape
            ``(n_tx, 3)``.
        apply_lens_correction (bool, optional): Apply acoustic-lens correction
            to the receive travel times (slower but more accurate in the
            near-field).  Defaults to ``False``.
        lens_thickness (float, optional): Lens thickness in meters.
            Defaults to ``1e-3``.
        lens_sound_speed (float, optional): Speed of sound inside the lens in
            m/s.  Defaults to ``1000``.
        fnum_window_fn (callable, optional): Window function applied to the
            normalized angle for f-number masking.  Receives values in ``[0, 1]``
            and should return ``0`` for values ``> 1``.
            Defaults to :func:`fnum_window_fn_rect`.
        sos_map (Tensor, optional): 2-D speed-of-sound map of shape
            ``(Nz, Nx)`` in m/s.  When provided, delays are computed
            numerically (heterogeneous mode, multistatic only).
            Defaults to ``None``.
        sos_grid_x (Tensor, optional): x-coordinates of ``sos_map`` columns.
        sos_grid_z (Tensor, optional): z-coordinates of ``sos_map`` rows.

    Returns:
        Tensor: Time-of-flight corrected data of shape
        ``(n_tx, n_pix, n_el, n_ch)``.
    """
    assert len(data.shape) == 4, (
        "The input data should have 4 dimensions, "
        f"namely n_tx, n_ax, n_el, n_ch, got {len(data.shape)} dimensions: {data.shape}"
    )

    n_tx, n_ax, n_el, _ = ops.shape(data)
    n_pix = ops.shape(flatgrid)[0]

    _validate_delay_inputs(data, flatgrid, t0_delays, probe_geometry, tx_apodizations)

    # ---- Compute delays ------------------------------------------------
    # txdel: transmit delay from t=0 to wavefront reaching each pixel
    # rxdel: receive delay from each pixel back to each element
    # After this block both have a consistent layout:
    #   txdel: (n_pix, n_tx)   rxdel: (n_pix, n_el)
    if sos_map is None:
        txdel, rxdel = calculate_delays(
            flatgrid,
            t0_delays,
            tx_apodizations,
            probe_geometry,
            initial_times,
            sampling_frequency,
            sound_speed,
            focus_distances,
            polar_angles,
            t_peak,
            tx_waveform_indices,
            transmit_origins,
            apply_lens_correction,
            lens_thickness,
            lens_sound_speed,
        )
        # calculate_delays returns txdel (n_pix, n_tx), rxdel (n_pix, n_el)
    else:
        assert apply_lens_correction is False, (
            "Lens correction is not currently supported in heterogeneous SOS mode. "
            "Either set apply_lens_correction=False or set sos_map=None."
        )

        txdel, rxdel = calculate_delays_heterogeneous_medium(
            flatgrid,
            sos_map,
            sos_grid_x,
            sos_grid_z,
            t0_delays,
            probe_geometry,
            initial_times,
            sampling_frequency,
            t_peak,
            tx_waveform_indices,
        )
        # calculate_delays_heterogeneous_medium returns txdel (n_tx, n_pix), rxdel (n_el, n_pix)
        # Transpose both to the shared convention.
        txdel = ops.moveaxis(txdel, 1, 0)  # -> (n_pix, n_tx)
        rxdel = ops.moveaxis(rxdel, 1, 0)  # -> (n_pix, n_el)

    # ---- F-number mask (receive aperture) ------------------------------
    mask = ops.cond(
        f_number == 0,
        lambda: ops.ones((n_pix, n_el, 1)),
        lambda: fnumber_mask(flatgrid, probe_geometry, f_number, fnum_window_fn=fnum_window_fn),
    )
    if sos_map is not None:
        # Prevent gradients from flowing through the mask when optimising
        # through the heterogeneous beamformer (e.g. SOS estimation).
        mask = ops.stop_gradient(mask)

    # ---- Correct a single transmit (closure) ---------------------------
    def _correct_single_tx(data_tx, txdel_tx, mask_tx=None):
        """Apply delay-and-interpolate for one transmit event.

        Args:
            data_tx (Tensor): RF/IQ data for one transmit ``(n_ax, n_el, n_ch)``.
            txdel_tx (Tensor): Transmit delays ``(n_pix, 1)``.
            mask_tx (Tensor, optional): Per-pixel transmit mask ``(n_pix, 1)``.

        Returns:
            Tensor: TOF-corrected data ``(n_pix, n_el, n_ch)``.
        """
        # Total delay per pixel per element: (n_pix, n_el)
        delays = rxdel + txdel_tx

        # Interpolate data at the computed delay positions
        tof_tx = apply_delays(data_tx, delays, clip_min=0, clip_max=n_ax - 1)

        # Apply f-number mask(s)
        if mask_tx is not None:
            tof_tx = tof_tx * mask * mask_tx[:, :, None]
        else:
            tof_tx = tof_tx * mask

        # Phase rotation for IQ data (see complex_rotate docstring)
        if data_tx.shape[-1] == 2:
            total_delay_seconds = delays / sampling_frequency
            theta = 2 * np.pi * demodulation_frequency * total_delay_seconds
            tof_tx = complex_rotate(tof_tx, theta)

        return tof_tx

    # ---- Vectorize over transmits --------------------------------------
    # Reshape txdel from (n_pix, n_tx) -> (n_tx, n_pix, 1) for per-tx slicing
    txdel = ops.moveaxis(txdel, 1, 0)[..., None]

    if sos_map is None:
        return vmap(_correct_single_tx)(data, txdel)

    # Heterogeneous path: apply transmit f-number mask and use gradient
    # checkpointing to limit memory consumption.
    mask_tx = ops.moveaxis(mask, 1, 0)
    _correct_single_tx_ckpt = keras.remat(_correct_single_tx)
    return vmap(_correct_single_tx_ckpt)(data, txdel, mask_tx)


def _validate_delay_inputs(data, grid, t0_delays, probe_geometry, tx_apodizations):
    """Validate input shapes common to all delay computation functions.

    Args:
        data (Tensor): Input RF or IQ data of shape ``(n_tx, n_ax, n_el, n_ch)``.
        grid (Tensor): Pixel coordinates of shape ``(n_pix, 3)``.
        t0_delays (Tensor): Per-element transmit delays of shape
            ``(n_tx, n_el)``.
        probe_geometry (Tensor): Element positions of shape ``(n_el, 3)``.

    Raises:
        AssertionError: If any array is not 2-D or if any array has an incompatible shape.
    """
    n_tx, n_ax, n_el, n_ch = ops.shape(data)

    _check_raw_data(data)

    for arr in [grid, t0_delays, probe_geometry, tx_apodizations]:
        assert arr.ndim == 2, f"Expected a 2-D array, got shape {arr.shape}."

    assert ops.shape(grid)[1] == 3, f"Expected grid to have shape (n_pix, 3), got {grid.shape}."
    assert ops.shape(probe_geometry) == (n_el, 3), (
        f"Expected probe_geometry to have shape (n_el, 3), "
        f"got {probe_geometry.shape} != {(n_el, 3)}."
    )
    assert ops.shape(t0_delays) == (n_tx, n_el), (
        f"Expected t0_delays to have shape (n_tx, n_el), got {t0_delays.shape} != {(n_tx, n_el)}."
    )
    assert ops.shape(tx_apodizations) == (n_tx, n_el), (
        "Expected tx_apodizations to have shape (n_tx, n_el), "
        f"got {tx_apodizations.shape} != {(n_tx, n_el)}."
    )


def calculate_delays(
    grid,
    t0_delays,
    tx_apodizations,
    probe_geometry,
    initial_times,
    sampling_frequency,
    sound_speed,
    focus_distances,
    polar_angles,
    t_peak,
    tx_waveform_indices,
    transmit_origins,
    apply_lens_correction=False,
    lens_thickness=None,
    lens_sound_speed=None,
    n_iter=2,
):
    """Compute transmit and receive delays in samples to every pixel.

    The total round-trip delay for a pixel is the sum of two components:

    * **Transmit delay** — time from transmission until the wavefront
      reaches the pixel.
    * **Receive delay** — time from the pixel back to each transducer
      element.

    Both are returned in **sample units** (i.e. already multiplied by
    ``sampling_frequency``).

    Args:
        grid (Tensor): Pixel coordinates of shape ``(n_pix, 3)``.
        t0_delays (Tensor): Per-element transmit delays in seconds of shape
            ``(n_tx, n_el)``, shifted so that the smallest delay is 0.
        tx_apodizations (Tensor): Transmit apodization weights of shape
            ``(n_tx, n_el)``.
        probe_geometry (Tensor): Element positions of shape ``(n_el, 3)``.
        initial_times (Tensor): Per-transmit time offsets of shape ``(n_tx,)``.
        sampling_frequency (float): Sampling frequency in Hz.
        sound_speed (float): Assumed speed of sound in m/s.
        focus_distances (Tensor): Focus distances of shape ``(n_tx,)``.
            Use ``0`` or ``np.inf`` for plane-wave transmission.
        polar_angles (Tensor): Polar steering angles in radians of shape
            ``(n_tx,)``.
        t_peak (Tensor): Waveform peak times in seconds of shape
            ``(n_waveforms,)``.
        tx_waveform_indices (Tensor): Index into ``t_peak`` for each transmit
            of shape ``(n_tx,)``.
        transmit_origins (Tensor): Origin of each transmit beam of shape
            ``(n_tx, 3)``.
        apply_lens_correction (bool, optional): Apply acoustic-lens
            correction (slower but more accurate in the near-field).
            Defaults to ``False``.
        lens_thickness (float, optional): Lens thickness in meters.
        lens_sound_speed (float, optional): Speed of sound in the lens in
            m/s.
        n_iter (int, optional): Newton-Raphson iterations for lens
            correction.  Defaults to ``2``.

    Returns:
        tuple[Tensor, Tensor]:
            - **transmit_delays** — of shape ``(n_pix, n_tx)``.
            - **receive_delays** — of shape ``(n_pix, n_el)``.
    """

    if not apply_lens_correction:
        # Compute receive distances in meters of shape (n_pix, n_el)
        rx_distances = distance_Rx(grid, probe_geometry)

        # Convert distances to delays in seconds
        rx_delays = rx_distances / sound_speed
    else:
        # Compute lens-corrected travel times from each element to each pixel
        assert lens_thickness is not None, "lens_thickness must be provided for lens correction."
        assert lens_sound_speed is not None, (
            "lens_sound_speed must be provided for lens correction."
        )
        rx_delays = compute_lens_corrected_travel_times(
            probe_geometry,
            grid,
            lens_thickness,
            lens_sound_speed,
            sound_speed,
            n_iter=n_iter,
        )

    # Compute transmit delays
    tx_delays = vmap(transmit_delays, in_axes=(None, 0, 0, None, 0, 0, 0, None, 0), out_axes=1)(
        grid,
        t0_delays,
        tx_apodizations,
        rx_delays,
        focus_distances,
        polar_angles,
        initial_times,
        None,
        transmit_origins,
    )

    # Add the offset to the transmit peak time
    tx_delays += ops.take(t_peak, tx_waveform_indices)[None]

    # TODO: nan to num needed?
    # tx_delays = ops.nan_to_num(tx_delays, nan=0.0, posinf=0.0, neginf=0.0)
    # rx_delays = ops.nan_to_num(rx_delays, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert from seconds to samples
    tx_delays *= sampling_frequency
    rx_delays *= sampling_frequency

    return tx_delays, rx_delays


def apply_delays(data, delays, clip_min: int = -1, clip_max: int = -1):
    """Interpolate RF/IQ data at fractional sample positions.

    Because the exact delay to a pixel will almost never fall on an integer
    sample index, this function performs **linear interpolation** between the
    two nearest samples (floor and ceil of each delay value).

    Args:
        data (Tensor): RF or IQ data of shape ``(n_ax, n_el, n_ch)``.
        delays (Tensor): Delays **in samples** of shape ``(n_pix, n_el)``.
        clip_min (int, optional): Minimum allowed sample index.  ``-1`` means
            no clipping.  Defaults to ``-1``.
        clip_max (int, optional): Maximum allowed sample index.  ``-1`` means
            no clipping.  Defaults to ``-1``.

    Returns:
        Tensor: Interpolated samples of shape ``(n_pix, n_el, n_ch)``.
    """

    # Add a dummy channel dimension to the delays tensor to ensure it has the
    # same number of dimensions as the data. The new shape is (n_pix, n_el, 1)
    delays = delays[..., None]

    # Get the integer values above and below the exact delay values
    # Floor to get the integers below
    # (num_elements, num_pixels, 1)
    d0 = ops.floor(delays)

    # Cast to integer to be able to use as indices
    d0 = ops.cast(d0, "int32")
    # Add 1 to find the integers above the exact delay values
    d1 = d0 + 1

    # Apply clipping of delays clipping to ensure correct behavior on cpu
    if clip_min != -1 and clip_max != -1:
        clip_min = ops.cast(clip_min, d0.dtype)
        clip_max = ops.cast(clip_max, d0.dtype)
        d0 = ops.clip(d0, clip_min, clip_max)
        d1 = ops.clip(d1, clip_min, clip_max)

    if data.shape[-1] == 2:
        d0 = ops.concatenate([d0, d0], axis=-1)
        d1 = ops.concatenate([d1, d1], axis=-1)

    # Gather pixel values
    # Here we extract for each transducer element the sample containing the
    # reflection from each pixel. These are of shape `(n_pix, n_el, n_ch)`.
    data0 = ops.take_along_axis(data, d0, 0)
    data1 = ops.take_along_axis(data, d1, 0)

    # Compute interpolated pixel value
    d0 = ops.cast(d0, delays.dtype)  # Cast to float
    d1 = ops.cast(d1, delays.dtype)  # Cast to float
    data0 = ops.cast(data0, delays.dtype)  # Cast to float
    data1 = ops.cast(data1, delays.dtype)  # Cast to float
    reflection_samples = (d1 - delays) * data0 + (delays - d0) * data1

    return reflection_samples


def complex_rotate(iq, theta):
    r"""Phase-rotate IQ data by angle *theta*.

    When delaying IQ-demodulated data it is not sufficient to interpolate the
    I and Q channels independently — the carrier phase shift must be
    compensated as well.  This function applies the rotation:

    .. math::

        I_\Delta &= I' \cos\theta - Q' \sin\theta \\
        Q_\Delta &= Q' \cos\theta + I' \sin\theta

    Args:
        iq (Tensor): IQ data of shape ``(..., 2)``.
        theta (Tensor or float): Rotation angle in radians (broadcastable to
            ``iq[..., 0]``).

    Returns:
        Tensor: Rotated IQ data of shape ``(..., 2)``.

    .. dropdown:: Derivation

        The IQ data is related to the RF data as follows:

        .. math::

            x(t) &= I(t)\cos(\omega_c t) + Q(t)\cos(\omega_c t + \pi/2)\\
            &= I(t)\cos(\omega_c t) - Q(t)\sin(\omega_c t)


        If we want to delay the RF data `x(t)` by `Δt` we can substitute in
        :math:`t=t+\Delta t`. We also define :math:`I'(t) = I(t + \Delta t)`,
        :math:`Q'(t) = Q(t + \Delta t)`, and :math:`\theta=\omega_c\Delta t`.
        This gives us:

        .. math::

            x(t + \Delta t) &= I'(t) \cos(\omega_c (t + \Delta t))
            - Q'(t) \sin(\omega_c (t + \Delta t))\\
            &=  \overbrace{(I'(t)\cos(\theta)
            - Q'(t)\sin(\theta) )}^{I_\Delta(t)} \cos(\omega_c t)\\
            &- \overbrace{(Q'(t)\cos(\theta)
            + I'(t)\sin(\theta))}^{Q_\Delta(t)} \sin(\omega_c t)

        This means that to correctly interpolate the IQ data to the new components
        :math:`I_\Delta(t)` and :math:`Q_\Delta(t)`, it is not sufficient to just
        interpolate the I- and Q-channels independently. We also need to rotate the
        I- and Q-channels by the angle :math:`\theta`. This function performs this
        rotation.
    """
    assert iq.shape[-1] == 2, (
        "The last dimension of the input tensor should be 2, "
        f"got {iq.shape[-1]} dimensions and shape {iq.shape}."
    )
    # Select i and q channels
    i = iq[..., 0]
    q = iq[..., 1]

    # Compute rotated components
    ir = i * ops.cos(theta) - q * ops.sin(theta)
    qr = q * ops.cos(theta) + i * ops.sin(theta)

    # Reintroduce channel dimension
    ir = ir[..., None]
    qr = qr[..., None]

    return ops.concatenate([ir, qr], -1)


def distance_Rx(grid, probe_geometry):
    """Euclidean distance from every pixel to every transducer element.

    Args:
        grid (Tensor): Pixel positions ``(x, y, z)`` of shape ``(n_pix, 3)``.
        probe_geometry (Tensor): Element positions ``(x, y, z)`` of shape
            ``(n_el, 3)``.

    Returns:
        Tensor: Distances of shape ``(n_pix, n_el)``.
    """
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = ops.linalg.norm(grid[:, None, :] - probe_geometry[None, :, :], axis=-1)
    return dist


def transmit_delays(
    grid,
    t0_delays,
    tx_apodization,
    rx_delays,
    focus_distance,
    polar_angle,
    initial_time,
    azimuth_angle=None,
    transmit_origin=None,
):
    """Compute the transmit delay from transmission to each pixel.

    Uses the **first-arrival** time for pixels before the focus (or virtual
    source) and the **last-arrival** time for pixels beyond the focus.

    Args:
        grid (Tensor): Pixel positions ``(x, y, z)`` of shape ``(n_pix, 3)``.
        t0_delays (Tensor): Per-element transmit delays in seconds of shape
            ``(n_el,)``.
        tx_apodization (Tensor): Transmit apodization weights of shape
            ``(n_el,)``.
        rx_delays (Tensor): Travel times in seconds from elements to pixels
            of shape ``(n_pix, n_el)``.
        focus_distance (float): Focus distance in meters.
            Use ``0`` or ``np.inf`` for plane-wave transmission.
        polar_angle (float): Polar steering angle in radians.
        initial_time (float): Time offset for this transmit in seconds.
        azimuth_angle (float, optional): Azimuth steering angle in radians.
            Defaults to ``None`` (treated as 0).
        transmit_origin (Tensor, optional): Origin of the transmit beam of
            shape ``(3,)``.  Defaults to ``(0, 0, 0)``.

    Returns:
        Tensor: Transmit delays of shape ``(n_pix,)``.
    """
    # Add a large offset for elements that are not used in the transmit to
    # disqualify them from being the closest element
    offset = ops.where(tx_apodization == 0, np.inf, 0.0)

    # Compute total travel time from t=0 to each pixel via each element
    # rx_delays has shape (n_pix, n_el)
    # t0_delays has shape (n_el,)
    total_times = rx_delays + t0_delays[None, :]

    if azimuth_angle is None:
        azimuth_angle = ops.zeros_like(polar_angle)

    # Set origin to (0, 0, 0) if not provided
    if transmit_origin is None:
        transmit_origin = ops.zeros(3, dtype=grid.dtype)

    # Compute the 3D position of the focal point
    # The beam direction vector
    beam_direction = ops.stack(
        [
            ops.sin(polar_angle) * ops.cos(azimuth_angle),
            ops.sin(polar_angle) * ops.sin(azimuth_angle),
            ops.cos(polar_angle),
        ]
    )

    # Handle plane wave case where focus_distance is set to zero
    # We use np.inf to consider the first wavefront arrival for all pixels
    focus_distance = ops.where(focus_distance == 0.0, np.inf, focus_distance)

    # Compute focal point position: origin + focus_distance * beam_direction
    # For negative focus_distance (diverging/virtual source), this is behind the origin
    focal_point = transmit_origin + focus_distance * beam_direction  # shape (3,)

    # Deal with plane wave case where focus_distance is infinite and beam_direction is zero
    # (np.inf * 0.0 -> nan) so we convert nan to zero
    focal_point = ops.where(ops.isnan(focal_point), 0.0, focal_point)

    # Compute the position of each pixel relative to the focal point
    pixel_relative_to_focus = grid - focal_point[None, :]  # shape (n_pix, 3)

    # Project onto the beam direction to determine if pixel is before or after focus
    # Positive projection means pixel is in the direction of beam propagation (beyond focus)
    # Negative projection means pixel is behind the focus (before focus)
    projection_along_beam = ops.sum(
        pixel_relative_to_focus * beam_direction[None, :], axis=-1
    )  # shape (n_pix,)

    # For focused waves (positive focus_distance):
    #   - Use min time for pixels before focus (projection < 0)
    #   - Use max time for pixels beyond focus (projection > 0)
    # For diverging waves (negative focus_distance, virtual source):
    #   - The sign of focus_distance flips the logic
    #   - Use min time for pixels between transducer and virtual source
    #   - Use max time for pixels beyond transducer
    is_before_focus = ops.cast(ops.sign(focus_distance), "float32") * projection_along_beam < 0.0

    # Compute the effective time of the pixels to the wavefront by computing the
    # smallest time over all elements (first wavefront arrival) for pixels before
    # the focus, and the largest time (last wavefront contribution) for pixels
    # beyond the focus.
    tx_delay = ops.where(
        is_before_focus,
        ops.min(total_times + offset[None, :], axis=-1),
        ops.max(total_times - offset[None, :], axis=-1),
    )

    # Subtract the initial time offset for this transmit
    tx_delay = tx_delay - initial_time

    return tx_delay


def fnumber_mask(flatgrid, probe_geometry, f_number, fnum_window_fn):
    """Receive-aperture apodization mask based on the f-number.

    Computes a per-pixel, per-element mask that suppresses contributions
    from elements whose angle to a pixel exceeds the acceptance cone
    defined by the f-number.  The transition within the cone is controlled
    by *fnum_window_fn* (e.g. :func:`fnum_window_fn_rect`,
    :func:`fnum_window_fn_hann`, :func:`fnum_window_fn_tukey`).

    Args:
        flatgrid (Tensor): Flattened pixel grid of shape ``(n_pix, 3)``.
        probe_geometry (Tensor): Element positions of shape ``(n_el, 3)``.
        f_number (float): Receive f-number (depth / aperture).  A value
            of ``0`` disables masking.
        fnum_window_fn (callable): Window function mapping normalized
            angles in ``[0, 1]`` to weights.  Must return ``0`` for inputs
            ``> 1``.

    Returns:
        Tensor: Mask of shape ``(n_pix, n_el, 1)``.
    """

    grid_relative_to_probe = flatgrid[:, None] - probe_geometry[None]

    grid_relative_to_probe_norm = ops.linalg.norm(grid_relative_to_probe, axis=-1)

    grid_relative_to_probe_z = grid_relative_to_probe[..., 2] / (grid_relative_to_probe_norm + 1e-6)

    alpha = ops.arccos(grid_relative_to_probe_z)

    # The f-number is f_number = z/aperture = 1/(2 * tan(alpha))
    # Rearranging gives us alpha = arctan(1/(2 * f_number))
    # We can use this to compute the maximum angle alpha that is allowed
    max_alpha = ops.arctan(1 / (2 * f_number + keras.backend.epsilon()))

    normalized_angle = alpha / max_alpha
    mask = fnum_window_fn(normalized_angle)

    # Add dummy channel dimension
    mask = mask[..., None]

    return mask


def calculate_delays_heterogeneous_medium(
    grid,
    sos_map,
    sos_grid_x,
    sos_grid_z,
    t0_delays,
    probe_geometry,
    initial_times,
    sampling_frequency,
    t_peak,
    tx_waveform_indices,
    n_ray_points=100,
):
    """Compute delays using a spatially-varying speed-of-sound map.

    Integrates the slowness (1 / speed-of-sound) along straight rays
    between each element and each pixel to approximate heterogeneous
    travel times.

    For the homogeneous (constant speed-of-sound) variant see
    :func:`calculate_delays`. If you do not have a SOS map, it is
    recommended to use :func:`calculate_delays`.

    .. important::

       Only valid for **multistatic** acquisitions (``n_tx == n_el``).

    .. note::

        Currently only supports 2D grids, not yet compatible with 3D.
        Assumes the grid is in the x-z plane and the y dimension is zero.
        Please use :func:`calculate_delays` for 3D data.

    .. note::

        This function is not compatible with the torch backend.

    Args:
        grid (Tensor): Pixel coordinates of shape ``(n_pix, 3)``.
        sos_map (Tensor): Speed-of-sound map of shape ``(Nz, Nx)`` in m/s.
        sos_grid_x (Tensor): x-coordinates of ``sos_map`` columns.
        sos_grid_z (Tensor): z-coordinates of ``sos_map`` rows.
        t0_delays (Tensor): Transmit delays of shape ``(n_tx, n_el)``,
            shifted so that the smallest delay is 0.
        probe_geometry (Tensor): Element positions of shape ``(n_el, 3)``.
        initial_times (Tensor): Per-transmit time offsets of shape ``(n_tx,)``.
        sampling_frequency (float): Sampling frequency in Hz.
        t_peak (Tensor): Waveform peak times of shape ``(n_waveforms,)``.
        tx_waveform_indices (Tensor): Index into ``t_peak`` for each transmit
            of shape ``(n_tx,)``.
        n_ray_points (int, optional): Number of integration points along
            each element-to-pixel ray.  Higher values improve accuracy at
            the cost of computation time.  Defaults to ``100``.

    Returns:
        tuple[Tensor, Tensor]:
            - **tx_delays** — Transmit delays in samples ``(n_tx, n_pix)``.
            - **rx_delays** — Receive delays in samples ``(n_el, n_pix)``.
    """
    n_tx = ops.shape(t0_delays)[0]
    n_el = ops.shape(probe_geometry)[0]

    if keras.backend.backend() == "torch":
        raise NotImplementedError(
            "calculate_delays_heterogeneous_medium is not currently "
            "implemented for the torch backend."
        )

    assert n_tx == n_el, (
        "Computing delays with heterogeneous medium (a sos grid was provided) "
        "requires a multistatic dataset (n_tx == n_el), "
        f"got n_tx={n_tx}, n_el={n_el}."
    )

    ray_parameters = ops.linspace(1, 0, n_ray_points, endpoint=False)[::-1]
    slowness_map = 1 / sos_map

    grid_x = grid[:, 0]
    grid_z = grid[:, 2]

    element_x = probe_geometry[:, 0]
    element_z = probe_geometry[:, 2]

    def _interpolate_slowness(p, el_x, el_z):
        xp = p * (grid_x - el_x) + el_x
        zp = p * (grid_z - el_z) + el_z

        dx_sos = sos_grid_x[1] - sos_grid_x[0]
        dz_sos = sos_grid_z[1] - sos_grid_z[0]

        xit = (xp - sos_grid_x[0]) / dx_sos
        zit = (zp - sos_grid_z[0]) / dz_sos

        coords = ops.stack([zit, xit], axis=0)

        return keras.ops.image.map_coordinates(
            slowness_map,
            coords,
            order=1,
            fill_mode="nearest",
        )

    # Euclidean distance from each element to each pixel
    dx = ops.abs(element_x[:, None] - grid_x[None, :])
    dz = ops.abs(element_z[:, None] - grid_z[None, :])
    ray_lengths = ops.sqrt(dx**2 + dz**2)

    # Average slowness along each ray via numerical integration
    slowness = vmap(
        lambda el_x, el_z: vmap(lambda p: _interpolate_slowness(p, el_x, el_z))(ray_parameters)
    )(element_x, element_z)

    valid_mask = ~ops.isnan(slowness)
    masked_sum = ops.sum(ops.where(valid_mask, slowness, 0.0), axis=1)
    count = ops.cast(ops.sum(valid_mask, axis=1), masked_sum.dtype)
    mean_slowness = masked_sum / (count + 1e-9)

    tof = mean_slowness * ray_lengths
    rx_delays = tof * sampling_frequency
    tx_delays = (
        tof
        # The diagonal of t0_delays selects the appropriate transmit delay
        # for each element (n_tx, n_el) -> (n_tx,)
        - initial_times[:, None]
        # can take diag because of the multistatic assumption (n_tx == n_el)
        + ops.diag(t0_delays)[:, None]
        + ops.take(t_peak, tx_waveform_indices)[:, None]
    ) * sampling_frequency
    return tx_delays, rx_delays
