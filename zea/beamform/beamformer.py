"""Main beamforming functions for ultrasound imaging."""

import keras
import numpy as np
from keras import ops

from zea.beamform.lens_correction import compute_lens_corrected_travel_times
from zea.func.tensor import vmap


def fnum_window_fn_rect(normalized_angle):
    """Rectangular window function for f-number masking."""
    return ops.where(normalized_angle <= 1.0, 1.0, 0.0)


def fnum_window_fn_hann(normalized_angle):
    """Hann window function for f-number masking."""
    # Use a Hann window function to smoothly transition the mask
    return ops.where(
        normalized_angle <= 1.0,
        0.5 * (1 + ops.cos(np.pi * normalized_angle)),
        0.0,
    )


def fnum_window_fn_tukey(normalized_angle, alpha=0.5):
    """Tukey window function for f-number masking.

    Args:
        normalized_angle (ops.Tensor): Normalized angle values in the range [0, 1].
        alpha (float, optional): The alpha parameter for the Tukey window. 0.0 corresponds to a
            rectangular window, 1.0 corresponds to a Hann window. Defaults to 0.5.
    """
    # Use a Tukey window function to smoothly transition the mask
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
):
    """Time-of-flight correction for a flat grid.

    Args:
        data (ops.Tensor): Input RF/IQ data of shape `(n_tx, n_ax, n_el, n_ch)`.
        flatgrid (ops.Tensor): Pixel locations x, y, z of shape `(n_pix, 3)`
        t0_delays (ops.Tensor): Times at which the elements fire shifted such
            that the first element fires at t=0 of shape `(n_tx, n_el)`
        tx_apodizations (ops.Tensor): Transmit apodizations of shape `(n_tx, n_el)`
        sound_speed (float): Speed-of-sound.
        probe_geometry (ops.Tensor): Element positions x, y, z of shape (n_el, 3)
        initial_times (Tensor): The probe transmit time offsets of shape `(n_tx,)`.
        sampling_frequency (float): Sampling frequency.
        demodulation_frequency (float): Demodulation frequency.
        f_number (float): Focus number (ratio of focal depth to aperture size).
        polar_angles (ops.Tensor): The angles of the waves in radians of shape `(n_tx,)`
        focus_distances (ops.Tensor): The focus distance of shape `(n_tx,)`
        t_peak (ops.Tensor): Time of the peak of the pulse in seconds.
            Shape `(n_waveforms,)`.
        tx_waveform_indices (ops.Tensor): The indices of the waveform used for each
            transmit of shape `(n_tx,)`.
        transmit_origins (ops.Tensor): Transmit origins of shape (n_tx, 3).
        apply_lens_correction (bool, optional): Whether to apply lens correction to
            time-of-flights. This makes it slower, but more accurate in the near-field.
            Defaults to False.
        lens_thickness (float, optional): Thickness of the lens in meters. Used for
            lens correction. Defaults to 1e-3.
        lens_sound_speed (float, optional): Speed of sound in the lens in m/s. Used
            for lens correction Defaults to 1000.
        fnum_window_fn (callable, optional): F-number function to define the transition from
            straight in front of the element (fn(0.0)) to the largest angle within the f-number cone
            (fn(1.0)). The function should be zero for fn(x>1.0).

    Returns:
        (ops.Tensor): time-of-flight corrected data
        with shape: `(n_tx, n_pix, n_el, n_ch)`.
    """

    assert len(data.shape) == 4, (
        "The input data should have 4 dimensions, "
        f"namely n_tx, n_ax, n_el, n_ch, got {len(data.shape)} dimensions: {data.shape}"
    )

    n_tx, n_ax, n_el, _ = ops.shape(data)

    # Calculate delays
    # --------------------------------------------------------------------
    # txdel: The delay from t=0 to the wavefront reaching the pixel
    # txdel has shape (n_tx, n_pix)
    #
    # rxdel: The delay from the wavefront reaching the pixel to the scattered wave
    # reaching the transducer element.
    # rxdel has shape (n_el, n_pix)
    # --------------------------------------------------------------------

    txdel, rxdel = calculate_delays(
        flatgrid,
        t0_delays,
        tx_apodizations,
        probe_geometry,
        initial_times,
        sampling_frequency,
        sound_speed,
        n_tx,
        n_el,
        focus_distances,
        polar_angles,
        t_peak,
        tx_waveform_indices,
        transmit_origins,
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
    )

    n_pix = ops.shape(flatgrid)[0]
    mask = ops.cond(
        f_number == 0,
        lambda: ops.ones((n_pix, n_el, 1)),
        lambda: fnumber_mask(flatgrid, probe_geometry, f_number, fnum_window_fn=fnum_window_fn),
    )

    def _apply_delays(data_tx, txdel):
        """Applies the delays to TOF correct a single transmit.

        Args:
            data_tx (ops.Tensor): The RF/IQ data for a single transmit of shape
                `(n_ax, n_el, n_ch)`.
            txdel (ops.Tensor): The transmit delays for a single transmit in samples
                (not in seconds) of shape `(n_pix, 1)`.

        Returns:
            ops.Tensor: The time-of-flight corrected data of shape
            `(n_pix, n_el, n_ch)`.
        """
        # data_tx is of shape (num_elements, num_samples, 1 or 2)

        # Take receive delays and add the transmit delays for this transmit
        # The txdel tensor has one fewer dimensions because the transmit
        # delays are the same for all dimensions
        # delays is of shape (n_pix, n_el)
        delays = rxdel + txdel

        # Compute the time-of-flight corrected samples for each element
        # from each pixel of shape (n_pix, n_el, n_ch)

        tof_tx = apply_delays(data_tx, delays, clip_min=0, clip_max=n_ax - 1)

        # Apply the mask
        tof_tx = tof_tx * mask

        # Apply phase rotation if using IQ data
        # This is needed because interpolating the IQ data without phase rotation
        # is not equivalent to interpolating the RF data and then IQ demodulating
        # See the docstring from complex_rotate for more details
        apply_phase_rotation = data_tx.shape[-1] == 2
        if apply_phase_rotation:
            total_delay_seconds = delays[:, :] / sampling_frequency
            theta = 2 * np.pi * demodulation_frequency * total_delay_seconds
            tof_tx = complex_rotate(tof_tx, theta)
        return tof_tx

    # Reshape to (n_tx, n_pix, 1)
    txdel = ops.moveaxis(txdel, 1, 0)
    txdel = txdel[..., None]

    return vmap(_apply_delays)(data, txdel)


def calculate_delays(
    grid,
    t0_delays,
    tx_apodizations,
    probe_geometry,
    initial_times,
    sampling_frequency,
    sound_speed,
    n_tx,
    n_el,
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
    """Calculates the delays in samples to every pixel in the grid.

    The delay consists of two components: The transmit delay and the
    receive delay.

    The transmit delay is the delay between transmission and the
    wavefront reaching the pixel.

    The receive delay is the delay between the
    wavefront reaching a pixel and the reflections returning to a specific
    element.

    Args:
        grid (Tensor): The pixel coordinates to beamform to of shape `(n_pix, 3)`.
        t0_delays (Tensor): The transmit delays in seconds of shape
            `(n_tx, n_el)`, shifted such that the smallest delay is 0. Defaults to None.
        tx_apodizations (Tensor): The transmit apodizations of shape
            `(n_tx, n_el)`.
        probe_geometry (Tensor): The positions of the transducer elements of shape
            `(n_el, 3)`.
        initial_times (Tensor): The probe transmit time offsets of shape
            `(n_tx,)`.
        sampling_frequency (float): The sampling frequency of the probe in Hz.
        sound_speed (float): The assumed speed of sound in m/s.
        focus_distances (Tensor): The focus distances of shape `(n_tx,)`.
            If the focus distance is set to infinity, the beamformer will
            assume plane wave transmission.
        polar_angles (Tensor): The polar angles of the plane waves in radians
            of shape `(n_tx,)`.
        t_peak (Tensor): Time of the peak of the pulse in seconds of shape
            `(n_waveforms,)`.
        tx_waveform_indices (Tensor): The indices of the waveform used for each
            transmit of shape `(n_tx,)`.
        transmit_origins (Tensor): Transmit origins of shape (n_tx, 3).
        apply_lens_correction (bool, optional): Whether to apply lens correction to
            time-of-flights. This makes it slower, but more accurate in the near-field.
            Defaults to False.
        lens_thickness (float, optional): Thickness of the lens in meters. Used for
            lens correction.
        lens_sound_speed (float, optional): Speed of sound in the lens in m/s. Used
            for lens correction.
        n_iter (int, optional): Number of iterations for the Newton-Raphson method
            used in lens correction. Defaults to 2.


    Returns:
        transmit_delays (Tensor): The tensor of transmit delays to every pixel
            in samples (not in seconds), of shape `(n_pix, n_tx)`.
        receive_delays (Tensor): The tensor of receive delays from every pixel
            back to the transducer element in samples (not in seconds), of shape
            `(n_pix, n_el)`.
    """

    # Validate input shapes
    for arr in [t0_delays, grid, tx_apodizations, probe_geometry]:
        assert arr.ndim == 2
    assert probe_geometry.shape[0] == n_el
    assert t0_delays.shape[0] == n_tx

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
    """Applies time delays for a single transmit using linear interpolation.

    Most delays in d will not be by an integer number of samples, which means
    we have no measurement for that time instant. This function solves this by
    finding the sample before and after and interpolating the data to the
    desired delays in d using linear interpolation.

    Args:
        data (ops.Tensor): The RF or IQ data of shape `(n_ax, n_el, n_ch)`. This is
            the data we are drawing samples from to for each element for each pixel.
        delays (ops.Tensor): The delays in samples of shape `(n_pix, n_el)`. Contains
            one delay value for every pixel in the image for every transducer element.
        clip_min (int, optional): The minimum delay value to use. If set to -1 no
            clipping is applied. Defaults to -1.
        clip_max (int, optional): The maximum delay value to use. If set to -1 no
            clipping is applied. Defaults to -1.

    Returns:
        ops.Tensor: The samples received by each transducer element corresponding to the
            reflections of each pixel in the image of shape `(n_pix, n_el, n_ch)`.
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
    """Performs a simple phase rotation of I and Q component.

    Args:
        iq (ops.Tensor): The iq data of shape `(..., 2)`.
        theta (float): The complex angle to rotate by.

    Returns:
        Tensor: The rotated tensor of shape `(..., 2)`.

    .. dropdown:: Explanation

        The IQ data is related to the RF data as follows:

        .. math::

            x(t) &= I(t)\\cos(\\omega_c t) + Q(t)\\cos(\\omega_c t + \\pi/2)\\\\
            &= I(t)\\cos(\\omega_c t) - Q(t)\\sin(\\omega_c t)


        If we want to delay the RF data `x(t)` by `Δt` we can substitute in
        :math:`t=t+\\Delta t`. We also define :math:`I'(t) = I(t + \\Delta t)`,
        :math:`Q'(t) = Q(t + \\Delta t)`, and :math:`\\theta=\\omega_c\\Delta t`.
        This gives us:

        .. math::

            x(t + \\Delta t) &= I'(t) \\cos(\\omega_c (t + \\Delta t))
            - Q'(t) \\sin(\\omega_c (t + \\Delta t))\\\\
            &=  \\overbrace{(I'(t)\\cos(\\theta)
            - Q'(t)\\sin(\\theta) )}^{I_\\Delta(t)} \\cos(\\omega_c t)\\\\
            &- \\overbrace{(Q'(t)\\cos(\\theta)
            + I'(t)\\sin(\\theta))}^{Q_\\Delta(t)} \\sin(\\omega_c t)

        This means that to correctly interpolate the IQ data to the new components
        :math:`I_\\Delta(t)` and :math:`Q_\\Delta(t)`, it is not sufficient to just
        interpolate the I- and Q-channels independently. We also need to rotate the
        I- and Q-channels by the angle :math:`\\theta`. This function performs this
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
    """Computes distance to user-defined pixels from elements.

    Expects all inputs to be arrays specified in SI units.

    Args:
        grid (ops.Tensor): Pixel positions in x,y,z of shape `(n_pix, 3)`.
        probe_geometry (ops.Tensor): Element positions in x,y,z of shape `(n_el, 3)`.

    Returns:
        dist (ops.Tensor): Distance from each pixel to each element of shape
            `(n_pix, n_el)`.
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
    """
    Computes the transmit delay from transmission to each pixel in the grid.

    Uses the first-arrival time for pixels before the focus (or virtual source)
    and the last-arrival time for pixels beyond the focus.

    The receive delays can be precomputed since they do not depend on the
    transmit parameters.

    Args:
        grid (ops.Tensor): Flattened tensor of pixel positions in x,y,z of shape `(n_pix, 3)`
        t0_delays (Tensor): The transmit delays in seconds of shape (n_el,).
        tx_apodization (Tensor): The transmit apodization of shape (n_el,).
        rx_delays (Tensor): The travel times in seconds from elements to pixels
            of shape (n_pix, n_el).
        focus_distance (float): The focus distance in meters.
        polar_angle (float): The polar angle in radians.
        initial_time (float): The initial time for this transmit in seconds.
        azimuth_angle (float, optional): The azimuth angle in radians. Defaults to 0.0.
        transmit_origin (ops.Tensor, optional): The origin of the transmit beam of shape (3,).
            If None, defaults to (0, 0, 0). Defaults to None.

    Returns:
        Tensor: The transmit delays of shape `(n_pix,)`.
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
    """Apodization mask for the receive beamformer.

    Computes a mask to disregard pixels outside of the vision cone of a
    transducer element. Transducer elements can only accurately measure
    signals within some range of incidence angles. Waves coming in from the
    side do not register correctly leading to a worse image.

    Args:
        flatgrid (ops.Tensor): The flattened image grid `(n_pix, 3)`.
        probe_geometry (ops.Tensor): The transducer element positions of shape
            `(n_el, 3)`.
        f_number (int): The receive f-number. Set to zero to not use masking and
            return 1. (The f-number is the  ratio between distance from the transducer
            and the size of the aperture below which transducer elements contribute to
            the signal for a pixel.).
        fnum_window_fn (callable): F-number function to define the transition from
            straight in front of the element (fn(0.0)) to the largest angle within the f-number cone
            (fn(1.0)). The function should be zero for fn(x>1.0).


    Returns:
        Tensor: Mask of shape `(n_pix, n_el, 1)`
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
