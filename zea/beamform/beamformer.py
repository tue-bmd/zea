"""Main beamforming functions for ultrasound imaging."""

import numpy as np
from keras import ops

from zea.beamform.lens_correction import calculate_lens_corrected_delays
from zea.tensor_ops import safe_vectorize


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
    fnum,
    angles,
    vfocus,
    apply_phase_rotation=False,
    apply_lens_correction=False,
    lens_thickness=1e-3,
    lens_sound_speed=1000,
):
    """Time-of-flight correction for a flat grid.

    Args:
        data (ops.Tensor): Input RF/IQ data of shape `(n_tx, n_ax, n_el, n_ch)`.
        flatgrid (ops.Tensor): Pixel locations x, y, z of shape `(n_pix, 3)`
        t0_delays (ops.Tensor): Times at which the elements fire shifted such
            that the first element fires at t=0 of shape `(n_tx, n_el)`
        tx_apodizations (ops.Tensor): Transmit apodizations of shape
            `(n_tx, n_el)`
        sound_speed (float): Speed-of-sound.
        probe_geometry (ops.Tensor): Element positions x, y, z of shape
        (num_samples, 3)
        initial_times (ops.Tensor): Time-ofsampling_frequencyet per transmission of shape
            `(n_tx,)`.
        sampling_frequency (float): Sampling frequency.
        demodulation_frequency (float): Demodulation frequency.
        fnum (int, optional): Focus number. Defaults to 1.
        angles (ops.Tensor): The angles of the plane waves in radians of shape
            `(n_tx,)`
        vfocus (ops.Tensor): The focus distance of shape `(n_tx,)`
        apply_phase_rotation (bool, optional): Whether to apply phase rotation to
            time-of-flights. Defaults to False.
        apply_lens_correction (bool, optional): Whether to apply lens correction to
            time-of-flights. This makes it slower, but more accurate in the near-field.
            Defaults to False.
        lens_thickness (float, optional): Thickness of the lens in meters. Used for
            lens correction. Defaults to 1e-3.
        lens_sound_speed (float, optional): Speed of sound in the lens in m/s. Used
            for lens correction Defaults to 1000.

    Returns:
        (ops.Tensor): time-of-flight corrected data
        with shape: `(n_tx, n_pix, n_el, num_rf_iq_channels)`.
    """

    assert len(data.shape) == 4, (
        "The input data should have 4 dimensions, "
        f"namely num_transmits, num_elements, num_samples, "
        f"num_rf_iq_channels, got {len(data.shape)} dimensions: ."
        f"{data.shape}"
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

    delay_fn = calculate_lens_corrected_delays if apply_lens_correction else calculate_delays
    txdel, rxdel = delay_fn(
        flatgrid,
        t0_delays,
        tx_apodizations,
        probe_geometry,
        initial_times,
        sampling_frequency,
        sound_speed,
        n_tx,
        n_el,
        vfocus,
        angles,
        lens_thickness=lens_thickness,
        lens_sound_speed=lens_sound_speed,
    )

    n_pix = ops.shape(flatgrid)[0]
    mask = ops.cond(
        fnum == 0,
        lambda: ops.ones((n_pix, n_el, 1)),
        lambda: apod_mask(flatgrid, probe_geometry, fnum),
    )

    def _apply_delays(data_tx, txdel):
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

        # Phase correction
        if apply_phase_rotation:
            tshift = delays[:, :] / sampling_frequency
            tdemod = flatgrid[:, None, 2] * 2 / sound_speed
            theta = 2 * np.pi * demodulation_frequency * (tshift - tdemod)
            tof_tx = _complex_rotate(tof_tx, theta)
        return tof_tx

    # Reshape to (n_tx, n_pix, 1)
    txdel = ops.moveaxis(txdel, 1, 0)
    txdel = txdel[..., None]

    return safe_vectorize(
        _apply_delays,
        signature="(n_samples,n_el,n_ch),(n_pix,1)->(n_pix,n_el,n_ch)",
    )(data, txdel)


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
    **kwargs,
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

    Returns:
        transmit_delays (Tensor): The tensor of transmit delays to every pixel,
            shape `(n_pix, n_tx)`.
        receive_delays (Tensor): The tensor of receive delays from every pixel
            back to the transducer element, shape `(n_pix, n_el)`.
    """

    def _tx_distances(polar_angles, t0_delays, tx_apodizations, focus_distances):
        return distance_Tx_generic(
            grid,
            t0_delays,
            tx_apodizations,
            probe_geometry,
            focus_distances,
            polar_angles,
            sound_speed,
        )

    tx_distances = safe_vectorize(
        _tx_distances,
        signature="(),(n_el),(n_el),()->(n_pix)",
    )(polar_angles, t0_delays, tx_apodizations, focus_distances)
    tx_distances = ops.transpose(tx_distances, (1, 0))
    # tx_distances shape is now (n_pix, n_tx)

    # Compute receive distances
    def _rx_distances(probe_geometry):
        return distance_Rx(grid, probe_geometry)

    rx_distances = safe_vectorize(_rx_distances, signature="(3)->(n_pix)")(probe_geometry)
    rx_distances = ops.transpose(rx_distances, (1, 0))
    # rx_distances shape is now (n_pix, n_el)

    # Compute the delays [in samples] from the distances
    # The units here are ([m]/[m/s]-[s])*[1/s] resulting in a unitless quantity
    # TODO: Add pulse width to transmit delays
    tx_delays = (tx_distances / sound_speed - initial_times[None]) * sampling_frequency
    rx_delays = (rx_distances / sound_speed) * sampling_frequency

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
            reflections of each pixel in the image of shape `(n_el, n_pix, n_ch)`.
    """

    # Add a dummy channel dimension to the delays tensor to ensure it has the
    # same number of dimensions as the data. The new shape is (1, n_el, n_pix)
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
    # reflection from each pixel. These are of shape `(n_el, n_pix, n_ch)`.
    data0 = ops.take_along_axis(data, d0, 0)
    data1 = ops.take_along_axis(data, d1, 0)

    # Compute interpolated pixel value
    d0 = ops.cast(d0, delays.dtype)  # Cast to float
    d1 = ops.cast(d1, delays.dtype)  # Cast to float
    data0 = ops.cast(data0, delays.dtype)  # Cast to float
    data1 = ops.cast(data1, delays.dtype)  # Cast to float
    reflection_samples = (d1 - delays) * data0 + (delays - d0) * data1

    return reflection_samples


def _complex_rotate(iq, theta):
    """Performs a simple phase rotation of I and Q component.

    Args:
        iq (ops.Tensor): The iq data of shape `(..., 2)`.
        theta (float): The complex angle to rotate by.

    Returns:
        Tensor: The rotated tensor of shape `(..., 2)`.
    """
    # assert iq.shape[-1] == 2, (
    #     "The last dimension of the input tensor should be 2, "
    #     f"got {iq.shape[-1]} dimensions and shape {iq.shape}."
    # )
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

    Expects all inputs to be numpy arrays specified in SI units.

    Args:
        grid (ops.Tensor): Pixel positions in x,y,z of shape `(n_pix, 3)`.
        probe_geometry (ops.Tensor): Element positions in x,y,z of shape `(n_el, 3)`.

    Returns:
        dist (ops.Tensor): Distance from each pixel to each element of shape
            `(n_pix, n_el)`.
    """
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = ops.linalg.norm(grid - probe_geometry[None, ...], axis=-1)
    return dist


def distance_Tx_generic(
    grid,
    t0_delays,
    tx_apodization,
    probe_geometry,
    focus_distance,
    polar_angle,
    sound_speed=1540,
):
    """Generic transmit distance calculation.

    Computes distance to user-defined pixels for generic transmits based on
    the t0_delays.

    Args:
        grid (ops.Tensor): Flattened tensor of pixel positions in x,y,z of shape
            `(n_pix, 3)`
        t0_delays (ops.Tensor): The transmit delays in seconds of shape `(n_el,)`,
            shifted such that the smallest delay is 0. Defaults to None.
        tx_apodization (ops.Tensor): The transmit apodizations of shape
            `(n_el,)`.
        probe_geometry (ops.Tensor): The positions of the transducer elements of shape
            `(n_el, 3)`.
        sound_speed (float): The speed of sound in m/s. Defaults to 1540.

    Returns:
        Tensor: Distance from each pixel to each element in meters
        of shape `(n_pix,)`
    """
    # Get the individual x, y, and z components of the pixel coordinates
    x = grid[:, 0]
    y = grid[:, 1]
    z = grid[:, 2]

    # Reshape x, y, and z to shape (n_pix, 1)
    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    # Get the individual x, y, and z coordinates of the elements and add a
    # dummy dimension at the beginning to shape (1, n_el).
    ele_x = probe_geometry[None, :, 0]
    ele_y = probe_geometry[None, :, 1]
    ele_z = probe_geometry[None, :, 2]

    # Compute the differences dx, dy, and dz of shape (n_pix, n_el)
    dx = x - ele_x
    dy = y - ele_y
    dz = z - ele_z

    # Define an infinite offset for elements that do not fire to not consider them in
    # the transmit distance calculation.
    offset = ops.where(tx_apodization == 0, np.inf, 0.0)

    # Compute the distance between the elements and the pixels of shape
    # (n_pix, n_el)
    dist = t0_delays[None] * sound_speed + ops.sqrt(dx**2 + dy**2 + dz**2)

    # Compute the z-coordinate of the focal point
    focal_z = ops.cos(polar_angle) * focus_distance

    # Compute the effective distance of the pixels to the wavefront by computing the
    # largest distance over all the elements when the pixel is behind the virtual
    # source and the smallest distance otherwise.
    dist = ops.where(
        ops.cast(ops.sign(focus_distance), "float32") * (grid[:, 2] - focal_z) <= 0.0,
        ops.min(dist + offset[None], 1),
        ops.max(dist - offset[None], 1),
    )

    return dist


def apod_mask(grid, probe_geometry, f_number):
    """Apodization mask for the receive beamformer.

    Computes a binary mask to disregard pixels outside of the vision cone of a
    transducer element. Transducer elements can only accurately measure
    signals within some range of incidence angles. Waves coming in from the
    side do not register correctly leading to a worse image.

    Args:
        grid (ops.Tensor): The flattened image grid `(n_pix, 3)`.
        probe_geometry (ops.Tensor): The transducer element positions of shape
            `(n_el, 3)`.
        f_number (int): The receive f-number. Set to zero to not use masking and
            return 1. (The f-number is the  ratio between distance from the transducer
            and the size of the aperture below which transducer elements contribute to
            the signal for a pixel.).

    Returns:
        Tensor: Mask of shape `(n_pix, n_el, 1)`
    """
    n_pix = ops.shape(grid)[0]
    n_el = ops.shape(probe_geometry)[0]

    # Get the depth of every pixel
    z_pixel = grid[:, 2]
    # Get the lateral location of each pixel
    x_pixel = grid[:, 0]
    # Get the lateral location of each element
    x_element = ops.cast(probe_geometry[:, 0], dtype="float32")

    # Compute the aperture size for every pixel
    # The f-number is by definition f=z/aperture
    aperture = z_pixel / f_number

    # Use matrix multiplication to expand aperture tensor, x_pixel tensor, and
    # x_element tensor to shape (n_pix, n_el)
    ones_aperture = ops.ones(
        (1, n_el), dtype=ops.dtype(aperture)
    )  # getting error here? pip install -U keras ;)
    ones_x_pixel = ops.ones((1, n_el), dtype=ops.dtype(x_pixel))
    ones_x_element = ops.ones((n_pix, 1), dtype=ops.dtype(x_element))

    aperture = ops.matmul(aperture[..., None], ones_aperture)
    expanded_x_pixel = ops.matmul(x_pixel[..., None], ones_x_pixel)
    expanded_x_element = ops.matmul(ones_x_element, x_element[None])

    # Compute the lateral distance between elements and pixels
    # Of shape (n_pix, n_el)
    distance = ops.abs(expanded_x_pixel - expanded_x_element)

    # Compute binary mask for which the lateral pixel distance is less than
    # half
    # the aperture i.e. where the pixel lies within the vision cone of the
    # element
    mask = distance <= aperture / 2
    mask = ops.cast(mask, "float32")

    # Add dummy dimension for RF/IQ channel channel
    mask = mask[..., None]

    return mask
