"""Main sos-grid based beamforming functions for ultrasound imaging."""

import keras
import numpy as np
from keras import ops

from zea.func.tensor import vmap

def tof_correction_grid(
    data,
    sos_grid,
    x_sos_grid,
    z_sos_grid,
    flatgrid,
    t0_delays,
    probe_geometry,
    initial_times,
    sampling_frequency,
    demodulation_frequency,
    f_number,
    t_peak,
    tx_waveform_indices,
):
    """Time-of-flight correction for a flat grid.

    Args:
        data (ops.Tensor): Input RF/IQ data of shape `(n_tx, n_ax, n_el, n_ch)`.
        sos_grid (ops.Tensor): sound speed grid
        x_sos_grid (ops.Tensor): x coordinates of the sound speed grid
        z_sos_grid (ops.Tensor): z coordinates of the sound speed grid
        flatgrid (ops.Tensor): Pixel locations x, y, z of shape `(n_pix, 3)`
        t0_delays (ops.Tensor): Times at which the elements fire shifted such
            that the first element fires at t=0 of shape `(n_tx, n_el)`
        probe_geometry (ops.Tensor): Element positions x, y, z of shape (n_el, 3)
        initial_times (Tensor): The probe transmit time offsets of shape `(n_tx,)`.
        sampling_frequency (float): Sampling frequency.
        demodulation_frequency (float): Demodulation frequency.
        f_number (float): Focus number (ratio of focal depth to aperture size).
        t_peak (ops.Tensor): Time of the peak of the pulse in seconds.
            Shape `(n_waveforms,)`.
        tx_waveform_indices (ops.Tensor): The indices of the waveform used for each
            transmit of shape `(n_tx,)`.
    Returns:
        (ops.Tensor): time-of-flight corrected data
        with shape: `(n_tx, n_pix, n_el, n_ch)`.
    """

    assert len(data.shape) == 4, (
        "The input data should have 4 dimensions, "
        f"namely n_tx, n_ax, n_el, n_ch, got {len(data.shape)} dimensions: {data.shape}"
    )
    assert data.shape[0]==data.shape[2], (
        "The dataset should be multistatic"
    )

    def calculate_delays_grid(
        data,
        grid,
        sos_grid,
        x_sos_grid,
        z_sos_grid,
        t0_delays,
        probe_geometry,
        initial_times,
        sampling_frequency,
        f_number,
        t_peak=t_peak,
        tx_waveform_indices=tx_waveform_indices,
    ):
        """
        Uses supplied sos_grid to interpolate the delays in samples to every pixel in the grid
        , only valid for multistatic data.

            
        The delay consists of two components: The transmit delay and the
        receive delay.

        The transmit delay is the delay between transmission and the
        wavefront reaching the pixel.

        The receive delay is the delay between the
        wavefront reaching a pixel and the reflections returning to a specific
        element.

        Args:
            data (ops.Tensor): Input RF/IQ data of shape `(n_tx, n_ax, n_el, n_ch)`.
            grid (Tensor): The pixel coordinates to beamform to of shape `(n_pix, 3)`.
            sos_grid (Tensor): The matrix of sound speed values in m/s (Nx x Nz)
            x_sos_grid (Tensor): Vector of x-grid points in sound speed definition (sos_grid.shape[0],)
            z_sos_grid (Tensor): Vector of z-grid points in sound speed definition
            t0_delays (Tensor): The transmit delays in seconds of shape
                `(n_tx, n_el)`, shifted such that the smallest delay is 0. Defaults to None.
            probe_geometry (Tensor): The positions of the transducer elements of shape
                `(n_el, 3)`.
            initial_times (Tensor): The probe transmit time offsets of shape
                `(n_tx,)`.
            sampling_frequency (float): The sampling frequency of the probe in Hz.
            f_number (float): Focus number (ratio of focal depth to aperture size).
            t_peak (ops.Tensor): Time of the peak of the pulse in seconds.
                Shape `(n_waveforms,)`.
            tx_waveform_indices (ops.Tensor): The indices of the waveform used for each
                transmit of shape `(n_tx,)`.
        """

        npts = 100 #Number of points to interpolate on a ray
        pts = ops.linspace(1, 0, npts, endpoint=False)[::-1]
        s = 1 / sos_grid # slowness map
        x = grid[:, 0]
        z = grid[:, 2]

        xe = probe_geometry[:, 0]
        ze = probe_geometry[:, 2]

        def interpolate(p,xe,ze):
            xp = p * (x - xe) + xe  # True spatial location of path in x at t
            zp = p * (z - ze) + ze  # True spatial location of path in z at t

            # Convert spatial locations into indices in xc and zc coordinates (in slowness map)
            dxc, dzc = x_sos_grid[1] - x_sos_grid[0], z_sos_grid[1] - z_sos_grid[0]  # Assume a grid! Grid spacings
            # Get indices of xt, zt in slowness map. Clamp at borders
            xit = ops.clip((xp -  x_sos_grid[0]) / dxc, 0, s.shape[0] - 1)
            zit = ops.clip((zp - z_sos_grid[0]) / dzc, 0, s.shape[1] - 1)
            xi0 = ops.floor(xit)
            zi0 = ops.floor(zit)
            xi1 = xi0 + 1
            zi1 = zi0 + 1
            # Interpolate slowness at (xt, zt)
            s00 = s[xi0.astype("int32"), zi0.astype("int32")]
            s10 = s[xi1.astype("int32"), zi0.astype("int32")]
            s01 = s[xi0.astype("int32"), zi1.astype("int32")]
            s11 = s[xi1.astype("int32"), zi1.astype("int32")]
            w00 = (xi1 - xit) * (zi1 - zit)
            w10 = (xit - xi0) * (zi1 - zit)
            w01 = (xi1 - xit) * (zit - zi0)
            w11 = (xit - xi0) * (zit - zi0)
            return s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11

        # Compute the time-of-flight
        dx = ops.abs(xe[:,None] - x[None,:])
        dz = ops.abs(ze[:,None] - z[None,:])
        dtrue = ops.sqrt(dx**2 + dz**2)
        #slowness = vmap(interpolate)(pts)  
        slowness = vmap(
            lambda xe, ze: vmap(
                lambda p: interpolate(p, xe, ze)
            )(pts)
        )(xe,ze)

        mask = ~ops.isnan(slowness)
        masked_sum = ops.sum(ops.where(mask, slowness, 0.0), axis=1)
        count = ops.sum(mask, axis=1)
        mean_slowness = masked_sum / (count + 1e-9)    # avoid division by zero

        tof = mean_slowness* dtrue
        #tof = ops.nanmean(slowness, axis=0) * dtrue
        rx_delays = tof*sampling_frequency
        tx_delays = (tof 
                    + t0_delays[0,0]
                    - initial_times[:,None]
                    + ops.take(t_peak, tx_waveform_indices)[:,None]
        )*sampling_frequency
        return tx_delays, rx_delays
    
    txdel, rxdel = calculate_delays_grid(
        data,
        flatgrid,
        sos_grid,
        x_sos_grid,
        z_sos_grid,
        t0_delays,
        probe_geometry,
        initial_times,
        sampling_frequency,
        f_number,

    )
    rxdel = ops.moveaxis(rxdel, 1, 0)
    n_tx, n_ax, n_el, _ = ops.shape(data)
    n_pix = ops.shape(flatgrid)[0]
    mask = ops.stop_gradient(ops.cond(
        f_number == 0,
        lambda: ops.ones((n_pix, n_el, 1)),
        lambda: apod_mask_normal(flatgrid, probe_geometry, f_number),
    ))


    def apply_delays(data, txdel, mask_tx):
        """
        Fused time-of-flight correction with optional phase rotation using linear interpolation.

        Args:
            data_tx (ops.Tensor): The RF/IQ data for a single transmit of shape
                `(n_ax, n_el, n_ch)`.
            txdel (ops.Tensor): The transmit delays for a single transmit in samples
                (not in seconds) of shape `(n_pix, 1)`.
            mask_tx (ops.Tensor): mask for pixel (f_number based)
        Returns:
            ops.Tensor: The time-of-flight corrected data of shape
            `(n_pix, n_el, n_ch)`.
        """
        # total delays in samples
        delays = rxdel + txdel  # (n_pix, n_el)

        d0 = ops.floor(delays[...,None]).astype("int32")
        d0 = ops.clip(d0, 0, data.shape[0] - 1)
        d1 = ops.clip(d0 + 1, 0, data.shape[0] - 1)

        w1 =delays[...,None] - d0
        w0 = 1.0 - w1

        # gather samples in one go
        data0 = ops.take_along_axis(data, d0, 0)
        data1 = ops.take_along_axis(data, d1, 0)
        tof_tx = w0 * data0 + w1* data1  # (n_pix, n_el, n_ch)
        # Apply the mask in transmit and receive
        tof_tx = tof_tx * mask * mask_tx[:, :,None] 
        #phase rotation
        tshift = delays / sampling_frequency
        theta = 2 * np.pi * demodulation_frequency * tshift
        i = tof_tx[..., 0]
        q = tof_tx[..., 1]
        ir = i * ops.cos(theta) - q * ops.sin(theta)
        qr = q * ops.cos(theta) + i * ops.sin(theta)
        tof_tx = ops.concatenate([ir[..., None], qr[..., None]], -1)
        return tof_tx

    # Reshape to (n_tx, n_pix, 1)
    # txdel = ops.moveaxis(txdel, 1, 0)
    txdel = txdel[..., None]
    #txdel = ops.moveaxis(txdel, 1, 0)
    mask_tx = ops.moveaxis(mask, 1, 0)
    _apply_delays_ckpt = keras.remat(apply_delays)

    return ops.vectorize(
        _apply_delays_ckpt,
        signature="(n_samples,n_el,n_ch),(n_pix,1),(n_pix,1)->(n_pix,n_el,n_ch)",
    )(data, txdel,mask_tx)
   # return vmap(_apply_delays)(data, txdel,mask_tx)



def apod_mask(grid, probe_geometry, f_number):
    """Vectorized apodization mask (JAX version)."""
    x_pixel = grid[:, 0]  # (n_pix,)
    z_pixel = grid[:, 2]  # (n_pix,)
    x_element = probe_geometry[:, 0]  # (n_el,)

    aperture = z_pixel / f_number  # (n_pix,)

    # Broadcasting: (n_pix, n_el)
    distance = ops.abs(x_pixel[:, None] - x_element[None, :])
    mask = (distance <= aperture[:, None] * 0.5).astype("float32")
    return mask[..., None]  # (n_pix, n_el, 1)


def compute_normals_xz(probe_geometry):
    """Approximate normals for a 2D x-z probe geometry."""
    tangent = ops.zeros_like(probe_geometry)
    tangent = tangent.at[1:-1].set(probe_geometry[2:] - probe_geometry[:-2])
    tangent = tangent.at[0].set(probe_geometry[1] - probe_geometry[0])
    tangent = tangent.at[-1].set(probe_geometry[-1] - probe_geometry[-2])

    tangent_xz = tangent.at[:, 1].set(0.0)  # ignore y

    normals = ops.stack([-tangent_xz[:, 2], ops.zeros(tangent.shape[0]), tangent_xz[:, 0]], axis=1)
    normals /= ops.linalg.norm(normals, axis=1, keepdims=True)
    return normals

def apod_mask_normal(grid, probe_geometry, f_number):
    """Vectorized apodization mask (JAX version)."""
    normals = compute_normals_xz(probe_geometry)  # (n_el,3)
    # Difference vector
    vec = grid[:, None, :] - probe_geometry[None, :, :]  # (n_pix,n_el,3)
    # Distance along normal
    d_parallel = ops.sum(vec * normals[None, :, :], axis=-1)  # (n_pix,n_el)
    # Aperture based on f-number
    aperture = ops.abs(d_parallel) / f_number  # (n_pix,n_el)
    # Transverse distance
    transverse_vec = vec - d_parallel[..., None] * normals[None, :, :]
    d_perp = ops.linalg.norm(transverse_vec, axis=-1)
    
    # Mask
    mask = (d_perp <= 0.5 * aperture).astype("float32")
    return mask[..., None]  # (n_pix,n_el,1)