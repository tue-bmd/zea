"""Doppler functions for processing I/Q ultrasound data."""

import numpy as np
from keras import ops

from zea.func import tensor


def color_doppler(
    data,
    center_frequency,
    pulse_repetition_frequency,
    sound_speed,
    hamming_size=None,
    lag=1,
):
    """Compute Color Doppler from packet of I/Q Data.

    Args:
        data (ndarray): I/Q complex data of shape (n_frames, grid_size_z, grid_size_x).
            n_frames corresponds to the ensemble length used to compute
            the Doppler signal.
        center_frequency (float): Transmit center frequency in Hz.
        pulse_repetition_frequency (float): Pulse repetition frequency in Hz.
        sound_speed (float): Speed of sound in the medium in m/s.
        hamming_size (int or tuple, optional): Size of the Hamming window to apply
            for spatial averaging. If None, no window is applied.
            If an integer, it is applied to both dimensions. If a tuple, it should
            contain two integers for the row and column dimensions.
        lag (int, optional): Lag for the auto-correlation computation.
            Defaults to 1, meaning Doppler is computed from the current frame
            and the next frame.

    Returns:
        doppler_velocities (ndarray): Doppler velocity map of shape (grid_size_z, grid_size_x) in
            meters/second.

    """
    assert data.ndim == 3, "Data must be a 3-D array"
    if not (isinstance(lag, int) and lag >= 1):
        raise ValueError("lag must be an integer >= 1")
    n_frames = data.shape[0]
    assert n_frames > lag, "Data must have more frames than the lag"

    if hamming_size is None:
        hamming_size = np.array([1, 1], dtype=int)
    elif np.isscalar(hamming_size):
        hamming_size = np.array([int(hamming_size), int(hamming_size)], dtype=int)
    else:
        assert len(hamming_size) == 2, "hamming_size must be an integer or a tuple of two integers"
        hamming_size = np.array(hamming_size, dtype=int)
    if not np.all(hamming_size > 0):
        raise ValueError("hamming_size must contain integers > 0")

    # Auto-correlation method
    iq1 = data[: n_frames - lag]
    iq2 = data[lag:]
    autocorr = ops.sum(iq1 * ops.conj(iq2), axis=0)  # Ensemble auto-correlation

    # Spatial weighted average
    if hamming_size[0] != 1 and hamming_size[1] != 1:
        h_row = np.hamming(hamming_size[0])
        h_col = np.hamming(hamming_size[1])
        autocorr = tensor.apply_along_axis(
            lambda x: tensor.correlate(x, h_row, mode="same"), 0, autocorr
        )
        autocorr = tensor.apply_along_axis(
            lambda x: tensor.correlate(x, h_col, mode="same"), 1, autocorr
        )

    # Doppler velocity
    nyquist_velocity = sound_speed * pulse_repetition_frequency / (4 * center_frequency * lag)
    phase = ops.arctan2(ops.imag(autocorr), ops.real(autocorr))
    doppler_velocities = -nyquist_velocity * phase / np.pi
    return doppler_velocities
