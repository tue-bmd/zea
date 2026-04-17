"""Delay computation for ultrasound beamforming."""

import matplotlib.pyplot as plt
import numpy as np


def compute_t0_delays_planewave(probe_geometry, polar_angles, azimuth_angles=0, sound_speed=1540):
    """Computes the transmit delays for a planewave.

    .. note::

        The transmit delays are shifted such that the first element fires at ``t=0``.

    Args:
        probe_geometry (np.ndarray): The positions of the elements in the array of
            shape (n_el, 3).
        polar_angles (np.ndarray): The polar angles of the planewave in radians of shape (n_tx,).
        azimuth_angles (np.ndarray, optional): The azimuth angles of the planewave
            in radians of shape (n_tx,). Defaults to 0.
        sound_speed (float, optional): The speed of sound. Defaults to 1540.

    Returns:
        np.ndarray: The transmit delays for each element of shape (n_tx, n_el).
    """
    assert probe_geometry is not None, "Probe geometry must be provided to compute t0_delays."

    # Convert single angles to arrays for broadcasting
    polar_angles = np.atleast_1d(polar_angles)
    azimuth_angles = np.atleast_1d(azimuth_angles)

    # Compute v for all angles
    v = np.stack(
        [
            np.sin(polar_angles) * np.cos(azimuth_angles),
            np.sin(polar_angles) * np.sin(azimuth_angles),
            np.cos(polar_angles),
        ],
        axis=-1,
    )

    # Compute the projection of the element positions onto the wave vectors
    projection = np.sum(probe_geometry[:, None, :] * v, axis=-1).T

    # Convert from distance to time to compute the transmit delays.
    t0_delays_not_zero_aligned = projection / sound_speed

    # The smallest (possibly negative) time corresponds to the moment when
    # the first element fires.
    t_first_fire = np.min(t0_delays_not_zero_aligned, axis=1)

    # The transmit delays are the projection minus the offset. This ensures
    # that the first element fires at t=0.
    t0_delays = t0_delays_not_zero_aligned - t_first_fire[:, None]
    return t0_delays


def compute_t0_delays_focused(
    transmit_origins,
    focus_distances,
    probe_geometry,
    polar_angles,
    azimuth_angles=None,
    sound_speed=1540,
):
    """Computes the transmit delays for a focused transmit.

    .. note::

        The transmit delays are shifted such that the first element fires at ``t=0``.

    Args:
        transmit_origins (np.ndarray): The origin of the focused transmit of shape (n_tx, 3,).
        focus_distances (np.ndarray): The distance to the focus for each transmit of shape (n_tx,).
        probe_geometry (np.ndarray): The positions of the elements in the array of
            shape (element, 3).
        polar_angles (np.ndarray): The polar angles in radians of shape (n_tx,).
        azimuth_angles (np.ndarray, optional): The azimuth angles in
            radians of shape (n_tx,).
        sound_speed (float, optional): The speed of sound. Defaults to 1540.

    Returns:
        np.ndarray: The transmit delays for each element of shape (n_tx, element).
    """
    n_tx = len(focus_distances)
    assert polar_angles.shape == (n_tx,), (
        f"polar_angles must have length n_tx = {n_tx}. Got length {len(polar_angles)}."
    )
    assert transmit_origins.shape == (n_tx, 3), (
        f"transmit_origins must have shape (n_tx, 3). Got shape {transmit_origins.shape}."
    )
    assert probe_geometry.shape[1] == 3 and probe_geometry.ndim == 2, (
        f"probe_geometry must have shape (element, 3). Got shape {probe_geometry.shape}."
    )
    assert focus_distances.shape == (n_tx,), (
        f"focus_distances must have length n_tx = {n_tx}. Got length {len(focus_distances)}."
    )

    # Convert single angles to arrays for broadcasting
    polar_angles = np.atleast_1d(polar_angles)
    if azimuth_angles is None:
        azimuth_angles = np.zeros(len(polar_angles))
    else:
        azimuth_angles = np.atleast_1d(azimuth_angles)
    assert azimuth_angles.shape == (n_tx,), (
        f"azimuth_angles must have length n_tx = {n_tx}. Got length {len(azimuth_angles)}."
    )

    # Compute v for all angles
    v = np.stack(
        [
            np.sin(polar_angles) * np.cos(azimuth_angles),
            np.sin(polar_angles) * np.sin(azimuth_angles),
            np.cos(polar_angles),
        ],
        axis=-1,
    )

    # Add a new dimension for broadcasting
    # The shape is now (n_tx, 1, 3)
    v = np.expand_dims(v, axis=1)

    # Compute the location of the virtual source by adding the focus distance
    # to the origin along the wave vectors.
    virtual_sources = transmit_origins[:, None] + focus_distances[:, None, None] * v

    # Compute the distances between the virtual sources and each element
    dist = np.linalg.norm(virtual_sources - probe_geometry, axis=-1)

    # Adjust distances based on the direction of focus
    dist *= -np.sign(focus_distances[:, None])

    # Convert from distance to time to compute the
    # transmit delays/travel times.
    travel_times = dist / sound_speed

    # The smallest (possibly negative) time corresponds to the moment when
    # the first element fires.
    t_first_fire = np.min(travel_times, axis=1)

    # Shift the transmit delays such that the first element fires at t=0.
    t0_delays = travel_times - t_first_fire[:, None]

    return t0_delays


def plot_t0_delays(t0_delays):
    """Plot the t0_delays for each transducer element.

    Elements are on the x-axis, and the t0_delays are on the y-axis.
    We plot multiple lines for each angle/transmit in the scan object.

    Args:
        t0_delays (np.ndarray): The t0 delays for each element of shape (n_tx, n_el).

    """
    n_tx = t0_delays.shape[0]
    _, ax = plt.subplots()
    for tx in range(n_tx):
        ax.plot(t0_delays[tx], label=f"Transmit {tx}")
    ax.set_xlabel("Element number")
    ax.set_ylabel("t0 delay [s]")
    plt.show()
