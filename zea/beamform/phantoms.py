import numpy as np


def fish():
    """Returns a scatterer phantom for ultrasound simulation tests.

    Returns:
        ndarray: The scatterer positions of shape (n_scat, 3).
    """
    # The size is the height of the fish
    size = 11e-3
    z_offset = 2.0 * size

    # See https://en.wikipedia.org/wiki/Fish_curve
    def fish_curve(t, size=1):
        x = size * (np.cos(t) - np.sin(t) ** 2 / np.sqrt(2))
        y = size * np.cos(t) * np.sin(t)
        return x, y

    scat_x, scat_z = fish_curve(np.linspace(0, 2 * np.pi, 100), size=size)

    scat_x = np.concatenate(
        [
            scat_x,
            np.array([size * 0.7]),
            np.array([size * 1.1]),
            np.array([size * 1.4]),
            np.array([size * 1.2]),
        ]
    )
    scat_y = np.zeros_like(scat_x)
    scat_z = np.concatenate(
        [
            scat_z,
            np.array([-size * 0.1]),
            np.array([-size * 0.25]),
            np.array([-size * 0.6]),
            np.array([-size * 1.0]),
        ]
    )

    scat_z += z_offset
    return np.stack([scat_x, scat_y, scat_z], axis=1)
