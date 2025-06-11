"""
Apodization functions
"""
import numpy as np
from scipy import signal


def square_wave_apodization(n_el: int, block_size: float):
    """Returns a square wave apodization.

    Used for incoherent beamforming.

    Args:
        n_el (int): Total number of elements in array.
        block_size (float): In terms of number of elements that will be
            high/low. Can be a float.

    Returns:
        apod (ndarray): array of size n_el.

    Example:

        +1 +1 +1 +1             +1 +1 +1 +1
                    -1 -1 -1 -1

        <----------> block_size = 4, n_el = 12
    """

    lambd = 2 * block_size / n_el
    freq = 1 / lambd
    t = np.linspace(0, 1, n_el)
    apod = signal.square(2 * np.pi * freq * t, duty=0.5)

    # weirdly last element is flipped
    apod[-1] = -1 * apod[-1]
    return apod
