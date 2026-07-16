"""Scatterer seeding for the scatterer-prior inversion.

Samples initial scatterer positions from a beamformed image so that bright
regions receive more scatterers, optionally mixed with a uniform floor over
the field of view. Runs on the host in NumPy; the returned positions are fed
to :class:`zea.inverse.ScattererSimulator`.
"""

import numpy as np
from keras import ops
from scipy.signal import hilbert


def seed_scatterers(
    image,
    grid,
    n_scatterers,
    prob_exponent=2.5,
    uniform_frac=0.3,
    envelope=True,
    seed=None,
):
    """Sample scatterer positions from a beamformed image.

    A fraction ``1 - uniform_frac`` of the scatterers is drawn from the image
    envelope with probability proportional to ``envelope**prob_exponent``
    (bright pixels seed more scatterers; a lower exponent is more generous to
    dim regions), jittered uniformly within each pixel cell. The remaining
    fraction is placed uniformly over the field of view, which lets the
    subsequent optimization assign energy to regions the image misses.

    Args:
        image (Tensor or ndarray): Beamformed image of shape
            ``(grid_size_z, grid_size_x)``.
        grid (Tensor or ndarray): Pixel positions of shape
            ``(grid_size_z, grid_size_x, 3)``, e.g. ``parameters.grid``.
        n_scatterers (int): Total number of scatterers to sample.
        prob_exponent (float, optional): Sharpness of the envelope-based
            sampling. Defaults to ``2.5``.
        uniform_frac (float, optional): Fraction of scatterers placed
            uniformly over the field of view. Defaults to ``0.3``.
        envelope (bool, optional): Detect the envelope of the (RF) image along
            depth before sampling. Set to ``False`` when ``image`` is already
            an envelope/B-mode image. Defaults to ``True``.
        seed (int, optional): Seed for reproducible sampling.

    Returns:
        ndarray: Scatterer positions ``(x, y, z)`` of shape
        ``(n_scatterers, 3)``, float32.
    """
    image = np.asarray(ops.convert_to_numpy(image), dtype=np.float32)
    grid = np.asarray(ops.convert_to_numpy(grid), dtype=np.float32)
    if image.shape != grid.shape[:-1]:
        raise ValueError(f"Image shape {image.shape} does not match grid shape {grid.shape[:-1]}.")

    rng = np.random.default_rng(seed)
    n_uniform = int(uniform_frac * n_scatterers)
    n_image = n_scatterers - n_uniform

    if envelope:
        image = np.abs(hilbert(image, axis=0))
    else:
        image = np.abs(image)

    # Pixel spacing for the jitter and out-of-plane coordinate. The grid is
    # regular, so neighbor differences give the spacing.
    dz = grid[1, 0, 2] - grid[0, 0, 2] if grid.shape[0] > 1 else 0.0
    dx = grid[0, 1, 0] - grid[0, 0, 0] if grid.shape[1] > 1 else 0.0

    flat_probability = (image.ravel() / (image.max() + 1e-12) + 1e-8) ** prob_exponent
    flat_probability = flat_probability / flat_probability.sum()
    indices = rng.choice(flat_probability.size, size=n_image, p=flat_probability)
    positions_image = grid.reshape(-1, 3)[indices]
    jitter = np.stack(
        [
            rng.uniform(-abs(dx) / 2, abs(dx) / 2, n_image),
            np.zeros(n_image),
            rng.uniform(-abs(dz) / 2, abs(dz) / 2, n_image),
        ],
        axis=1,
    )
    positions_image = positions_image + jitter

    x_low, x_high = grid[..., 0].min(), grid[..., 0].max()
    z_low, z_high = grid[..., 2].min(), grid[..., 2].max()
    positions_uniform = np.stack(
        [
            rng.uniform(x_low, x_high, n_uniform),
            np.full(n_uniform, grid[..., 1].mean()),
            rng.uniform(z_low, z_high, n_uniform),
        ],
        axis=1,
    )

    positions = np.concatenate([positions_image, positions_uniform], axis=0)
    return positions.astype(np.float32)
