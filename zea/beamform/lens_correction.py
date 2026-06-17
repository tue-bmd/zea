r"""Lens-corrected delay computation for ultrasound beamforming.

The acoustic lens fitted over most ultrasound probes has a lower speed of
sound than the surrounding medium (tissue / water), ~1000 m/s versus 1540 m/s,
which shortens the travel time near the face of the transducer and alters
the effective focus. We assume a flat lens with uniform thickness and
speed of sound.

The corrected one-way travel time from each transducer element to each image
pixel is computed by finding the lateral crossing point :math:`x_l` on the
lens surface that minimises total travel time (Fermat's principle):

.. math::

    T(x_l) = \frac{\sqrt{(x_l - x_e)^2 + z_l^2}}{c_\text{lens}}
            + \frac{\sqrt{(x_l - x_s)^2 + (z_l - z_s)^2}}{c_\text{medium}}

where :math:`(x_e, 0)` is the element position, :math:`(x_s, z_s)` is the
pixel position, and :math:`z_l` is the lens thickness.  Setting
:math:`\partial T / \partial x_l = 0` recovers Snell's law:

.. math::

    \frac{\sin\theta_\text{lens}}{c_\text{lens}}
    = \frac{\sin\theta_\text{medium}}{c_\text{medium}}

The root is found iteratively via Newton-Raphson.

.. note::

    This is more physically accurate than the scalar ``lensCorrection`` field
    used by Verasonics, which adds a single constant delay offset uniformly
    across all elements and ignores angle-dependent refraction.
"""

from keras import ops


def compute_lens_corrected_travel_times(
    element_pos, pixel_pos, lens_thickness, c_lens, c_medium, n_iter=1
):
    """Compute the travel time of the shortest path between the element and the pixel.

    .. note::

        This function assumes a flat array geometry.

    Args:
        element_pos (ndarray): The position of the element of shape (n_el, 3).
        pixel_pos (ndarray): The position of the pixel of shape (n_pixels, 3).
        lens_thickness (float): The thickness of the lens in meters.
        c_lens (float): The speed of sound in the lens in m/s.
        c_medium (float): The speed of sound in the medium in m/s.
        n_iter (int): The number of iterations to run the Newton-Raphson method.

    Returns:
        ndarray: The travel times of shape (n_pixels, n_el).
    """

    pixel_pos = pixel_pos[:, None] - element_pos[None]

    # Project the 3D problem to a 2D problem by shifting all pixels to have the element
    # in the origin and then projecting the positions to the plane spanned by
    # [pixel_x-element_x, pixel_z-element_z, 0], [0, 0, 1]
    xs = ops.norm(pixel_pos[..., :2], axis=-1)
    zs = pixel_pos[..., -1]

    pixel_pos_2d = ops.stack([xs, zs], axis=-1)
    element_pos_2d = ops.zeros((1, element_pos.shape[0], 2))

    xl = compute_xl(
        element_pos_2d,
        pixel_pos_2d,
        lens_thickness,
        c_lens,
        c_medium,
        n_iter,
    )

    # Form the position of the lens crossing point by adding the z-coordinate of the
    # lens to the point
    pos_lenscrossing = ops.stack([xl, lens_thickness * ops.ones_like(xl)], axis=-1)

    indices = ops.array([0, -1])
    element_pos = ops.take(element_pos, indices, axis=-1)

    # Compute the travel time of the shortest path
    travel_time = compute_travel_time(
        element_pos_2d, pos_lenscrossing, c_lens
    ) + compute_travel_time(pos_lenscrossing, pixel_pos_2d, c_medium)
    return travel_time


def compute_xl(element_pos_2d, pixel_pos_2d, lens_thickness, c_lens, c_medium, n_iter):
    """Computes the lateral point on the lens that the shortest path goes through based
    on Fermat's principle.

    Args:
        element_pos_2d (float): The 2D position of the element.
        pixel_pos_2d (float): The 2D position of the pixel.
        lens_thickness (float): The thickness of the lens in meters.
        c_lens (float): The speed of sound in the lens in m/s.
        c_medium (float): The speed of sound in the medium in m/s.
        n_iter (int): The number of iterations to run the Newton-Raphson method.

    Returns:
        float: The x-coordinate of the lateral point on the lens.
    """
    xs = pixel_pos_2d[..., 0]
    zs = pixel_pos_2d[..., 1]
    xe = element_pos_2d[..., 0]
    ze = element_pos_2d[..., 1]

    # Apply Newton-Raphson method to find the lateral point on the lens that the shortest
    # path goes through
    xl_init = lens_thickness * (xs - xe) / (zs - ze) + xe
    xl = xl_init
    for _ in range(n_iter):
        xl = xl + dxl(xe, ze, xl, xs, zs, lens_thickness, c_lens, c_medium)

        # Clip the lateral point to be in between the element and the pixel
        xl = ops.clip(xl, ops.minimum(xs, xe), ops.maximum(xs, xe))

    return xl


def dxl(xe, ze, xl, xs, zs, zl, c_lens, c_medium):
    """Computes the update step for the lateral point on the lens that the shortest path
    using the Newton-Raphson method.

    Notes
    -----
    This result was derived by defining the total travel time through the lens and the
    medium as a function of the lateral point on the lens and then taking the
    derivative. We then have a function whose root is the lateral point on the lens that
    the shortest path goes through. We then compute the derivative and update the
    lateral point on the lens using the Newton-Raphson method:
    x_new = x - f(x) / f'(x).
    """

    eps = 1e-6

    numerator = -((xe - xl) / (c_lens * ops.sqrt((xe - xl) ** 2 + (ze - zl) ** 2))) + (
        (xl - xs) / (c_medium * ops.sqrt((xl - xs) ** 2 + (zl - zs) ** 2)) + eps
    )

    denominator = (
        -((xe - xl) ** 2 / (c_lens * ((xe - xl) ** 2 + (ze - zl) ** 2) ** (3 / 2) + eps))
        + (1 / (c_lens * ops.sqrt((xe - xl) ** 2 + (ze - zl) ** 2)))
        - ((xl - xs) ** 2 / (c_medium * ((xl - xs) ** 2 + (zl - zs) ** 2) ** (3 / 2) + eps))
        + (1 / (c_medium * ops.sqrt((xl - xs) ** 2 + (zl - zs) ** 2) + eps))
    )

    result = -numerator / (denominator + eps)

    # Handle NaNs
    result = ops.nan_to_num(result)

    # Clip the update step to prevent divergence
    # This value is chosen to be small enough to prevent divergence but large enough to
    # cover the distance accross a normal ultrasound aperture in a single step.
    result = ops.clip(result, -10e-3, 10e-3)

    return result


def compute_travel_time(pos_a, pos_b, c):
    """Compute the travel time between two points."""
    return ops.linalg.norm(pos_a - pos_b, axis=-1) / c
