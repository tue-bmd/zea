"""Geometry helpers for beamforming.

Lightweight, backend-agnostic utilities derived purely from probe and transmit geometry.
This is a *leaf* module (only depends on ``keras`` / ``numpy``) so it can be
imported from :mod:`zea.beamform.beamformer`, :mod:`zea.simulator` and
:mod:`zea.beamform.pfield` without introducing an import cycle.

The transmit helpers (:func:`transmit_apex_distance`, :func:`insonified_edge_rays`)
work in the ``x-z`` imaging plane and in zea's transducer frame: ``z = 0`` at the
middle of the array, ``+z`` into the medium, and angles measured from ``+z``,
positive toward ``+x``. They back the default beamforming grid of
:class:`zea.Parameters`.
"""

import keras
import numpy as np
from keras import ops

STEERING_TOLERANCE = 1e-6
"""Polar angles at or below this magnitude [rad] count as unsteered. Converted data
stores a nominally unsteered scan as float noise rather than exact zeros."""

SINC_HALF_AMPLITUDE = 0.6033545644016143
"""The argument ``x`` at which ``sinc(x) = sin(pi x) / (pi x)`` drops to 0.5 (-6 dB)."""


def element_pitch(probe_geometry):
    """Median distance between neighbouring elements in metres.

    Unlike :meth:`zea.probes.Probe.get_pitch` this also works for curved arrays, since
    it measures the distance along the array rather than along ``x``.

    Args:
        probe_geometry (np.ndarray): Element positions of shape ``(n_el, 3)``, ordered
            along the array.

    Returns:
        float: The median element spacing, or ``nan`` for a single element.
    """
    probe_geometry = np.asarray(probe_geometry, dtype=np.float64)
    if len(probe_geometry) < 2:
        return float("nan")
    return float(np.median(np.linalg.norm(np.diff(probe_geometry, axis=0), axis=1)))


def transmit_apex_distance(transmit_origins, polar_angles, focus_distances, tol):
    """Find the point on the central axis that a set of transmits fans out from.

    Two transmit schemes define such a point:

    - **Steered transmits** whose axes (from ``transmit_origins`` along ``polar_angles``)
      cross ``x = 0`` in one point. That is the array middle for transmits launched
      from there, a virtual apex behind the array for transmits spread over a flat
      array, and the centre of curvature for transmits along a curved array's normals.
    - **Unsteered diverging waves** that share one virtual source
      (``transmit_origins + focus_distances`` along ``+z``) on the central axis.

    Args:
        transmit_origins (np.ndarray): Transmit origins of shape ``(n_tx, 3)`` [m].
        polar_angles (np.ndarray): Steering angles of shape ``(n_tx,)`` [rad].
        focus_distances (np.ndarray): Focus distances of shape ``(n_tx,)`` [m]
            (negative for diverging waves).
        tol (float): How far apart [m] the crossings may be and still count as one
            point. :class:`zea.Parameters` uses half the element pitch, the same
            tolerance as :func:`zea.probes.fit_curved_probe_radius`.

    Returns:
        float | None: The distance [m] from the middle of the array back to that point
        (``>= 0``), or ``None`` when the transmits define no single point.
    """
    origins = np.asarray(transmit_origins, dtype=np.float64)
    angles = np.asarray(polar_angles, dtype=np.float64)
    focus = np.asarray(focus_distances, dtype=np.float64)

    steered = np.abs(angles) > STEERING_TOLERANCE
    if np.any(steered):
        # Where each steered axis crosses x = 0, as a distance behind the array
        crossings = origins[steered, 0] / np.tan(angles[steered]) - origins[steered, 2]
        middle = float(np.median(crossings))
        if np.ptp(crossings) <= tol and middle >= -tol:
            return max(middle, 0.0)
        return None

    if np.all(np.isfinite(focus) & (focus < 0)):
        source_x = origins[:, 0]
        source_z = origins[:, 2] + focus
        if np.max(np.abs(source_x)) <= tol and np.ptp(source_z) <= tol:
            return float(-np.median(source_z))
    return None


def insonified_edge_rays(
    probe_geometry, transmit_origins, polar_angles, focus_distances, element_normals=None
):
    """Edge rays bounding the region each transmit insonifies, in the ``x-z`` plane.

    - **Focused** transmit (positive, finite focus): its axis, from its origin along its
      angle. It insonifies a narrow beam, so it adds no width.
    - **Plane** wave (``inf`` or ``0`` focus): from the two end elements of the array
      along the steering angle, tilted with each end element's normal (which makes an
      unsteered plane wave on a curved array span the array's angular range).
    - **Diverging** wave (negative focus): from its virtual source through the two end
      elements of the array.

    Args:
        probe_geometry (np.ndarray): Element positions of shape ``(n_el, 3)``.
        transmit_origins (np.ndarray): Transmit origins of shape ``(n_tx, 3)``.
        polar_angles (np.ndarray): Steering angles of shape ``(n_tx,)`` [rad].
        focus_distances (np.ndarray): Focus distances of shape ``(n_tx,)`` [m].
        element_normals (np.ndarray, optional): Unit element normals of shape
            ``(n_el, 3)``. Defaults to :func:`compute_element_normals`, which estimates
            the end elements' normals from their last neighbour, half an element short on
            a curved array; pass the exact normals when they are known.

    Returns:
        tuple[np.ndarray, np.ndarray]: Start points and unit directions of the rays,
        both of shape ``(n_rays, 2)`` as ``(x, z)``.
    """
    probe_geometry = np.asarray(probe_geometry, dtype=np.float64)
    origins = np.asarray(transmit_origins, dtype=np.float64)[:, [0, 2]]
    angles = np.asarray(polar_angles, dtype=np.float64)
    focus = np.asarray(focus_distances, dtype=np.float64)

    if element_normals is None:
        element_normals = compute_element_normals(ops.convert_to_tensor(probe_geometry))
    normals = np.asarray(ops.convert_to_numpy(element_normals), dtype=np.float64)
    ends = [int(np.argmin(probe_geometry[:, 0])), int(np.argmax(probe_geometry[:, 0]))]
    end_points = probe_geometry[ends][:, [0, 2]]
    end_tilts = np.arctan2(normals[ends, 0], normals[ends, 2])

    points, directions = [], []
    for origin, angle, f in zip(origins, angles, focus):
        axis = np.array([np.sin(angle), np.cos(angle)])
        if np.isfinite(f) and f > 0:
            points.append(origin)
            directions.append(axis)
        elif np.isfinite(f) and f < 0:
            source = origin + f * axis
            for end in end_points:
                points.append(end)
                directions.append((end - source) / np.linalg.norm(end - source))
        else:
            for end, tilt in zip(end_points, end_tilts):
                points.append(end)
                directions.append(np.array([np.sin(angle + tilt), np.cos(angle + tilt)]))
    return np.asarray(points), np.asarray(directions)


def rays_at_depth(points, directions, depth):
    """Where rays reach a depth ``z = depth``.

    Rays that start below ``depth`` are evaluated at their start point.

    Args:
        points (np.ndarray): Ray start points of shape ``(n_rays, 2)`` as ``(x, z)``.
        directions (np.ndarray): Unit directions of shape ``(n_rays, 2)``.
        depth (float): The depth [m].

    Returns:
        np.ndarray: Points of shape ``(n_rays, 2)``; ``nan`` for rays that never reach
        ``depth`` because they point sideways or backwards.
    """
    points = np.asarray(points, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    reaching = directions[:, 1] > STEERING_TOLERANCE
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.maximum((depth - points[:, 1]) / directions[:, 1], 0.0)
    out = points + t[:, None] * directions
    out[~reaching] = np.nan
    return out


def element_acceptance_angle(element_width, wavelength):
    """Half-angle within which a single element receives, at -6 dB.

    Uses the same rigid-baffle directivity model as
    :func:`zea.func.ultrasound.directivity`, ``sinc(element_width / wavelength *
    sin(theta))``, which drops to 0.5 at ``sin(theta) = 0.603 * wavelength /
    element_width``. Elements narrower than about 0.6 wavelength never drop that far
    and receive over the whole half-space.

    Args:
        element_width (float): Element width [m].
        wavelength (float): Wavelength [m].

    Returns:
        float: The acceptance half-angle [rad], at most ``pi / 2``.
    """
    sine = SINC_HALF_AMPLITUDE * wavelength / element_width
    return float(np.pi / 2 if sine >= 1 else np.arcsin(sine))


def compute_element_normals(probe_geometry, eps=1e-12):
    """Estimate per-element unit surface normals from element positions.

    zea's built-in receive f-number apodization
    (:func:`zea.beamform.beamformer.fnumber_mask`) measures the acceptance
    cone relative to each element's surface normal. For a flat linear array
    every element faces ``+z`` and the normal is trivial, but for a
    curved/convex array the peripheral elements are physically tilted outward,
    so their true look-direction is *not* ``+z``. Assuming ``+z`` for those
    elements needlessly clips the receive aperture at the lateral edges of the
    sector. This function derives each element's outward normal directly from
    ``probe_geometry`` so no extra probe metadata (radius of curvature, per
    element angles, ...) is required.

    The array is treated as a 1-D curve of elements ordered along the aperture
    lying in the ``x-z`` imaging plane (zea's convention; the medium is at
    ``+z``). For each element the local tangent ``d`` is estimated by central
    finite differences of neighbouring element positions (one-sided at the two
    ends). The outward normal is the depth axis ``+z`` with its component along
    the tangent removed::

        n = normalize(z_hat - (z_hat.d) / (d.d) * d)

    This yields **exactly** ``(0, 0, 1)`` for a flat array (any spacing,
    ordered or not — the ``z`` column is identically zero, so the rejection
    leaves ``z_hat`` untouched), so existing linear-array reconstructions are
    unchanged bit-for-bit. For a convex arc it recovers the exact radial
    outward normal at the interior elements.

    Args:
        probe_geometry (Tensor): Element positions ``(x, y, z)`` of shape
            ``(n_el, 3)`` in metres, ordered along the array.
        eps (float): Small value guarding the tangent-length division.

    Returns:
        Tensor: Unit outward normals of shape ``(n_el, 3)``.

    Notes:
        * Intended for 1-D arrays (linear / phased / curved) in the ``x-z``
          plane. For a genuine 3-D matrix array (elements spread in ``y``) a
          single index-ordered tangent is not meaningful, so the function
          falls back to ``+z`` for every element when the geometry has a
          non-negligible ``y`` extent — reproducing today's behaviour.
        * A single-element array falls back to ``+z``.
    """
    z_hat = ops.cast(ops.convert_to_tensor([0.0, 0.0, 1.0]), probe_geometry.dtype)
    # (n_el, 3) field of +z, used for every fallback path.
    z_field = z_hat[None] * ops.ones_like(probe_geometry)

    n_el = probe_geometry.shape[0]
    if n_el is not None and n_el < 2:
        # A lone element carries no tangent information.
        return z_field

    # Central finite differences of element positions (one-sided at the ends):
    #   d[0]        = p[1] - p[0]
    #   d[i]        = p[i+1] - p[i-1]      (interior)
    #   d[-1]       = p[-1] - p[-2]
    fwd = probe_geometry[1:] - probe_geometry[:-1]  # (n_el - 1, 3)
    tangent = ops.concatenate([fwd[:1], fwd[1:] + fwd[:-1], fwd[-1:]], axis=0)  # (n_el, 3)

    # Reject the tangent component from +z (scale-free: no tangent normalisation,
    # so a flat array with z==0 gives exactly z_hat regardless of pitch/eps).
    z_dot_d = ops.sum(z_hat[None] * tangent, axis=-1, keepdims=True)  # (n_el, 1)
    d_dot_d = ops.sum(tangent * tangent, axis=-1, keepdims=True)  # (n_el, 1)
    rejected = z_hat[None] - (z_dot_d / ops.maximum(d_dot_d, eps)) * tangent  # (n_el, 3)

    # Normalise without eps on the good path (nn == 1 exactly for a flat array),
    # falling back to +z where the tangent is (near-)parallel to +z.
    nn = ops.linalg.norm(rejected, axis=-1, keepdims=True)
    normals = ops.where(nn < 1e-6, z_field, rejected / ops.maximum(nn, 1e-12))

    # Guard against genuine 3-D matrix arrays: if the elements have a
    # non-negligible extent in y, an index-ordered 1-D tangent is meaningless,
    # so reproduce today's "+z for everyone" behaviour. Convex 1-D probes
    # (y ~= 0) pass straight through. Scalar condition -> broadcasts over both.
    x, y, z = probe_geometry[:, 0], probe_geometry[:, 1], probe_geometry[:, 2]
    y_span = ops.max(y) - ops.min(y)
    inplane_span = ops.maximum(ops.max(x) - ops.min(x), ops.max(z) - ops.min(z))
    is_planar_xz = y_span <= 1e-3 * inplane_span + keras.backend.epsilon()
    normals = ops.where(is_planar_xz, normals, z_field)

    return normals
