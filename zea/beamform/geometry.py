"""Geometry helpers for beamforming.

Lightweight, backend-agnostic utilities derived purely from probe geometry.
This is a *leaf* module (only depends on ``keras`` / ``numpy``) so it can be
imported from :mod:`zea.beamform.beamformer`, :mod:`zea.simulator` and
:mod:`zea.beamform.pfield` without introducing an import cycle.
"""

import keras
from keras import ops


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
