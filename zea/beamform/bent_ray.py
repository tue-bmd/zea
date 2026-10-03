r"""Refraction-aware (bent-ray) travel times in a heterogeneous speed-of-sound map.

Pulse-echo speed-of-sound (SoS) estimation by autofocusing (see
:class:`~zea.ops.CommonMidpointPhaseError`) optimizes the SoS map that is used to
compute the beamforming delays. The default heterogeneous delay model in zea
(:func:`zea.beamform.beamformer.calculate_delays_heterogeneous_medium`) integrates the
slowness along *straight* element-to-pixel rays, which neglects refraction. This module
implements the *UltraBend* propagation model, which replaces the straight rays by
bent rays traced through the current SoS map.

.. citation:: duelmer2026ultrabend

The method works in three steps:

1. **Ray marching.** For every element a fan of rays is launched and integrated
   through the slowness field :math:`s(\mathbf{x}) = 1 / c(\mathbf{x})` with a
   second-order Runge-Kutta scheme. The rays are the characteristics of the eikonal
   equation :math:`\lVert \nabla T_e(\mathbf{x}) \rVert = s(\mathbf{x})`; in
   arc-length parametrization the position :math:`\mathbf{q}`, unit direction
   :math:`\mathbf{d}` and travel time :math:`\tau^\text{ray}` evolve as

   .. math::

       \frac{d\mathbf{q}}{d\ell} = \mathbf{d}, \qquad
       \frac{d\mathbf{d}}{d\ell} = \frac{\nabla s - (\nabla s \cdot \mathbf{d})\,
       \mathbf{d}}{s}, \qquad
       \frac{d\tau^\text{ray}}{d\ell} = s.

2. **Deposition.** At every ray sample, nearby nodes :math:`\mathbf{x}_f` of a
   regular travel-time grid receive a local arrival-time estimate, corrected for
   the along-ray offset :math:`d_\parallel` and the spherical path-length
   difference due to the orthogonal offset :math:`d_\perp`

   .. math::

       \tau^\text{loc}(\mathbf{x}_f, \ell) = \tau^\text{ray}(\ell)
       + \left(\sqrt{(\ell + d_\parallel)^2 + d_\perp^2} - \ell\right)
       s(\mathbf{q}(\ell)),

   i.e. the path-length difference to a spherical wavefront through the ray sample.
   The paper uses its second-order expansion :math:`d_\parallel + d_\perp^2 / 2\ell`,
   which is inaccurate close to the element where the offsets are comparable to
   :math:`\ell`. Each estimate gets a Gaussian weight
   :math:`w \propto \exp(-(d_\parallel^2 + d_\perp^2) / 2\sigma^2)` inside a fixed
   support radius. The estimates are combined into the first-arrival
   time with a soft-min

   .. math::

       \alpha = w \exp\left(-\beta \left(\tau^\text{loc} - \min \tau^\text{loc}
       \right)\right), \qquad
       \tau_e(\mathbf{x}_f) = \frac{\sum_\ell \alpha\, \tau^\text{loc}}
       {\sum_\ell \alpha}.

   The paper evaluates this in two tracing passes (first the minimum, then the
   weighted sum). Here it is evaluated in a single pass with a running minimum and
   rescaled accumulators (the same trick as an online log-sum-exp), which gives the
   identical result.

3. **Replay adjoint.** Differentiating through the ray tracer would require storing
   every intermediate ray state. Instead, following path replay backpropagation
   (Vicini et al., 2021), the backward pass traces the same rays again with the ray
   geometry, deposition weights and soft-min reference times held fixed, and scatters
   the travel-time adjoints back onto the slowness grid with the bilinear
   interpolation weights. By Fermat's principle the travel time is stationary with
   respect to small perturbations of the ray path, so neglecting the geometry
   derivative is a first-order accurate approximation. Memory use is independent of
   the number of ray steps.

Travel-time grid nodes that no ray reaches (e.g. outside the launched fan) fall back to
the straight-ray travel time, so the result is always defined.

The main entry points are :func:`compute_bent_ray_travel_times`, which returns a
travel-time map of shape ``(n_el, n_z, n_x)``, and :func:`sample_travel_time_map`,
which interpolates such a map at arbitrary pixel positions. In a pipeline, use
:class:`zea.ops.BentRayTravelTimes` followed by :class:`zea.ops.TOFCorrection`.

.. note::

    Only 2-D media in the x-z plane are supported. Element positions and pixels are
    projected onto that plane (``y`` is ignored).
"""

import numpy as np
from keras import ops

from zea.beamform.geometry import compute_element_normals

# Sentinel "infinite" time in seconds, kept finite so that ``0 * BIG`` is well defined
# on every backend (unlike ``0 * inf``).
_BIG_TIME = 1e3


def _uniform_axis(coords):
    """Return ``(origin, spacing, n)`` of a uniformly spaced 1-D coordinate tensor."""
    coords = ops.convert_to_tensor(coords, "float32")
    return coords[0], coords[1] - coords[0], coords.shape[0]


def bilinear_stencil(x, z, x0, dx, nx, z0, dz, nz):
    """Flat node indices and bilinear weights of points on a regular x-z grid.

    Points outside the grid are clamped to its boundary, which corresponds to
    nearest-neighbor extrapolation of the interpolated field.

    Args:
        x (Tensor): Lateral coordinates of the query points, any shape ``S``.
        z (Tensor): Axial coordinates of the query points, shape ``S``.
        x0 (float): Lateral coordinate of the first grid column.
        dx (float): Lateral grid spacing.
        nx (int): Number of grid columns.
        z0 (float): Axial coordinate of the first grid row.
        dz (float): Axial grid spacing.
        nz (int): Number of grid rows.

    Returns:
        tuple[Tensor, Tensor]:
            - **indices** -- int32 indices into the row-major flattened ``(nz, nx)``
              grid of shape ``S + (4,)``.
            - **weights** -- bilinear weights of shape ``S + (4,)`` that sum to one.
    """
    fx = ops.clip((x - x0) / dx, 0.0, nx - 1.0)
    fz = ops.clip((z - z0) / dz, 0.0, nz - 1.0)
    ix0 = ops.clip(ops.cast(ops.floor(fx), "int32"), 0, max(nx - 2, 0))
    iz0 = ops.clip(ops.cast(ops.floor(fz), "int32"), 0, max(nz - 2, 0))
    ix1 = ops.minimum(ix0 + 1, nx - 1)
    iz1 = ops.minimum(iz0 + 1, nz - 1)
    wx = fx - ops.cast(ix0, fx.dtype)
    wz = fz - ops.cast(iz0, fz.dtype)

    indices = ops.stack([iz0 * nx + ix0, iz0 * nx + ix1, iz1 * nx + ix0, iz1 * nx + ix1], -1)
    weights = ops.stack(
        [(1 - wz) * (1 - wx), (1 - wz) * wx, wz * (1 - wx), wz * wx],
        axis=-1,
    )
    return indices, weights


def _gather_bilinear(field_flat, indices, weights):
    """Interpolate a flattened field (``(n_nodes,)`` or ``(n_nodes, k)``) at a stencil."""
    values = ops.take(field_flat, indices, axis=0)
    if len(field_flat.shape) == 1:
        return ops.sum(values * weights, axis=-1)
    return ops.sum(values * weights[..., None], axis=-2)


def _central_gradient(field, dz, dx):
    """Gradient ``(d/dz, d/dx)`` of a 2-D field, one-sided at the borders."""

    def _diff(f, spacing):
        # Differences along axis 0; edges use one-sided differences.
        if f.shape[0] < 2:
            return ops.zeros_like(f)
        forward = (f[1:] - f[:-1]) / spacing
        central = (f[2:] - f[:-2]) / (2.0 * spacing)
        return ops.concatenate([forward[:1], central, forward[-1:]], axis=0)

    return _diff(field, dz), ops.transpose(_diff(ops.transpose(field), dx))


def _segment_min(values, segment_ids, num_segments):
    """Segment-wise minimum. Empty segments get a very large value."""
    return -ops.segment_max(-values, segment_ids, num_segments=num_segments)


class _BentRayTracer:
    """Ray marcher shared by the forward pass and the replay adjoint.

    All geometry is computed from a fixed (non-differentiated) slowness field. The
    tracer is stateless apart from its configuration, so the backward pass can replay
    exactly the same ray steps as the forward pass.
    """

    def __init__(
        self,
        slowness,
        sos_grid_x,
        sos_grid_z,
        element_xz,
        element_normals_xz,
        grid_x,
        grid_z,
        n_rays,
        max_angle,
        step_size,
        support_radius,
        sigma,
        softmin_scale,
    ):
        self.slowness = slowness
        self.nz_s, self.nx_s = slowness.shape
        self.x0_s, self.dx_s, _ = _uniform_axis(sos_grid_x)
        self.z0_s, self.dz_s, _ = _uniform_axis(sos_grid_z)
        self.n_sos = self.nz_s * self.nx_s

        grad_z, grad_x = _central_gradient(slowness, self.dz_s, self.dx_s)
        # Stack slowness and its gradient so a single gather serves all three.
        self.fields = ops.stack(
            [ops.reshape(slowness, (-1,)), ops.reshape(grad_x, (-1,)), ops.reshape(grad_z, (-1,))],
            axis=-1,
        )

        self.x0, self.dx, self.nx = _uniform_axis(grid_x)
        self.z0, self.dz, self.nz = _uniform_axis(grid_z)
        self.n_targets = self.nz * self.nx
        self.n_el = element_xz.shape[0]
        self.n_rays = n_rays
        self.n_segments = self.n_el * self.n_targets

        cell = ops.minimum(ops.abs(self.dx), ops.abs(self.dz))
        self.radius = support_radius * cell
        self.sigma = sigma * cell
        self.beta = 1.0 / softmin_scale
        self.step = 0.5 * cell if step_size is None else ops.cast(step_size, "float32")

        # Per-axis stencil half-width. A node within ``radius`` of the ray sample is at
        # most ``support_radius + 0.5`` cells from the nearest node along each axis.
        half = int(np.floor(support_radius + 0.5))
        offsets = np.arange(-half, half + 1, dtype="int32")
        oz, ox = np.meshgrid(offsets, offsets, indexing="ij")
        self.offset_z = ops.convert_to_tensor(oz.reshape(-1))
        self.offset_x = ops.convert_to_tensor(ox.reshape(-1))

        # Rays are traced while inside the bounding box of the travel-time grid and the
        # elements, padded by the support radius.
        xs = ops.concatenate(
            [ops.stack([self.x0, self.x0 + (self.nx - 1) * self.dx]), element_xz[:, 0]], 0
        )
        zs = ops.concatenate(
            [ops.stack([self.z0, self.z0 + (self.nz - 1) * self.dz]), element_xz[:, 1]], 0
        )
        self.box = (
            ops.min(xs) - self.radius,
            ops.max(xs) + self.radius,
            ops.min(zs) - self.radius,
            ops.max(zs) + self.radius,
        )
        diagonal = ops.sqrt((self.box[1] - self.box[0]) ** 2 + (self.box[3] - self.box[2]) ** 2)
        self.n_steps = ops.cast(ops.ceil(diagonal / self.step), "int32")

        # Launch a fan of rays around each element's normal.
        angles = ops.convert_to_tensor(np.linspace(-max_angle, max_angle, n_rays), "float32")
        cos_a, sin_a = ops.cos(angles)[None], ops.sin(angles)[None]
        nx_e, nz_e = element_normals_xz[:, 0:1], element_normals_xz[:, 1:2]
        self.init_dx = nx_e * cos_a + nz_e * sin_a
        self.init_dz = -nx_e * sin_a + nz_e * cos_a
        self.init_qx = ops.broadcast_to(element_xz[:, 0:1], (self.n_el, n_rays))
        self.init_qz = ops.broadcast_to(element_xz[:, 1:2], (self.n_el, n_rays))
        self.element_index = ops.arange(self.n_el, dtype="int32")[:, None, None]

    def initial_state(self):
        """Ray state ``(qx, qz, dx, dz, tau, active)``, each of shape ``(n_el, n_rays)``."""
        zeros = ops.zeros((self.n_el, self.n_rays), "float32")
        return (self.init_qx, self.init_qz, self.init_dx, self.init_dz, zeros, zeros + 1.0)

    def sos_stencil(self, qx, qz):
        """Bilinear stencil of positions on the slowness grid."""
        return bilinear_stencil(
            qx, qz, self.x0_s, self.dx_s, self.nx_s, self.z0_s, self.dz_s, self.nz_s
        )

    def _sample(self, qx, qz):
        """Slowness and its gradient at the given positions."""
        fields = _gather_bilinear(self.fields, *self.sos_stencil(qx, qz))
        return fields[..., 0], fields[..., 1], fields[..., 2]

    @staticmethod
    def _bend(s, gx, gz, dx, dz):
        """Change in ray direction per unit arc length."""
        g_par = gx * dx + gz * dz
        return (gx - g_par * dx) / s, (gz - g_par * dz) / s

    @staticmethod
    def _normalize(dx, dz):
        norm = ops.sqrt(dx**2 + dz**2)
        return dx / norm, dz / norm

    def step_rays(self, state):
        """Advance all rays by one RK2 (midpoint) step.

        Returns:
            tuple: The new state, and the midpoint positions ``(qx_mid, qz_mid)``
            at which the slowness for the travel-time increment was sampled.
        """
        qx, qz, dx, dz, tau, active = state
        h = self.step

        s, gx, gz = self._sample(qx, qz)
        kx, kz = self._bend(s, gx, gz, dx, dz)
        qx_mid, qz_mid = qx + 0.5 * h * dx, qz + 0.5 * h * dz
        dx_mid, dz_mid = self._normalize(dx + 0.5 * h * kx, dz + 0.5 * h * kz)

        s_mid, gx_mid, gz_mid = self._sample(qx_mid, qz_mid)
        kx, kz = self._bend(s_mid, gx_mid, gz_mid, dx_mid, dz_mid)
        qx_new, qz_new = qx + h * dx_mid, qz + h * dz_mid
        dx_new, dz_new = self._normalize(dx + h * kx, dz + h * kz)
        tau_new = tau + h * s_mid

        x_lo, x_hi, z_lo, z_hi = self.box
        inside = (qx_new >= x_lo) & (qx_new <= x_hi) & (qz_new >= z_lo) & (qz_new <= z_hi)
        active_new = active * ops.cast(inside, active.dtype)
        return (qx_new, qz_new, dx_new, dz_new, tau_new, active_new), (qx_mid, qz_mid)

    def deposit(self, state, arc_length):
        """Local arrival-time estimates of a ray sample at nearby grid nodes.

        Args:
            state (tuple): Ray state after the step.
            arc_length (float): Arc length of the rays at this sample.

        Returns:
            dict: ``segment`` (target index), ``tau_loc`` (Eq. 5), ``weight`` (Eq. 6,
            zero for invalid candidates) and ``correction`` (the coefficient of
            :math:`s(\\mathbf{q})` in Eq. 5), each of shape ``(n_el, n_rays, n_stencil)``,
            plus the slowness stencil at the ray positions.
        """
        qx, qz, dx, dz, tau, active = state

        ix = ops.cast(ops.round((qx - self.x0) / self.dx), "int32")[..., None] + self.offset_x
        iz = ops.cast(ops.round((qz - self.z0) / self.dz), "int32")[..., None] + self.offset_z
        in_grid = (ix >= 0) & (ix < self.nx) & (iz >= 0) & (iz < self.nz)
        ix = ops.clip(ix, 0, self.nx - 1)
        iz = ops.clip(iz, 0, self.nz - 1)

        rel_x = self.x0 + ops.cast(ix, "float32") * self.dx - qx[..., None]
        rel_z = self.z0 + ops.cast(iz, "float32") * self.dz - qz[..., None]
        d_par = rel_x * dx[..., None] + rel_z * dz[..., None]
        d_perp = -rel_x * dz[..., None] + rel_z * dx[..., None]
        dist2 = d_par**2 + d_perp**2

        stencil = self.sos_stencil(qx, qz)
        s = _gather_bilinear(self.fields[:, 0], *stencil)
        # Path-length difference to a spherical wavefront centered at arc length ``ell``
        # behind the ray sample. Eq. 5 of the paper is its second-order expansion
        # ``d_par + d_perp**2 / (2 ell)``, which breaks down close to the element.
        ell = ops.maximum(arc_length, self.step)
        correction = ops.sqrt((ell + d_par) ** 2 + d_perp**2) - ell
        tau_loc = tau[..., None] + correction * s[..., None]

        valid = in_grid & (dist2 <= self.radius**2) & (active[..., None] > 0)
        weight = ops.where(valid, ops.exp(-dist2 / (2.0 * self.sigma**2)), 0.0)
        tau_loc = ops.where(valid, tau_loc, _BIG_TIME)
        segment = self.element_index * self.n_targets + iz * self.nx + ix
        return {
            "segment": ops.reshape(segment, (-1,)),
            "tau_loc": ops.reshape(tau_loc, (-1,)),
            "weight": ops.reshape(weight, (-1,)),
            "correction": correction,
            "weight_3d": weight,
            "stencil": stencil,
        }

    def arc_length(self, i):
        """Arc length of the rays after step ``i`` (zero-based)."""
        return ops.cast(i + 1, "float32") * self.step

    def forward(self):
        """Trace all rays and compute the soft-min travel times (Eqs. 7-8).

        Returns:
            tuple[Tensor, Tensor, Tensor]: Flattened ``(n_el * n_z * n_x,)`` tensors with
            the soft-min travel time, the running minimum (reference) time and the
            normalization :math:`\\sum \\alpha` relative to that minimum.
        """
        n_seg = self.n_segments

        def body(i, carry):
            state, t_min, norm, weighted = carry
            state, _ = self.step_rays(state)
            dep = self.deposit(state, self.arc_length(i))
            seg, tau_loc, weight = dep["segment"], dep["tau_loc"], dep["weight"]

            new_min = ops.minimum(t_min, _segment_min(tau_loc, seg, n_seg))
            rescale = ops.exp(-self.beta * (t_min - new_min))
            alpha = weight * ops.exp(-self.beta * (tau_loc - ops.take(new_min, seg)))
            norm = norm * rescale + ops.segment_sum(alpha, seg, num_segments=n_seg)
            weighted = weighted * rescale + ops.segment_sum(
                alpha * tau_loc, seg, num_segments=n_seg
            )
            return state, new_min, norm, weighted

        zeros = ops.zeros((n_seg,), "float32")
        carry = (self.initial_state(), zeros + _BIG_TIME, zeros, zeros)
        _, t_min, norm, weighted = ops.fori_loop(0, self.n_steps, body, carry)
        travel_time = weighted / ops.where(norm > 0, norm, 1.0)
        return travel_time, t_min, norm

    def _ray_adjoint(self, dep, t_min, adjoint_per_norm):
        """Per-candidate adjoint of the local arrival times (Eq. 9)."""
        seg = dep["segment"]
        alpha = dep["weight"] * ops.exp(-self.beta * (dep["tau_loc"] - ops.take(t_min, seg)))
        adjoint = alpha * ops.take(adjoint_per_norm, seg)
        return ops.reshape(adjoint, dep["weight_3d"].shape)

    def _scatter(self, values, stencil):
        """Scatter per-ray values onto the slowness grid (adjoint of interpolation)."""
        indices, weights = stencil
        contributions = values[..., None] * weights
        return ops.segment_sum(
            ops.reshape(contributions, (-1,)),
            ops.reshape(indices, (-1,)),
            num_segments=self.n_sos,
        )

    def backward(self, travel_time_adjoint, t_min, norm):
        """Replay adjoint: gradient of the travel times w.r.t. the slowness grid.

        The ray geometry, deposition weights and soft-min reference times are held
        fixed, which makes the travel times linear in the slowness. The gradient then
        has two parts: the local correction term of Eq. 5 at each deposited ray sample,
        and the accumulated path time, where each slowness sample along a ray affects
        all deposits made later on that ray. The latter needs the total adjoint per
        ray, so the rays are replayed twice.

        Args:
            travel_time_adjoint (Tensor): Adjoint of the flattened travel times.
            t_min (Tensor): Soft-min reference times from :meth:`forward`.
            norm (Tensor): Soft-min normalization from :meth:`forward`.

        Returns:
            Tensor: Gradient w.r.t. the slowness grid of shape ``(nz_s, nx_s)``.
        """
        adjoint_per_norm = ops.where(
            norm > 0, travel_time_adjoint / ops.where(norm > 0, norm, 1.0), 0.0
        )
        zeros_rays = ops.zeros((self.n_el, self.n_rays), "float32")

        # Replay 1: total adjoint deposited by each ray over its whole path.
        def total_body(i, carry):
            state, total = carry
            state, _ = self.step_rays(state)
            dep = self.deposit(state, self.arc_length(i))
            adjoint = self._ray_adjoint(dep, t_min, adjoint_per_norm)
            return state, total + ops.sum(adjoint, axis=-1)

        _, total = ops.fori_loop(0, self.n_steps, total_body, (self.initial_state(), zeros_rays))

        # Replay 2: scatter both gradient terms onto the slowness grid. The slowness
        # sampled during step i enters every deposit from step i onwards, so it receives
        # the remaining (suffix) adjoint of its ray.
        def grad_body(i, carry):
            state, consumed, grad = carry
            state, (qx_mid, qz_mid) = self.step_rays(state)
            remaining = (total - consumed) * self.step
            grad = grad + self._scatter(remaining, self.sos_stencil(qx_mid, qz_mid))

            dep = self.deposit(state, self.arc_length(i))
            adjoint = self._ray_adjoint(dep, t_min, adjoint_per_norm)
            local = ops.sum(adjoint * dep["correction"], axis=-1)
            grad = grad + self._scatter(local, dep["stencil"])
            return state, consumed + ops.sum(adjoint, axis=-1), grad

        carry = (self.initial_state(), zeros_rays, ops.zeros((self.n_sos,), "float32"))
        _, _, grad = ops.fori_loop(0, self.n_steps, grad_body, carry)
        return ops.reshape(grad, (self.nz_s, self.nx_s))


def _to_xz(positions):
    """Project ``(n, 3)`` positions (or ``(n, 2)`` x-z positions) onto the x-z plane."""
    positions = ops.cast(ops.convert_to_tensor(positions), "float32")
    if positions.shape[-1] == 3:
        return ops.stack([positions[:, 0], positions[:, 2]], axis=-1)
    return positions


def straight_ray_travel_times(
    sos_map, sos_grid_x, sos_grid_z, element_positions, points, n_ray_points=100
):
    """Travel times along straight rays through a speed-of-sound map.

    The slowness is averaged over ``n_ray_points`` samples on the straight segment
    between each element and each point, and multiplied by the segment length. This is
    the propagation model of :func:`zea.beamform.beamformer.calculate_delays_heterogeneous_medium`.

    Args:
        sos_map (Tensor): Speed-of-sound map of shape ``(nz_s, nx_s)`` in m/s.
        sos_grid_x (Tensor): Uniformly spaced x-coordinates of the ``sos_map`` columns.
        sos_grid_z (Tensor): Uniformly spaced z-coordinates of the ``sos_map`` rows.
        element_positions (Tensor): Element positions of shape ``(n_el, 3)``.
        points (Tensor): Query points of shape ``(n_pix, 3)``.
        n_ray_points (int, optional): Number of samples along each ray.
            Defaults to ``100``.

    Returns:
        Tensor: Travel times in seconds of shape ``(n_el, n_pix)``.
    """
    sos_map = ops.cast(sos_map, "float32")
    nz_s, nx_s = sos_map.shape
    x0, dx, _ = _uniform_axis(sos_grid_x)
    z0, dz, _ = _uniform_axis(sos_grid_z)
    slowness_flat = ops.reshape(1.0 / sos_map, (-1,))

    elements = _to_xz(element_positions)
    points = _to_xz(points)
    ray_parameters = ops.convert_to_tensor(
        (np.arange(n_ray_points, dtype="float32") + 1.0) / n_ray_points
    )

    n_sos = nz_s * nx_s
    n_el, n_pix = elements.shape[0], points.shape[0]

    # The travel times are linear in the slowness, so the adjoint is written out
    # explicitly. This keeps the loop out of automatic differentiation, which some
    # backends (e.g. TensorFlow with XLA) cannot do for loops of unknown length.
    # All tensors are passed as arguments: a custom gradient must not close over
    # tensors traced outside of it (e.g. by an enclosing jit).
    @ops.custom_gradient
    def _integrate(slowness_flat, elements, points, ray_parameters, x0, dx, z0, dz):
        delta = points[None] - elements[:, None]  # (n_el, n_pix, 2)
        lengths = ops.sqrt(ops.sum(delta**2, axis=-1))

        def _stencil(i):
            p = ray_parameters[i]
            x = elements[:, None, 0] + p * delta[..., 0]
            z = elements[:, None, 1] + p * delta[..., 1]
            return bilinear_stencil(x, z, x0, dx, nx_s, z0, dz, nz_s)

        def _accumulate(i, total):
            return total + _gather_bilinear(slowness_flat, *_stencil(i))

        total = ops.fori_loop(0, n_ray_points, _accumulate, ops.zeros((n_el, n_pix)))
        travel_times = total / n_ray_points * lengths

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            weight = ops.cast(upstream, "float32") * lengths / n_ray_points

            def _scatter(i, grad_slowness):
                indices, weights = _stencil(i)
                return grad_slowness + ops.segment_sum(
                    ops.reshape(weight[..., None] * weights, (-1,)),
                    ops.reshape(indices, (-1,)),
                    num_segments=n_sos,
                )

            grad_slowness = ops.fori_loop(0, n_ray_points, _scatter, ops.zeros((n_sos,), "float32"))
            return (grad_slowness,) + tuple(
                ops.zeros_like(t) for t in (elements, points, ray_parameters, x0, dx, z0, dz)
            )

        return travel_times, grad

    return _integrate(slowness_flat, elements, points, ray_parameters, x0, dx, z0, dz)


def compute_bent_ray_travel_times(
    sos_map,
    sos_grid_x,
    sos_grid_z,
    element_positions,
    grid_x,
    grid_z,
    element_normals=None,
    n_rays=256,
    max_angle=np.deg2rad(60.0),
    step_size=None,
    support_radius=1.0,
    sigma=0.5,
    softmin_scale=50e-9,
    fill_unreached=True,
):
    r"""One-way bent-ray travel times from each element to a regular grid.

    Implements the UltraBend propagation model; see the module documentation of
    :mod:`zea.beamform.bent_ray` for the method. The function is differentiable with
    respect to ``sos_map``. Its gradient is computed with a replay adjoint, so memory
    use does not grow with the number of ray steps.

    .. note::

        The rays must be dense enough to reach every grid node: at distance
        :math:`r` from an element neighboring rays are :math:`r \cdot 2\,
        \text{max\_angle} / (n_\text{rays} - 1)` apart, which should stay below about
        twice the support radius (``support_radius`` times the grid spacing). Nodes
        that no ray reaches use the straight-ray travel time when ``fill_unreached`` is
        set.

    Args:
        sos_map (Tensor): Speed-of-sound map of shape ``(nz_s, nx_s)`` in m/s.
        sos_grid_x (Tensor): Uniformly spaced x-coordinates of the ``sos_map``
            columns. Outside the map, the speed of sound is extrapolated with nearest
            neighbor.
        sos_grid_z (Tensor): Uniformly spaced z-coordinates of the ``sos_map`` rows.
        element_positions (Tensor): Element positions of shape ``(n_el, 3)``.
        grid_x (Tensor): Uniformly spaced x-coordinates of the travel-time grid.
        grid_z (Tensor): Uniformly spaced z-coordinates of the travel-time grid.
        element_normals (Tensor, optional): Unit element normals of shape
            ``(n_el, 3)``, the center of each element's fan of rays. Defaults to
            ``None``, in which case they are derived from ``element_positions`` with
            :func:`zea.beamform.geometry.compute_element_normals` (``+z`` for a flat
            array).
        n_rays (int, optional): Number of rays launched per element.
            Defaults to ``256``.
        max_angle (float, optional): Half opening angle in radians of the fan of
            rays, relative to the element normal. Defaults to 60 degrees.
        step_size (float, optional): Ray-marching step in meters. Defaults to
            ``None``, which uses half the travel-time grid spacing.
        support_radius (float, optional): Radius around each ray sample within which
            grid nodes receive an arrival-time estimate, in units of the travel-time
            grid spacing. Defaults to ``1.0``.
        sigma (float, optional): Standard deviation of the Gaussian deposition weight,
            in units of the travel-time grid spacing. Defaults to ``0.5``.
        softmin_scale (float, optional): Softness :math:`1/\beta` of the soft-min over
            arrival-time estimates, in seconds. Defaults to ``50e-9``.
        fill_unreached (bool, optional): Replace the travel time of nodes that no ray
            reached by the straight-ray travel time. Otherwise those nodes are ``0``.
            Defaults to ``True``.

    Returns:
        Tensor: Travel times in seconds of shape ``(n_el, nz, nx)``, with ``nz`` and
        ``nx`` the lengths of ``grid_z`` and ``grid_x``.
    """
    sos_map = ops.cast(ops.convert_to_tensor(sos_map), "float32")
    sos_grid_x = ops.cast(ops.convert_to_tensor(sos_grid_x), "float32")
    sos_grid_z = ops.cast(ops.convert_to_tensor(sos_grid_z), "float32")
    grid_x = ops.cast(ops.convert_to_tensor(grid_x), "float32")
    grid_z = ops.cast(ops.convert_to_tensor(grid_z), "float32")
    element_positions = ops.cast(ops.convert_to_tensor(element_positions), "float32")
    if element_normals is None:
        element_normals = compute_element_normals(element_positions)
    element_xz = _to_xz(element_positions)
    normals_xz = _to_xz(element_normals)
    normals_xz = normals_xz / ops.sqrt(ops.sum(normals_xz**2, axis=-1, keepdims=True))

    n_el, nz, nx = element_xz.shape[0], grid_z.shape[0], grid_x.shape[0]
    n_seg = n_el * nz * nx
    tracer_kwargs = dict(
        n_rays=int(n_rays),
        max_angle=float(max_angle),
        step_size=step_size,
        support_radius=float(support_radius),
        sigma=float(sigma),
        softmin_scale=float(softmin_scale),
    )

    @ops.custom_gradient
    def _trace(slowness, sos_grid_x, sos_grid_z, element_xz, normals_xz, grid_x, grid_z):
        tracer = _BentRayTracer(
            ops.stop_gradient(slowness),
            sos_grid_x,
            sos_grid_z,
            element_xz,
            normals_xz,
            grid_x,
            grid_z,
            **tracer_kwargs,
        )
        travel_time, t_min, norm = tracer.forward()

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            # Only the travel times (not the ``reached`` mask) carry a gradient.
            upstream = ops.cast(upstream[:n_seg], "float32")
            grad_slowness = tracer.backward(upstream, t_min, norm)
            return (
                grad_slowness,
                ops.zeros_like(sos_grid_x),
                ops.zeros_like(sos_grid_z),
                ops.zeros_like(element_xz),
                ops.zeros_like(normals_xz),
                ops.zeros_like(grid_x),
                ops.zeros_like(grid_z),
            )

        reached = ops.cast(norm > 0, travel_time.dtype)
        return ops.concatenate([travel_time, reached], axis=0), grad

    traced = _trace(1.0 / sos_map, sos_grid_x, sos_grid_z, element_xz, normals_xz, grid_x, grid_z)
    travel_time = ops.reshape(traced[:n_seg], (n_el, nz, nx))
    reached = ops.stop_gradient(ops.reshape(traced[n_seg:], (n_el, nz, nx))) > 0

    if not fill_unreached:
        return travel_time

    gz, gx = ops.meshgrid(grid_z, grid_x, indexing="ij")
    nodes = ops.stack([ops.reshape(gx, (-1,)), ops.reshape(gz, (-1,))], axis=-1)
    straight = straight_ray_travel_times(sos_map, sos_grid_x, sos_grid_z, element_xz, nodes)
    straight = ops.reshape(straight, (n_el, nz, nx))
    return ops.where(reached, travel_time, straight)


def sample_travel_time_map(travel_time_map, grid_x, grid_z, element_positions, points):
    """Interpolate per-element travel-time maps at arbitrary points.

    Near an element a travel-time map is strongly curved, which makes plain bilinear
    interpolation inaccurate. Therefore, a homogeneous reference travel time
    :math:`s_\\text{ref} \\lVert \\mathbf{x} - \\mathbf{e} \\rVert` (with a per-element
    least-squares fit of :math:`s_\\text{ref}`) is subtracted before interpolation and
    added back exactly afterwards. Points outside the grid are clamped to its border.

    Args:
        travel_time_map (Tensor): Travel times in seconds of shape ``(n_el, nz, nx)``.
        grid_x (Tensor): Uniformly spaced x-coordinates of the map columns.
        grid_z (Tensor): Uniformly spaced z-coordinates of the map rows.
        element_positions (Tensor): Element positions of shape ``(n_el, 3)``.
        points (Tensor): Query points of shape ``(n_pix, 3)``.

    Returns:
        Tensor: Travel times in seconds of shape ``(n_el, n_pix)``.
    """
    travel_time_map = ops.cast(travel_time_map, "float32")
    n_el, nz, nx = travel_time_map.shape
    x0, dx, _ = _uniform_axis(grid_x)
    z0, dz, _ = _uniform_axis(grid_z)
    elements = _to_xz(element_positions)
    points = _to_xz(points)

    gz, gx = ops.meshgrid(ops.cast(grid_z, "float32"), ops.cast(grid_x, "float32"), indexing="ij")
    node_distance = ops.sqrt(
        (gx[None] - elements[:, 0, None, None]) ** 2 + (gz[None] - elements[:, 1, None, None]) ** 2
    )
    reference_slowness = ops.stop_gradient(
        ops.sum(travel_time_map * node_distance, axis=(1, 2))
        / ops.maximum(ops.sum(node_distance**2, axis=(1, 2)), 1e-12)
    )
    residual = travel_time_map - reference_slowness[:, None, None] * node_distance
    residual = ops.reshape(ops.transpose(residual, (1, 2, 0)), (nz * nx, n_el))

    stencil = bilinear_stencil(points[:, 0], points[:, 1], x0, dx, nx, z0, dz, nz)
    residual_at_points = ops.transpose(_gather_bilinear(residual, *stencil))  # (n_el, n_pix)
    point_distance = ops.sqrt(ops.sum((points[None, :, :] - elements[:, None, :]) ** 2, axis=-1))
    return residual_at_points + reference_slowness[:, None] * point_distance
