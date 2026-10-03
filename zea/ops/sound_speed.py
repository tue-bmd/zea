"""Operations for propagation modeling in heterogeneous speed-of-sound media."""

import numpy as np
from keras import ops

from zea.beamform.bent_ray import compute_bent_ray_travel_times
from zea.internal.registry import ops_registry
from zea.ops.base import Operation

__all__ = ["BentRayTravelTimes"]


@ops_registry("bent_ray_travel_times")
class BentRayTravelTimes(Operation):
    """Refraction-aware travel times from each element through a speed-of-sound map.

    Traces bent rays from every element through ``sos_map`` and returns the one-way
    travel time from each element to a regular grid (``travel_time_map`` of shape
    ``(n_el, Nz, Nx)``), using the UltraBend propagation model (see
    :func:`zea.beamform.bent_ray.compute_bent_ray_travel_times`).

    .. citation:: duelmer2026ultrabend

    :class:`TOFCorrection` picks up ``travel_time_map`` and interpolates the delays
    of every pixel from it, so the rays are traced once per speed-of-sound map rather
    than once per pixel patch. Place this operation *before* the
    :class:`PatchedGrid`::

        Pipeline(
            [
                BentRayTravelTimes(),
                PatchedGrid([TOFCorrection(), CommonMidpointPhaseError()]),
                ReshapeGrid(),
            ]
        )

    The travel times are differentiable with respect to ``sos_map`` (using a replay
    adjoint), so the pipeline above can be used for speed-of-sound estimation by
    autofocusing, with refraction taken into account.

    The travel-time grid is given by ``travel_time_grid_x`` and ``travel_time_grid_z``
    if those are passed to the pipeline. Otherwise, it spans the speed-of-sound grid,
    refined ``grid_upsample`` times. Pixels outside the travel-time grid get the travel
    time of the nearest grid point at the border, so make sure the grid covers the
    field of view.

    .. important::
        Only for multistatic datasets (``n_tx == n_el``), e.g. synthetic aperture data,
        and 2-D imaging in the x-z plane.
    """

    # The travel-time map is not ``data``: it is always written to these keys, which
    # TOFCorrection reads.
    OUTPUT_KEYS = ["travel_time_map", "travel_time_grid_x", "travel_time_grid_z"]

    def __init__(
        self,
        n_rays=256,
        max_angle=np.deg2rad(60.0),
        step_size=None,
        support_radius=1.0,
        sigma=0.5,
        softmin_scale=50e-9,
        grid_upsample=4,
        fill_unreached=True,
        **kwargs,
    ):
        """
        Args:
            n_rays (int): Number of rays launched per element. Defaults to ``256``.
            max_angle (float): Half opening angle in radians of the fan of rays
                around each element's normal. Defaults to 60 degrees.
            step_size (float or None): Ray-marching step in meters. Defaults to
                ``None``, which uses half the travel-time grid spacing.
            support_radius (float): Radius around each ray sample, in travel-time grid
                cells, in which grid nodes receive an arrival-time estimate.
                Defaults to ``1.0``.
            sigma (float): Standard deviation of the Gaussian deposition weight in
                travel-time grid cells. Defaults to ``0.5``.
            softmin_scale (float): Softness of the soft-min over arrival-time
                estimates in seconds. Defaults to ``50e-9``.
            grid_upsample (int): Refinement of the speed-of-sound grid that gives the
                default travel-time grid. Defaults to ``4``.
            fill_unreached (bool): Use straight-ray travel times for grid nodes that
                no ray reaches. Defaults to ``True``.
        """
        super().__init__(**kwargs)
        self.n_rays = n_rays
        self.max_angle = max_angle
        self.step_size = step_size
        self.support_radius = support_radius
        self.sigma = sigma
        self.softmin_scale = softmin_scale
        self.grid_upsample = grid_upsample
        self.fill_unreached = fill_unreached

    @property
    def output_keys(self):
        """The keys this operation writes."""
        return list(self.OUTPUT_KEYS)

    def _refine(self, coords):
        n = (ops.shape(coords)[0] - 1) * self.grid_upsample + 1
        return ops.linspace(coords[0], coords[-1], n)

    def call(
        self,
        sos_map=None,
        sos_grid_x=None,
        sos_grid_z=None,
        probe_geometry=None,
        travel_time_grid_x=None,
        travel_time_grid_z=None,
        **kwargs,
    ):
        """Compute the travel-time map.

        Args:
            sos_map (Tensor): Speed-of-sound map of shape ``(Nz_s, Nx_s)`` in m/s.
            sos_grid_x (Tensor): Uniformly spaced x-coordinates of the ``sos_map``
                columns.
            sos_grid_z (Tensor): Uniformly spaced z-coordinates of the ``sos_map``
                rows.
            probe_geometry (Tensor): Element positions of shape ``(n_el, 3)``.
            travel_time_grid_x (Tensor, optional): Uniformly spaced x-coordinates of
                the travel-time grid.
            travel_time_grid_z (Tensor, optional): Uniformly spaced z-coordinates of
                the travel-time grid.

        Returns:
            dict: ``travel_time_map`` of shape ``(n_el, Nz, Nx)`` in seconds, and the
            ``travel_time_grid_x`` and ``travel_time_grid_z`` it is defined on.
        """
        if sos_map is None or sos_grid_x is None or sos_grid_z is None:
            raise ValueError(
                "BentRayTravelTimes requires `sos_map`, `sos_grid_x` and `sos_grid_z`."
            )
        if travel_time_grid_x is None:
            travel_time_grid_x = self._refine(ops.cast(sos_grid_x, "float32"))
        if travel_time_grid_z is None:
            travel_time_grid_z = self._refine(ops.cast(sos_grid_z, "float32"))

        travel_time_map = compute_bent_ray_travel_times(
            sos_map,
            sos_grid_x,
            sos_grid_z,
            probe_geometry,
            travel_time_grid_x,
            travel_time_grid_z,
            n_rays=self.n_rays,
            max_angle=self.max_angle,
            step_size=self.step_size,
            support_radius=self.support_radius,
            sigma=self.sigma,
            softmin_scale=self.softmin_scale,
            fill_unreached=self.fill_unreached,
        )
        return {
            "travel_time_map": travel_time_map,
            "travel_time_grid_x": travel_time_grid_x,
            "travel_time_grid_z": travel_time_grid_z,
        }
