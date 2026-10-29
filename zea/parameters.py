"""Structure containing the parameters defining an ultrasound acquisition.

This module provides the :class:`Parameters` class, a flexible structure
for managing all parameters related to an ultrasound acquisition (merged probe
and scan parameters).

Features
^^^^^^^^

- **Flexible initialization:** The :class:`Parameters` class supports lazy initialization,
  allowing you to specify any combination of supported parameters. You can pass only
  the parameters you have, and the rest will be computed or set to defaults as needed.

- **Automatic computation:** Many scan properties (such as
  grid, number of pixels, wavelength, etc.) are computed automatically from the
  provided parameters. This enables you to work with minimal input and still obtain
  all necessary scan configuration details.

- **Dependency tracking and lazy evaluation:** Derived properties are computed only
  when accessed, and are automatically invalidated and recomputed if their dependencies
  change. This ensures efficient memory usage and avoids unnecessary computations.

- **Parameter validation:** All parameters are type-checked and validated against
  a predefined schema, reducing errors and improving robustness.

- **Selection of transmits:** The scan supports flexible selection of transmit events,
  using the :meth:`set_transmits` method. You can select all, a specific number,
  or specific transmit indices. The selection is stored and can be accessed via
  the :attr:`selected_transmits` property.

Comparison to ``zea.Config`` and ``zea.Probe``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :class:`zea.config.Config`: A general-purpose parameter dictionary for experiment and pipeline
  configuration. It is not specific to ultrasound acquisition and does not compute
  derived parameters.

- :class:`zea.probes.Probe`: Contains only probe-specific parameters (e.g., geometry, frequency).

- :class:`zea.Parameters`: Combines all parameters relevant to an ultrasound acquisition,
  including probe, acquisition, and scan region. It also provides automatic computation
  of derived properties and dependency management.

Example Usage
^^^^^^^^^^^^^

.. doctest::

    >>> from zea import Config, File, Probe, Parameters

    >>> # The usual entry point: load the merged probe + scan parameters from a file
    >>> path = (
    ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
    ... )
    >>> with File(path) as f:
    ...     parameters = f.load_parameters()
    >>> type(parameters).__name__
    'Parameters'

    >>> # You can also build one from a Probe's parameters ...
    >>> probe = Probe.from_name("verasonics_l11_4v")
    >>> parameters = Parameters(
    ...     probe_geometry=probe.probe_geometry,
    ...     center_frequency=probe.probe_center_frequency,
    ...     element_width=probe.element_width,
    ...     grid_size_z=256,
    ...     n_tx=11,
    ... )

    >>> # ... from a Config's parameters ...
    >>> config = Config.from_path("hf://zeahub/configs/config_picmus_rf.yaml")
    >>> parameters = Parameters(n_tx=11, **config.parameters)

    >>> # ... or fully manually
    >>> parameters = Parameters(
    ...     grid_size_x=128,
    ...     grid_size_z=256,
    ...     xlims=(-0.02, 0.02),
    ...     zlims=(0.0, 0.06),
    ...     ylims=(0.0, 0.0),
    ...     center_frequency=6.25e6,
    ...     sound_speed=1540.0,
    ...     sampling_frequency=25e6,
    ...     n_el=128,
    ...     n_tx=11,
    ...     probe_geometry=probe.probe_geometry,
    ... )

    >>> # Access a derived property (computed lazily)
    >>> grid = parameters.grid  # shape: (grid_size_z, grid_size_x, 3)

    >>> # Select a subset of transmit events
    >>> _ = parameters.set_transmits(3)  # Use 3 evenly spaced transmits
    >>> _ = parameters.set_transmits([0, 2, 4])  # Use specific transmit indices
    >>> _ = parameters.set_transmits("all")  # Use all transmits

"""

from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
from keras import ops

from zea import log
from zea.beamform.geometry import (
    STEERING_TOLERANCE,
    element_acceptance_angle,
    element_pitch,
    insonified_edge_rays,
    rays_at_depth,
    transmit_apex_distance,
)
from zea.beamform.pfield import compute_pfield
from zea.beamform.pixelgrid import (
    cartesian_pixel_grid,
    check_for_aliasing,
    polar_pixel_grid,
    scanline_aligned_apodization,
    scanline_pixel_grid,
)
from zea.data.spec import ProbeSpec, ScanSpec
from zea.display import compute_scan_convert_2d_coordinates
from zea.func.ultrasound import compute_time_to_peak
from zea.internal.parameters import BaseParameters, MissingDependencyError, cache_with_dependencies
from zea.internal.utils import deprecated
from zea.probes import Probe, fit_curved_probe_radius


class Parameters(BaseParameters):
    """Contains and computes all parameters relevant to an ultrasound acquisition.

    A :class:`Parameters` object holds **all** parameters relevant to an
    acquisition — merged probe and scan parameters — and computes derived
    quantities (grid, wavelength, pfield, scan-conversion coordinates, ...)
    lazily with dependency tracking and caching.  Obtain one from a file via
    :meth:`zea.data.file.File.load_parameters`.

    The set of valid file-backed parameters is derived from
    :class:`~zea.data.spec.ScanSpec` and :class:`~zea.data.spec.ProbeSpec`
    (single source of truth), extended with recon/beamforming parameters that
    are not stored in the file.  Arbitrary custom parameters may also be set;
    they are stored as-is and are not used to compute any derived parameters
    (e.g. the beamforming grid). They are simply passed through — for example
    to a pipeline call (see :class:`~zea.internal.parameters.BaseParameters`).

    """

    sound_speed: float
    """Speed of sound in the medium [m/s]. Defaults to 1540.0."""

    sampling_frequency: float
    """Sampling frequency of the received signal [Hz]."""

    center_frequency: float | np.ndarray
    """Transmit center frequency [Hz]. Scalar, or shape (n_tx,) per transmit."""

    probe_center_frequency: float
    """Nominal center frequency of the probe [Hz]."""

    probe_bandwidth_percent: float
    """Probe bandwidth as a percentage of the center frequency. Defaults to 200.0."""

    n_el: int
    """Number of elements in the transducer array."""

    n_ax: int
    """Number of axial samples in the received signal."""

    n_ch: int
    """Number of channels (1 for RF, 2 for IQ data)."""

    n_frames: int
    """Number of frames in the dataset."""

    probe_geometry: np.ndarray
    """Element positions [m], shape (n_el, 3)."""

    element_width: float
    """Width of each transducer element [m]."""

    element_height: float
    """Height (elevation) of each transducer element [m]."""

    selected_transmits: list[int] | None
    """Indices of the currently selected transmit events, or ``None`` when not yet
    resolved (e.g. image-only files where ``n_tx`` is unknown).

    Assigning to this attribute (or calling :meth:`set_transmits`) accepts any of
    the following, which are resolved to concrete indices via :meth:`find_transmits`:

    - ``None`` or ``"all"``: use all transmits. If the total transmit count
      ``n_tx`` is not yet known (image-only files), ``set_transmits(None)`` leaves
      this as ``None`` until ``n_tx`` is set.
    - ``"center"``: use only the center transmit.
    - ``int``: select this many evenly spaced transmits.
    - ``list``/``np.ndarray``: use these specific transmit indices.
    - ``slice``: use transmits given by the slice (e.g. ``slice(0, 10, 2)``).
    """

    waveforms_one_way: np.ndarray | None
    """One-way transmit waveforms, shape (n_tx, n_samples_one_way). None if absent."""

    waveforms_two_way: np.ndarray | None
    """Two-way transmit waveforms, shape (n_tx, n_samples_two_way). None if absent."""

    pixels_per_wavelength: int
    """Number of grid pixels per wavelength. Defaults to 4."""

    grid_type: str
    """Beamforming grid type, ``"cartesian"`` or ``"polar"``. Defaults to
    ``"cartesian"``. Combine with ``enable_scanline=True`` for line-by-line
    (scanline) imaging instead of a shared compounded grid. A polar grid fans out from
    :attr:`grid_apex`."""

    dynamic_range: np.ndarray
    """Dynamic range for image display in dB, shape (2,) as (min_dB, max_dB)."""

    resolution: float | None
    """Resolution for scan conversion [mm/pixel]. If None, computed from the image."""

    polar_margin: float | None
    """Fixed angle [rad] to widen the default :attr:`polar_limits` by on each side,
    replacing the automatic margin derived from the transmits. Defaults to ``None``
    (automatic)."""

    fill_value: float
    """Value used for out-of-bounds pixels during scan conversion."""

    attenuation_coef: float
    """Attenuation coefficient [dB/(MHz*cm)]. Defaults to 0.0."""

    apply_lens_correction: bool
    """Whether to apply lens correction to the transmit delays. Defaults to False."""

    focal_region_length: float
    """Full length in meters of the region around the focal plane of focused transmits.

    In this region the first-arrival and last-arrival transmit delays are linearly blended,
    smoothing the focal-plane transition while preserving the same model outside the region.
    This targets the focal-depth artifact described by Rindal et al. (IUS 2018). Defaults to
    0.0 (disabled, i.e. conventional first/last-arrival switching everywhere). A typical
    starting value is 2e-3 m (corresponding to +/-1e-3 m around the focal plane).
    """

    lens_thickness: float
    """Thickness of the lens [m]."""

    lens_sound_speed: float
    """Speed of sound in the lens material [m/s]."""

    elevation_tilt_limits: tuple | np.ndarray
    """Elevation tilt ``atan(y / z)`` [rad] of the first and last sample of a 3D frustum,
    shape (2,). Used to scan convert 3D volumes, together with :attr:`rlims` and
    :attr:`polar_limits` (the lateral tilt); see :func:`zea.display.scan_convert_3d`."""

    pfield_kwargs: dict
    """Extra keyword arguments for the pressure-field computation.

    See :func:`zea.beamform.pfield.compute_pfield`. Defaults to ``{}``.
    """

    enable_scanline: bool
    """Whether to beamform in scanline (line-by-line) mode: one pixel column
    per transmit instead of a shared compounded ``grid_type`` grid, paired with
    ``flat_aligned_apodization`` (fed to :class:`~zea.ops.AlignedApodization`).
    Defaults to ``False``.

    Scanline imaging needs focused transmits (every focus distance > 0), and the grid
    type must match how the lines run:

    - ``grid_type="polar"`` for steered lines whose axes meet in one point (sector /
      phased-array or curved-array geometry): the polar grid with one column per
      transmit angle, fanning out from :attr:`grid_apex`. :attr:`polar_limits` is the
      range of the transmit angles.
    - ``grid_type="cartesian"`` for unsteered, parallel lines (linear-scan geometry):
      a vertical column under each transmit (see
      :func:`~zea.beamform.pixelgrid.scanline_pixel_grid`).

    Any other combination raises a ``ValueError`` when the grid is built.

    .. note::
        This only builds the scanline *grid*. For true classical scanline
        beamforming, also construct the beamformer with
        ``Beamform(enable_aligned_apodization=True)`` so that
        ``flat_aligned_apodization`` masks each pixel to its owning transmit;
        without it every transmit is still compounded onto the
        one-column-per-transmit grid.
    """

    flat_receive_apodization: np.ndarray
    """Optional custom receive-aperture apodization of shape ``(n_pix, n_el)``.

    Per-pixel, per-element (receive-channel) weights, fed to
    :class:`~zea.ops.ReceiveApodization` when the beamformer is built with
    ``Beamform(enable_receive_apodization=True)``. This is the user-supplied
    counterpart of the built-in f-number mask
    (:func:`~zea.beamform.beamformer.fnumber_mask`) and is applied *in addition*
    to it (set ``f_number=0`` to use a fully custom receive apodization alone).
    ``None`` (default) makes :class:`~zea.ops.ReceiveApodization` a no-op.

    Distinct from ``flat_aligned_apodization``, which weights the *transmit*
    axis (compounding), not the receive channels.
    """

    scan_schema = deepcopy(ScanSpec.SCHEMA)
    probe_schema = deepcopy(Probe.SCHEMA)
    for key in Probe._NON_PARAMETERS:
        probe_schema.pop(key)

    # Valid parameters are derived from the scan and probe specs + a few
    # beamforming-only parameters.
    VALID_PARAMS: ClassVar[dict[str, dict[str, Any]]] = {
        **scan_schema,
        **probe_schema,
        "grid_size_x": {"dtype": np.int32},
        "grid_size_y": {"dtype": np.int32},
        "grid_size_z": {"dtype": np.int32},
        "xlims": {"dtype": np.float32, "shape": (2,)},
        "ylims": {"dtype": np.float32, "shape": (2,)},
        "zlims": {"dtype": np.float32, "shape": (2,)},
        "pixels_per_wavelength": {"dtype": np.int32, "default": 4},
        "pfield_kwargs": {"dtype": dict, "default": {}},
        "apply_lens_correction": {"dtype": bool, "default": False},  # native dtype on purpose
        "enable_scanline": {"dtype": bool, "default": False},  # native dtype on purpose
        "flat_aligned_apodization": {"dtype": (type(None), np.ndarray)},
        "flat_receive_apodization": {"dtype": (type(None), np.ndarray), "default": None},
        "focal_region_length": {"dtype": np.float32, "default": 0.0},
        "grid_type": {"dtype": str, "default": "cartesian"},
        "polar_limits": {"dtype": np.float32, "shape": (2,)},
        "dynamic_range": {"dtype": np.float32, "shape": (2,)},
        "selected_transmits": {
            "dtype": (type(None), str, int, list, slice, np.ndarray),
            "default": None,
        },
        "n_frames": {"dtype": np.int32},
        "n_el": {"dtype": np.int32},
        "n_tx": {"dtype": np.int32},
        "n_ax": {"dtype": int},  # native dtype on purpose
        "n_ch": {"dtype": np.int32},
        "attenuation_coef": {"dtype": np.float32, "default": 0.0},
        # native dtype on purpose; None derives it from the probe (see the f_number property)
        "f_number": {"dtype": (float, type(None)), "default": None},
        "polar_margin": {"dtype": (np.float32, type(None)), "default": None},
        "t_peak": {"dtype": np.float32},
        "elevation_tilt_limits": {"dtype": np.float32, "shape": (2,)},
        "rlims": {"dtype": np.float32, "shape": (2,)},
        "fill_value": {"dtype": float},
        "resolution": {"dtype": (np.float32, type(None)), "default": None},
    }

    # Former parameters that are now derived. Without this they would be accepted as
    # custom (passthrough) parameters and silently ignored.
    _REMOVED_PARAMS: ClassVar[dict[str, str]] = {
        "distance_to_apex": (
            "`distance_to_apex` is no longer a parameter: the point the polar grid fans out "
            "from is derived from the transmits and the probe. Read `grid_apex` instead "
            "(the old distance_to_apex is -grid_apex[2])."
        ),
    }

    # Former names of parameters, mapped onto the current name with a deprecation warning.
    _RENAMED_PARAMS: ClassVar[dict[str, str]] = {
        "rho_range": "rlims",
        "theta_range": "polar_limits",
        "phi_range": "elevation_tilt_limits",
    }

    @classmethod
    def _renamed(cls, name):
        """The current name of ``name``, warning when it is a deprecated one."""
        if name not in cls._RENAMED_PARAMS:
            return name
        new = cls._RENAMED_PARAMS[name]
        log.deprecated(f"Parameter `{name}` is deprecated, use `{new}` instead.")
        return new

    # Add some defaults that are not stored in a file
    VALID_PARAMS["sound_speed"]["default"] = 1540.0
    VALID_PARAMS["probe_bandwidth_percent"]["default"] = 200.0
    # Give these a default of None (rather than leaving them unset) so that they can be
    # declared as dependencies of computed properties (e.g. t_peak, n_waveforms) without
    # tripping MissingDependencyError
    VALID_PARAMS["waveforms_one_way"]["default"] = None
    VALID_PARAMS["waveforms_two_way"]["default"] = None

    @cache_with_dependencies("probe_geometry")
    def aperture_size(self):
        """Calculate the aperture size (x,y,z) based on the probe geometry."""
        if "probe_geometry" in self._params:
            x_coords = self.probe_geometry[:, 0]
            y_coords = self.probe_geometry[:, 1]
            z_coords = self.probe_geometry[:, 2]
            aperture_width = x_coords.max() - x_coords.min()
            aperture_height = y_coords.max() - y_coords.min()
            aperture_depth = z_coords.max() - z_coords.min()
            return np.array([aperture_width, aperture_height, aperture_depth])
        return None

    def __getattr__(self, item):
        if item in Parameters._REMOVED_PARAMS:
            raise AttributeError(Parameters._REMOVED_PARAMS[item])
        if item in Parameters._RENAMED_PARAMS:
            return getattr(self, Parameters._renamed(item))
        return super().__getattr__(item)

    def update(self, params=None, *, force=False, **kwargs):
        """Update parameters from a mapping and/or keyword arguments.

        See :meth:`zea.internal.parameters.BaseParameters.update`.
        """
        merged = {}
        for key, value in {**(dict(params) if params else {}), **kwargs}.items():
            if key in self._REMOVED_PARAMS:
                raise ValueError(self._REMOVED_PARAMS[key])
            merged[self._renamed(key)] = value
        super().update(merged, force=force)

    @cache_with_dependencies(optional=("probe_geometry",))
    def radius_of_curvature(self):
        """The probe's radius of curvature [m], fitted from :attr:`probe_geometry`.

        ``inf`` for a flat array, or when there is no probe geometry. This is a property
        of the probe; for a curved array it is also the default distance from the array
        to :attr:`grid_apex`, the centre of curvature.
        """
        probe_geometry = self._params.get("probe_geometry")
        if probe_geometry is None:
            return np.float32(np.inf)
        try:
            return np.float32(fit_curved_probe_radius(probe_geometry))
        except ValueError:  # not a curved array
            return np.float32(np.inf)

    def _transmit_origins_or_zeros(self):
        """Transmit origins for the geometry defaults. Zeros when the file has none, as the
        beamformer assumes, without repeating its missing-value warning."""
        if self._params.get("transmit_origins") is None:
            return np.zeros((self.n_tx, 3))
        return self.transmit_origins

    def _transmit_or_probe_apex(self):
        """Distance [m] behind the middle of the array of the point the transmits fan out
        from or, when they define none, of a curved array's centre of curvature.

        ``None`` when neither defines one. See
        :func:`~zea.beamform.geometry.transmit_apex_distance`.
        """
        angles = self.polar_angles
        probe_geometry = self._params.get("probe_geometry")
        if angles is not None and probe_geometry is not None:
            distance = transmit_apex_distance(
                self._transmit_origins_or_zeros(),
                angles,
                self.focus_distances,
                tol=0.5 * element_pitch(probe_geometry),
            )
            if distance is not None:
                return distance
        if np.isfinite(self.radius_of_curvature):
            return float(self.radius_of_curvature)
        return None

    def _element_normals(self):
        """Exact element normals for a curved array, pointing away from its centre of
        curvature; ``None`` (estimate them) otherwise."""
        radius = float(self.radius_of_curvature)
        if not np.isfinite(radius):
            return None
        centre = np.array([0.0, 0.0, -radius])
        return (np.asarray(self.probe_geometry, dtype=np.float64) - centre) / radius

    @cache_with_dependencies(
        "polar_limits",
        "radius_of_curvature",
        optional=("probe_geometry", "polar_angles", "focus_distances", "transmit_origins"),
    )
    def grid_apex(self):
        """The point ``(x, y, z)`` [m] a polar grid fans out from, of shape (3,).

        It lies on the central axis, at or behind the array: ``(0, 0, -d)`` with
        ``d >= 0``. Radii are measured from it, so ``rlims = zlims - grid_apex[2]``.
        It is derived, not set: the grid fans out from where the transmits fan out from.
        In order of precedence:

        1. Steered focused transmit axes that meet in one point: that point. This is the middle
           of the array for transmits launched from there, a virtual apex behind a flat
           array whose transmits are spread over it, and the centre of curvature for
           transmits along a curved array's normals.
        2. Unsteered diverging waves that share one virtual source: that source.
        3. A curved array: its centre of curvature (see :attr:`radius_of_curvature`).
        4. A flat array with a sector (non-zero :attr:`polar_limits`): the virtual apex
           whose sector edges pass through the ends of the array,
           ``d = (aperture / 2) / tan(max |polar_limits|)``.
        5. Otherwise the middle of the array.

        Missing ``transmit_origins`` count as zeros, as in the beamformer.
        """
        distance = self._transmit_or_probe_apex()
        if distance is None:
            limits = self.polar_limits
            max_angle = 0.0 if limits is None else float(np.max(np.abs(limits)))
            probe_geometry = self._params.get("probe_geometry")
            if probe_geometry is not None and 0 < max_angle < np.pi / 2:
                half_aperture = float(np.ptp(probe_geometry[:, 0])) / 2
                distance = half_aperture / np.tan(max_angle)
            else:
                distance = 0.0
        # Adding 0.0 turns -0.0 (from negating a zero distance) into 0.0
        return np.array([0.0, 0.0, -distance], dtype=np.float32) + np.float32(0.0)

    @cache_with_dependencies(
        "xlims",
        "ylims",
        "zlims",
        "grid_size_x",
        "grid_size_z",
        "grid_size_y",
        "grid_type",
        "is_3d",
        "polar_limits",
        "grid_apex",
        "transmit_origins",
        "focus_distances",
        "polar_angles",
        "azimuth_angles",
        "enable_scanline",
    )
    def grid(self):
        """The beamforming grid of shape (grid_size_z, grid_size_x, [grid_size_y], 3)."""
        if self.enable_scanline:
            # One pixel column per transmit: scanline imaging is the special case
            # of pixel-based DAS where each pixel belongs to exactly one transmit.
            polar_angles = self._scanline_polar_angles()
            if self.grid_type == "polar":
                # The polar grid at the transmit angles. Its rows sit at one radius from
                # grid_apex, where the transmit axes meet, so scan conversion places them.
                return polar_pixel_grid(
                    self.polar_limits,
                    self.zlims,
                    self.grid_size_z,
                    self.n_tx,
                    grid_apex=self.grid_apex,
                    polar_angles=polar_angles,
                )
            return scanline_pixel_grid(
                self.transmit_origins,
                self.focus_distances,
                polar_angles,
                self.zlims,
                self.grid_size_z,
                azimuth_angles=self.azimuth_angles,
                grid_type="cartesian",
            )
        elif self.grid_type == "polar":
            if self.is_3d:
                raise NotImplementedError(
                    "3D polar grids are not yet supported. Set grid_type='cartesian', "
                    "or drop ylims/grid_size_y to build a 2D polar grid."
                )
            return polar_pixel_grid(
                self.polar_limits,
                self.zlims,
                self.grid_size_z,
                self.grid_size_x,
                grid_apex=self.grid_apex,
            )
        elif self.grid_type == "cartesian":
            grid = cartesian_pixel_grid(
                self.xlims,
                self.zlims,
                self.ylims,
                grid_size_z=self.grid_size_z,
                grid_size_x=self.grid_size_x,
                grid_size_y=self.grid_size_y,
            )
            try:
                check_for_aliasing(self)
            except MissingDependencyError:
                # No wavelength (e.g. missing center frequency); cannot assess aliasing.
                pass
            return grid
        else:
            raise ValueError(
                f"Unsupported grid type: {self.grid_type}. Supported types are "
                "'cartesian' and 'polar'."
            )

    def _scanline_polar_angles(self):
        """The transmit angles the columns of a scanline grid follow, after checking that
        the transmits suit scanline imaging (see :attr:`enable_scanline`)."""
        if self.grid_type not in ("cartesian", "polar"):
            raise ValueError(
                f"Unsupported grid type: {self.grid_type}. Supported types are "
                "'cartesian' and 'polar'."
            )
        focus = np.asarray(self.focus_distances, dtype=np.float64)
        if not np.all(np.isfinite(focus) & (focus > 0)):
            raise ValueError(
                "enable_scanline=True needs focused transmits (every focus distance > 0), "
                f"got focus_distances from {focus.min():.4g} to {focus.max():.4g} m. Plane "
                "and diverging waves insonify more than one line; image them with a regular "
                "grid (enable_scanline=False)."
            )
        polar_angles = self.polar_angles
        if polar_angles is None:
            log.warning_once(
                "No ``polar_angles`` provided, using zeros",
                key=(id(self), "polar_angles"),
            )
            polar_angles = np.zeros(self.n_tx)
        steered = bool(np.any(np.abs(polar_angles) > STEERING_TOLERANCE))
        if self.grid_type == "cartesian" and steered:
            raise ValueError(
                "A cartesian scanline grid needs unsteered, parallel transmits, but these "
                "are steered. Use grid_type='polar' for steered lines that meet in one point."
            )
        if self.grid_type == "polar":
            if not steered:
                raise ValueError(
                    "A polar scanline grid needs steered transmits whose axes meet in one "
                    "point, but these are unsteered and parallel. Use grid_type='cartesian'."
                )
            apex = transmit_apex_distance(
                self.transmit_origins,
                polar_angles,
                focus,
                tol=0.5 * element_pitch(self.probe_geometry),
            )
            if apex is None:
                raise ValueError(
                    "A polar scanline grid needs transmit axes that meet in one point (the "
                    "grid apex), but these do not. Use a regular grid "
                    "(enable_scanline=False)."
                )
        return polar_angles

    @cache_with_dependencies("enable_scanline", "n_tx", "grid_size_z", "selected_transmits")
    def flat_aligned_apodization(self):
        """Per-pixel, per-transmit compounding apodization weight of shape (n_pix, n_tx).

        Can be set explicitly to any weighting, in which case it is stored over
        the full transmit axis and read back sliced by ``selected_transmits``.
        This requires ``enable_scanline`` to be ``False``: a scanline grid has
        one column per *selected* transmit, so its pixel axis is defined by the
        transmit selection and a mask stored over the full transmit axis cannot
        stay aligned with it.
        When left unset, it is the one-hot mask (see
        :func:`~zea.beamform.pixelgrid.scanline_aligned_apodization`) that
        isolates each pixel's owning transmit if ``enable_scanline`` is ``True``,
        and ``None`` otherwise, which makes :class:`~zea.ops.AlignedApodization`
        a no-op.

        This weights the *transmit* axis (compounding), not the receive channels;
        for custom receive-aperture apodization see ``flat_receive_apodization``.
        """
        value = self._params.get("flat_aligned_apodization")
        if value is not None:
            if self.enable_scanline:
                raise ValueError(
                    "``flat_aligned_apodization`` cannot be set explicitly when "
                    "``enable_scanline`` is True: the scanline grid holds one column per "
                    "selected transmit, so its pixel axis follows the transmit selection and "
                    "a mask stored over the full transmit axis cannot stay aligned with it. "
                    "Set ``enable_scanline=False`` to supply a custom mask."
                )
            _ = self.n_tx  # raises a clear error if no transmit selection is resolved yet
            if value.shape[1] != self.n_tx_total:
                raise ValueError(
                    "``flat_aligned_apodization`` is stored over the full transmit axis, so "
                    f"it must have {self.n_tx_total} columns, got shape {value.shape}."
                )
            return value[:, self.selected_transmits]
        if not self.enable_scanline:
            return None
        return scanline_aligned_apodization(self.n_tx, self.grid_size_z)

    @cache_with_dependencies(
        "xlims",
        "wavelength",
        "pixels_per_wavelength",
        "enable_scanline",
        "n_tx",
        "grid_type",
        optional=("polar_limits", "grid_apex", "zlims"),
    )
    def grid_size_x(self):
        """Grid width in pixels, set to prevent aliasing if not provided.

        For a cartesian grid, the number of lateral (x) pixels across :attr:`xlims`. For a
        polar grid, the number of rays, sized by the outer arc length
        ``(zlims[1] - grid_apex[2]) * (polar_limits[1] - polar_limits[0])``. When
        ``enable_scanline`` is ``True``, this is fixed to ``n_tx`` (one column per
        transmit).
        """
        if self.enable_scanline:
            return self.n_tx

        grid_size_x = self._params.get("grid_size_x")
        if grid_size_x is not None:
            return grid_size_x

        pixel = self.wavelength / self.pixels_per_wavelength
        limits = self.polar_limits if self.grid_type == "polar" else None
        if limits is not None:
            outer_radius = float(np.max(self.zlims)) - float(self.grid_apex[2])
            width = outer_radius * abs(float(limits[1]) - float(limits[0]))
        else:
            width = self.xlims[1] - self.xlims[0]
        return max(int(np.ceil(width / pixel)), 1)

    @cache_with_dependencies(
        "ylims",
        "wavelength",
        "pixels_per_wavelength",
    )
    def grid_size_y(self):
        """Grid height in pixels. For a cartesian grid, this is the vertical (y) pixels in the grid,
        set to prevent aliasing if not provided. For a polar grid, this can be thought of as
        the number for rays in the azimuthal direction.
        """
        grid_size_y = self._params.get("grid_size_y")
        if grid_size_y is not None:
            return grid_size_y

        height = self.ylims[1] - self.ylims[0]
        min_grid_size_y = int(np.ceil(height / (self.wavelength / self.pixels_per_wavelength)))
        return max(min_grid_size_y, 1)

    @cache_with_dependencies(
        "zlims",
        "wavelength",
        "pixels_per_wavelength",
    )
    def grid_size_z(self):
        """Grid depth in pixels. This is the number of axial (z) pixels in the grid,
        set to prevent aliasing if not provided. When ``enable_scanline`` is ``True``,
        this is the number of depth samples per transmit line."""
        grid_size_z = self._params.get("grid_size_z")
        if grid_size_z is not None:
            return grid_size_z

        depth = self.zlims[1] - self.zlims[0]
        min_grid_size_z = int(np.ceil(depth / (self.wavelength / self.pixels_per_wavelength)))
        return max(min_grid_size_z, 1)

    @cache_with_dependencies("sound_speed", "center_frequency")
    def wavelength(self):
        """Calculate the wavelength based on sound speed and transmit center frequency."""
        return self.sound_speed / self.center_frequency

    @cache_with_dependencies(
        optional=("probe_geometry", "element_width", "center_frequency", "sound_speed")
    )
    def f_number(self):
        """Receive f-number: the acceptance cone of each element in the built-in receive
        mask (:func:`~zea.beamform.beamformer.fnumber_mask`). ``0`` disables the mask.

        If not explicitly provided, it follows from the probe: ``1 / (2 tan(alpha))``, with
        ``alpha`` the angle at which an element's directivity drops to -6 dB (see
        :func:`~zea.beamform.geometry.element_acceptance_angle`), at the center frequency.
        The element width falls back to 90 % of the pitch. Elements narrower than about
        0.6 wavelength receive over the whole half-space, which gives 0 (typical for
        phased arrays). Without a probe geometry or center frequency it is 1.0.

        This sets which pixels each element contributes to; it does not change the grid.
        """
        value = self._params.get("f_number")
        if value is not None:
            return value

        probe_geometry = self._params.get("probe_geometry")
        center_frequency = self._params.get("center_frequency")
        if probe_geometry is None or center_frequency is None or len(probe_geometry) < 2:
            return 1.0
        element_width = self._params.get("element_width")
        if element_width is None:
            element_width = 0.9 * element_pitch(probe_geometry)
        wavelength = float(self.sound_speed) / float(np.mean(center_frequency))
        angle = element_acceptance_angle(float(element_width), wavelength)
        if angle >= np.pi / 2:
            return 0.0
        return float(1 / (2 * np.tan(angle)))

    @cache_with_dependencies(
        "zlims",
        "probe_geometry",
        "grid_type",
        "enable_scanline",
        optional=(
            "polar_limits",
            "grid_apex",
            "polar_angles",
            "focus_distances",
            "transmit_origins",
        ),
    )
    def xlims(self):
        """The lateral (x) limits of the grid in meters.

        If not explicitly provided, the limits describe the grid:

        - **Polar grid** (including scanline): the width of the sector, from
          :attr:`grid_apex`, :attr:`zlims` and :attr:`polar_limits`. An explicit value sets
          the width of the scan-converted image instead (crop or pad, see
          :attr:`extent`); it does not change the sector.
        - **Cartesian scanline grid**: the lateral extent of its columns.
        - **Cartesian grid**: what the transmits insonify down to the deepest pixel, and
          at least the probe width. Each transmit's region is bounded by the same edge
          rays that set the default :attr:`polar_limits` (see
          :func:`~zea.beamform.geometry.insonified_edge_rays`).
        """
        xlims = self._params.get("xlims")
        if xlims is not None:
            return xlims

        probe_x = self.probe_geometry[:, 0]
        if self.grid_type == "polar":
            limits = self.polar_limits
            if limits is None:
                return (float(np.min(probe_x)), float(np.max(probe_x)))
            apex = self.grid_apex
            radii = [float(z) - float(apex[2]) for z in self.zlims]
            x = [float(apex[0]) + r * np.sin(float(t)) for r in radii for t in limits]
            return (min(x), max(x))

        focus_distances = self.focus_distances
        polar_angles = self.polar_angles
        if polar_angles is None:
            polar_angles = np.zeros_like(focus_distances)
        origins = self._transmit_origins_or_zeros()

        if self.enable_scanline:
            focus = np.where(np.isfinite(focus_distances), focus_distances, 0.0)
            columns = origins[:, 0] + focus * np.sin(polar_angles)
            return (float(np.min(columns)), float(np.max(columns)))

        points, directions = insonified_edge_rays(
            self.probe_geometry,
            origins,
            polar_angles,
            focus_distances,
            element_normals=self._element_normals(),
        )
        edges_x = rays_at_depth(points, directions, float(np.max(self.zlims)))[:, 0]
        x = np.concatenate([probe_x, edges_x[np.isfinite(edges_x)]])
        return (float(np.min(x)), float(np.max(x)))

    @cache_with_dependencies("zlims", "grid_type", "azimuth_limits", "probe_geometry", "grid_apex")
    def ylims(self):
        """The y-limits of the beamforming grid [m]. If not explicitly set, it is computed based
        on the azimuth limits and probe geometry.
        """
        ylims = self._params.get("ylims")
        if ylims is not None:
            return ylims

        # If ylims not set, compute based on azimuth limits and probe geometry
        radius = max(self.zlims) - float(self.grid_apex[2])
        ylims_azimuth = (
            (0.0, 0.0)  # avoid numerical imprecision with np.cos(np.pi/2)
            if self.azimuth_limits is None or self.azimuth_limits[0] == self.azimuth_limits[1]
            else (
                radius * np.cos(-np.pi / 2 + self.azimuth_limits[0]),
                radius * np.cos(-np.pi / 2 + self.azimuth_limits[1]),
            )
        )
        ylims_plane = (min(self.probe_geometry[:, 1]), max(self.probe_geometry[:, 1]))
        ylims = (
            min(ylims_azimuth[0], ylims_plane[0]),
            max(ylims_azimuth[1], ylims_plane[1]),
        )
        return ylims

    @cache_with_dependencies("sound_speed", "sampling_frequency", "n_ax")
    def zlims(self):
        """The z-limits of the beamforming grid [m]."""
        zlims = self._params.get("zlims")
        if zlims is None:
            return [0, self.sound_speed * self.n_ax / self.sampling_frequency / 2]
        return zlims

    @cache_with_dependencies(
        "grid", "grid_type", optional=("xlims", "zlims", "radius_of_curvature")
    )
    def extent(self):
        """The extent of the image in the format: (xmin, xmax, ymin, ymax, zmin, zmax).

        For a cartesian grid, the extent of the grid itself. For a polar grid, the window
        the scan-converted image fills, which :class:`~zea.ops.ScanConvert` takes from here:

        - ``x`` spans :attr:`xlims`, which defaults to the width of the sector; an explicit
          value crops or pads the image.
        - ``z`` spans the grid, except that on a flat array it starts at ``zlims[0]``. That
          crops the pixels a virtual apex places behind the array. A curved array needs no
          crop, since its sector follows the array.
        """
        # self.grid holds transducer-frame coordinates for every grid type (polar grids
        # fan out from grid_apex), so this needs no correction.
        grid = self.grid
        xmin, xmax = grid[..., 0].min(), grid[..., 0].max()
        ymin, ymax = grid[..., 1].min(), grid[..., 1].max()
        zmin, zmax = grid[..., 2].min(), grid[..., 2].max()

        if self.grid_type == "polar" and grid.ndim == 3:
            xmin, xmax = self.xlims
            if not np.isfinite(self.radius_of_curvature):
                zmin = max(zmin, min(self.zlims))

        return np.array([xmin, xmax, ymin, ymax, zmin, zmax])

    @cache_with_dependencies("extent")
    def extent_imshow(self):
        """The extent of the beamforming grid in the format: (xmin, xmax, ymin, ymax, zmin, zmax).

        Returns:
            np.ndarray: The extent of the beamforming grid in the format (xmin, xmax, zmax, zmin).
                This format can be used directly in matplotlib's ``plt.imshow``.
        """
        xlims_0, xlims_1, ylims_0, ylims_1, zlims_0, zlims_1 = self.extent
        if ylims_0 != ylims_1:
            log.warning("Are you sure you want to use 2D imshow extent for a 3D grid?")
        return np.array([xlims_0, xlims_1, zlims_1, zlims_0])

    @cache_with_dependencies("grid")
    def flatgrid(self):
        """The beamforming grid of shape (grid_size_z*grid_size_x*grid_size_y, 3)."""
        return self.grid.reshape(-1, 3)

    @cache_with_dependencies("grid_size_x", "grid_size_y", "grid_size_z")
    def is_3d(self):
        """Whether the scan grid is 3D (True) or 2D (False)."""
        return self.grid_size_y > 1 and self.grid_size_x > 1 and self.grid_size_z > 1

    @property
    def n_tx_total(self):
        """The total number of transmits in the full dataset."""
        return self._params["n_tx"]

    @cache_with_dependencies("selected_transmits")
    def n_tx(self):
        """The number of currently selected transmits."""
        if self.selected_transmits is None:
            raise ValueError(
                'No transmits selected. Call `set_transmits(None/"all"/...)` to '
                "resolve a selection (this requires the total transmit count `n_tx` "
                "to be set first)."
            )
        return len(self.selected_transmits)

    def set_transmits(self, selection):
        """Set the selected transmits based on a selection.

        Args:
            selection: Specifies which transmits to select:

            - None: Use all transmits
            - "all": Use all transmits
            - "center": Use only the center transmit
            - "focused": Use only focused transmits (positive finite
                ``focus_distances``)
            - "diverging": Use only diverging transmits (negative finite
                ``focus_distances``)
            - "plane": Use only plane wave transmits (``np.inf``, the
                preferred marker, or ``0``)
            - int: Select this many evenly spaced transmits
            - list/array: Use these specific transmit indices
            - slice: Use transmits specified by the slice (e.g., slice(0, 10, 2))

        Returns:
            The current instance for method chaining.
        """
        if selection is None and self._params.get("n_tx") is None:
            # n_tx not yet known (e.g. file with image-only data); store as-is.
            idx = None
        else:
            idx = self.find_transmits(selection)
            if len(idx) == 0:
                log.warning(f"No transmits found for selection '{selection}'.")

        self._params["selected_transmits"] = idx
        self._invalidate("selected_transmits")

        return self

    def find_transmits(self, selection) -> list:
        """Find transmit events based on a selection.

        This method provides flexible ways to select transmit events:

        Args:
            selection: Specifies which transmits to select:

            - None: Use all transmits
            - "all": Use all transmits
            - "center": Use only the center transmit
            - "focused": Use only focused transmits (positive finite
                ``focus_distances``)
            - "diverging": Use only diverging transmits (negative finite
                ``focus_distances``)
            - "plane": Use only plane wave transmits (``np.inf``, the
                preferred marker, or ``0``)
            - int: Select this many evenly spaced transmits
            - list/array: Use these specific transmit indices
            - slice: Use transmits specified by the slice (e.g., slice(0, 10, 2))

        Returns:
            The selected transmit indices.

        Raises:
            ValueError: If the selection is invalid or incompatible with the scan.
        """
        n_tx_total = self._params.get("n_tx")
        if n_tx_total is None:
            raise ValueError("n_tx must be set.")

        # Handle array-like - convert to list of indices
        if isinstance(selection, np.ndarray):
            if len(selection.shape) == 0:
                # Handle scalar numpy array
                selection = int(selection)
            elif len(selection.shape) == 1:
                selection = selection.tolist()
            else:
                raise ValueError(f"Invalid array shape: {selection.shape}")

        # Handle None and "all" - use all transmits
        if selection is None or selection == "all":
            return list(range(n_tx_total))

        # Handle "center" - use center transmit
        if selection == "center":
            return [n_tx_total // 2]

        if selection == "focused":
            value = self._params.get("focus_distances")
            if value is None:
                raise ValueError("No focus distances provided, cannot select focused transmits")
            # Plane waves use inf (or 0); a finite positive focus marks a focused transmit.
            return np.where((value > 0) & np.isfinite(value))[0].tolist()

        if selection == "diverging":
            value = self._params.get("focus_distances")
            if value is None:
                raise ValueError("No focus distances provided, cannot select diverging transmits")
            return np.where(value < 0)[0].tolist()

        if selection == "plane":
            value = self._params.get("focus_distances")
            if value is None:
                raise ValueError("No focus distances provided, cannot select plane wave transmits")
            return np.where((value == 0) | np.isinf(value))[0].tolist()

        # Handle integer - select evenly spaced transmits
        if isinstance(selection, (int, np.integer)):
            selection = int(selection)  # Convert numpy integer to Python int
            if selection <= 0:
                raise ValueError(f"Number of transmits must be positive, got {selection}")

            if selection > n_tx_total:
                raise ValueError(
                    f"Requested {selection} transmits exceeds available transmits ({n_tx_total})"
                )

            if selection == 1:
                return [n_tx_total // 2]
            else:
                # Compute evenly spaced indices
                tx_indices = np.linspace(0, n_tx_total - 1, selection)
                return list(np.rint(tx_indices).astype(int))

        # Handle slice - convert to list of indices
        if isinstance(selection, slice):
            selection = list(range(n_tx_total))[selection]

        # Handle list of indices
        if isinstance(selection, list):
            # Validate indices
            if not all(isinstance(i, (int, np.integer)) for i in selection):
                raise ValueError("All transmit indices must be integers")

            if any(i < 0 or i >= n_tx_total for i in selection):
                raise ValueError(f"Transmit indices must be between 0 and {n_tx_total - 1}")

            # Convert numpy integers to Python ints
            return [int(i) for i in selection]

        raise ValueError(f"Unsupported selection type: {type(selection)}")

    @cache_with_dependencies("center_frequency")
    def demodulation_frequency(self):
        """The demodulation frequency in Hz."""
        if self._params.get("demodulation_frequency") is not None:
            return self._params["demodulation_frequency"]

        return self.center_frequency

    @cache_with_dependencies("selected_transmits")
    def polar_angles(self):
        """Polar angles for each transmit event in radians of shape (n_tx,).
        These angles are often used in 2D imaging."""
        value = self._params.get("polar_angles")
        if value is None:
            return None

        return value[self.selected_transmits]

    @cache_with_dependencies(
        "polar_angles",
        "enable_scanline",
        optional=(
            "polar_margin",
            "probe_geometry",
            "zlims",
            "focus_distances",
            "transmit_origins",
            "radius_of_curvature",
        ),
    )
    def polar_limits(self):
        """The angles [rad] of the first and last column of a polar grid.

        If not explicitly provided, the range of the transmit angles (:attr:`polar_angles`),
        widened on each side to cover what the transmits insonify down to the deepest pixel,
        ``zlims[1]``, as seen from :attr:`grid_apex`. Each transmit's region is bounded by
        two edge rays (see :func:`~zea.beamform.geometry.insonified_edge_rays`): a focused
        transmit adds no margin, a plane wave adds the spread from the ends of the array,
        and a diverging wave the spread from its virtual source through the ends of the
        array. On a flat array whose transmits define no apex (plane or diverging waves),
        the apex is where the outermost edge ray, traced back from its array end, crosses
        the central axis, so the sector covers each region from the array down.
        Set :attr:`polar_margin` to widen by a fixed angle instead.

        With ``enable_scanline=True`` this is exactly the range of the transmit angles,
        since the columns are the transmits; an explicit value is ignored there.
        """
        value = self._params.get("polar_limits")
        angles = self.polar_angles
        if self.enable_scanline and angles is not None:
            if value is not None:
                log.warning_once(
                    "``polar_limits`` is ignored with enable_scanline=True: the columns of a "
                    "scanline grid follow the transmit angles.",
                    key=(id(self), "polar_limits_scanline"),
                )
            return (float(np.min(angles)), float(np.max(angles)))
        if value is not None or angles is None:
            return value

        low, high = float(np.min(angles)), float(np.max(angles))
        margin = self._params.get("polar_margin")
        if margin is not None:
            return (low - float(margin), high + float(margin))

        probe_geometry = self._params.get("probe_geometry")
        try:
            depth = float(np.max(self.zlims))
        except MissingDependencyError:
            depth = None
        if probe_geometry is None or depth is None:
            return (low, high)

        points, directions = insonified_edge_rays(
            probe_geometry,
            self._transmit_origins_or_zeros(),
            angles,
            self.focus_distances,
            element_normals=self._element_normals(),
        )
        # Seen from the apex the transmits define or, on a flat array, from the virtual apex
        # whose sector edge runs along the outermost edge ray through the array end (where
        # grid_apex places it). Without either, from the array middle.
        apex = self._transmit_or_probe_apex()
        if apex is None:
            max_angle = float(np.max(np.abs(np.arctan2(directions[:, 0], directions[:, 1]))))
            half_aperture = float(np.ptp(probe_geometry[:, 0])) / 2
            apex = half_aperture / np.tan(max_angle) if 0 < max_angle < np.pi / 2 else 0.0
        reached = rays_at_depth(points, directions, depth)
        if apex > 0:
            # The start points bound the near field; from the array middle itself they
            # would lie at +-90 degrees.
            reached = np.concatenate([points, reached])
        edge_angles = np.arctan2(reached[:, 0], reached[:, 1] + apex)
        edge_angles = edge_angles[np.isfinite(edge_angles)]
        return (
            float(np.min(edge_angles, initial=low)),
            float(np.max(edge_angles, initial=high)),
        )

    @cache_with_dependencies("selected_transmits", "n_tx")
    def azimuth_angles(self):
        """Azimuth angles for each transmit event in radians
        of shape (n_tx,). These angles are often used in 3D imaging."""
        value = self._params.get("azimuth_angles")
        if value is None:
            log.warning_once(
                "No ``azimuth_angles`` provided, using zeros",
                key=(id(self), "azimuth_angles"),
            )
            return np.zeros(self.n_tx)

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits", "n_tx")
    def azimuth_limits(self):
        """The limits of the azimuth angles.

        Returns ``None`` if azimuth angles were
        not provided (e.g. standard 2D imaging).

        The presence check uses the raw parameter to avoid triggering the
        :attr:`azimuth_angles` "using zeros" fallback warning for 2D data.
        """
        value = self._params.get("azimuth_limits")
        if value is None and self._params.get("azimuth_angles") is not None:
            azimuth_angles = self.azimuth_angles
            value = azimuth_angles.min(), azimuth_angles.max()
            diff = value[1] - value[0]
            # add 15% margin to the limits
            value = (value[0] - 0.15 * diff, value[1] + 0.15 * diff)
        return value

    @cache_with_dependencies("selected_transmits", "n_el", "n_tx")
    def t0_delays(self):
        """Transmit delays in seconds of
        shape (n_tx, n_el), shifted such that the smallest delay is 0."""
        value = self._params.get("t0_delays")
        if value is None:
            log.warning_once(
                "No ``t0_delays`` provided, using zeros",
                key=(id(self), "t0_delays"),
            )
            return np.zeros((self.n_tx, self.n_el))

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits", "n_el", "n_tx")
    def tx_apodizations(self):
        """Transmit apodizations of shape (n_tx, n_el)."""
        value = self._params.get("tx_apodizations")
        if value is None:
            log.warning_once(
                "No ``tx_apodizations`` provided, using ones",
                key=(id(self), "tx_apodizations"),
            )
            return np.ones((self.n_tx, self.n_el))

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def rx_aperture_indices(self):
        """Probe element index of each receive channel of shape (n_tx, n_rx).

        ``None`` for the usual case where every element receives on every transmit,
        in which case the mapping is the identity and :attr:`n_rx` equals
        :attr:`n_el`. Only acquisitions with a receive sub-aperture (a sliding
        receive window, or a multiplexed front-end with fewer channels than
        elements) carry it, since the mapping is then not recoverable from the
        channel data alone. Sliced by :attr:`selected_transmits`, so it stays
        aligned with :attr:`t0_delays` and :attr:`tx_apodizations`.

        The transmit counterpart needs no such field: the transmit aperture is
        recorded per element in :attr:`tx_apodizations`, whose zeros mark the
        elements that did not fire.
        """
        value = self._params.get("rx_aperture_indices")
        if value is None:
            return None

        return value[self.selected_transmits]

    @cache_with_dependencies("rx_aperture_indices", "n_el")
    def n_rx(self):
        """Number of receive channels in the channel data.

        Equal to :attr:`n_el` unless the acquisition used a receive sub-aperture,
        which :attr:`rx_aperture_indices` then maps onto the probe's elements.
        """
        indices = self.rx_aperture_indices
        if indices is None:
            return self.n_el

        return int(indices.shape[1])

    @cache_with_dependencies("selected_transmits", "n_tx")
    def focus_distances(self):
        """Focus distances in meters for each event of shape (n_tx,).

        The sign and magnitude encode the transmit type: a positive finite value
        is a focused transmit (focal point that distance in front of the array),
        a negative finite value is a diverging transmit (virtual source that
        distance behind the array), and ``np.inf`` is a plane wave. ``np.inf`` is
        the preferred, canonical plane-wave value in zea; ``0.0`` is also accepted
        (see :meth:`find_transmits`). When no focus distances are provided, zeros
        are used (i.e. treated as plane waves).
        """
        value = self._params.get("focus_distances")
        if value is None:
            log.warning_once(
                "No ``focus_distances`` provided, using zeros",
                key=(id(self), "focus_distances"),
            )
            return np.zeros(self.n_tx)

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits", "n_tx")
    def transmit_origins(self):
        """Transmit origins of shape (n_tx, 3)."""
        value = self._params.get("transmit_origins")
        if value is None:
            log.warning_once(
                "No ``transmit_origins`` provided, using zeros",
                key=(id(self), "transmit_origins"),
            )
            return np.zeros((self.n_tx, 3))

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits", "n_tx")
    def initial_times(self):
        """Initial times in seconds for each event of shape (n_tx,)."""
        value = self._params.get("initial_times")
        if value is None:
            log.warning_once(
                "No ``initial_times`` provided, using zeros",
                key=(id(self), "initial_times"),
            )
            return np.zeros(self.n_tx)

        return value[self.selected_transmits]

    @cache_with_dependencies("waveforms_one_way", "waveforms_two_way")
    def n_waveforms(self):
        """The number of unique transmit waveforms."""

        if self.waveforms_one_way is not None:
            return self.waveforms_one_way.shape[0]

        if self.waveforms_two_way is not None:
            return self.waveforms_two_way.shape[0]

        return 1

    @cache_with_dependencies("center_frequency", "selected_transmits", "waveforms_two_way")
    def t_peak(self):
        """The time of the peak of the pulse in seconds of shape (n_tx,).

        If not set explicitly and ``waveforms_two_way`` (the two-way,
        pulse-echo transmit waveform) is available, this is estimated from it
        via :func:`~zea.func.ultrasound.compute_time_to_peak`. Otherwise
        it defaults to ``1 / center_frequency``.
        """
        t_peak = self._params.get("t_peak")
        if t_peak is None:
            waveforms = self._params.get("waveforms_two_way")
            if waveforms is not None:
                t_peak = ops.convert_to_numpy(
                    compute_time_to_peak(waveforms, self.center_frequency)
                )
            else:
                t_peak = np.full(self.n_tx_total, 1 / self.center_frequency)

        return t_peak[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def time_to_next_transmit(self):
        """The time between subsequent transmit events of shape (n_frames, n_tx)."""
        value = self._params.get("time_to_next_transmit")
        if value is None:
            return None

        return value[:, self.selected_transmits]

    @cache_with_dependencies("n_ax")
    def tgc_gain_curve(self):
        """Time gain compensation (TGC) curve of shape (n_ax,)."""
        value = self._params.get("tgc_gain_curve")
        if value is None:
            log.warning_once(
                "No ``tgc_gain_curve`` provided, using ones",
                key=(id(self), "tgc_gain_curve"),
            )
            return np.ones(self.n_ax)
        return value[: self.n_ax]

    @cache_with_dependencies(
        "sound_speed",
        "center_frequency",
        "probe_bandwidth_percent",
        "probe_geometry",
        "tx_apodizations",
        "grid",
        "t0_delays",
        "pfield_kwargs",
        optional=("element_width",),
    )
    def pfield(self) -> np.ndarray:
        """Compute or return the pressure field (pfield) for weighting
        of shape (n_tx, grid_size_z, grid_size_x)."""
        if self.is_3d:
            raise NotImplementedError("Pfield computation is not yet implemented for 3D scans.")

        pfield = compute_pfield(
            sound_speed=self.sound_speed,
            center_frequency=self.center_frequency,
            probe_geometry=self.probe_geometry,
            tx_apodizations=self.tx_apodizations,
            grid=self.grid,
            t0_delays=self.t0_delays,
            probe_bandwidth_percent=self.probe_bandwidth_percent,
            element_width=self._params.get("element_width"),
            **self.pfield_kwargs,
        )
        return ops.convert_to_numpy(pfield)

    @cache_with_dependencies("pfield")
    def flat_pfield(self):
        """Flattened pfield for weighting of shape (n_pix, n_tx)."""
        return self.pfield.reshape(self.n_tx, -1).swapaxes(0, 1)

    @cache_with_dependencies("zlims", "grid_apex")
    def rlims(self):
        """Radius ``(rho_min, rho_max)`` [m] of the first and last row of a polar grid.

        Rho is measured from :attr:`grid_apex`, so it is
        ``zlims - grid_apex[2]`` (:attr:`zlims` are depths below the transducer), matching
        the rows of :func:`~zea.beamform.pixelgrid.polar_pixel_grid`. Used for scan
        conversion, together with :attr:`polar_limits` for the columns."""
        value = self._params.get("rlims")
        if value is None:
            apex_z = float(self.grid_apex[2])
            return (float(self.zlims[0]) - apex_z, float(self.zlims[1]) - apex_z)
        return value

    @cache_with_dependencies(
        "rlims",
        "polar_limits",
        "resolution",
        "grid_size_z",
        "grid_size_x",
        optional=("grid_apex", "extent"),
    )
    def coordinates_2d(self):
        """The coordinates for scan conversion.

        These index the polar image, so they live in the frame of :attr:`grid_apex`;
        :attr:`rlims` already carries the offset. The output covers :attr:`extent`.
        When ``rlims`` is set explicitly, it describes a polar image of its own rather than
        this grid, so the output covers its bounding box instead.
        """
        grid_apex = extent = None
        if self._params.get("rlims") is None:
            try:
                grid_apex, extent = self.grid_apex, self.extent
            except MissingDependencyError:
                pass
        coords, _ = compute_scan_convert_2d_coordinates(
            (self.grid_size_z, self.grid_size_x),
            self.rlims,
            self.polar_limits,
            self.resolution,
            grid_apex=grid_apex,
            extent=extent,
        )
        return coords

    @cache_with_dependencies("is_3d", "coordinates_2d")
    def coordinates(self):
        """Get the coordinates for scan conversion."""
        if self.is_3d:
            raise NotImplementedError(
                "Scan conversion of 3D grids is not supported. Use "
                "zea.display.scan_convert_3d directly on the volume."
            )
        return self.coordinates_2d

    @cache_with_dependencies("time_to_next_transmit")
    def pulse_repetition_frequency(self):
        """The pulse repetition frequency (PRF) [Hz]. Assumes a constant PRF."""
        if self.time_to_next_transmit is None:
            raise ValueError(
                "Time to next transmit must be set to compute pulse repetition frequency"
            )

        pulse_repetition_interval = np.mean(self.time_to_next_transmit)

        return 1 / pulse_repetition_interval

    @cache_with_dependencies("time_to_next_transmit")
    def frames_per_second(self):
        """The number of frames per second [Hz]. Assumes a constant frame rate.

        Frames per second computed based on time between transmits within a frame.
        Ignores time between frames (e.g. due to processing).

        Uses the time it took to do all transmits (per frame). So if you only use some portion
        of the transmits, the fps will still be calculated based on all.
        """
        time_to_next_transmit = self._params.get("time_to_next_transmit")
        if time_to_next_transmit is None:
            raise ValueError("Time to next transmit must be set to compute frames per second")

        # Check if fps is constant
        uniq = np.unique(time_to_next_transmit, axis=0)  # frame axis
        if uniq.shape[0] != 1:
            log.warning("Time to next transmit is not constant")

        # Compute fps
        time = np.mean(np.sum(time_to_next_transmit, axis=1))
        fps = 1 / time
        return fps

    def to_scan_dict(self) -> dict:
        """Return scan parameters as a plain dict.

        Suitable for passing directly to :meth:`~zea.File.create` as the
        ``scan`` argument, or to :func:`~zea.data.file_operations.save_file`
        alongside :meth:`to_probe_dict`.

        Only fields defined in :class:`~zea.data.spec.ScanSpec` that are
        currently stored on this object are included (``None`` values are
        omitted).  Values are read through property access so that any active
        :attr:`selected_transmits` filtering is applied (e.g. after calling
        :meth:`set_transmits`).

        Returns:
            dict: Scan parameter dict keyed by :class:`~zea.data.spec.ScanSpec`
                field names.
        """
        result = {}
        for field in ScanSpec.fields():
            # Only include fields that were explicitly stored as a non-None value.
            # Skipping None here is critical: fields like ``tgc_gain_curve`` that
            # are absent in the source file are stored as None in ``_params``.
            # Calling ``getattr`` on them would trigger computed defaults (e.g.
            # ones), which would be written to the new file and break round-trips.
            if field not in self._params or self._params[field] is None:
                continue
            try:
                value = getattr(self, field)  # applies selected_transmits filtering
            except MissingDependencyError:
                continue
            if value is not None:
                result[field] = value
        return result

    def to_probe_dict(self) -> dict:
        """Return file-backed probe parameters as a plain dict.

        Suitable for passing directly to :meth:`~zea.File.create` as the
        ``probe`` argument, or to :func:`~zea.data.file_operations.save_file`
        alongside :meth:`to_scan_dict`.

        Only fields defined in :class:`~zea.data.spec.ProbeSpec` that are
        currently stored on this object are included (``None`` values are
        omitted).

        Returns:
            dict: Probe parameter dict keyed by :class:`~zea.data.spec.ProbeSpec`
                field names.
        """
        return {
            field: self._params[field]
            for field in ProbeSpec.fields()
            if field in self._params and self._params[field] is not None
        }

    def __setattr__(self, name: str, value):
        if name in self._REMOVED_PARAMS:
            raise ValueError(self._REMOVED_PARAMS[name])
        name = self._renamed(name)
        if name == "selected_transmits":
            # If setting selected_transmits, call set_transmits to handle logic
            self.set_transmits(value)
        else:
            return super().__setattr__(name, value)


class Scan(Parameters):
    """Deprecated alias for :class:`Parameters`.

    ``Scan`` was renamed to :class:`zea.Parameters` (which now holds the merged
    probe and scan parameters). This subclass is kept temporarily to ease the
    transition: instantiating it emits a :class:`DeprecationWarning` pointing to
    :class:`zea.Parameters`. It will be removed in a future release.
    """

    @deprecated(replacement="zea.Parameters")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
