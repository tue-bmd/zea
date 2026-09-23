"""The transducer elements of the simulators: their resolved geometry and physics
(:class:`ElementModel`), the one-way transmit and receive responses of every element to every
scatterer (:func:`element_responses`), and the straight rays through the sound speed and
attenuation maps."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from keras import ops

from zea.beamform.lens_correction import compute_lens_path_lengths
from zea.func.ultrasound import directivity, straight_ray_mean, straight_ray_slowness
from zea.internal.core import concrete


def min_distance(sound_speed, center_frequency):
    """Half a wavelength: the simulators clamp the element-scatterer distance to it for the
    phase and the spreading, as SIMUS does, so that the 1 / r of a scatterer on an element stays
    finite. The angles keep the true geometry."""
    return sound_speed / (2.0 * center_frequency)


def attenuate(f, attenuation_coef, dist, power=1.0):
    """Amplitude left after attenuation in the frequency domain.

    Args:
        f (array-like): The input frequencies.
        attenuation_coef (float): The attenuation coefficient in dB/cm/MHz^power.
        dist (float): The distance the signal has traveled.
        power (float): Exponent of the frequency dependence: ``attenuation_coef * f**power``
            dB/cm with ``f`` in MHz.

    Returns:
        array-like: The spectrum of the attenuation.
    """
    # The floor keeps f**power finite in the gradient at the zero bin.
    f_mhz = ops.maximum(ops.abs(f) * 1e-6, 1e-12) ** power
    return ops.exp(-ops.log(10) * attenuation_coef / 20 * dist * 100 * f_mhz)


def spread(dist, exponent=1.0, mindist=1e-3, reference=1e-3):
    """Geometric spreading of the wavefront.

    Args:
        dist (array-like): The distance the wave has traveled.
        exponent (float): 1 for spherical, 0.5 for cylindrical. An elevation lens focuses the
            transmitted energy to a slab, resulting in a cylindrical transmit and a spherical
            receive path.
        mindist (float): Distances below it are clamped to it. The simulators pass half a
            wavelength, :func:`min_distance`.
        reference (float): Distance of unit gain.

    Returns:
        array-like: An amplitude factor in the shape of `dist`.
    """
    dist = ops.maximum(dist, mindist)
    return (reference / dist) ** exponent


def obliquity_factor(cos_angle, baffle_impedance_ratio):
    """Obliquity factor of an element in a baffle of finite impedance, at the cosine of the
    angle to its normal: 1 in a rigid baffle (ratio 0), the cosine in a soft one (``inf``), and
    cos / (cos + ratio) in between, with the ratio the medium impedance over the baffle's
    (Selfridge et al. 1980, as in SIMUS). With a non-rigid baffle, directions behind the
    element get 0, not a pole.
    """
    if baffle_impedance_ratio == 0:
        return ops.ones_like(cos_angle)
    cos_angle = ops.maximum(cos_angle, 0.0)
    if baffle_impedance_ratio == float("inf"):
        return cos_angle
    return cos_angle / (cos_angle + baffle_impedance_ratio)


# ---------------------------------------------------------------------------------------------
# Element model
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementModel:
    """The element positions and the physics of one element, with the defaults of
    :func:`simulate_rf` resolved and the scalars cast to float32. Built once per simulator call
    by :func:`element_model`, and read by :func:`element_responses` in every frequency block.

    Attributes:
        geometry: Element positions [m], (n_el, 3) float32.
        sound_speed, element_width, element_height, attenuation_coef, attenuation_power,
            lens_thickness, lens_sound_speed, lens_attenuation_coef, min_dist: The scalars of
            :func:`simulate_rf` as float32 (a lens speed of None is 0, unused without the lens).
        apply_lens_correction, two_dimensional: Static switches.
        baffle_impedance_ratio: Static float, see :func:`obliquity_factor`.
        element_normals: Outward normals (n_el, 3) float32, or None for +z.
        n_sub_elements: Static (n_lateral, n_elevation), see :func:`_resolve_sub_elements`.
        elevation_focus: Static focal distance [m], or None.
        directivity_frequency: Frequency [Hz] the directivity is evaluated at, or None for
            every bin.
    """

    geometry: Any
    sound_speed: Any
    element_width: Any
    element_height: Any
    attenuation_coef: Any
    attenuation_power: Any
    lens_thickness: Any
    lens_sound_speed: Any
    apply_lens_correction: bool
    two_dimensional: bool
    baffle_impedance_ratio: float
    element_normals: Any
    n_sub_elements: tuple
    elevation_focus: float | None
    lens_attenuation_coef: Any
    min_dist: Any
    directivity_frequency: Any = None


def element_model(
    probe_geometry,
    sound_speed,
    center_frequency,
    pulses,
    *,
    element_width,
    element_height,
    attenuation_coef,
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    two_dimensional=False,
    baffle_impedance_ratio=0.0,
    element_normals=None,
    n_sub_elements=None,
    elevation_focus=None,
    lens_attenuation_coef=0.0,
    attenuation_power=1.0,
    simplified_directivity=False,
):
    """Validate the element arguments of :func:`simulate_rf` and resolve their defaults: the
    element width from the pitch, the height from the width, and the sub-element counts from
    the top of the band of ``pulses``."""
    _validate_two_dimensional(two_dimensional, elevation_focus, probe_geometry)
    _validate_baffle(baffle_impedance_ratio)
    _validate_element_normals(element_normals, int(probe_geometry.shape[0]))
    element_width = _resolve_element_width(probe_geometry, element_width)
    element_height = _resolve_element_height(probe_geometry, element_width, element_height)
    _validate_lens(
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        elevation_focus,
        element_height,
    )
    n_sub_elements = _resolve_sub_elements(
        n_sub_elements,
        elevation_focus,
        element_width,
        element_height,
        sound_speed,
        max(pulse.band[1] for pulse in pulses),
        two_dimensional,
    )
    return ElementModel(
        geometry=ops.cast(probe_geometry, "float32"),
        sound_speed=_as_f32(sound_speed),
        element_width=_as_f32(element_width),
        element_height=_as_f32(element_height),
        attenuation_coef=_as_f32(attenuation_coef),
        attenuation_power=_as_f32(attenuation_power),
        lens_thickness=_as_f32(lens_thickness),
        lens_sound_speed=_as_f32(lens_sound_speed),
        apply_lens_correction=bool(apply_lens_correction),
        two_dimensional=bool(two_dimensional),
        baffle_impedance_ratio=float(baffle_impedance_ratio),
        element_normals=None if element_normals is None else _as_f32(element_normals),
        n_sub_elements=n_sub_elements,
        elevation_focus=None if elevation_focus is None else float(elevation_focus),
        lens_attenuation_coef=_as_f32(lens_attenuation_coef),
        min_dist=min_distance(_as_f32(sound_speed), float(center_frequency)),
        directivity_frequency=_as_f32(center_frequency) if simplified_directivity else None,
    )


def _scene_positions(positions, model):
    """The positions (n, 3) as float32, moved into the imaging plane in 2D."""
    positions = ops.cast(positions, "float32")
    if model.two_dimensional:
        positions = _snap_elevation(positions, model.geometry)
    return positions


def _snap_elevation(positions, geometry):
    """The positions moved into the imaging plane, the probe's elevation center."""
    center = ops.mean(geometry[:, 1])
    return ops.stack(
        [positions[:, 0], ops.zeros_like(positions[:, 0]) + center, positions[:, 2]], axis=-1
    )


# ---------------------------------------------------------------------------------------------
# Element responses
# ---------------------------------------------------------------------------------------------


def element_responses(positions, model, freqs, slowness=None, attenuation=None):
    """Transmit and receive one-way responses [f, s, e] of the elements of ``model`` to the
    scatterers at ``positions``, and the one-way travel time [s, e] from the element centers.

    Each element is the mean of its sub-elements, each with its own path, phase and sinc
    directivity, so the response holds in the near field. The path is clamped at
    ``model.min_dist`` for the phase and the spreading only. ``slowness`` and ``attenuation``
    are the mean slowness and attenuation coefficient [s, e] of the straight rays through the
    maps (:func:`_ray_means`), or None for the homogeneous medium of ``model``; they time and
    attenuate the medium leg of every path, while the lens leg, the directivity and the
    spreading keep the homogeneous geometry. In 2D the positions lie in the imaging plane
    (:func:`_scene_positions`): there is no elevation directivity, and the transmit spreads
    cylindrically, as behind an ideal lens.
    """
    dtype = positions.dtype
    frame = _element_frame(model.element_normals, dtype)
    # A conformal lens on a curved probe: its face is normal to each element.
    lens_normals = None if model.element_normals is None else frame.normal
    sub = _sub_elements(model, frame, dtype)
    f3 = freqs[:, None, None]
    min_time = model.min_dist / model.sound_speed

    def response(j):
        """Receive response of sub-element ``j``, preceded by its transmit response in 2D,
        where the two differ."""
        path = _sub_element_path(positions, sub, j, model, lens_normals, slowness)
        amplitude = _sub_element_amplitude(f3, path, frame, sub, model, attenuation)
        delay = ops.maximum(path.time, min_time)[None] - sub.advance[j]
        phase = ops.exp(ops.array(-2j * np.pi, "complex64") * ops.cast(delay * f3, "complex64"))

        def with_spreading(exponent):
            gain = spread(path.spread[None], exponent, model.min_dist)
            return ops.cast(amplitude * gain, "complex64") * phase

        rx = with_spreading(1.0)
        if not model.two_dimensional:
            return (rx,)
        # An ideal elevation lens: cylindrical spread on the way out, spherical back.
        return with_spreading(0.5), rx

    responses = _mean_over_sub_elements(response, sub.n)
    tau = _one_way_time(
        positions,
        model.geometry,
        model.apply_lens_correction,
        model.lens_thickness,
        model.lens_sound_speed,
        model.sound_speed,
        slowness,
        lens_normals,
    )
    return responses[0], responses[-1], tau


@dataclass(frozen=True)
class _Frame:
    """Lateral, elevation and normal unit vectors of the elements, each (n_el, 3) or (1, 3)."""

    lateral_axis: Any
    elevation_axis: Any
    normal: Any


def _element_frame(element_normals, dtype="float32"):
    """Element axes for the given normals, or for +z when None.

    The elevation axis is +y projected onto the element plane, and the lateral axis completes
    the frame. Normals along y, where that projection vanishes, are rejected by
    :func:`_validate_element_normals`.
    """
    if element_normals is None:
        eye = ops.cast(ops.convert_to_tensor(np.eye(3, dtype=np.float32)), dtype)
        return _Frame(eye[0:1], eye[1:2], eye[2:3])
    normal = ops.cast(element_normals, dtype)
    normal = normal / ops.linalg.norm(normal, axis=-1, keepdims=True)
    y = ops.cast(ops.convert_to_tensor(np.array([0.0, 1.0, 0.0], np.float32)), dtype)
    elevation_axis = y - normal[:, 1:2] * normal
    elevation_axis = elevation_axis / ops.linalg.norm(elevation_axis, axis=-1, keepdims=True)
    lateral_axis = ops.cross(elevation_axis, normal)
    return _Frame(lateral_axis, elevation_axis, normal)


@dataclass(frozen=True)
class _SubElements:
    """The ``n`` sub-elements of every element, ``width`` by ``height`` [m] each. ``offsets``
    (n, n_el or 1, 3) are their centers relative to the element center, ``advance`` (n,) the
    focusing advance [s] of an ideal elevation lens at each, and ``thickness`` (n,) the lens
    thickness [m] under each, None without the lens."""

    n: int
    width: Any
    height: Any
    offsets: Any
    advance: Any
    thickness: Any


def _sub_elements(model, frame, dtype):
    """Sub-element geometry of ``model``, with the offsets laid out along the axes of ``frame``."""
    n_lateral, n_elevation = model.n_sub_elements
    u, v = _sub_element_offsets(n_lateral, n_elevation, model.element_width, model.element_height)
    u, v = ops.cast(u, dtype), ops.cast(v, dtype)
    advance, thickness = _elevation_focusing(v, model)
    return _SubElements(
        n=n_lateral * n_elevation,
        width=model.element_width / n_lateral,
        height=model.element_height / n_elevation,
        offsets=u[:, None, None] * frame.lateral_axis[None]
        + v[:, None, None] * frame.elevation_axis[None],
        advance=advance,
        thickness=thickness,
    )


def _sub_element_offsets(n_lateral, n_elevation, element_width, element_height):
    """Centroid offsets (u, v) of the sub-elements in the element frame, each (n_sub,)."""
    u = (ops.arange(n_lateral, dtype="float32") - (n_lateral - 1) / 2) * (
        ops.cast(element_width, "float32") / n_lateral
    )
    v = (ops.arange(n_elevation, dtype="float32") - (n_elevation - 1) / 2) * (
        ops.cast(element_height, "float32") / n_elevation
    )
    return ops.reshape(ops.tile(u[:, None], (1, n_elevation)), (-1,)), ops.tile(v, (n_lateral,))


def _elevation_focusing(v, model):
    """Focusing advance [s] and lens thickness [m] of the sub-elements at elevation offsets
    ``v``. Without the lens the focus is an ideal advance per sub-element and the thickness is
    None; with it the advance is zero and the thickness profile, thinned towards the edges,
    does the focusing."""
    advance = ops.zeros_like(v)
    if not model.apply_lens_correction:
        if model.elevation_focus is not None:
            focus = ops.cast(model.elevation_focus, v.dtype)
            advance = (ops.sqrt(focus**2 + v**2) - focus) / model.sound_speed
        return advance, None
    if model.elevation_focus is None:
        return advance, ops.full_like(v, model.lens_thickness)
    sag = _lens_sag(v, model.elevation_focus, model.sound_speed, model.lens_sound_speed)
    return advance, model.lens_thickness - sag


def _lens_sag(v, elevation_focus, sound_speed, lens_sound_speed):
    """Thickness removed from the lens at elevation offset ``v`` to focus at ``elevation_focus``.

    A slower lens is thickest at the center, a faster one thinnest (negative sag).
    """
    focus = ops.cast(elevation_focus, v.dtype)
    path = ops.sqrt(focus**2 + v**2) - focus
    return path * lens_sound_speed / (sound_speed - lens_sound_speed)


@dataclass(frozen=True)
class _Path:
    """The paths from one sub-element of every element to every scatterer, (n_scat, n_el, ...):
    the straight ``vector`` [m], the ``medium`` and ``lens`` legs [m] (``lens`` None without
    the lens), the travel ``time`` [s], and ``spread``, the distance with the same spreading."""

    vector: Any
    medium: Any
    lens: Any
    time: Any
    spread: Any


def _sub_element_path(positions, sub, j, model, lens_normals, slowness):
    """Path from sub-element ``j`` to every scatterer: straight, or refracted through the lens
    thickness under it. The medium leg runs at ``slowness`` when given."""
    vector = positions[:, None] - model.geometry[None] - sub.offsets[j][None]
    if not model.apply_lens_correction:
        distance = ops.linalg.norm(vector, axis=-1)
        time = _medium_time(distance, model.sound_speed, slowness)
        return _Path(vector=vector, medium=distance, lens=None, time=time, spread=distance)
    thickness = sub.thickness[j]
    lens, medium = compute_lens_path_lengths(
        model.geometry + sub.offsets[j],
        positions,
        lens_thickness=thickness,
        c_lens=model.lens_sound_speed,
        c_medium=model.sound_speed,
        n_iter=3,
        element_normals=lens_normals,
    )
    return _Path(
        vector=vector,
        medium=medium,
        lens=lens,
        time=lens / model.lens_sound_speed + _medium_time(medium, model.sound_speed, slowness),
        spread=_lens_spread_distance(
            lens, medium, thickness, model.sound_speed, model.lens_sound_speed
        ),
    )


def _medium_time(length, sound_speed, slowness=None):
    """Travel time [s] over a medium leg: at the ray's mean ``slowness``, or at ``1 / c``."""
    return length / sound_speed if slowness is None else length * slowness


def _lens_spread_distance(lens_len, medium_len, thickness, sound_speed, lens_sound_speed):
    """Distance whose 1/r spreading is the ray-tube divergence of the path refracted at the face.

    The wave leaves the face as if from a source lens_len * c_lens / c below it (apparent
    depth). The refracted wavefront is astigmatic: that radius holds across the plane of
    incidence, and within it the radius is scaled by cos^2 of the medium angle over cos^2 of
    the lens angle.
    """
    ratio = lens_sound_speed / sound_speed
    lens_len = ops.maximum(lens_len, 1e-9)
    cos_lens_sq = ops.clip((thickness / lens_len) ** 2, 1e-6, 1.0)
    cos_medium_sq = ops.maximum(1.0 - (1.0 - cos_lens_sq) / ratio**2, 1e-6)
    r_across = lens_len * ratio
    r_within = r_across * cos_medium_sq / cos_lens_sq
    return lens_len * ops.sqrt(
        (r_across + medium_len) * (r_within + medium_len) / (r_across * r_within)
    )


def _sub_element_amplitude(f3, path, frame, sub, model, attenuation):
    """Real amplitude [f, s, e] along ``path`` at the frequencies ``f3`` (n_freq, 1, 1): the sinc
    directivity of the sub-element in the geometric direction, the attenuation of the lens and
    medium legs, and the baffle obliquity. ``attenuation`` is a per-ray coefficient [s, e] for
    the medium leg, or None for the model's."""
    theta, phi, obliquity = _element_angles(path.vector, frame)
    f_dir = f3 if model.directivity_frequency is None else model.directivity_frequency
    amplitude = directivity(f_dir, theta[None], sub.width, model.sound_speed)
    if not model.two_dimensional:
        amplitude = amplitude * directivity(f_dir, phi[None], sub.height, model.sound_speed)
    if path.lens is not None:
        amplitude = amplitude * attenuate(f3, model.lens_attenuation_coef, path.lens[None])
    attenuation_coef = model.attenuation_coef if attenuation is None else attenuation[None]
    amplitude = amplitude * attenuate(
        f3, attenuation_coef, path.medium[None], model.attenuation_power
    )
    return amplitude * obliquity_factor(obliquity, model.baffle_impedance_ratio)[None]


def _element_angles(relative, frame):
    """Lateral and elevation angles and cos of the angle to the element normal.

    The sines of theta and phi are the direction cosines lateral / r and elevation / r, as in
    the Fraunhofer pattern of a rectangular aperture. Projected angles arctan2(lateral, axial)
    would narrow the elevation pattern for laterally offset scatterers.

    Args:
        relative (array-like): Scatterer positions relative to the elements, (n_scat, n_el, 3).
        frame (_Frame): Element axes from :func:`_element_frame`.

    Returns:
        theta, phi, obliquity: arrays of shape (n_scat, n_el).
    """
    lateral = ops.sum(relative * frame.lateral_axis[None], axis=-1)
    elevation = ops.sum(relative * frame.elevation_axis[None], axis=-1)
    axial = ops.sum(relative * frame.normal[None], axis=-1)
    dist = ops.maximum(ops.linalg.norm(relative, axis=-1), 1e-12)
    theta = ops.arcsin(ops.clip(lateral / dist, -1.0, 1.0))
    phi = ops.arcsin(ops.clip(elevation / dist, -1.0, 1.0))
    obliquity = axial / dist
    return theta, phi, obliquity


def _mean_over_sub_elements(response, n_sub):
    """Mean of the tuples of arrays ``response(j)`` over the ``n_sub`` sub-elements, summed in
    a loop so that one sub-element's responses are live at a time."""
    if n_sub == 1:
        return response(0)

    def body(j, total):
        return tuple(t + r for t, r in zip(total, response(j)))

    total = ops.fori_loop(1, n_sub, body, response(0))
    scale = ops.array(1.0 / n_sub, "complex64")
    return tuple(t * scale for t in total)


def _one_way_time(
    positions,
    geometry,
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    sound_speed,
    slowness=None,
    lens_normals=None,
):
    """One-way travel time [s, e] from each element center to each position: straight, or the
    shortest path through the lens, whose face follows ``lens_normals`` when given.

    ``slowness`` is the mean slowness of each straight ray (:func:`_ray_slowness`), or None for
    ``1 / sound_speed``.
    """
    if not apply_lens_correction:
        length = ops.linalg.norm(positions[:, None] - geometry[None], axis=-1)
        return _medium_time(length, sound_speed, slowness)
    lens_len, medium_len = compute_lens_path_lengths(
        geometry,
        positions,
        lens_thickness=lens_thickness,
        c_lens=lens_sound_speed,
        c_medium=sound_speed,
        n_iter=3,
        element_normals=lens_normals,
    )
    return lens_len / lens_sound_speed + _medium_time(medium_len, sound_speed, slowness)


# ---------------------------------------------------------------------------------------------
# Rays through the maps
# ---------------------------------------------------------------------------------------------


def _ray_means(
    positions,
    model,
    sos_map,
    attenuation_map,
    map_grid_x,
    map_grid_z,
    map_grid_y,
    n_samples,
):
    """Mean slowness and mean attenuation coefficient [s, e] of the straight rays from the
    elements of ``model`` to the positions, each None without its map."""
    start = _ray_starts(
        model.geometry, model.apply_lens_correction, model.lens_thickness, model.element_normals
    )
    grids = (map_grid_x, map_grid_z, map_grid_y)
    slowness = _ray_slowness(positions, start, model.sound_speed, sos_map, *grids, n_samples)
    if attenuation_map is None:
        return slowness, None
    attenuation = straight_ray_mean(
        positions,
        start,
        attenuation_map,
        map_grid_x,
        map_grid_z,
        model.attenuation_coef,
        grid_y=map_grid_y,
        n_samples=int(n_samples),
    )
    return slowness, attenuation


def _ray_slowness(positions, start, sound_speed, sos_map, grid_x, grid_z, grid_y, n_samples):
    """Mean slowness [s, e] of the straight rays from ``start`` to the positions, or None
    without a map."""
    if sos_map is None:
        return None
    return straight_ray_slowness(
        positions,
        start,
        sos_map,
        grid_x,
        grid_z,
        sound_speed,
        map_grid_y=grid_y,
        n_samples=int(n_samples),
    )


def _ray_starts(geometry, apply_lens_correction, lens_thickness, element_normals):
    """Start points of the rays through the maps: the elements, or the lens face with a lens
    (the lens leg is timed and attenuated separately)."""
    if not apply_lens_correction:
        return geometry
    normal = _element_frame(element_normals, geometry.dtype).normal
    return geometry + ops.cast(lens_thickness, geometry.dtype) * normal


# ---------------------------------------------------------------------------------------------
# Validation and argument resolution, on concrete values only
# ---------------------------------------------------------------------------------------------


def _as_f32(x):
    return ops.cast(0.0 if x is None else x, "float32")


def _validate_two_dimensional(two_dimensional, elevation_focus, probe_geometry, tol=1e-6):
    """2D is a 1D probe with an ideal elevation lens, so it excludes the 3D lens model."""
    if not two_dimensional:
        return
    if elevation_focus is not None:
        raise ValueError(
            "two_dimensional collapses elevation (cylindrical transmit spread, no elevation "
            "directivity); elevation_focus models the lens in 3D through the elevation "
            "sub-elements. Pick one."
        )
    geometry = concrete(probe_geometry)
    if geometry is None:
        return
    elevation = np.asarray(geometry)[:, 1]
    if elevation.max() - elevation.min() > tol:
        raise ValueError(
            "two_dimensional=True needs a 1D probe, but the elements span "
            f"{elevation.min():.2e} to {elevation.max():.2e} m in elevation."
        )


def _validate_maps(sos_map, attenuation_map, map_grid_x, map_grid_z, map_grid_y=None):
    """Static checks of the maps: x and z grids given (y for a 3D map) with matching shapes and,
    when concrete, uniform ascending grids, positive speeds and non-negative attenuation."""
    grids = (("map_grid_x", map_grid_x), ("map_grid_z", map_grid_z), ("map_grid_y", map_grid_y))
    maps = (("sos_map", sos_map), ("attenuation_map", attenuation_map))
    given = [name for name, m in maps if m is not None]
    if not given:
        grids_given = [name for name, grid in grids if grid is not None]
        if grids_given:
            raise ValueError(f"{', '.join(grids_given)} given without sos_map or attenuation_map.")
        return
    if map_grid_x is None or map_grid_z is None:
        raise ValueError(f"{' and '.join(given)} needs the coordinates map_grid_x and map_grid_z.")
    grids = [("map_grid_z", map_grid_z), ("map_grid_x", map_grid_x)]
    if map_grid_y is not None:
        grids.append(("map_grid_y", map_grid_y))
    expected = tuple(int(ops.shape(grid)[0]) for _, grid in grids)
    for name, m in maps:
        if m is None:
            continue
        shape = tuple(int(d) for d in ops.shape(m))
        if shape != expected:
            raise ValueError(
                f"{name} of shape {shape} does not match its grids: expected (Nz, Nx) for a "
                f"2D map or (Nz, Nx, Ny) with map_grid_y for a 3D map, here {expected}."
            )
    for name, grid in grids:
        values = concrete(grid)
        if values is None:
            continue
        steps = np.diff(np.asarray(values, np.float64))
        if len(values) < 2 or steps.min() <= 0:
            raise ValueError(f"{name} must be ascending with at least two points.")
        if not np.allclose(steps, steps[0], rtol=1e-3):
            raise ValueError(f"{name} must be uniformly spaced.")
    values = concrete(sos_map)
    if values is not None and (not np.all(np.isfinite(values)) or np.any(values <= 0)):
        raise ValueError("sos_map must hold finite, positive sound speeds.")
    values = concrete(attenuation_map)
    if values is not None and (not np.all(np.isfinite(values)) or np.any(values < 0)):
        raise ValueError("attenuation_map must hold finite, non-negative coefficients.")


def _validate_lens(
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    sound_speed,
    elevation_focus,
    element_height,
):
    """Static checks of a focusing lens: distinct speeds, and a face above the elements."""
    if not apply_lens_correction:
        return
    if lens_sound_speed is None:
        raise ValueError("apply_lens_correction=True requires lens_sound_speed.")
    if elevation_focus is None:
        return
    values = [concrete(x) for x in (lens_thickness, lens_sound_speed, sound_speed, element_height)]
    if any(v is None for v in values):
        return
    thickness, c_lens, c, height = (float(v) for v in values)
    if c_lens == c:
        raise ValueError("A lens at the medium's sound speed cannot focus; set lens_sound_speed.")
    sag = (np.sqrt(float(elevation_focus) ** 2 + (height / 2) ** 2) - float(elevation_focus)) * (
        c_lens / (c - c_lens)
    )
    if thickness - sag <= 0:
        raise ValueError(
            f"lens_thickness {thickness:.2e} m is too thin to focus at {elevation_focus} m: the "
            f"lens needs at least {sag:.2e} m at the center."
        )


def _resolve_element_width(probe_geometry, element_width):
    """Return the element width, inferring it from the probe pitch when not given."""
    if element_width is not None:
        return element_width
    try:
        geometry = ops.convert_to_numpy(probe_geometry)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(
            "Element width is not provided, and automatic inference is not available for "
            "traced/symbolic probe geometry (for example under JAX JIT or TensorFlow graph "
            "mode). Please provide `element_width` explicitly in the scan/probe parameters."
        ) from exc

    try:
        from zea.probes import Probe

        pitch = Probe.get_pitch(geometry)
    except (ValueError, IndexError, AttributeError) as exc:
        raise ValueError(
            "Element width is not provided and automatic estimation failed from probe "
            "geometry. Please provide `element_width` explicitly or ensure the probe "
            "geometry is a 1-D uniformly spaced linear array. "
            f"Details: {exc}"
        ) from exc
    return pitch * 0.9  # 90% of the pitch


def _resolve_element_height(probe_geometry, element_width, element_height, tol=1e-6):
    """Element height, inferred when not given: an eighth of the probe width for a 1D probe (at
    least the element width), and the element width for a 2D probe or a single element. The
    probe width is n_el times the pitch, measured along the probe whatever its tilt. A traced
    geometry gives a traced height, a concrete one a Python float."""
    if element_height is not None:
        return element_height
    n_el = int(probe_geometry.shape[0])
    if n_el < 2:
        return element_width
    geometry = ops.cast(probe_geometry, "float32")
    x, y, z = geometry[:, 0], geometry[:, 1], geometry[:, 2]
    # The extent along the principal axis of the elements in the (x, z) plane: the length of a
    # linear probe at any tilt (tilted elements come with element normals), the chord of a curved
    # one. The tilt is the principal direction of the 2x2 covariance, in closed form.
    dx, dz = x - ops.mean(x), z - ops.mean(z)
    tilt = 0.5 * ops.arctan2(2 * ops.sum(dx * dz), ops.sum(dx * dx) - ops.sum(dz * dz))
    lateral = x * ops.cos(tilt) + z * ops.sin(tilt)
    probe_width = (ops.max(lateral) - ops.min(lateral)) * (n_el / (n_el - 1))
    one_dimensional = ops.max(y) - ops.min(y) <= tol
    height = ops.where(one_dimensional, ops.maximum(probe_width / 8, element_width), element_width)
    value = concrete(height)
    return height if value is None else float(value)


def _resolve_sub_elements(
    n_sub_elements,
    elevation_focus,
    element_width,
    element_height,
    sound_speed,
    max_frequency,
    two_dimensional=False,
):
    """Sub-elements per element as (n_lateral, n_elevation).

    "auto" is the SIMUS rule ceil(size / lambda_min), lambda_min at ``max_frequency`` [Hz], the
    top of the band, so that the far-field assumption holds per sub-element. None and an int
    keep one elevation sub-element unless there is an elevation focus, which needs the elevation
    subdivision to act at all. In 2D there is a single elevation sub-element.
    """
    if isinstance(n_sub_elements, (tuple, list)):
        n_lateral, n_elevation = (int(n) for n in n_sub_elements)
        return max(n_lateral, 1), 1 if two_dimensional else max(n_elevation, 1)
    focused = elevation_focus is not None
    if n_sub_elements != "auto" and not focused:
        return (1 if n_sub_elements is None else max(int(n_sub_elements), 1)), 1
    values = [concrete(x) for x in (sound_speed, element_width, element_height)]
    if any(v is None for v in values):
        raise ValueError(
            "The sub-element count cannot be derived from a traced sound speed or element "
            "size; pass n_sub_elements=(n_lateral, n_elevation) explicitly."
        )
    c, width, height = (float(v) for v in values)
    lambda_min = c / float(max_frequency)
    n_elevation = 1 if two_dimensional else max(int(np.ceil(height / lambda_min)), 1)
    if n_sub_elements == "auto":
        return max(int(np.ceil(width / lambda_min)), 1), n_elevation
    return (1 if n_sub_elements is None else max(int(n_sub_elements), 1)), n_elevation


def _validate_element_normals(element_normals, n_el, tol=1e-6):
    """Reject normals of the wrong shape, of zero length, or along the y axis: the element frame
    projects +y onto the element plane, which is undefined there (a probe converted with its y
    and z columns swapped does this). Traced normals are only checked for shape."""
    if element_normals is None:
        return
    shape = tuple(int(n) for n in element_normals.shape)
    if len(shape) != 2 or shape[1] != 3 or shape[0] not in (1, n_el):
        raise ValueError(f"element_normals must have shape ({n_el}, 3), got {shape}.")
    normals = concrete(element_normals)
    if normals is None:
        return
    length = np.linalg.norm(normals.astype(np.float64), axis=-1)
    if np.any(length <= tol):
        raise ValueError(
            f"element_normals of elements {np.flatnonzero(length <= tol)[:8]} have zero length."
        )
    along_y = np.abs(normals[:, 1]) / length >= 1 - tol
    if np.any(along_y):
        raise ValueError(
            f"element_normals of elements {np.flatnonzero(along_y)[:8]} point along the y "
            "(elevation) axis, where the element frame is undefined. Elements face +z by default; "
            "were the y and z columns swapped?"
        )


def _validate_baffle(baffle_impedance_ratio):
    if not baffle_impedance_ratio >= 0:
        raise ValueError(
            f"baffle_impedance_ratio ({baffle_impedance_ratio}) must be non-negative: 0 for a "
            "rigid baffle, inf for a soft one."
        )
