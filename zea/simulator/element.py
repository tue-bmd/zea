"""Transducer elements definitions for the simulators. Includes the resolved geometry and physics
(:class:`ElementModel`), the one-way transmit and receive responses of every element to every
scatterer (:func:`element_responses`), and the elevation slab of a 1D probe."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from keras import ops

from zea import log
from zea.beamform.lens_correction import (
    compute_lens_corrected_travel_times,
    compute_lens_path_lengths,
)
from zea.func.ultrasound import directivity
from zea.internal.core import concrete


def min_distance(sound_speed, center_frequency):
    """The distance the simulator clamps the element-scatterer distance to for the phase and spread,
    so that the 1 / r of a scatterer on an element stays finite."""
    return sound_speed / (2.0 * center_frequency)


def attenuate(f, attenuation_coef, dist):
    """
    Applies attenuation to the signal in the frequency domain.

    Args:
        f (array-like): The input frequencies.
        attenuation_coef (float): The attenuation coefficient in dB/cm/MHz.
        dist (float): The distance the signal has traveled.

    Returns:
        array-like: The spectrum of the attenuation.
    """
    return ops.exp(-ops.log(10) * attenuation_coef / 20 * dist * 100 * ops.abs(f) * 1e-6)


def spread(dist, exponent=1.0, mindist=1e-3, reference=1e-3):
    """Geometric spreading of the wavefront.

    Args:
        dist (array-like): The distance the wave has traveled.
        exponent (float): 1 for spherical, 0.5 for cylindrical.
        mindist (float): Distance to clamp to; typically half a wavelength (:func:`min_distance`).
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
    (Pesque et al., IEEE Ultrasonics Symposium 1984, as in SIMUS; the soft-baffle cosine is
    Selfridge et al. 1980). With a non-rigid baffle, directions behind the element get 0, not
    a pole.
    """
    if baffle_impedance_ratio == 0:
        return ops.ones_like(cos_angle)
    cos_angle = ops.maximum(cos_angle, 0.0)
    if baffle_impedance_ratio == float("inf"):
        return cos_angle
    return cos_angle / (cos_angle + baffle_impedance_ratio)


def _validate_scatter_exponent(scatter_exponent):
    """Reject exponents that make the weighting non-finite: the DC bin is zero, so a
    negative exponent gives infinite gain there, and the NaN spreads over the whole frame."""
    if not np.isfinite(scatter_exponent) or scatter_exponent < 0:
        raise ValueError(
            f"scatter_exponent ({scatter_exponent}) must be finite and non-negative. "
            "2 is Rayleigh scattering (e.g. blood), myocardium is approximately 1.5, "
            "soft tissue 0.6-0.8."
        )


@dataclass(frozen=True)
class ElementModel:
    """The transducer as :func:`element_responses` sees it: the element positions and the
    physics of one element, with the defaults of :func:`simulate_rf` resolved. Built by
    :func:`element_model` once per simulator call.

    Attributes:
        geometry: Element positions [m], (n_el, 3) float32.
        sound_speed, element_width, element_height, attenuation_coef, lens_thickness,
            lens_sound_speed, lens_attenuation_coef, min_dist: The scalars of
            :func:`simulate_rf`, resolved.
        apply_lens_correction, elevation_slab_2d: Static switches.
        baffle_impedance_ratio: Static float, see :func:`obliquity_factor`.
        element_normals: Unit normals (n_el, 3), or None for +z.
        n_sub_elements: Static (n_width, n_height), see :func:`_resolve_sub_elements`.
        elevation_focus: Static focal distance [m], or None.
    """

    geometry: Any
    sound_speed: Any
    element_width: Any
    element_height: Any
    attenuation_coef: Any
    lens_thickness: Any
    lens_sound_speed: Any
    apply_lens_correction: bool
    elevation_slab_2d: bool
    baffle_impedance_ratio: float
    element_normals: Any
    n_sub_elements: tuple
    elevation_focus: float | None
    lens_attenuation_coef: Any
    min_dist: Any


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
    elevation_slab_2d=False,
    baffle_impedance_ratio=0.0,
    element_normals=None,
    n_sub_elements=None,
    elevation_focus=None,
    lens_attenuation_coef=0.0,
):
    """The :class:`ElementModel` of the element arguments of :func:`simulate_rf`, validated and
    with the defaults resolved: the element width from the pitch, the height from the width,
    and the sub-element counts from the top of the band of ``pulses``."""
    _validate_elevation(elevation_slab_2d, elevation_focus)
    _validate_baffle(baffle_impedance_ratio)
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
    )
    return ElementModel(
        geometry=ops.cast(probe_geometry, "float32"),
        sound_speed=sound_speed,
        element_width=element_width,
        element_height=element_height,
        attenuation_coef=attenuation_coef,
        lens_thickness=lens_thickness,
        lens_sound_speed=lens_sound_speed,
        apply_lens_correction=bool(apply_lens_correction),
        elevation_slab_2d=bool(elevation_slab_2d),
        baffle_impedance_ratio=float(baffle_impedance_ratio),
        element_normals=element_normals,
        n_sub_elements=n_sub_elements,
        elevation_focus=None if elevation_focus is None else float(elevation_focus),
        lens_attenuation_coef=lens_attenuation_coef,
        min_dist=min_distance(sound_speed, center_frequency),
    )


def _validate_baffle(baffle_impedance_ratio):
    if not baffle_impedance_ratio >= 0:
        raise ValueError(
            f"baffle_impedance_ratio ({baffle_impedance_ratio}) must be non-negative: 0 for a "
            "rigid baffle, inf for a soft one."
        )


def _validate_elevation(elevation_slab_2d, elevation_focus):
    if elevation_slab_2d and elevation_focus is not None:
        raise ValueError(
            "elevation_slab_2d is the cheap 2D approximation (slab pruning and cylindrical "
            "spread); elevation_focus models the lens in 3D through the elevation "
            "sub-elements. Pick one."
        )


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
            f"lens needs at least {sag:.2e} m at the centre."
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
    """Return the element height, inferring it when not given: an eighth of the width of a 1D
    probe (elements without elevation extent; n_el times the pitch), at least the element
    width, and the element width for a 2D probe or a single element. Works on a traced
    geometry, as a traced height; a Python float otherwise."""
    if element_height is not None:
        return element_height
    n_el = int(probe_geometry.shape[0])
    if n_el < 2:
        return element_width
    geometry = ops.cast(probe_geometry, "float32")
    x, y = geometry[:, 0], geometry[:, 1]
    probe_width = (ops.max(x) - ops.min(x)) * (n_el / (n_el - 1))
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
):
    """Sub-elements per element as (n_width, n_height).

    "auto" is the SIMUS rule ceil(size / lambda_min), lambda_min at ``max_frequency`` [Hz], the
    top of the band. None and an int keep one sub-element over the height unless there is an
    elevation focus, which needs the subdivision over the height to act at all.
    """
    if isinstance(n_sub_elements, (tuple, list)):
        n_width, n_height = (int(n) for n in n_sub_elements)
        return max(n_width, 1), max(n_height, 1)
    focused = elevation_focus is not None
    if n_sub_elements != "auto" and not focused:
        return (1 if n_sub_elements is None else max(int(n_sub_elements), 1)), 1
    values = [concrete(x) for x in (sound_speed, element_width, element_height)]
    if any(v is None for v in values):
        raise ValueError(
            "The sub-element count cannot be derived from a traced sound speed or element "
            "size; pass n_sub_elements=(n_width, n_height) explicitly."
        )
    c, width, height = (float(v) for v in values)
    lambda_min = c / float(max_frequency)
    n_height = max(int(np.ceil(height / lambda_min)), 1)
    if n_sub_elements == "auto":
        return max(int(np.ceil(width / lambda_min)), 1), n_height
    return (1 if n_sub_elements is None else max(int(n_sub_elements), 1)), n_height


# ---------------------------------------------------------------------------------------------
# Element responses
# ---------------------------------------------------------------------------------------------


def element_responses(positions, model, freqs):
    """Transmit and receive one-way responses [scatterer, element, frequency_bin] of the
    elements of ``model`` to the scatterers at ``positions``, and the one-way travel time
    [scatterer, element] from the element centres.

    Each element is the mean of its sub-elements, so the response holds in the near field.
    The distances are clamped at ``model.min_dist`` for the phase and the spreading only.
    """
    dtype = positions.dtype
    frame = _element_frame(model.element_normals, dtype)
    # A conformal lens on a curved probe: its face is normal to each element.
    lens_normals = None if model.element_normals is None else frame.normal
    sub = _sub_elements(model, frame, dtype)
    f3 = freqs[None, None, :]

    def response(j):
        path = _sub_element_path(positions, sub, j, model, lens_normals)
        amplitude = _sub_element_amplitude(f3, path, frame, sub, model)
        distance = ops.maximum(path.phase, model.min_dist)[..., None]
        delay = distance / model.sound_speed - sub.advance[j]
        phase = ops.exp(ops.array(-2j * np.pi, "complex64") * ops.cast(delay * f3, "complex64"))

        def with_spreading(exponent):
            gain = spread(path.spread[..., None], exponent, model.min_dist)
            return ops.cast(amplitude * gain, "complex64") * phase

        rx = with_spreading(1.0)
        # An elevation lens focuses the transmit to a slab: cylindrical spread on the way out.
        tx = with_spreading(0.5) if model.elevation_slab_2d else rx
        return tx, rx

    shape = (ops.shape(positions)[0], ops.shape(model.geometry)[0], ops.shape(freqs)[0])
    tx, rx = _mean_over_sub_elements(response, sub.n, shape)
    return tx, rx, _travel_time(positions, model, lens_normals)


@dataclass(frozen=True)
class _Frame:
    """Unit vectors along the width and height of the elements and their normal, each
    (n_el, 3) or (1, 3)."""

    width_axis: Any
    height_axis: Any
    normal: Any


def _element_frame(element_normals, dtype="float32"):
    """The :class:`_Frame` of elements with the given normals, or +z for None.

    The height axis is +y projected onto the element plane, the width axis completes the frame.
    """
    if element_normals is None:
        eye = ops.cast(ops.convert_to_tensor(np.eye(3, dtype=np.float32)), dtype)
        return _Frame(eye[0:1], eye[1:2], eye[2:3])
    normal = ops.cast(element_normals, dtype)
    normal = normal / ops.linalg.norm(normal, axis=-1, keepdims=True)
    y = ops.cast(ops.convert_to_tensor(np.array([0.0, 1.0, 0.0], np.float32)), dtype)
    height_axis = y - normal[:, 1:2] * normal
    height_axis = height_axis / ops.linalg.norm(height_axis, axis=-1, keepdims=True)
    width_axis = ops.cross(height_axis, normal)
    return _Frame(width_axis, height_axis, normal)


@dataclass(frozen=True)
class _SubElements:
    """The ``n`` sub-elements of every element, ``width`` by ``height`` [m] each. ``offsets``
    (n, n_el or 1, 3) are their centres relative to the element centre, ``advance`` (n,) the
    focusing advance [s] of an ideal elevation lens at each, and ``thickness`` (n,) the lens
    thickness [m] under each, None without the lens."""

    n: int
    width: Any
    height: Any
    offsets: Any
    advance: Any
    thickness: Any


def _sub_elements(model, frame, dtype):
    """The :class:`_SubElements` of ``model`` in its element ``frame``."""
    n_width, n_height = model.n_sub_elements
    u, v = _sub_element_offsets(n_width, n_height, model.element_width, model.element_height)
    u, v = ops.cast(u, dtype), ops.cast(v, dtype)
    advance, thickness = _elevation_focusing(v, model)
    return _SubElements(
        n=n_width * n_height,
        width=model.element_width / n_width,
        height=model.element_height / n_height,
        offsets=u[:, None, None] * frame.width_axis[None]
        + v[:, None, None] * frame.height_axis[None],
        advance=advance,
        thickness=thickness,
    )


def _sub_element_offsets(n_width, n_height, element_width, element_height):
    """Centroid offsets (u, v) of the sub-elements in the element frame, each (n_sub,)."""
    u = (ops.arange(n_width, dtype="float32") - (n_width - 1) / 2) * (
        ops.cast(element_width, "float32") / n_width
    )
    v = (ops.arange(n_height, dtype="float32") - (n_height - 1) / 2) * (
        ops.cast(element_height, "float32") / n_height
    )
    return ops.reshape(ops.tile(u[:, None], (1, n_height)), (-1,)), ops.tile(v, (n_width,))


def _elevation_focusing(v, model):
    """How the elevation focus acts on the sub-elements at height offsets ``v``: as the ideal
    focusing advance [s] of each without the lens, or with it as the lens thickness [m] under
    each, thinned towards the edges. Returns ``(advance, thickness)``, the latter None without
    the lens."""
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
    """Thickness removed from the lens at height offset ``v`` to focus at ``elevation_focus``.

    A slower lens is thickest at the centre, a faster one thinnest (negative sag).
    """
    focus = ops.cast(elevation_focus, v.dtype)
    path = ops.sqrt(focus**2 + v**2) - focus
    return path * lens_sound_speed / (sound_speed - lens_sound_speed)


@dataclass(frozen=True)
class _Path:
    """The paths from one sub-element of every element to every scatterer, (n_scat, n_el, ...):
    the straight ``vector`` [m], the ``medium`` and ``lens`` legs [m] (``lens`` None without
    the lens), ``phase``, the medium distance with the same travel time, and ``spread``, the
    distance with the same spreading."""

    vector: Any
    medium: Any
    lens: Any
    phase: Any
    spread: Any


def _sub_element_path(positions, sub, j, model, lens_normals):
    """The :class:`_Path` from sub-element ``j``: straight, or with the lens the shortest path
    through its local thickness, whose leg counts ``sound_speed / lens_sound_speed`` times in
    ``phase``."""
    vector = positions[:, None] - model.geometry[None] - sub.offsets[j][None]
    if not model.apply_lens_correction:
        distance = ops.linalg.norm(vector, axis=-1)
        return _Path(vector=vector, medium=distance, lens=None, phase=distance, spread=distance)
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
        phase=lens * (model.sound_speed / model.lens_sound_speed) + medium,
        spread=_lens_spread_distance(
            lens, medium, thickness, model.sound_speed, model.lens_sound_speed
        ),
    )


def _lens_spread_distance(lens_len, medium_len, thickness, sound_speed, lens_sound_speed):
    """Distance whose 1/r spreading is the ray-tube divergence of the path refracted at the face.

    The phase path scales the lens leg by c / c_lens, but the wave leaves the face as if from a
    source lens_len * c_lens / c below it (apparent depth). The refracted wavefront is
    astigmatic: that radius holds across the plane of incidence, and within it the radius is
    scaled by cos^2 of the medium angle over cos^2 of the lens angle.
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


def _sub_element_amplitude(f3, path, frame, sub, model):
    """The real amplitude over the frequencies ``f3`` (1, 1, n_freq) along ``path``: the sinc
    directivity of the sub-element in the geometric direction, the attenuation over the lens
    and medium legs, and the baffle obliquity."""
    theta, phi, obliquity = _element_angles(path.vector, frame)
    c = model.sound_speed
    amplitude = directivity(f3, theta[..., None], sub.width, c) * directivity(
        f3, phi[..., None], sub.height, c
    )
    if path.lens is not None:
        amplitude = amplitude * attenuate(f3, model.lens_attenuation_coef, path.lens[..., None])
    amplitude = amplitude * attenuate(f3, model.attenuation_coef, path.medium[..., None])
    return amplitude * obliquity_factor(obliquity, model.baffle_impedance_ratio)[..., None]


def _element_angles(relative, frame):
    """Angles in the width and height directions and cos of the angle to the element normal.

    The sines of theta and phi are the direction cosines along the width and the height over
    r, as in the Fraunhofer pattern of a rectangular aperture (and SIMUS). Projected angles
    arctan2(width, axial) would narrow the height pattern for scatterers off the width axis.

    Args:
        relative (array-like): Scatterer positions relative to the elements, (n_scat, n_el, 3).
        frame (_Frame): Element axes from :func:`_element_frame`.

    Returns:
        theta, phi, obliquity: arrays of shape (n_scat, n_el).
    """
    along_width = ops.sum(relative * frame.width_axis[None], axis=-1)
    along_height = ops.sum(relative * frame.height_axis[None], axis=-1)
    axial = ops.sum(relative * frame.normal[None], axis=-1)
    dist = ops.maximum(ops.linalg.norm(relative, axis=-1), 1e-12)
    theta = ops.arcsin(ops.clip(along_width / dist, -1.0, 1.0))
    phi = ops.arcsin(ops.clip(along_height / dist, -1.0, 1.0))
    obliquity = axial / dist
    return theta, phi, obliquity


def _mean_over_sub_elements(response, n_sub, shape):
    """The mean of the (tx, rx) pairs ``response(j)``, each of ``shape``, over the ``n_sub``
    sub-elements, summed in a loop so that one sub-element's responses are live at a time."""
    if n_sub == 1:
        return response(0)

    def body(j, carry):
        tx, rx = response(j)
        return carry[0] + tx, carry[1] + rx

    zeros = ops.zeros(shape, "complex64")
    tx, rx = ops.fori_loop(0, n_sub, body, (zeros, zeros))
    scale = ops.array(1.0 / n_sub, "complex64")
    return tx * scale, rx * scale


def _travel_time(positions, model, lens_normals):
    """One-way travel time [s] from the element centres to every scatterer, (n_scat, n_el):
    straight, or the shortest path through the lens."""
    if not model.apply_lens_correction:
        distance = ops.linalg.norm(positions[:, None] - model.geometry[None], axis=-1)
        return distance / model.sound_speed
    return compute_lens_corrected_travel_times(
        model.geometry,
        positions,
        lens_thickness=model.lens_thickness,
        c_lens=model.lens_sound_speed,
        c_medium=model.sound_speed,
        n_iter=3,
        element_normals=lens_normals,
    )


def elevation_slab_mask(scatterer_positions, probe_geometry, element_height):
    """Zero out the scatterers an elevation lens never insonifies.

    Returns:
        array-like: 1 inside the slab and 0 outside, of shape (n_scat,).
    """
    if element_height is None:
        raise ValueError("elevation_slab_2d=True requires element_height to be provided.")
    elevation_center = ops.mean(probe_geometry[:, 1])
    offset = ops.abs(scatterer_positions[:, 1] - elevation_center)
    return ops.cast(offset <= element_height / 2, "float32")


def select_elevation_slab(
    scatterer_positions, scatterer_magnitudes, probe_geometry, element_height
):
    """Drop the scatterers an elevation lens never insonifies.

    Not jittable: the output length is data dependent. Under jit use
    :func:`elevation_slab_mask`, which zeroes magnitudes instead and keeps a static shape.

    Returns:
        tuple: the (positions, magnitudes) inside the slab.
    """
    mask = elevation_slab_mask(scatterer_positions, probe_geometry, element_height)
    keep = ops.convert_to_numpy(mask) > 0
    return scatterer_positions[keep], scatterer_magnitudes[keep]


def elevation_slab_bucket(
    scatterer_positions=None,
    scatterer_magnitudes=None,
    probe_geometry=None,
    element_height=None,
    elevation_slab_2d=False,
    bucket_growth=2.0,
    **kwargs,
):
    """
    Prune scatterers outside of the elevation slab. Round up to a power of 2 so jit can cache
    the approximate shape.

    Returns:
        dict: pruned scatterers, or ``{}`` if the input is traced or pruning is disabled.
    """
    del kwargs
    if not elevation_slab_2d or element_height is None:
        return {}
    if scatterer_positions is None or scatterer_magnitudes is None or probe_geometry is None:
        return {}

    try:
        positions = ops.convert_to_numpy(scatterer_positions)
        magnitudes = ops.convert_to_numpy(scatterer_magnitudes)
        geometry = ops.convert_to_numpy(probe_geometry)
    except (RuntimeError, ValueError, TypeError):
        return {}  # traced, fall back to masking

    batched = positions.ndim == 3
    if not batched:
        positions, magnitudes = positions[None], magnitudes[None]

    n_scat = positions.shape[1]
    center = geometry[:, 1].mean()
    inside = np.abs(positions[..., 1] - center) <= element_height / 2

    # ops.map needs a uniform shape when using batched mode
    n_keep = int(inside.sum(axis=1).max())
    if n_keep >= n_scat:
        return {}
    steps = np.ceil(np.log(max(n_keep, 1)) / np.log(bucket_growth))
    bucket = min(n_scat, max(1, int(bucket_growth**steps)))

    index = np.zeros((positions.shape[0], bucket), dtype=np.int64)
    pad_mask = np.ones((positions.shape[0], bucket), dtype=bool)
    for item, row in enumerate(inside):
        kept = np.flatnonzero(row)[:bucket]
        index[item, : len(kept)] = kept
        pad_mask[item, : len(kept)] = False

    positions = np.take_along_axis(positions, index[..., None], axis=1)
    magnitudes = np.where(pad_mask, 0.0, np.take_along_axis(magnitudes, index, axis=1))

    if not batched:
        positions, magnitudes = positions[0], magnitudes[0]
    return {"scatterer_positions": positions, "scatterer_magnitudes": magnitudes}


def _warn_if_elevation_extent(probe_geometry, tol=1e-6):
    """Warn if an elevation lens is used with a seemingly non-1D array probe."""
    try:
        elevation = ops.convert_to_numpy(probe_geometry)[:, 1]
    except (RuntimeError, ValueError, TypeError):
        return  # traced, cannot inspect
    if elevation.max() - elevation.min() > tol:
        log.warning(
            "elevation_slab_2d=True models a 1D probe with a simplified cylindrical elevation lens,"
            " but the probe is not 1D "
            f"(element elevation min, max: {elevation.min()}, {elevation.max()}) "
            "This is probably a mistake."
        )


def _apply_elevation_slab(
    scatterer_positions, scatterer_magnitudes, probe_geometry, element_height
):
    """Prune to the elevation slab, falling back to masking if positions are traced.

    Under jit `elevation_slab_bucket` has usually pruned already, so the mask only re-zeroes
    padding.
    """
    try:
        return select_elevation_slab(
            scatterer_positions, scatterer_magnitudes, probe_geometry, element_height
        )
    except (RuntimeError, ValueError, TypeError):
        mask = elevation_slab_mask(scatterer_positions, probe_geometry, element_height)
        return scatterer_positions, scatterer_magnitudes * mask


def prepare_scatterers(scatterer_positions, scatterer_magnitudes, model):
    """Prepare scatterers for simulation: drop (or zero, under jit) those outside the elevation
    slab when ``model.elevation_slab_2d``, and cast to float32."""
    if model.elevation_slab_2d:
        _warn_if_elevation_extent(model.geometry)
        scatterer_positions, scatterer_magnitudes = _apply_elevation_slab(
            scatterer_positions, scatterer_magnitudes, model.geometry, model.element_height
        )
    return ops.cast(scatterer_positions, "float32"), ops.cast(scatterer_magnitudes, "float32")
