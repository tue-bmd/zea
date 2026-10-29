"""Reference values for the pixel grid explorer, computed by zea itself.

The explorer (``pixel_grid_explorer.html``) re-implements the grid geometry of
:class:`zea.Parameters` in JavaScript. It embeds the values computed here and checks
its own port against them when the page loads; ``tests/test_pixel_grid_explorer.py``
runs the same comparison against the current zea.

Each case is a page state (the keys the page's ``resolve(s)`` reads), and its expected
values come from building that state as a :class:`zea.Parameters` object.
"""

import numpy as np

import zea
from zea.display import compute_scan_convert_2d_coordinates
from zea.probes import create_curved_probe_geometry, create_probe_geometry

CURVED_PITCH = 2 * 40e-3 * np.radians(30) / 127

BASE = dict(
    probe="flat",
    n_el=64,
    pitch=0.3e-3,
    radius=40e-3,
    grid_type="polar",
    enable_scanline=False,
    zlims=None,
    n_ax=1300,
    polar_limits=None,
    polar_margin=None,
    xlims=None,
    grid_size_x=64,
    grid_size_z=128,
    center_frequency=2.5e6,
    sampling_frequency=10e6,
    sound_speed=1540.0,
    f_number=None,
    element_width=None,
    pixels_per_wavelength=4,
    n_tx=65,
    tx_span_deg=45.0,
    tx_origins="zeros",
    focus_distance=60e-3,
)
CURVED = dict(
    probe="curved",
    n_el=128,
    pitch=CURVED_PITCH,
    center_frequency=3e6,
    sampling_frequency=12e6,
    n_ax=1560,
)


def cases():
    """Page states covering the rules for the apex, sector, extents and scanline grids."""
    focused, plane, diverging = 60e-3, None, -20e-3
    out = []
    for grid_type in ("polar", "cartesian"):
        for origins in ("zeros", "array"):
            for focus in (focused, plane, diverging):
                out.append(
                    dict(BASE, grid_type=grid_type, tx_origins=origins, focus_distance=focus)
                )
    out += [
        # Fully default grid sizes and depth
        dict(
            BASE, grid_size_x=None, grid_size_z=None, n_tx=21, tx_span_deg=20.0, focus_distance=None
        ),
        # One diverging wave, and unsteered plane waves with an overridden sector
        dict(BASE, n_tx=1, tx_span_deg=0.0, focus_distance=-12e-3),
        dict(BASE, n_tx=1, tx_span_deg=0.0, focus_distance=None, polar_limits=[-0.6, 0.7]),
        # Overrides
        dict(BASE, n_tx=21, tx_span_deg=20.0, focus_distance=None, polar_margin=0.05),
        dict(BASE, zlims=[0.01, 0.07], xlims=[-0.02, 0.03]),
        dict(BASE, grid_type="cartesian", xlims=[-0.02, 0.025], focus_distance=None),
        dict(BASE, f_number=1.5, center_frequency=5e6, sampling_frequency=20e6),
        dict(BASE, center_frequency=5e6, sampling_frequency=20e6, element_width=0.29e-3),
        # Curved arrays
        dict(BASE, **CURVED, n_tx=1, tx_span_deg=0.0, focus_distance=None),
        dict(BASE, **CURVED, polar_limits=[-0.6, 0.6], zlims=[0.01, 0.1], n_tx=1, tx_span_deg=0.0),
        dict(BASE, **CURVED, tx_origins="array", n_tx=64, tx_span_deg=30.0),
        dict(BASE, **CURVED, tx_origins="zeros", n_tx=33, tx_span_deg=30.0, focus_distance=None),
        dict(BASE, **CURVED, grid_type="cartesian", n_tx=11, tx_span_deg=10.0, focus_distance=None),
        # Scanline grids: accepted ...
        dict(BASE, enable_scanline=True),
        dict(BASE, enable_scanline=True, tx_origins="array", n_tx=33),
        dict(BASE, **CURVED, enable_scanline=True, tx_origins="array", n_tx=64, tx_span_deg=30.0),
        dict(
            BASE,
            grid_type="cartesian",
            enable_scanline=True,
            tx_origins="array",
            n_tx=17,
            tx_span_deg=0.0,
            focus_distance=30e-3,
        ),
        # ... and rejected
        dict(BASE, enable_scanline=True, focus_distance=None),
        dict(BASE, enable_scanline=True, focus_distance=diverging),
        dict(BASE, grid_type="cartesian", enable_scanline=True),
        dict(BASE, enable_scanline=True, tx_span_deg=0.0, n_tx=9),
    ]
    return out


def transmits(case, probe_geometry):
    """The page's transmit set (mirrors ``makeTransmits`` in the page)."""
    n_tx = case["n_tx"]
    span = np.radians(case["tx_span_deg"])
    angles = np.zeros(n_tx) if n_tx == 1 else np.linspace(-span, span, n_tx)
    origins = np.zeros((n_tx, 3))
    if case["tx_origins"] == "array":
        if case["probe"] == "curved":
            radius = case["radius"]
            origins[:, 0] = radius * np.sin(angles)
            origins[:, 2] = radius * np.cos(angles) - radius
        else:
            half = (probe_geometry[:, 0].max() - probe_geometry[:, 0].min()) / 2
            if n_tx > 1 and span > 0:
                # The axes meet behind the array; the outermost transmit leaves an end element
                origins[:, 0] = half / np.tan(span) * np.tan(angles)
            else:
                origins[:, 0] = 0.0 if n_tx == 1 else np.linspace(-half, half, n_tx)
    focus = case["focus_distance"]
    focus = np.full(n_tx, np.inf if focus is None else focus)
    return angles.astype(np.float32), origins.astype(np.float32), focus.astype(np.float32)


def parameters_for(case):
    """Build the :class:`zea.Parameters` a page state describes."""
    if case["probe"] == "curved":
        probe_geometry = create_curved_probe_geometry(case["n_el"], case["pitch"], case["radius"])
    else:
        probe_geometry = create_probe_geometry(case["n_el"], case["pitch"])
    angles, origins, focus = transmits(case, probe_geometry)
    kwargs = dict(
        probe_geometry=probe_geometry,
        grid_type=case["grid_type"],
        enable_scanline=case["enable_scanline"],
        center_frequency=case["center_frequency"],
        sampling_frequency=case["sampling_frequency"],
        sound_speed=case["sound_speed"],
        pixels_per_wavelength=case["pixels_per_wavelength"],
        n_el=case["n_el"],
        n_tx=case["n_tx"],
        n_ax=case["n_ax"],
        polar_angles=angles,
        transmit_origins=origins,
        focus_distances=focus,
    )
    optional = (
        "zlims",
        "polar_limits",
        "polar_margin",
        "xlims",
        "grid_size_x",
        "grid_size_z",
        "f_number",
        "element_width",
    )
    for key in optional:
        if case[key] is not None:
            kwargs[key] = case[key]
    return zea.Parameters(**kwargs)


def expected_for(case):
    """The values the page must reproduce for one state."""
    parameters = parameters_for(case)
    try:
        grid = np.asarray(parameters.grid, dtype=np.float64)
    except ValueError:
        if case["enable_scanline"]:
            return {"scanline_error": True}
        raise

    n_z, n_x = grid.shape[:2]
    samples = {}
    for i, j in [(0, 0), (0, n_x - 1), (n_z - 1, 0), (n_z - 1, n_x - 1), (n_z // 2, n_x // 2)]:
        samples[f"{i},{j}"] = [float(grid[i, j, 0]), float(grid[i, j, 2])]

    out = {
        "grid_apex": [float(v) for v in parameters.grid_apex],
        "polar_limits": [float(v) for v in parameters.polar_limits],
        "xlims": [float(v) for v in parameters.xlims],
        "zlims": [float(v) for v in parameters.zlims],
        "rho_range": [float(v) for v in parameters.rho_range],
        "f_number": float(parameters.f_number),
        "grid_size_x": int(parameters.grid_size_x),
        "grid_size_z": int(parameters.grid_size_z),
        "grid_shape": [int(n_z), int(n_x)],
        "extent": [float(v) for v in parameters.extent],
        "grid_samples": samples,
    }
    if case["grid_type"] == "polar":
        _, scan_convert = compute_scan_convert_2d_coordinates(
            (parameters.grid_size_z, parameters.grid_size_x),
            parameters.rho_range,
            parameters.theta_range,
            grid_apex=parameters.grid_apex,
            extent=parameters.extent,
        )
        out["scan_convert"] = {
            key: [float(np.asarray(v)) for v in scan_convert[key]] for key in ("x_lim", "z_lim")
        }
        out["scan_convert"]["resolution"] = float(np.asarray(scan_convert["resolution"]))
    return out


def reference():
    """All cases with their expected values, as the page embeds them."""
    return [{"config": case, "expected": expected_for(case)} for case in cases()]
