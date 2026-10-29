"""Diagrams of the polar pixel grid of a `zea.Parameters` object.

Shows how ``zlims``, ``xlims``, ``rlims``, ``polar_limits``, ``grid_apex``,
``extent`` and ``probe_geometry`` relate, for a curved array and for a flat phased array.
Every quantity is read back from a `zea.Parameters` object rather than drawn by hand,
so the figures keep showing what the code does. The interactive explorer
(``docs/explorer``) covers every other combination.

Writes a light and a dark rendering of each figure to ``docs/_static/``. Uses
matplotlib's built-in mathtext with Computer Modern, so no LaTeX installation is needed.
Run from the repository root::

    python docs/figures/pixelgrid.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Polygon, Rectangle

import zea
from zea.probes import create_curved_probe_geometry, create_probe_geometry

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cmr10"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True  # cmr10 has no unicode minus glyph
plt.rcParams["lines.linewidth"] = 0.8

OUTPUT_DIR = Path(__file__).parents[1] / "_static"

# Monochrome on a transparent background: elements are told apart by line weight,
# dash pattern and their labels rather than by hue. ``backdrop`` is the colour the
# small patches behind labels are painted in, matching furo's page background so
# they mask the lines underneath without showing an edge.
THEMES = {
    "Light": {"ink": "#000000", "backdrop": "#ffffff"},
    "Dark": {"ink": "#ffffff", "backdrop": "#131416"},
}

TT = "cmtt10"  # typewriter font for parameter names, like \texttt in LaTeX
RECESSIVE = 0.45  # alpha for reference lines that should not compete with the geometry
DASHED = (0, (4, 3))

MM = 1e3  # plot everything in mm


def curved_array():
    """A curved array (40 mm radius, +-30 degrees) imaged on a +-35 degree sector."""
    n_el = 128
    return zea.Parameters(
        probe_geometry=create_curved_probe_geometry(
            n_el=n_el, pitch=2 * 40e-3 * np.radians(30) / (n_el - 1), radius=40e-3
        ),
        grid_type="polar",
        polar_limits=np.radians([-35.0, 35.0]),
        zlims=(10e-3, 100e-3),
        grid_size_x=64,
        grid_size_z=128,
        center_frequency=3e6,
        sampling_frequency=12e6,
        n_el=n_el,
        n_tx=1,
        n_ax=2048,
    )


def phased_array():
    """A flat phased array with focused lines steered +-45 degrees from its middle."""
    n_el, n_tx, span = 64, 65, np.radians(45)
    return zea.Parameters(
        probe_geometry=create_probe_geometry(n_el=n_el, pitch=0.3e-3),
        grid_type="polar",
        polar_angles=np.linspace(-span, span, n_tx),
        transmit_origins=np.zeros((n_tx, 3)),
        focus_distances=np.full(n_tx, 60e-3),
        zlims=(15e-3, 75e-3),
        grid_size_x=64,
        grid_size_z=128,
        center_frequency=2.5e6,
        sampling_frequency=10e6,
        n_el=n_el,
        n_tx=n_tx,
        n_ax=2048,
    )


def ray(theta):
    """Unit vector (x, z) of a ray at polar angle ``theta`` from the +z axis toward +x."""
    return np.array([np.sin(theta), np.cos(theta)])


def draw(parameters, theme, path):
    """Draw the grid of ``parameters`` in the given color theme and save it to ``path``."""
    ink = theme["ink"]
    backdrop = dict(facecolor=theme["backdrop"], edgecolor="none", alpha=0.9, pad=1.5)
    label = dict(color=ink, fontsize=9)
    code = dict(color=ink, fontsize=9, family=TT)

    grid = parameters.grid * MM  # (grid_size_z, grid_size_x, 3)
    zlims = np.asarray(parameters.zlims) * MM
    xlims = np.asarray(parameters.xlims) * MM
    polar_limits = np.asarray(parameters.polar_limits)
    apex = np.asarray(parameters.grid_apex)[[0, 2]] * MM
    rlims = np.asarray(parameters.rlims) * MM
    extent = np.asarray(parameters.extent)[[0, 1, 4, 5]] * MM  # x0, x1, z0, z1
    probe = np.asarray(parameters.probe_geometry)[:, [0, 2]] * MM

    span = max(extent[1] - extent[0], extent[3] - apex[1])
    unit = span / 100  # label offsets scale with the figure
    left = extent[0] - 16 * unit
    top = min(apex[1], probe[:, 1].min()) - 14 * unit

    fig, ax = plt.subplots(figsize=(6.5, 5.6))

    # Transducer surface (z = 0) and the central axis (x = 0) the angles are measured from
    ax.axhline(0, color=ink, lw=0.8, ls=DASHED, alpha=RECESSIVE)
    ax.text(left, -unit, r"$z=0$", **label, ha="left", va="bottom", alpha=RECESSIVE)
    ax.plot(
        [0, 0],
        [top + 4 * unit, extent[3] + 5 * unit],
        color=ink,
        lw=0.6,
        ls=DASHED,
        alpha=RECESSIVE,
    )
    ax.text(unit, top + 4 * unit, r"$x=0$", **label, ha="left", va="bottom", alpha=RECESSIVE)

    # Axis orientation: +x to the right, +z (into the medium) downward
    origin = np.array([left + unit, top - 2 * unit])
    for direction, name in [((8 * unit, 0), r"$x$"), ((0, 8 * unit), r"$z$")]:
        tip = origin + direction
        ax.annotate("", xy=tip, xytext=origin, arrowprops=dict(arrowstyle="->", color=ink, lw=0.7))
        ax.text(*(tip + (1.5 * unit, 0)), name, **label, ha="left", va="center")

    # Pixel grid: a sparse polar mesh inside the region, with the outline on top
    for i in range(0, grid.shape[0], 8):
        ax.plot(grid[i, :, 0], grid[i, :, 2], color=ink, lw=0.35, alpha=0.35)
    for j in range(0, grid.shape[1], 4):
        ax.plot(grid[:, j, 0], grid[:, j, 2], color=ink, lw=0.35, alpha=0.35)
    outline = np.concatenate([grid[0, :, :], grid[:, -1, :], grid[-1, ::-1, :], grid[::-1, 0, :]])
    ax.fill(outline[:, 0], outline[:, 2], color=ink, alpha=0.06, zorder=0)
    ax.plot(outline[:, 0], outline[:, 2], color=ink, lw=1.6)

    # On a flat array the image starts at zlims[0]: hatch what lies above it
    if extent[2] > grid[..., 2].min() + 1e-9:
        hatch = Polygon(
            outline[:, [0, 2]],
            closed=True,
            fill=False,
            hatch="////",
            edgecolor=ink,
            lw=0,
            alpha=0.5,
        )
        ax.add_patch(hatch)
        hatch.set_clip_path(
            Rectangle(
                (extent[0] - span, top - span),
                3 * span,
                extent[2] - top + span,
                transform=ax.transData,
            )
        )

    # The two edge rays, drawn back to the apex they fan out from
    for edge in (grid[0, 0, [0, 2]], grid[0, -1, [0, 2]]):
        ax.plot(*np.stack([apex, edge], axis=-1), color=ink, lw=0.6, ls=DASHED, alpha=RECESSIVE)
    ax.text(
        0, extent[3] - 22 * unit, "pixel grid", **label, ha="center", va="bottom", bbox=backdrop
    )
    ax.text(
        0, extent[3] - 21 * unit, "grid_type='polar'", **code, ha="center", va="top", bbox=backdrop
    )

    # polar_limits: signed angles from the central axis, positive toward +x
    arc_radius = 0.4 * rlims[1]
    for theta, name in zip(polar_limits, ("polar_limits[0]", "polar_limits[1]")):
        start, end = sorted([90.0, 90.0 - np.degrees(theta)])  # matplotlib: CCW from +x
        ax.add_patch(
            Arc(apex, 2 * arc_radius, 2 * arc_radius, theta1=start, theta2=end, color=ink, lw=0.9)
        )
        ax.annotate(
            "",
            xy=apex + arc_radius * ray(theta),
            xytext=apex + arc_radius * ray(0.97 * theta),
            arrowprops=dict(arrowstyle="-|>", color=ink, lw=0.9, mutation_scale=7),
        )
        ax.text(
            *(apex + (arc_radius + 2.5 * unit) * ray(theta / 2)),
            name,
            **code,
            ha="right" if theta < 0 else "left",
            va="top",
            bbox=backdrop,
        )

    # rlims: radii from the apex, as dimension lines alongside the right edge ray
    theta = polar_limits[1]
    direction = ray(theta)
    outward = np.array([np.cos(theta), -np.sin(theta)])
    rotation = np.degrees(np.arctan2(-direction[1], direction[0])) % 180
    rotation = rotation - 180 if rotation > 90 else rotation
    for r, offset, name in [(rlims[0], 8, "rlims[0]"), (rlims[1], 15, "rlims[1]")]:
        start = apex + offset * unit * outward
        end = apex + r * direction + offset * unit * outward
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="<->", color=ink, lw=0.9))
        for anchor, dimension_point in [(apex, start), (apex + r * direction, end)]:
            ax.plot(*np.stack([anchor, dimension_point], axis=-1), color=ink, lw=0.4)
        # A short dimension cannot hold its label, which then sits past the far end
        at = (start + end) / 2 if r > 30 * unit else end + 12 * unit * direction
        ax.text(
            *at,
            name,
            **code,
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
            bbox=backdrop,
        )

    # Probe geometry
    ax.plot(probe[:, 0], probe[:, 1], color=ink, lw=1.6, solid_capstyle="round")
    step = max(len(probe) // 9, 1)
    ax.plot(probe[::step, 0], probe[::step, 1], "o", color=ink, ms=2.2)
    # Labels on the left, clear of the rlims dimension lines on the right
    ax.annotate(
        "probe_geometry",
        xy=probe[0],
        xytext=(probe[0, 0] - 14 * unit, min(probe[0, 1], 0) - 7 * unit),
        **code,
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="-", color=ink, lw=0.6, shrinkB=2),
    )

    # grid_apex, with its distance behind the array when it is not the array middle
    ax.plot(*apex, "o", color=ink, ms=4, zorder=3)
    if -apex[1] > 5 * unit:
        x = -4 * unit
        ax.annotate(
            "", xy=(x, 0), xytext=(x, apex[1]), arrowprops=dict(arrowstyle="<->", color=ink, lw=0.9)
        )
        ax.plot([apex[0], x - unit], [apex[1], apex[1]], color=ink, lw=0.4)
        ax.text(
            x - unit,
            0.72 * apex[1],
            "-grid_apex[2]",
            **code,
            ha="right",
            va="center",
            bbox=backdrop,
        )
        ax.text(
            apex[0],
            apex[1] - 2.5 * unit,
            "grid_apex",
            **code,
            ha="center",
            va="bottom",
            bbox=backdrop,
        )
    else:
        ax.annotate(
            "grid_apex",
            xy=apex,
            xytext=(-8 * unit, top + 3 * unit),
            **code,
            ha="right",
            va="center",
            arrowprops=dict(arrowstyle="-", color=ink, lw=0.6, shrinkA=2, shrinkB=3),
        )

    # zlims: depths below the transducer surface, measured on the central axis
    for z, name in zip(zlims, ("zlims[0]", "zlims[1]")):
        ax.plot([extent[0] - 2 * unit, 0], [z, z], color=ink, lw=0.6, ls=DASHED, alpha=RECESSIVE)
        ax.text(extent[0] - 4 * unit, z, name, **code, ha="right", va="center")

    # The scan-converted image: Parameters.extent, spanning xlims
    ax.add_patch(
        Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            facecolor="none",
            edgecolor=ink,
            lw=1.1,
            ls=(0, (5, 3)),
            zorder=-1,
        )
    )
    below = extent[3] + 2.5 * unit
    for x, name, ha in [(xlims[0], "xlims[0]", "left"), (xlims[1], "xlims[1]", "right")]:
        ax.text(x, below, name, **code, ha=ha, va="top")
    ax.text(
        extent[1],
        below + 6 * unit,
        "scan-converted image\n(Parameters.extent)",
        **label,
        ha="right",
        va="top",
    )
    ax.text(
        extent[0],
        below + 6 * unit,
        "rlims = zlims - grid_apex[2]",
        color=ink,
        fontsize=8,
        family=TT,
        ha="left",
        va="top",
    )

    ax.set_aspect("equal")
    ax.set_xlim(left - 2 * unit, extent[1] + 22 * unit)
    ax.set_ylim(below + 18 * unit, top - 6 * unit)
    ax.set_axis_off()

    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


FIGURES = {"pixelgrid": curved_array, "pixelgrid-phased-array": phased_array}

if __name__ == "__main__":
    for name, make in FIGURES.items():
        parameters = make()
        for theme_name, theme_colors in THEMES.items():
            draw(parameters, theme_colors, OUTPUT_DIR / f"{name}-{theme_name}.svg")
