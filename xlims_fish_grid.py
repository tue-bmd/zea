"""Fish phantom under the automatic xlims, per transmit type and probe curvature."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Parameters, display
from zea.beamform import phantoms
from zea.beamform.delays import compute_t0_delays_focused, compute_t0_delays_planewave
from zea.probes import create_curved_probe_geometry
from zea.simulator import simulate_rf

N_EL = 80
APERTURE = 20e-3
CENTER_FREQUENCY = 3e6
SAMPLING_FREQUENCY = 12e6
SOUND_SPEED = 1540.0
ZLIMS = (5e-3, 40e-3)
# ANGLES = np.deg2rad([-20.0, 0.0])
# ANGLES = np.deg2rad([0.0,])
ANGLES = np.deg2rad([-20.0, 20.0])
DYNAMIC_RANGE = (-50.0, 0.0)
TILTS = np.deg2rad([15.0, 30.0])  # curved rows: element tilt at the array edges

COLUMNS = {
    "focused (25 mm)": np.full(len(ANGLES), 25e-3),
    # "plane wave": np.full(len(ANGLES), np.inf),
    # "diverging (-10 mm)": np.full(len(ANGLES), -10e-3),
}


def flat_geometry():
    x = np.linspace(-APERTURE / 2, APERTURE / 2, N_EL)
    return np.stack([x, np.zeros(N_EL), np.zeros(N_EL)], axis=1)


def curved_geometry(tilt):
    radius = APERTURE / (2 * tilt)  # arc of length APERTURE spanning +-tilt
    return create_curved_probe_geometry(n_el=N_EL, pitch=APERTURE / (N_EL - 1), radius=radius)


def parameters_for(probe_geometry, focus_distances):
    if np.all(np.isinf(focus_distances)):
        t0_delays = compute_t0_delays_planewave(
            probe_geometry=probe_geometry, polar_angles=ANGLES, sound_speed=SOUND_SPEED
        )
    else:
        t0_delays = compute_t0_delays_focused(
            transmit_origins=np.zeros((len(ANGLES), 3)),
            focus_distances=focus_distances,
            probe_geometry=probe_geometry,
            polar_angles=ANGLES,
            sound_speed=SOUND_SPEED,
        )
    return Parameters(
        n_tx=len(ANGLES),
        n_el=N_EL,
        n_ch=1,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=SAMPLING_FREQUENCY,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=np.ones((len(ANGLES), N_EL)) * np.hanning(N_EL)[None],
        element_width=float(np.linalg.norm(probe_geometry[1] - probe_geometry[0])),
        focus_distances=focus_distances,
        polar_angles=ANGLES,
        initial_times=np.zeros(len(ANGLES)),
        n_ax=512 + 128,
        # zlims=ZLIMS,
        selected_transmits="all",
        sound_speed=SOUND_SPEED,
        apply_lens_correction=False,
        attenuation_coef=0.0,
    )


def beamformed_image(parameters, scatterers):
    rf_data = simulate_rf(
        scatterer_positions=scatterers,
        scatterer_magnitudes=np.ones(len(scatterers), dtype=np.float32),
        probe_geometry=parameters.probe_geometry,
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        sound_speed=parameters.sound_speed,
        n_ax=parameters.n_ax,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=parameters.sampling_frequency,
        t0_delays=parameters.t0_delays,
        initial_times=parameters.initial_times,
        element_width=parameters.element_width,
        attenuation_coef=parameters.attenuation_coef,
        tx_apodizations=parameters.tx_apodizations,
        # noise_level_db=-30.,
        # tgc_max_db=30.0,
        t_peak=parameters.t_peak,
    )
    # add a tiny bit of noise:
    rf_data += np.random.normal(0.0, 1e-3, size=rf_data.shape).astype(np.float32)
    pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False, baseband=False)
    inputs = pipeline.prepare_parameters(parameters, dynamic_range=DYNAMIC_RANGE)
    image = pipeline(**{**inputs, pipeline.key: rf_data})[pipeline.output_key]
    return np.asarray(display.to_8bit(image, dynamic_range=DYNAMIC_RANGE))


def draw_steering(ax, probe_geometry, zmax, focus_distances):
    """Two rays per transmit, leaving the left and right end of the aperture, aimed at the focus.

    Focused: rays converge at the focus and cross beyond it.
    Diverging: focus is a virtual point behind the array, so rays fan out.
    Plane wave: focus is at infinity, rays stay parallel at the steering angle.
    """
    edges = probe_geometry[
        [int(np.argmin(probe_geometry[:, 0])), int(np.argmax(probe_geometry[:, 0]))]
    ]
    colors = [f"xkcd:{x}" for x in ["salmon", "teal", "lavender", "gold", "lightblue"]]
    for i, (angle, focus_distance) in enumerate(zip(ANGLES, focus_distances)):
        for x0, _, z0 in edges:
            if np.isinf(focus_distance):
                slope = np.tan(angle)
            else:
                focus_x = focus_distance * np.sin(angle)
                focus_z = focus_distance * np.cos(angle)
                slope = (x0 - focus_x) / (z0 - focus_z)
            ax.plot(
                [x0 * 1e3, (x0 + (zmax - z0) * slope) * 1e3],
                [z0 * 1e3, zmax * 1e3],
                color=colors[i % len(colors)],
                alpha=0.5,
                lw=1.5,
            )


def main():
    scatterers = phantoms.fish()
    # filter out only the scatterers near x=0:
    # scatterers = scatterers[np.abs(scatterers[:, 0]) < 1e-3]

    rows = {"flat probe": flat_geometry()}
    for tilt in TILTS:
        rows[f"curved probe (+-{np.rad2deg(tilt):.0f} deg)"] = curved_geometry(tilt)

    fig, axes = plt.subplots(
        len(rows), len(COLUMNS), figsize=(4 * len(COLUMNS), 4 * len(rows)), constrained_layout=True
    )
    if len(COLUMNS) == 1:
        axes = axes[:, None]
    for row, (row_name, probe_geometry) in enumerate(rows.items()):
        for col, (col_name, focus_distances) in enumerate(COLUMNS.items()):
            parameters = parameters_for(probe_geometry, focus_distances)
            image = beamformed_image(parameters, scatterers)
            xlims, zlims = parameters.xlims, parameters.zlims

            ax = axes[row, col]
            ax.imshow(
                image,
                cmap="gray",
                aspect="equal",
                extent=[xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3],
            )
            draw_steering(ax, probe_geometry, max(zlims), focus_distances)
            ax.plot(
                probe_geometry[:, 0] * 1e3,
                probe_geometry[:, 2] * 1e3,
                ".",
                color="deepskyblue",
                ms=2,
            )
            ax.set_xlim(xlims[0] * 1e3, xlims[1] * 1e3)
            ax.set_ylim(zlims[1] * 1e3, min(0.0, probe_geometry[:, 2].min() * 1e3))
            ax.set_title(
                f"{col_name}\nxlims {xlims[0] * 1e3:.1f} to {xlims[1] * 1e3:.1f} mm", fontsize=9
            )
            # if col == 0:
            #     ax.set_ylabel(f"{row_name}\ndepth [mm]", fontsize=9)
            # ax.set_xlabel("x [mm]", fontsize=8)
            ax.tick_params(labelsize=7)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xlims_fish_grid.png")
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    main()
