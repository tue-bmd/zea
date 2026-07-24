import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["ZEA_DISABLE_CACHE"] = "1"

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Parameters, Probe, init_device
from zea.beamform import phantoms
from zea.beamform.delays import compute_t0_delays_focused
from zea.simulator import simulate_rf
from zea.visualize import set_mpl_style

# In[4]:


init_device(verbose=False)
set_mpl_style()


center_frequency = 3e6
n_el = 64
n_tx = 1
wavelength = 1540.0 / center_frequency
probe_geometry = np.stack(
    [
        (np.arange(n_el) - (n_el - 1) / 2) * wavelength,
        np.zeros(n_el),
        np.zeros(n_el),
    ],
    axis=1,
)
transmit_origins = np.array([0, 0, 0])[None] * np.ones((n_tx, 1))
focus_distances = np.ones(n_tx) * 20e-3
angles = np.array([0]) * np.pi / 180


def get_image(focal_region_margin=0.0):
    # Define a linear probe

    probe = Probe(
        probe_geometry=probe_geometry,
        probe_center_frequency=5e6,
    )

    # Define a planewave scan

    # 40 degrees in radians
    sound_speed = 1540.0

    # Set grid and image size
    xlims = (-20e-3, 20e-3)
    zlims = (10e-3, 35e-3)
    width, height = xlims[1] - xlims[0], zlims[1] - zlims[0]
    wavelength = sound_speed / probe.probe_center_frequency
    grid_size_x = int(width / (0.5 * wavelength)) + 1
    grid_size_z = int(height / (0.5 * wavelength)) + 1

    t0_delays = compute_t0_delays_focused(
        transmit_origins=transmit_origins,
        focus_distances=focus_distances,
        probe_geometry=probe_geometry,
        polar_angles=angles,
        sound_speed=sound_speed,
    )
    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

    parameters = Parameters(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=probe.probe_center_frequency,
        sampling_frequency=probe.probe_center_frequency * 4,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=1024,
        xlims=xlims,
        zlims=zlims,
        grid_size_x=grid_size_x,
        grid_size_z=grid_size_z,
        lens_sound_speed=1000,
        lens_thickness=1e-3,
        n_ch=1,
        selected_transmits="all",
        sound_speed=sound_speed,
        apply_lens_correction=False,
        attenuation_coef=0.0,
        focal_region_margin=focal_region_margin,
        transmit_origins=transmit_origins,
    )

    # Create the phantom scatterer positions and magnitudes
    positions = phantoms.fish() + np.array([-8e-3, 0, 0])[None]
    magnitudes = np.ones(len(positions), dtype=np.float32)

    simulation_args = {
        "scatterer_positions": positions,
        "scatterer_magnitudes": magnitudes,
        "probe_geometry": probe.probe_geometry,
        "apply_lens_correction": parameters.apply_lens_correction,
        "lens_thickness": parameters.lens_thickness,
        "lens_sound_speed": parameters.lens_sound_speed,
        "sound_speed": parameters.sound_speed,
        "n_ax": parameters.n_ax,
        "center_frequency": probe.probe_center_frequency,
        "sampling_frequency": parameters.sampling_frequency,
        "t0_delays": parameters.t0_delays,
        "initial_times": parameters.initial_times,
        "element_width": parameters.element_width,
        "attenuation_coef": parameters.attenuation_coef,
        "tx_apodizations": parameters.tx_apodizations,
    }

    rf_data = simulate_rf(**simulation_args)
    print("Simulated RF data shape:", rf_data.shape)

    pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False, baseband=False)
    inputs = pipeline.prepare_parameters(parameters, dynamic_range=(-50, 0))
    inputs[pipeline.key] = rf_data
    outputs = pipeline(**inputs)
    image = outputs[pipeline.output_key]

    image = zea.display.to_8bit(image, dynamic_range=(-50, 0))

    return image


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, focal_region_margin in enumerate([0.0, 5e-3]):
    image = get_image(focal_region_margin=focal_region_margin)
    axes[i].imshow(image, cmap="gray", extent=[-20, 20, 35, 10])
    axes[i].set_title(f"Focal Region Margin: {focal_region_margin * 1e3:.1f} mm")
    axes[i].set_xlabel("Lateral (mm)")
    axes[i].plot(
        probe_geometry[:, 0] * 1e3,
        np.zeros(probe_geometry.shape[0]),
        "C0-",
        linewidth=2,
        label="Probe Elements",
    )

axes[1].set_yticks([])
axes[0].set_ylabel("Axial (mm)")
# axes[1].legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("simulation_plot_images.png")
