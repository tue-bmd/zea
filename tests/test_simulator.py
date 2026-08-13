"""
Simulate and beamform the fish phantom with a planewave, then check that every scatterer shows
up as a bright dot. For approximations, check that the image is close to the exact mode results.
"""

from functools import partial

import numpy as np
import pytest

import zea
from zea import Parameters, Probe, display
from zea.beamform import phantoms
from zea.beamform.delays import compute_t0_delays_planewave
from zea.metrics import psnr
from zea.simulator import simulate_rf, simulate_rf_fast

N_EL = 80
APERTURE = 32e-3
N_TX = 5
CENTER_FREQUENCY = 3e6  # Hz
SOUND_SPEED = 1540.0  # m/s
XLIMS = (-24e-3, 24e-3)
ZLIMS = (10e-3, 35e-3)
DYNAMIC_RANGE = (-50.0, 0.0)

SIMULATORS = {
    "exact": partial(simulate_rf, factored=False),
    "factored": partial(simulate_rf, factored=True),
    "fast": simulate_rf_fast,
}

# Minimum PSNR [dB] of each approximation against the exact simulator, after beamforming.
MIN_PSNR = {"factored": 40.0, "fast": 20.0}


def _parameters(probe_geometry):
    angles = np.linspace(-5, 5, N_TX) * np.pi / 180
    wavelength = SOUND_SPEED / CENTER_FREQUENCY
    return Parameters(
        n_tx=N_TX,
        n_el=N_EL,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=CENTER_FREQUENCY * 4,
        probe_geometry=probe_geometry,
        t0_delays=compute_t0_delays_planewave(
            probe_geometry=probe_geometry, polar_angles=angles, sound_speed=SOUND_SPEED
        ),
        tx_apodizations=np.ones((N_TX, N_EL)) * np.hanning(N_EL)[None],
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        focus_distances=np.ones(N_TX) * np.inf,
        polar_angles=angles,
        initial_times=np.ones(N_TX) * 1e-6,
        n_ax=1024,
        xlims=XLIMS,
        zlims=ZLIMS,
        grid_size_x=int((XLIMS[1] - XLIMS[0]) / (0.5 * wavelength)) + 1,
        grid_size_z=int((ZLIMS[1] - ZLIMS[0]) / (0.5 * wavelength)) + 1,
        lens_sound_speed=1000,
        lens_thickness=1e-3,
        n_ch=1,
        selected_transmits="all",
        sound_speed=SOUND_SPEED,
        apply_lens_correction=False,
        attenuation_coef=0.0,
    )


@pytest.fixture(scope="module")
def fish_scan():
    probe_geometry = np.stack(
        [np.linspace(-APERTURE / 2, APERTURE / 2, N_EL), np.zeros(N_EL), np.zeros(N_EL)], axis=1
    )
    probe = Probe(probe_geometry=probe_geometry, probe_center_frequency=CENTER_FREQUENCY)
    parameters = _parameters(probe_geometry)
    positions = phantoms.fish()

    simulation_args = {
        "scatterer_positions": positions,
        "scatterer_magnitudes": np.ones(len(positions), dtype=np.float32),
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
        "t_peak": parameters.t_peak,
    }

    pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False, baseband=False)
    inputs = pipeline.prepare_parameters(parameters, dynamic_range=DYNAMIC_RANGE)

    def beamform(rf_data):
        """Beamformed 8-bit B-mode image, as in the simulation example notebook."""
        kwargs = {**inputs, pipeline.key: rf_data}
        image = pipeline(**kwargs)[pipeline.output_key]
        return np.asarray(display.to_8bit(image, dynamic_range=DYNAMIC_RANGE), dtype=np.float32)

    return positions, simulation_args, beamform


@pytest.fixture(scope="module")
def images(fish_scan):
    _, simulation_args, beamform = fish_scan
    return {mode: beamform(fn(**simulation_args)) for mode, fn in SIMULATORS.items()}


def _dot_brightness(image, positions):
    z = np.linspace(ZLIMS[0], ZLIMS[1], image.shape[0])
    x = np.linspace(XLIMS[0], XLIMS[1], image.shape[1])

    values = []
    for pos_x, _, pos_z in positions:
        values.append(image[np.argmin(np.abs(z - pos_z)), np.argmin(np.abs(x - pos_x))])
    return np.array(values)


@pytest.mark.parametrize("mode", list(SIMULATORS))
def test_simulated_fish_dots_visible(fish_scan, images, mode):
    positions, _, _ = fish_scan
    image = images[mode]
    image_norm = image / image.mean()
    values = _dot_brightness(image_norm, positions)

    darkest = values.min()
    # The center-frequency gains of the time-domain mode flatten the dots against the background.
    assert darkest > (1.5 if mode == "fast" else 2.0), (
        f"{mode} simulator: dimmest scatterer is only {darkest:.1f}x the average brightness"
    )


@pytest.mark.parametrize("mode", list(MIN_PSNR))
def test_simulator_mode_psnr_against_exact(images, mode):
    """The approximate modes stay close to the exact simulator after beamforming."""
    value = float(psnr(images["exact"][..., None], images[mode][..., None], max_val=255))
    min_psnr = MIN_PSNR[mode]
    assert value > min_psnr, (
        f"{mode} mode: PSNR against `exact` is low! {value:.1f} dB, expected {min_psnr:.0f} dB"
    )
