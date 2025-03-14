"""Tests for the Operation and Pipeline classes in ops_v2.py.

# TODO: Run tests for all backends
# TODO: merge with original ops
"""

# pylint: disable=arguments-differ, abstract-class-instantiated, pointless-string-statement
import os

os.environ["KERAS_BACKEND"] = "numpy"
import json

import keras
import numpy as np
import pytest
import matplotlib.pyplot as plt

from usbmd import ops_v2 as ops
from usbmd.config.config import Config
from usbmd.core import DataTypes
from usbmd.probes import Dummy, Probe
from usbmd.registry import ops_v2_registry as ops_registry
from usbmd.scan import Scan, compute_t0_delays_planewave, compute_t0_delays_focused
from usbmd.utils.visualize import set_mpl_style


def _get_default_pipeline(ultrasound_scan):
    """Returns a default pipeline for ultrasound simulation."""
    operations = [
        ops.Simulate(
            apply_lens_correction=ultrasound_scan.apply_lens_correction,
            n_ax=ultrasound_scan.n_ax,
        ),
        ops.TOFCorrection(apply_lens_correction=ultrasound_scan.apply_lens_correction),
        ops.PfieldWeighting(),
        ops.DelayAndSum(),
        ops.EnvelopeDetect(axis=-2),
        ops.LogCompress(output_key="image"),
        ops.Normalize(key="image", output_key="image"),
    ]
    pipeline = ops.Pipeline(operations=operations, jit_options=None)
    return pipeline


def _get_linear_probe():
    """Returns a probe for ultrasound simulation tests."""
    n_el = 128
    aperture = 40e-3
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )

    return Probe(
        probe_geometry=probe_geometry,
        center_frequency=7e6,
        sampling_frequency=28e6,
    )


def _get_phased_array_probe():
    """Returns a probe for ultrasound simulation tests."""
    n_el = 80
    aperture = 20e-3
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )

    return Probe(
        probe_geometry=probe_geometry,
        center_frequency=3.12e6,
        sampling_frequency=12.5e6,
    )


def _get_probe(kind):
    if kind == "linear":
        return _get_linear_probe()
    elif kind == "phased_array":
        return _get_phased_array_probe()
    else:
        raise ValueError(f"Unknown probe kind: {kind}")


def _get_constant_scan_kwargs():
    return {
        "Nx": 256,
        "Nz": 256,
        "lens_sound_speed": 1000,
        "lens_thickness": 1e-3,
        "n_ch": 1,
        "selected_transmits": "all",
        "n_ax": 513,
        "sound_speed": 1540.0,
        "apply_lens_correction": False,
        "attenuation_coef": 0.0,
    }


def _get_planewave_scan(ultrasound_probe):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 5

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]
    probe_geometry = ultrasound_probe.probe_geometry

    angles = np.linspace(10, -10, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * np.inf
    t0_delays = compute_t0_delays_planewave(
        probe_geometry=probe_geometry, polar_angles=angles, sound_speed=sound_speed
    )

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        xlims=(-15e-3, 15e-3),
        zlims=(0, 35e-3),
        **constant_scan_kwargs,
    )


def _get_multistatic_scan(ultrasound_probe):
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 5

    tx_apodizations = np.zeros((n_tx, n_el))
    for n, idx in enumerate(np.linspace(0, n_el - 1, n_tx, dtype=int)):
        tx_apodizations[n, idx] = 1
    probe_geometry = ultrasound_probe.probe_geometry

    focus_distances = np.zeros(n_tx)
    t0_delays = np.zeros((n_tx, n_el))

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        focus_distances=focus_distances,
        polar_angles=np.zeros(n_tx),
        initial_times=np.ones(n_tx) * 1e-6,
        xlims=(-15e-3, 15e-3),
        zlims=(0, 35e-3),
        **_get_constant_scan_kwargs(),
    )


def _get_diverging_scan(ultrasound_probe):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 5

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

    angles = np.linspace(10, -10, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * -15e-3
    t0_delays = compute_t0_delays_focused(
        origins=np.zeros((n_tx, 3)),
        focus_distances=focus_distances,
        probe_geometry=ultrasound_probe.probe_geometry,
        polar_angles=angles,
        sound_speed=sound_speed,
    )
    element_width = np.linalg.norm(
        ultrasound_probe.probe_geometry[1] - ultrasound_probe.probe_geometry[0]
    )

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=ultrasound_probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=element_width,
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        xlims=(-15e-3, 15e-3),
        zlims=(0, 35e-3),
        **constant_scan_kwargs,
    )


def _get_focused_scan(ultrasound_probe):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 5

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

    angles = np.linspace(10, -10, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * 15e-3
    t0_delays = compute_t0_delays_focused(
        origins=np.zeros((n_tx, 3)),
        focus_distances=focus_distances,
        probe_geometry=ultrasound_probe.probe_geometry,
        polar_angles=angles,
        sound_speed=sound_speed,
    )
    element_width = np.linalg.norm(
        ultrasound_probe.probe_geometry[1] - ultrasound_probe.probe_geometry[0]
    )

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=ultrasound_probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=element_width,
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        xlims=(-15e-3, 15e-3),
        zlims=(0, 35e-3),
        **constant_scan_kwargs,
    )


def _get_linescan_scan(ultrasound_probe):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 5

    center_elements = np.linspace(0, n_el + 1, n_tx, dtype=int)
    center_elements = center_elements[1:-1]
    tx_apodizations = np.zeros((n_tx, n_el))
    aperture_size_elements = 24
    origins = []
    for n, idx in enumerate(center_elements):
        el0 = np.clip(idx - aperture_size_elements // 2, 0, n_el)
        el1 = np.clip(idx + aperture_size_elements // 2, 0, n_el)
        tx_apodizations[n, el0:el1] = np.hanning(el1 - el0)[None]
        origins.append(ultrasound_probe.probe_geometry[idx])
    origins = np.stack(origins, axis=0)

    angles = np.zeros(n_tx)

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * 15e-3
    t0_delays = compute_t0_delays_focused(
        origins=origins,
        focus_distances=focus_distances,
        probe_geometry=ultrasound_probe.probe_geometry,
        polar_angles=angles,
        sound_speed=sound_speed,
    )
    element_width = np.linalg.norm(
        ultrasound_probe.probe_geometry[1] - ultrasound_probe.probe_geometry[0]
    )

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=ultrasound_probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=element_width,
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        xlims=(-15e-3, 15e-3),
        zlims=(0, 35e-3),
        **constant_scan_kwargs,
    )


def _get_scan(ultrasound_probe, kind):
    if kind == "planewave":
        return _get_planewave_scan(ultrasound_probe)
    elif kind == "multistatic":
        return _get_multistatic_scan(ultrasound_probe)
    elif kind == "diverging":
        return _get_diverging_scan(ultrasound_probe)
    elif kind == "focused":
        return _get_focused_scan(ultrasound_probe)
    elif kind == "linescan":
        return _get_linescan_scan(ultrasound_probe)
    else:
        raise ValueError(f"Unknown scan kind: {kind}")


@pytest.fixture
def ultrasound_scatterers():
    """Returns scatterer positions and magnitudes for ultrasound simulation tests."""
    scat_x, scat_z = np.meshgrid(
        np.linspace(-10e-3, 10e-3, 5),
        np.linspace(10e-3, 30e-3, 5),
        indexing="ij",
    )
    scat_x, scat_z = np.ravel(scat_x), np.ravel(scat_z)
    # scat_x, scat_z = np.array([-10e-3, 0e-3]), np.array([10e-3, 20e-3])
    n_scat = len(scat_x)
    scat_positions = np.stack(
        [
            scat_x,
            np.zeros_like(scat_x),
            scat_z,
        ],
        axis=1,
    )

    return {
        "positions": scat_positions.astype(np.float32),
        "magnitudes": np.ones(n_scat, dtype=np.float32),
        "n_scat": n_scat,
    }


@pytest.mark.parametrize(
    "probe_kind, scan_kind",
    [
        ("linear", "planewave"),
        ("phased_array", "planewave"),
        ("linear", "multistatic"),
        ("phased_array", "multistatic"),
        ("linear", "diverging"),
        ("phased_array", "diverging"),
        ("linear", "focused"),
        ("phased_array", "focused"),
        ("linear", "linescan"),
    ],
)
def test_default_ultrasound_pipeline(
    probe_kind,
    scan_kind,
    ultrasound_scatterers,
):
    """Tests the default ultrasound pipeline."""

    ultrasound_probe = _get_probe(probe_kind)
    ultrasound_scan = _get_scan(ultrasound_probe, scan_kind)
    default_pipeline = _get_default_pipeline(ultrasound_scan)
    # all dynamic parameters are set in the call method of the operations
    # or equivalently in the pipeline call (which is passed to the operations)
    output_default = default_pipeline(
        ultrasound_scan,
        ultrasound_probe,
        scatterer_positions=ultrasound_scatterers["positions"],
        scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
        dynamic_range=(-50, 0),
        input_range=(-50, 0),
        output_range=(0, 255),
    )

    image = output_default["image"][0]
    set_mpl_style()
    plt.figure()
    extent = [
        ultrasound_scan.xlims[0] * 1e3,
        ultrasound_scan.xlims[1] * 1e3,
        ultrasound_scan.zlims[1] * 1e3,
        ultrasound_scan.zlims[0] * 1e3,
    ]
    plt.imshow(image, cmap="gray", aspect="auto", extent=extent)
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.title(f"{probe_kind} {scan_kind}")
    plt.savefig(f"{probe_kind}_{scan_kind}.png")
    plt.close()
    # Check that the pipeline produced the expected outputs
    assert "image" in output_default
    assert output_default["image"].shape[0] == 1  # Batch dimension
    # Verify the normalized image has values between 0 and 255
    assert np.nanmin(output_default["image"]) >= 0.0
    assert np.nanmax(output_default["image"]) <= 255.0
