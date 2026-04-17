"""Module to create dummy Scan objects for testing and simulation purposes."""

import numpy as np

from zea.beamform.delays import compute_t0_delays_focused, compute_t0_delays_planewave
from zea.probes import Probe
from zea.scan import Scan


def _get_linear_probe():
    """Returns a probe for ultrasound simulation tests."""
    n_el = 128
    aperture = 30e-3
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
        center_frequency=2.5e6,
        sampling_frequency=10e6,
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


def _get_n_ax(ultrasound_probe):
    """Returns the number of ax for ultrasound simulation tests based on the center
    frequency. A probe with a higher center frequency needs more samples to cover
    the image depth.
    """
    is_low_frequency_probe = ultrasound_probe.center_frequency < 4e6

    if is_low_frequency_probe:
        return 510

    return 1024


def _get_probe(kind) -> Probe:
    if kind == "linear":
        return _get_linear_probe()
    elif kind == "phased_array":
        return _get_phased_array_probe()
    else:
        raise ValueError(f"Unknown probe kind: {kind}")


def _get_constant_scan_kwargs():
    return {
        "lens_sound_speed": 1000,
        "lens_thickness": 1e-3,
        "n_ch": 1,
        "selected_transmits": "all",
        "sound_speed": 1540.0,
        "apply_lens_correction": False,
        "attenuation_coef": 0.0,
    }


def _get_lims_and_gridsize(center_frequency, sound_speed):
    """Returns the limits and gridsize for ultrasound simulation tests."""
    xlims, zlims = (-20e-3, 20e-3), (0, 35e-3)
    width, height = xlims[1] - xlims[0], zlims[1] - zlims[0]
    wavelength = sound_speed / center_frequency
    gridsize = (
        int(width / (0.5 * wavelength)) + 1,
        int(height / (0.5 * wavelength)) + 1,
    )
    return {"xlims": xlims, "zlims": zlims, "grid_size_x": gridsize[0], "grid_size_z": gridsize[1]}


def _get_planewave_scan(ultrasound_probe, grid_type, **kwargs):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]
    probe_geometry = ultrasound_probe.probe_geometry

    angles = np.linspace(10, -10, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    t0_delays = compute_t0_delays_planewave(
        probe_geometry=probe_geometry, polar_angles=angles, sound_speed=sound_speed
    )

    # Focus distances can be overriden via kwargs
    if "focus_distances" not in kwargs:
        kwargs["focus_distances"] = np.ones(n_tx) * np.inf

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(ultrasound_probe.center_frequency, sound_speed),
        **constant_scan_kwargs,
        **kwargs,
    )


def _get_multistatic_scan(ultrasound_probe, grid_type, **kwargs):
    n_el = ultrasound_probe.n_el
    n_tx = 8

    tx_apodizations = np.zeros((n_tx, n_el))
    for n, idx in enumerate(np.linspace(0, n_el - 1, n_tx, dtype=int)):
        tx_apodizations[n, idx] = 1
    probe_geometry = ultrasound_probe.probe_geometry

    focus_distances = np.zeros(n_tx)
    t0_delays = np.zeros((n_tx, n_el))

    constant_scan_kwargs = _get_constant_scan_kwargs()

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
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(
            ultrasound_probe.center_frequency, constant_scan_kwargs["sound_speed"]
        ),
        **constant_scan_kwargs,
        **kwargs,
    )


def _get_diverging_scan(ultrasound_probe, grid_type, **kwargs):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

    angles = np.linspace(10, -10, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * -15e-3
    transmit_origins = np.zeros((n_tx, 3))
    t0_delays = compute_t0_delays_focused(
        transmit_origins=transmit_origins,
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
        transmit_origins=transmit_origins,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(ultrasound_probe.center_frequency, sound_speed),
        **constant_scan_kwargs,
        **kwargs,
    )


def _get_focused_scan(ultrasound_probe, grid_type, **kwargs):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

    angles = np.linspace(30, -30, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * 15e-3
    transmit_origins = np.zeros((n_tx, 3))
    t0_delays = compute_t0_delays_focused(
        transmit_origins=transmit_origins,
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
        transmit_origins=transmit_origins,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(ultrasound_probe.center_frequency, sound_speed),
        **constant_scan_kwargs,
        **kwargs,
    )


def _get_linescan_scan(ultrasound_probe, grid_type, **kwargs):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    center_elements = np.linspace(0, n_el + 1, n_tx + 2, dtype=int)
    center_elements = center_elements[1:-1]
    tx_apodizations = np.zeros((n_tx, n_el))
    aperture_size_elements = 24

    # Define subapertures
    transmit_origins = []
    for n, idx in enumerate(center_elements):
        el0 = np.clip(idx - aperture_size_elements // 2, 0, n_el)
        el1 = np.clip(idx + aperture_size_elements // 2, 0, n_el)
        tx_apodizations[n, el0:el1] = np.hanning(el1 - el0)[None]
        transmit_origins.append(ultrasound_probe.probe_geometry[idx])
    transmit_origins = np.stack(transmit_origins, axis=0)

    # All angles should be zero because each line fires straight ahead
    angles = np.zeros(n_tx)

    sound_speed = constant_scan_kwargs["sound_speed"]

    focus_distances = np.ones(n_tx) * 15e-3
    t0_delays = compute_t0_delays_focused(
        transmit_origins=transmit_origins,
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
        transmit_origins=transmit_origins,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(ultrasound_probe.center_frequency, sound_speed),
        **constant_scan_kwargs,
        **kwargs,
    )


def _get_scan(ultrasound_probe, kind, grid_type="cartesian", **kwargs) -> Scan:
    if kind == "planewave":
        return _get_planewave_scan(ultrasound_probe, grid_type, **kwargs)
    elif kind == "multistatic":
        return _get_multistatic_scan(ultrasound_probe, grid_type, **kwargs)
    elif kind == "diverging":
        return _get_diverging_scan(ultrasound_probe, grid_type, **kwargs)
    elif kind == "focused":
        return _get_focused_scan(ultrasound_probe, grid_type, **kwargs)
    elif kind == "linescan":
        return _get_linescan_scan(ultrasound_probe, grid_type, **kwargs)
    else:
        raise ValueError(f"Unknown scan kind: {kind}")


def get_scan(kind="planewave", probe_kind="linear", grid_type="cartesian", **kwargs) -> Scan:
    """Returns a scan for ultrasound simulation tests."""
    ultrasound_probe = _get_probe(probe_kind)
    return _get_scan(ultrasound_probe, kind, grid_type, **kwargs)
