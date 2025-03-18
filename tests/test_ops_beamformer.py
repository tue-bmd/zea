"""Tests for the ops beamformer.
"""

# pylint: disable=import-outside-toplevel

import numpy as np

from usbmd.config import load_config_from_yaml
from usbmd.config.validation import check_config
from usbmd.ops_v2 import Pipeline, Simulate
from usbmd.probes import Verasonics_l11_4v
from usbmd.scan import PlaneWaveScan

from . import backend_equality_check


def _get_params(reconstruction_mode):
    """Get the necessary objects for the test"""
    # probe = get_probe(config)
    config = load_config_from_yaml(r"./tests/config_test.yaml")
    config = check_config(config)

    probe = Verasonics_l11_4v()
    probe_parameters = probe.get_parameters()
    scan = PlaneWaveScan(
        probe_geometry=probe.probe_geometry,
        n_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        n_ax=2047,
        sampling_frequency=probe_parameters["sampling_frequency"],
        center_frequency=probe_parameters["center_frequency"],
        polar_angles=np.array([0.0]),
    )
    scan._focus_distances = (
        np.array([0.0]) if reconstruction_mode == "generic" else np.array([np.inf])
    )

    # Set scan grid parameters
    # The grid is updated automatically when it is accessed after the scan parameters
    # have been changed.
    dx = scan.wvln
    dz = scan.wvln
    scan.Nx = int(np.ceil((scan.xlims[1] - scan.xlims[0]) / dx)) // 4
    scan.Nz = int(np.ceil((scan.zlims[1] - scan.zlims[0]) / dz)) // 4

    # use pipeline here so it is easy to propagate the scan parameters
    simulator = Pipeline(
        [Simulate(apply_lens_correction=scan.apply_lens_correction, n_ax=scan.n_ax)]
    )

    # Generate pseudorandom input tensor
    # Define scatterers
    scat_x, scat_z = np.meshgrid(
        np.linspace(-10e-3, 10e-3, 5),
        np.linspace(5e-3, 30e-3, 5),
        indexing="ij",
    )
    scat_x, scat_z = np.ravel(scat_x), np.ravel(scat_z)
    n_scat = len(scat_x)
    scat_positions = np.stack(
        [
            scat_x,
            np.zeros_like(scat_x),
            scat_z,
        ],
        axis=1,
    )
    output = simulator(
        scan,
        probe,
        scatterer_positions=scat_positions.astype(np.float32),
        scatterer_magnitudes=np.ones(n_scat, dtype=np.float32),
    )
    data = output["raw_data"]

    return config, probe, scan, data


@backend_equality_check(
    decimal=[0, 2, 3], timeout=600, backends=["torch", "tensorflow", "jax"]
)
def test_tof_correction(reconstruction_mode="generic"):
    """Test TOF Correction between backends.
    Also ensures that the output is the same when it is split into patches

    Note:
        The timeout is set to 120 seconds because the TOF correction can be slow.
        Allowing a higher tolerance for torch at the moment.

    """

    import keras
    from keras import ops

    from usbmd import beamformer

    # pylint: disable=unused-variable
    config, probe, scan, inputs = _get_params(reconstruction_mode)

    # round inputs a bit to avoid numerical issues
    inputs = np.round(inputs, 2)

    batch_item = 0  # Only one batch item
    kwargs = dict(  # pylint: disable=use-dict-literal
        data=inputs[batch_item],
        grid=scan.grid,
        t0_delays=scan.t0_delays,
        tx_apodizations=scan.tx_apodizations,
        sound_speed=scan.sound_speed,
        probe_geometry=probe.probe_geometry,
        initial_times=scan.initial_times,
        sampling_frequency=scan.sampling_frequency,
        demodulation_frequency=scan.demodulation_frequency,
        fnum=scan.f_number,
        angles=scan.polar_angles,
        vfocus=scan.focus_distances,
    )
    for key, item in kwargs.items():
        # If item is a floating point numpy array, convert to float32
        if isinstance(item, np.ndarray):
            if hasattr(item, "dtype") and np.issubdtype(item.dtype, np.floating):
                item = item.astype(np.float32)
            # Convert to tensor if numpy array
            kwargs[key] = ops.convert_to_tensor(item)

    outputs = []
    for patches in [1, 10]:
        output = beamformer.tof_correction(
            **kwargs,
            apply_phase_rotation=bool(scan.demodulation_frequency),
            patches=patches,
        )
        outputs.append(ops.convert_to_numpy(output))
    np.testing.assert_allclose(
        outputs[0], outputs[1], err_msg="Different results for patches=1 and patches=10"
    )

    keras.utils.clear_session(free_memory=True)  # Free memory
    return outputs[0]
