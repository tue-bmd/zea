"""Tests for the ops beamformer.
"""

import keras
import numpy as np

from tests.test_processing import equality_libs_processing
from usbmd import beamformer
from usbmd.config import load_config_from_yaml
from usbmd.config.validation import check_config
from usbmd.probes import Verasonics_l11_4v
from usbmd.scan import PlaneWaveScan
from usbmd.utils.simulator import UltrasoundSimulator


def _get(reconstruction_mode):
    """Mostly copied from tests/test_pytorch_beamforming.py"""
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
    dx = scan.wvln / 4
    dz = scan.wvln / 4
    scan.Nx = int(np.ceil((scan.xlims[1] - scan.xlims[0]) / dx))
    scan.Nz = int(np.ceil((scan.zlims[1] - scan.zlims[0]) / dz))

    simulator = UltrasoundSimulator(probe, scan)

    # Generate pseudorandom input tensor
    data = simulator.generate(200)

    inputs = np.expand_dims(data[0], axis=(1, -1))
    return config, probe, scan, data, inputs


@equality_libs_processing()
def test_tof_correction(reconstruction_mode="generic"):
    """Test TOF Correction between backends.
    Also ensures that the output is the same when it is split into patches"""

    from keras import ops  # pylint: disable=import-outside-toplevel

    _, probe, scan, _, inputs = _get(reconstruction_mode)

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
        sampling_frequency=scan.fs,
        fdemod=scan.fdemod,
        fnum=scan.f_number,
        angles=scan.polar_angles,
        vfocus=scan.focus_distances,
    )
    for key, item in kwargs.items():
        # If item is a floating point numpy array, convert to float32
        if hasattr(item, "dtype") and np.issubdtype(item.dtype, np.floating):
            item = item.astype(np.float32)
        # Convert to tensor if numpy array
        if isinstance(item, np.ndarray):
            kwargs[key] = ops.convert_to_tensor(item)

    outputs = []
    for patches in [1, 10]:
        output = beamformer.tof_correction(
            **kwargs,
            apply_phase_rotation=bool(scan.fdemod),
            patches=patches,
        )
        outputs.append(ops.convert_to_numpy(output))
    np.testing.assert_allclose(
        outputs[0], outputs[1], err_msg="Different results for patches=1 and patches=10"
    )

    keras.utils.clear_session(free_memory=True)  # Free memory
    return outputs[0]


if __name__ == "__main__":
    test_tof_correction()
