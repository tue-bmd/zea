"""Test the tf implementation of the beamformers.
"""

import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras

from usbmd.backend.tensorflow.layers.beamformers import get_beamformer
from usbmd.backend.tensorflow.utils.utils import tf_snapshot
from usbmd.config import load_config_from_yaml
from usbmd.config.validation import check_config
from usbmd.probes import Verasonics_l11_4v
from usbmd.scan import PlaneWaveScan
from usbmd.utils.simulator import UltrasoundSimulator

# Add project folder to path to find config files
wd = Path(__file__).parent.parent
sys.path.append(str(wd))


# test
@pytest.mark.parametrize(
    "reconstruction_mode, patches",
    [("generic", None), ("generic", 4), ("pw", None), ("pw", 4)],
)
def test_das_beamforming(
    reconstruction_mode, patches, debug=False, compare_gt=True, jit=False
):
    """Performs DAS beamforming on random data to verify that no errors occur. Does
    not check correctness of the output.

    Args:
        debug (bool, optional): Set to True to enable debugging options (plotting). Defaults to
        False. compare_gt (bool, optional): Set to True to compare against GT. Defaults to True.

    Returns:
        numpy array: beamformed output
    """

    config = load_config_from_yaml(r"./tests/config_test.yaml")
    config = check_config(config)
    config.ml_library = "tensorflow"
    if jit:
        config.model.beamformer.jit = True  # pylint: disable=no-member

    if patches:
        config.model.beamformer.patches = patches  # pylint: disable=no-member

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
        angles=np.array(
            [
                0,
            ]
        ),
    )
    # We override the focus parameter for now to force the beamformer to use the generic delay
    # calculation if reconstruction_mode == "generic".
    scan._focus_distances = (
        np.array([0]) if reconstruction_mode == "generic" else np.array([np.inf])
    )

    # Set scan grid parameters
    # The grid is updated automatically when it is accessed after the scan parameters
    # have been changed.
    dx = scan.wvln / 4
    dz = scan.wvln / 4
    scan.Nx = int(np.ceil((scan.xlims[1] - scan.xlims[0]) / dx))
    scan.Nz = int(np.ceil((scan.zlims[1] - scan.zlims[0]) / dz))

    simulator = UltrasoundSimulator(probe, scan)
    beamformer = get_beamformer(probe, scan, config)

    # Ensure reproducible results
    tf.random.set_seed(0)
    np.random.seed(0)

    # Generate pseudorandom input tensor
    data = simulator.generate(200)

    inputs = np.expand_dims(data[0], axis=(1, -1))

    # Perform beamforming and convert to numpy array
    outputs = beamformer(inputs)

    # plot results
    if debug:
        fig, axs = plt.subplots(1, 3)
        aspect_ratio = (data[1].shape[1] / data[1].shape[2]) / (
            data[0].shape[1] / data[0].shape[2]
        )
        axs[0].imshow(np.abs(inputs.squeeze().T), aspect=aspect_ratio)
        axs[0].set_title("RF data")
        axs[1].imshow(np.squeeze(outputs))
        axs[1].set_title("Beamformed")
        axs[2].imshow(cv2.GaussianBlur(data[1].squeeze(), (5, 5), cv2.BORDER_DEFAULT))
        axs[2].set_title("Ground Truth")
        fig.show()

    y_true = cv2.GaussianBlur(data[1].squeeze(), (5, 5), cv2.BORDER_DEFAULT)
    y_pred = np.squeeze(outputs)

    y_true = y_true / y_true.max()
    y_pred = y_pred / y_pred.max()

    MSE = np.mean(np.square(y_true - y_pred))
    print(f"MSE: {MSE}")

    # Free all GPU memory
    keras.backend.clear_session()

    if compare_gt:
        assert MSE < 0.01
    else:
        return y_pred


def test_jit_compile():
    """Test that the jit compilation works and gives the same result as the non-jit	version."""
    jit_output = test_das_beamforming(
        reconstruction_mode="pw", patches=None, debug=False, compare_gt=False, jit=True
    )
    non_jit_output = test_das_beamforming(
        reconstruction_mode="pw", patches=None, debug=False, compare_gt=False, jit=False
    )
    # Numerical difference between XLA and non-XLA compiled models are expected, we are here
    # only checking if images are similar on a global scale. Users should always manually check
    # the output of the model.
    assert np.allclose(jit_output, non_jit_output, atol=1e-2)


def test_dynamic_beamforming():
    """Test that the beamformer can be called with different inputs and configurations."""
    config = load_config_from_yaml(r"./tests/config_test.yaml")
    config.ml_library = "tensorflow"

    probe = Verasonics_l11_4v()
    probe_parameters = probe.get_parameters()

    scan = PlaneWaveScan(
        probe_geometry=probe.probe_geometry,
        n_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        n_ax=2046,
        sampling_frequency=probe_parameters["sampling_frequency"],
        center_frequency=probe_parameters["center_frequency"],
        angles=np.array(
            [
                0,
            ]
        ),
    )

    probe2 = pickle.loads(pickle.dumps(probe))
    scan2 = pickle.loads(pickle.dumps(scan))
    scan2.fc = 5e6
    scan2.fs = 4 * 5e6
    probe2.probe_type = "new_probe_type"

    sound_speed = 1600

    # Generate pseudorandom input tensor
    simulator = UltrasoundSimulator(probe, scan)
    data = simulator.generate(20)
    inputs = np.expand_dims(data[0], axis=(1, -1))

    # Initialize beamformer
    beamformer = get_beamformer(probe, scan, config)

    # Check that the beamformer can be called with different inputs
    beamformer(inputs)
    beamformer(inputs, probe=probe, scan=scan)
    beamformer(inputs, probe=probe2, scan=scan2)
    beamformer(inputs, probe=probe2, scan=scan2, sound_speed=sound_speed)


def test_snapshot():
    """Test that the snapshot function works."""
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
        angles=np.array(
            [
                0,
            ]
        ),
    )

    probe_snapshot = tf_snapshot(probe)
    scan_snapshot = tf_snapshot(scan)

    def check_snapshot(snapshot, obj):
        for key, value in snapshot.items():
            assert isinstance(value, tf.Tensor)
            np.testing.assert_allclose(value.numpy(), getattr(obj, key))

    check_snapshot(probe_snapshot, probe)
    check_snapshot(scan_snapshot, scan)
