"""Example script for training deep learning based beamforming.

Author(s): Ben Luijten
Date: 09/12/2022

Summary: This script trains a deep learning based beamformer (ABLE) on a single angle plane wave
input, towards an 11 plane wave target. The target is created using the same scan parameters as the
input, but with a different number of angles and DAS beamforming.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from usbmd.datasets import get_dataset
from usbmd.probes import get_probe
from usbmd.processing import Process
from usbmd.setup_usbmd import setup
from usbmd.tensorflow_ultrasound.layers.beamformers import get_beamformer
from usbmd.tensorflow_ultrasound.losses import SMSLE
from usbmd.utils.utils import update_dictionary


def train_beamformer(config):
    """Train function that initializes the dataset, beamformer model and optimizer, creates the
    target data, and then trains the model."""

    ## Dataloading and parameter initialization
    # Intialize dataset
    dataset = get_dataset(config.data)

    # Initialize scan based on dataset
    scan_class = dataset.get_scan_class()
    default_scan_params = dataset.get_default_scan_parameters()
    config_scan_params = config.scan
    config.model.beamformer.patches = 4

    # dict merging of manual config and dataset default scan parameters
    scan_params = update_dictionary(default_scan_params, config_scan_params)

    # Setting the grid size to automatic mode based on pixels per wavelength
    scan_params["Nx"] = None
    scan_params["Nz"] = None

    scan = scan_class(**scan_params, modtype=config.data.modtype)

    # initialize probe
    probe = get_probe(dataset.get_probe_name())

    # Create target data
    # pylint: disable=unexpected-keyword-arg
    target_beamformer = get_beamformer(probe, scan, config)
    print("Creating target data...")
    data = dataset[0][
        scan.selected_transmits
    ]  # Select the transmits as defined in the config
    targets = target_beamformer.predict(np.expand_dims(data, axis=0), batch_size=1)

    ## Create the beamforming model
    # Only use the center angle for training
    config.scan.selected_transmits = 1
    config.model.beamformer.type = "able"
    config.model.beamformer.patches = (
        1  # No patching needed for the single angle beamformer
    )
    config_scan_params = config.scan

    # dict merging of manual config and dataset default scan parameters
    scan_params = update_dictionary(default_scan_params, config_scan_params)

    # Setting the grid size to automatic mode based on pixels per wavelength
    scan_params["Nx"] = None
    scan_params["Nz"] = None

    scan = scan_class(**scan_params, modtype=config.data.modtype)
    scan.angles = np.array([0])

    inputs = np.expand_dims(dataset[0][37:38], axis=0)

    beamformer = get_beamformer(probe, scan, config)
    beamformer.summary()

    # Get DAS beamformer as reference
    config.model.beamformer.type = "das"
    das_beamformer = get_beamformer(probe, scan, config)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    beamformer.compile(
        optimizer=optimizer,
        loss=SMSLE(),
        metrics=SMSLE(),
        run_eagerly=False,
        jit_compile=True,
    )

    ## Augment the data and train the model
    # repeat the inputs and targets N times with noise
    N = 10
    inputs = np.repeat(inputs, N, axis=0)
    targets = np.repeat(targets, N, axis=0)

    # add noise to the inputs based on SNR
    SNR = 5
    noise = np.random.normal(0, 1, inputs.shape)
    noise = noise / np.linalg.norm(noise) * np.linalg.norm(inputs) / SNR
    inputs += noise

    # Train the model
    history = beamformer.fit(inputs, targets, epochs=50, batch_size=1, verbose=1)
    prediction = np.array(beamformer(inputs))
    das = np.array(das_beamformer(inputs))

    # Create a Process class to convert the data to an image
    process = Process(config, scan, probe)
    img_target = process.run(targets[0], "beamformed_data", "image")
    img_prediction = process.run(prediction[0], "beamformed_data", "image")
    img_das = process.run(das[0], "beamformed_data", "image")

    # plot the resulting image
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_target, cmap="gray")
    plt.title("Target")
    plt.subplot(1, 3, 2)
    plt.imshow(img_prediction, cmap="gray")
    plt.title("Prediction")
    plt.subplot(1, 3, 3)
    plt.imshow(img_das, cmap="gray")
    plt.title("DAS")

    return history, beamformer


if __name__ == "__main__":
    # Load config
    path_to_config_file = Path.cwd() / "configs/config_picmus_iq.yaml"
    config = setup(path_to_config_file)

    # Train
    _, beamformer = train_beamformer(config)

    plt.show()
