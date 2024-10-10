"""Example script for training deep learning based beamforming.

Author(s): Ben Luijten
Date: 09/12/2022

Summary: This script trains a deep learning based beamformer (ABLE) on a single angle plane wave
input, towards a multi angle plane wave target. As such, the target is created using the same scan
parameters as the input, but with a different number of angles and DAS beamforming.

This script should be compatible with plane wave data in USBMD format.

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from usbmd.backend.tensorflow.layers.beamformers import get_beamformer
from usbmd.backend.tensorflow.losses import SMSLE
from usbmd.data import get_dataset
from usbmd.probes import get_probe
from usbmd.processing import Process
from usbmd.setup_usbmd import setup
from usbmd.utils import update_dictionary


def train_beamformer(config):
    """Train function that initializes the dataset, beamformer model and optimizer, creates the
    target data, and then trains the model."""

    ## Dataloading and parameter initialization
    # Intialize dataset
    dataset = get_dataset(config.data)

    # Initialize scan based on dataset
    scan_class = dataset.get_scan_class()
    default_scan_params = dataset.get_scan_parameters_from_file()
    config_scan_params = config.scan
    config.model.beamformer.patches = 4

    # dict merging of manual config and dataset default scan parameters
    scan_params = update_dictionary(default_scan_params, config_scan_params)
    scan = scan_class(**scan_params)

    # initialize probe
    probe = get_probe(dataset.get_probe_name())

    # Create target data
    # pylint: disable=unexpected-keyword-arg
    target_beamformer = get_beamformer(probe, scan, config)
    print("Creating target data...")
    data = dataset[0][scan.selected_transmits][None, ...]
    targets = target_beamformer.predict(data, batch_size=1)

    ## Create the beamforming model
    # Only use the center angle for training
    config.scan.selected_transmits = "center"
    config.model.beamformer.type = "able"
    config.model.beamformer.patches = 4
    config_scan_params = config.scan

    # dict merging of manual config and dataset default scan parameters
    scan_params = update_dictionary(default_scan_params, config_scan_params)
    scan = scan_class(**scan_params)

    inputs = dataset[0][scan.selected_transmits][None, ...]

    beamformer = get_beamformer(probe, scan, config)

    # Get DAS beamformer as reference
    config.model.beamformer.type = "das"
    das_beamformer = get_beamformer(probe, scan, config)

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    beamformer.compile(
        optimizer=optimizer,
        loss=SMSLE(),
        metrics=SMSLE(),
        jit_compile=True,
    )

    ## Augment the data and train the model
    # repeat the inputs and targets N times with noise
    N = 32
    train_inputs = np.repeat(inputs, N, axis=0)
    train_targets = np.repeat(targets, N, axis=0)

    # Add noise to the inputs based on SNR
    SNR = 20
    noise = np.random.normal(0, 1, train_inputs.shape)
    noise = noise / np.linalg.norm(noise) * np.linalg.norm(train_inputs) / SNR
    train_inputs += noise

    # Train the model
    history = beamformer.fit(
        train_inputs, train_targets, epochs=100, batch_size=1, verbose=1
    )
    prediction = beamformer.predict(inputs, batch_size=1)
    das = das_beamformer.predict(inputs, batch_size=1)

    # Create a Process class to convert the data to an image
    process = Process(config, scan, probe)
    process.set_pipeline(dtype="beamformed_data", to_dtype="image")
    img_target = process.run(targets[0])

    img_prediction = process.run(prediction[0])
    img_das = process.run(das[0])

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
    plt.show()

    return history, beamformer


if __name__ == "__main__":
    # Load config
    path_to_config_file = Path.cwd() / "configs/config_picmus_rf.yaml"
    config = setup(path_to_config_file)

    # Train
    _, beamformer = train_beamformer(config)
