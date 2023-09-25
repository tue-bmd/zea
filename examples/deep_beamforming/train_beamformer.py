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

from usbmd.common import set_data_paths
from usbmd.datasets import get_dataset
from usbmd.probes import get_probe
from usbmd.tensorflow_ultrasound.layers.beamformers import get_beamformer
from usbmd.tensorflow_ultrasound.losses import smsle
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage
from usbmd.setup import setup
from usbmd.utils.utils import update_dictionary
from usbmd.utils.video import ScanConverterTF


def train(config):
    """Train function that initializes the dataset, beamformer model and optimizer, creates the
    target data, and then trains the model."""

    ## Dataloading and parameter initialization
    # Intialize dataset
    dataset = get_dataset(config.data)
    data = dataset[0]

    # Initialize scan based on dataset
    scan_class = dataset.get_scan_class()
    default_scan_params = dataset.get_default_scan_parameters()
    config_scan_params = config.scan

    # dict merging of manual config and dataset default scan parameters
    scan_params = update_dictionary(default_scan_params, config_scan_params)

    # Reducing the pixels per wavelength to 1 to reduce memory usage at the cost of resolution
    scan_params['pixels_per_wvln'] = 1
    # Setting the grid size to automatic mode based on pixels per wavelength
    scan_params['Nx'] = None
    scan_params['Nz'] = None

    scan = scan_class(**scan_params, modtype=config.data.modtype)

    # initialize probe
    probe = get_probe(dataset.get_probe_name())

    # Create target data
    target_beamformer = get_beamformer(probe, scan, config)

    targets = target_beamformer(np.expand_dims(data[scan.n_angles], axis=0))

    ## Create the beamforming model
    # Only use the center angle for training
    config.scan.n_angles = 1
    config.model.beamformer.type = 'able'
    config_scan_params = config.scan

    # dict merging of manual config and dataset default scan parameters
    scan_params = update_dictionary(default_scan_params, config_scan_params)

    # Reducing the pixels per wavelength to 1 to reduce memory usage at the cost of resolution
    scan_params['pixels_per_wvln'] = 1
    # Setting the grid size to automatic mode based on pixels per wavelength
    scan_params['Nx'] = None
    scan_params['Nz'] = None

    scan = scan_class(**scan_params, modtype=config.data.modtype)
    scan.angles = np.array([0])

    inputs = np.expand_dims(data[37:38], axis=0)

    beamformer = get_beamformer(probe, scan, config)
    beamformer.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    beamformer.compile(optimizer=optimizer,
                       loss=smsle,
                       metrics=smsle,
                       run_eagerly=False)

    ## Augment the data and train the model
    # repeat the inputs and targets N times with noise
    N = 100
    inputs = np.repeat(inputs, N, axis=0)
    targets = np.repeat(targets, N, axis=0)

    # add noise to the inputs based on SNR
    SNR = 20
    noise = np.random.normal(0, 1, inputs.shape)
    noise = noise / np.linalg.norm(noise) * np.linalg.norm(inputs) / SNR
    inputs += noise

    # Train the model
    history = beamformer.fit(
        inputs,
        targets,
        epochs=10,
        batch_size=1,
        verbose=1)

    scan_converter = ScanConverterTF(grid=scan.grid)


    targets = scan_converter.convert(targets)
    predictions = scan_converter.convert(beamformer(inputs))

    # plot the resulting image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(targets[0], cmap='gray')
    plt.title('Target')
    plt.subplot(1, 2, 2)
    plt.imshow(predictions[0], cmap='gray')
    plt.title('Prediction')
    plt.show()

    return history, beamformer


if __name__ == '__main__':
    # Load config
    path_to_config_file = Path.cwd() / 'configs/config_picmus_iq.yaml'
    config = setup(file=path_to_config_file)
    config.data.user = set_data_paths(local=True)

    # Set GPU usage
    set_gpu_usage(config.get('device'))

    # Train
    _, beamformer = train(config)
