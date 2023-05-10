"""Example script for training deep learning based beamforming.
Author(s): Ben Luijten
Date: 09/12/2022
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from usbmd.common import set_data_paths
from usbmd.datasets import get_dataset
from usbmd.probes import get_probe
from usbmd.tensorflow_ultrasound.layers.beamformers import create_beamformer
from usbmd.tensorflow_ultrasound.losses import smsle
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage
from usbmd.ui import setup
from usbmd.utils.utils import update_dictionary


def create_targets(data, scan, probe, n_angles):
    config.data.n_angles = n_angles
    scan.n_angles = n_angles
    scan.N_tx = n_angles

    # Target beamformer
    beamformer = create_beamformer(probe, scan, config)

    # Create targets
    targets = beamformer(data)

    return targets

def create_inputs(data, scan, probe, config, n_angles):
    inputs = data[n_angles]
    inputs = np.expand_dims(inputs, axis=0)
    return inputs


def train(config):

    # intialize dataset
    dataset = get_dataset(config.data)

    # Initialize scan based on dataset
    scan_class = dataset.get_scan_class()
    default_scan_params = dataset.get_default_scan_parameters()
    config_scan_params = config.scan

    # dict merging of manual config and dataset default scan parameters
    scan_params = update_dictionary(default_scan_params, config_scan_params)
    scan = scan_class(**scan_params, modtype=config.data.modtype)

    # initialize probe
    probe = get_probe(dataset.get_probe_name())

    # Create input and target data
    inputs = create_inputs(dataset[0], scan, probe, config, scan.n_angles)
    targets = create_targets(dataset[0], scan, probe, 11)

    beamformer = create_beamformer(probe, scan, config)

    beamformer.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    beamformer.compile(optimizer=optimizer,
                  loss=smsle,
                  metrics=smsle,
                  run_eagerly=False)

    data = dataset[0]
    data = data[scan.n_angles]
    data = np.expand_dims(data, axis=0)

    # Train the model
    history = beamformer.fit(
        data,
        epochs=100,
        batch_size=1,
        verbose=1)

    return beamformer


if __name__ == '__main__':
    # Set GPU usage
    set_gpu_usage()

    # Load config
    path_to_config_file = Path.cwd() / 'configs/config_picmus_iq.yaml'
    config = setup(file=path_to_config_file)
    config.data.user = set_data_paths(local=True)
    config.data.n_angles = 75

    beamformer = train(config)







# def train(config, dataset_directory):
#     """Loads parameters and data, and trains the model"""

#     # Initialization
#     # Load the probe and grid based on config
#     probe = get_probe_from_config(config)
#     scan = initialize_scan_from_probe(probe)
#     grid = scan.grid

#     # Set number of channels (RF/IQ)
#     # This will be added to the probe class later on
#     probe.N_ch = 2 if config.data.get('IQ') else 1
#     probe.N_tx = config.data.n_angles

#     probe.fc = 6250000
#     probe.fdemod = 0
#     probe.bandwidth = None

#     # Create the beamforming model
#     # A grid is added as "auxiallary input" such that we can train on different grid patches
#     model = create_beamformer(probe, scan, config, aux_inputs=['grid'])

#     model.summary()

#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#     model.compile(optimizer=optimizer,
#                   loss=smsle,
#                   metrics=smsle,
#                   run_eagerly=False)

#     # Data loading
#     dataloader


#     dataloader = UltrasoundLoader(
#         config,
#         dataset_directory,
#         probe,
#         grid,
#         val_size=0.1)

#     N_batches = len(dataloader.x_train)
#     x_test, y_test = dataloader.load_test()

#     # Create TF dataloader
#     tf_train_gen = tf.data.Dataset.from_generator(
#         dataloader.load_batches,
#         output_types=((tf.float32, tf.float32), tf.float32),
#         output_shapes=((tf.TensorShape(model.inputs[0].shape[1:]),
#                         tf.TensorShape(model.inputs[1].shape[1:])),
#                         tf.TensorShape(model.outputs[0].shape[1:]))
#     ).batch(config.model.batch_size)

#     # Train the model
#     model.fit(tf_train_gen,
#               steps_per_epoch=N_batches,
#               epochs=20,
#               callbacks=[],
#               max_queue_size=10,
#               workers=1,
#               verbose=1)

#     # Inference
#     test_idx = 2

#     # Create full-image inference model, no aux grid input needed here
#     config.model.patch_shape = (grid.shape[0], grid.shape[1])
#     infer_model = create_beamformer(probe, grid, config)
#     infer_model.set_weights(model.get_weights())
#     y_pred = infer_model(
#         [np.expand_dims(x_test[test_idx], 0), np.expand_dims(grid, 0)])

#     # Convert to images
#     proc = Process(None, probe=probe)
#     proc.downsample_factor = 1
#     y_pred_img = proc.run(y_pred['beamformed'][0].numpy(),
#                           dtype='envelope_data', to_dtype='image')
#     y_test_img = proc.run(
#         y_test[test_idx], dtype='envelope_data', to_dtype='image')

#     # Show an example of the test data
#     aspect = (grid.shape[1]/grid.shape[0])/(np.diff(config.scan.xlims)/np.diff(config.scan.zlims))

#     fig, axs = plt.subplots(1, 2)
#     axs[0].imshow(y_pred_img.T, cmap='gray', aspect = aspect)
#     axs[0].set_title('Deep Learning')
#     axs[1].imshow(y_test_img, cmap='gray', aspect = aspect)
#     axs[1].set_title('Target')
#     fig.show()

#     return model


# if __name__ == '__main__':
#     # pylint: disable=no-member
#     # Choose gpu, or select automatically
#     set_gpu_usage()

#     # Load config file
#     path_to_config_file = Path.cwd() / 'examples/deep_beamforming/example_config_nMAP.yaml'
#     config = setup(path_to_config_file)

#     model = train(config)
