"""Example script for training deep learning based beamforming.
Author(s): Ben Luijten
Date: 09/12/2022
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from usbmd.common import set_data_paths
from usbmd.probes import get_probe
from usbmd.processing import Process
# make sure you have Pip installed usbmd (see README)
from usbmd.tensorflow_ultrasound.dataloader import UltrasoundLoader
from usbmd.tensorflow_ultrasound.layers.beamformers import create_beamformer
from usbmd.tensorflow_ultrasound.losses import smsle
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage
from usbmd.ui import setup
from usbmd.utils.pixelgrid import get_grid


def train(config, dataset_directory):
    """Loads parameters and data, and trains the model"""

    # Initialization
    # Load the probe and grid based on config
    probe = get_probe(config)
    grid = get_grid(config, probe)

    # Set number of channels (RF/IQ)
    # This will be added to the probe class later on
    probe.N_ch = 2 if config.data.get('IQ') else 1
    probe.N_tx = config.data.n_angles

    probe.fc = 6250000
    probe.fdemod = 0
    probe.bandwidth = None

    # Create the beamforming model
    # A grid is added as "auxiallary input" such that we can train on different grid patches
    model = create_beamformer(probe,
                              grid,
                              config,
                              aux_inputs=['grid'])

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  loss=smsle,
                  metrics=smsle,
                  run_eagerly=False)

    # Data loading
    dataloader = UltrasoundLoader(
        config,
        dataset_directory,
        probe,
        grid,
        val_size=0.1)

    N_batches = len(dataloader.x_train)
    x_test, y_test = dataloader.load_test()

    # Create TF dataloader
    tf_train_gen = tf.data.Dataset.from_generator(
        dataloader.load_batches,
        output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=((tf.TensorShape(model.inputs[0].shape[1:]),
                        tf.TensorShape(model.inputs[1].shape[1:])),
                       tf.TensorShape(model.outputs[0].shape[1:]))
    ).batch(config.model.batch_size)

    # Train the model
    model.fit(tf_train_gen,
              steps_per_epoch=N_batches,
              epochs=20,
              callbacks=[],
              max_queue_size=10,
              workers=1,
              verbose=1)

    # Inference

    test_ix = 2

    # Create full-image inference model, no aux grid input needed here
    config.model.patch_shape = (grid.shape[0], grid.shape[1])
    infer_model = create_beamformer(probe, grid, config)
    infer_model.set_weights(model.get_weights())
    y_pred = infer_model(
        [np.expand_dims(x_test[test_ix], 0), np.expand_dims(grid, 0)])

    # Convert to images
    proc = Process(None, probe=probe)
    proc.downsample_factor = 1
    y_pred_img = proc.run(y_pred['beamformed'][0].numpy(),
                          dtype='envelope_data', to_dtype='image')
    y_test_img = proc.run(
        y_test[test_ix], dtype='envelope_data', to_dtype='image')

    # Show an example of the test data
    aspect = (grid.shape[1]/grid.shape[0])/(np.diff(config.scan.xlims)/np.diff(config.scan.zlims))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(y_pred_img.T, cmap='gray', aspect = aspect)
    axs[0].set_title('Deep Learning')
    axs[1].imshow(y_test_img, cmap='gray', aspect = aspect)
    axs[1].set_title('Target')
    fig.show()

    return model


if __name__ == '__main__':
    # pylint: disable=no-member
    # Choose gpu, or select automatically
    set_gpu_usage()

    # Load config file
    path_to_config_file = Path.cwd() / 'examples/deep_beamforming/example_config_nMAP.yaml'
    config = setup(path_to_config_file)

    data_root = set_data_paths(config.data.local)['data_root']
    dataset_directory = data_root / config.data.dataset

    model = train(config, dataset_directory)
