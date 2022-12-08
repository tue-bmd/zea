"""
Example script for training deep learning based beamforming
"""
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# make sure you have Pip installed usbmd (see README)
from usbmd.tensorflow_ultrasound.dataloader import UltrasoundLoader 
from usbmd.tensorflow_ultrasound.layers.beamformers_v2 import create_beamformer
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage
from usbmd.ui import setup
from usbmd.utils.pixelgrid import get_grid
from usbmd.probes import get_probe
from usbmd.tensorflow_ultrasound.losses import smsle
from usbmd.processing import Process


def train(config):

    ## Initialization
    # Load the probe and grid based on config
    probe = get_probe(config)
    grid = get_grid(config, probe)

    # Set number of channels (RF/IQ)
    # This will be added to the probe class later on
    probe.N_ch = 2 if config.data.get('IQ') else 1
    probe.N_tx = 1

    ## Create the beamforming model
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

    ## Data loading
    dataloader = UltrasoundLoader(config,
        probe,
        grid,
        val_size=0.1)

    N_batches = len(dataloader.x_train)
    x_test, y_test = dataloader.load_test()

    # Create TF dataloader
    tf_train_gen = tf.data.Dataset.from_generator(dataloader.load_batches,
                                                output_types = ((tf.float32, tf.float32), tf.float32),
                                                output_shapes = ((tf.TensorShape(model.inputs[0].shape[1:]),
                                                                tf.TensorShape(model.inputs[1].shape[1:])),
                                                                tf.TensorShape(model.outputs[0].shape[1:]))
    ).batch(config.model.batch_size)

    ## Train the model
    model.fit(tf_train_gen,
              steps_per_epoch = N_batches,
              epochs = 10,
              callbacks=[],
              max_queue_size=10,
              workers=1,
              verbose=1)

    ## Inference
    data = next(iter(tf_train_gen))
    pred = model(data[0])['beamformed']

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(pred[0])
    axs[1].imshow(data[1][0])
    plt.show()

    # Create full-image inference model, not aux grid input needed here
    config.model.patch_shape = (grid.shape[0], grid.shape[1])
    infer_model = create_beamformer(probe, grid, config)
    infer_model.set_weights(model.get_weights())
    pred = infer_model([np.expand_dims(x_test[10],0), np.expand_dims(grid,0)])
    
    proc = Process(None, probe=probe)
    proc.downsample_factor = 1
    img = proc.run(pred['beamformed'], dtype='beamformed_data', to_dtype='image')
    plt.imshow(img[0].T, cmap='gray')
    plt.show()

    return model


if __name__ == '__main__':

    # Choose gpu, or select automatically
    set_gpu_usage(gpu_ids=0)

    # Load config file
    path_to_config_file = Path.cwd() / 'examples/deep_beamforming/example_config.yaml'
    config = setup(path_to_config_file)

    model = train(config)
