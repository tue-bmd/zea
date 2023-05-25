# pylint: disable=no-member
# pylint: disable=not-an-iterable

"""This module contains the function to initialize the models for the webserver demo."""


import os

import numpy as np
import tensorflow as tf

from usbmd.probes import Verasonics_l11_4v
from usbmd.scan import PlaneWaveScan
from usbmd.tensorflow_ultrasound.layers.beamformers import create_beamformer
from usbmd.utils.config import load_config_from_yaml


def trt_opt(model, name=None):
    """Function that

    Args:
        model (tf.Model): Tensorflow model
        name (string, optional): Model name. Defaults to None.

    Returns:
        tf.Model: TRT optimized model
    """

    if name:
        modelname = f'models/{name}'

    try:
        trt_input = 'trt_input'

        dummy_input = [tf.zeros(input.shape) for input in model.inputs]
        _ = model(dummy_input)

        model.save(trt_input)

        params = tf.experimental.tensorrt.ConversionParams(
            precision_mode='FP32')

        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=trt_input, conversion_params=params)
        trt_model = converter.convert()
        converter.summary()
        _ = trt_model(dummy_input[0])
        tf.saved_model.save(trt_model, modelname)

        if name:
            trt_model = tf.saved_model.load(modelname)

        def wrapped_model(input_data):
            output = trt_model(tf.convert_to_tensor(
                input_data, dtype='float32'))
            return list(output.values())[0]

        return wrapped_model

    except RuntimeError:
        print('Model was not optimized using TRT')
        return model


def get_models():
    """Function that creates all models, will be replaced in later versions"""
    model_dict = {}
    model_dict['DAS_1PW'], grid = create_DAS([5])
    model_dict['DAS_5PW'], grid = create_DAS([1, 3, 5, 7, 9])
    model_dict['DAS_11PW'], grid = create_DAS(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # model_dict['ABLE_1PW'], grid = create_ABLE(
    #     'configs/inference/l11-4v_ABLE_1PW.yaml')
    # model_dict['ABLE_5PW'], grid = create_ABLE(
    #     'configs/inference/l11-4v_ABLE_5PW.yaml')
    # model_dict['ABLE_11PW'], grid = create_ABLE(
    #     'configs/inference/l11-4v_ABLE_11PW.yaml')
    return model_dict, grid


def model_from_file(path):
    """Load a model from a file path"""
    trt_model = tf.saved_model.load(path)

    def wrapped_model(input_data):
        output = trt_model(tf.convert_to_tensor(input_data, dtype='float32'))
        return list(output.values())[0]

    return wrapped_model


def create_DAS(n_angles):
    """Creates a delay-and-sum model with n_angles PWs"""
    print(f'Creating DAS {len(n_angles)} PW model')
    config = load_config_from_yaml('configs/config_webserver.yaml')
    config.data.n_angles = n_angles
    probe = Verasonics_l11_4v()
    scan = PlaneWaveScan(
        N_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(5e-3, 55e-3),
        N_ax=576,
        fs=6.25e6,
        fc=6.25e6,
        angles=np.deg2rad(np.linspace(-18, 18, 11)[config.data.n_angles]),
        modtype=config.data.modtype,
        Nx=config.scan.get('Nx'),
        Nz=config.scan.get('Nz')
    )

    model = create_beamformer(
        probe,
        scan,
        config,
        aux_inputs=config.model.beamformer.get('aux_inputs')
    )

    try:
        model = tf.function(model, jit_compile=True)
    except:
        print('Could not compile model, running uncompiled')

    # Build model by passing a dictionary of dummy data with correct dtype and shape
    dummy_input = {
        inp.name.strip('_input'): tf.zeros(inp.shape, dtype=inp.dtype) for inp in model.inputs
    }
    _ = model(dummy_input)

    return model, scan.grid


def create_ABLE(config_path):
    """creates an ABLE model from a config file"""
    print(f'Creating ABLE model from {config_path}')
    config = load_config_from_yaml(config_path)
    probe = Verasonics_l11_4v()
    scan = PlaneWaveScan(
        N_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(5e-3, 55e-3),
        N_ax=576,
        fs=6.25e6,
        fc=6.25e6,
        angles=np.deg2rad(np.linspace(-18, 18, 11)[config.scan.n_angles]),
        modtype=config.data.modtype,
        Nx=config.scan.get('Nx'),
        Nz=config.scan.get('Nz')
    )

    model = create_beamformer(
        probe,
        scan,
        config,
        aux_inputs=config.model.beamformer.get('aux_inputs')
    )

    try:
        path = 'trained_models/0911_1805_realtime_ABLE_1PW_MV/model'
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Could not find file: {path}")
        model.load_weights(path, by_name=False)
    except FileNotFoundError as e:
        raise e
    model = tf.function(model, jit_compile=True)

    dummy_input = {
        inp.name.strip('_input'): tf.zeros(inp.shape, dtype=inp.dtype) for inp in model.inputs
    }
    _ = model(dummy_input)
    return model, scan.grid


def distributed_model(probe, scan, config, gpus):
    """ Funtion that splits the beamforming in N grids, to be distributed across multiple GPU's

    Args:
        probe (Probe): Probe object
        scan (Scan): Scan object
        config (Config): Config object
        gpus (list): list of GPU's to use

    Returns:
        model (tf.keras.Model): Keras model of the distributed beamformer
    """

    N = len(gpus)

    subgrids = np.split(scan.grid, N, axis=1)
    sub_beamformers = []

    subgrid_outputs = []
    for gpu, subgrid in zip(gpus, subgrids):

        scan.grid = subgrid
        scan.Nx = subgrid.shape[1]
        scan.Nz = subgrid.shape[0]

        with tf.device(gpu.name.strip('/physical_device:')):
            sub_beamformers.append(create_beamformer(
                probe,
                scan,
                config,
                aux_inputs=config.model.beamformer.get('aux_inputs')
            ))

    # Full model inputs
    inputs = {}
    for inp in sub_beamformers[0].inputs:

        if inp.name == 'input_grid':
            shape = (inp.shape[1], N*inp.shape[2], inp.shape[3])
        else:
            shape = inp.shape[1:]

        inputs[inp.name.strip('input_')] = tf.keras.Input(
            shape=shape, name=inp.name, batch_size=1)

    for i, sub_beamformer in enumerate(sub_beamformers):
        with tf.device(gpus[i].name.strip('/physical_device:')):
            sub_inputs = inputs.copy()
            if 'grid' in sub_inputs:
                sub_inputs['grid'] = sub_inputs['grid'][:,
                                                        :, i*scan.Nx:(i+1)*scan.Nx, :]
            subgrid_outputs.append(sub_beamformer(sub_inputs))

    outputs = tf.concat(subgrid_outputs, -1)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model_dict = get_models()
