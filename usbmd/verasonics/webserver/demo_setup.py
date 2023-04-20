import numpy as np
import tensorflow as tf

from usbmd.probes import Verasonics_l11_4v
from usbmd.scan import PlaneWaveScan
from usbmd.tensorflow_ultrasound.layers.beamformers import create_beamformer
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage
from usbmd.utils.config import load_config_from_yaml

set_gpu_usage()

def trt_opt(model, name=None):

    if name:
        modelname = f'models/%s' % name

    try:
        trt_input = 'trt_input'
        trt_output = 'trt_output'

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
            output = trt_model(tf.convert_to_tensor(input_data, dtype='float32'))
            return list(output.values())[0]

        return wrapped_model

    except RuntimeError:
        print('Model was not optimized using TRT')
        return model


def get_models():
    model_dict = {}
    model_dict['DAS_1PW'], grid = create_DAS_1PW()
    # model_dict['DAS_5PW'] = create_DAS_5PW()
    # model_dict['DAS_11PW'] = create_DAS_11PW()
    # model_dict['ABLE_1PW'] = create_ABLE_1PW()
    # model_dict['ABLE_5PW'] = create_ABLE_5PW()
    # model_dict['ABLE_11PW'] = create_ABLE_11PW()
    return model_dict, grid


def model_from_file(path):

    trt_model = tf.saved_model.load(path)

    def wrapped_model(input_data):
            output = trt_model(tf.convert_to_tensor(input_data, dtype='float32'))
            return list(output.values())[0]

    return wrapped_model

def create_DAS_1PW():
    config = load_config_from_yaml('configs/config_webserver.yaml')
    config.data.n_angles = 1
    probe = Verasonics_l11_4v()
    probe_parameters = probe.get_default_scan_parameters()

    scan = PlaneWaveScan(
        N_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        N_ax=576,
        fs=7.24e6,
        fc=7.24e6,
        angles=np.array([0,]),
        modtype=config.data.modtype,
        Nx = config.scan.get('Nx'),
        Nz = config.scan.get('Nz')
        )

    model = create_beamformer(
        probe,
        scan,
        config,
        aux_inputs=config.model.beamformer.get('aux_inputs')
    )
    model = trt_opt(model, name = 'DAS_1PW')
    return model, scan.grid

# def create_DAS_5PW():
#     cfg = load_config_from_yaml('configs/config_webserver.yaml')
#     cfg.data.n_angles = [1,3,5,7,9]
#     probe = get_probe(cfg)
#     probe.N_ax = 576
#     probe.fs = probe.fs/4
#     grid = get_grid(cfg, probe)
#     model = create_beamformer(probe, grid, cfg)
#     model = trt_opt(model, name = 'DAS_5PW')
#     return model

# def create_DAS_11PW():
#     cfg = load_config_from_yaml('configs/config_webserver.yaml')
#     cfg.data.n_angles = [0,1,2,3,4,5,6,7,8,9,10]
#     probe = get_probe(cfg)
#     probe.N_ax = 576
#     probe.fs = probe.fs/4
#     grid = get_grid(cfg, probe)
#     model = create_beamformer(probe, grid, cfg)
#     model = trt_opt(model, name = 'DAS_11PW')
#     return model

# def create_ABLE_1PW():
#     cfg = load_config_from_yaml('python/configs/inference/l11-4v_ABLE_1PW.yaml')
#     probe = get_probe(cfg)
#     grid = get_grid(cfg, probe)
#     model = create_beamformer(probe, grid, cfg)
#     model.load_weights(cfg.model_path)

#     model = trt_opt(model, name = 'ABLE_1PW')

#     #model.compile(jit_compile=False)
#     return model

# def create_ABLE_5PW():
#     cfg = load_config_from_yaml('python/configs/inference/l11-4v_ABLE_5PW.yaml')
#     probe = get_probe(cfg)
#     grid = get_grid(cfg, probe)
#     model = create_beamformer(probe, grid, cfg)
#     model.load_weights(cfg.model_path)
#     model = trt_opt(model, name = 'ABLE_5PW')
#     #model.compile(jit_compile=False)
#     return model

# def create_ABLE_11PW():
#     cfg = load_config_from_yaml('python/configs/inference/l11-4v_ABLE_11PW.yaml')
#     probe = get_probe(cfg)
#     grid = get_grid(cfg, probe)
#     model = create_beamformer(probe, grid, cfg)
#     model.load_weights(cfg.model_path)
#     model = trt_opt(model, name = 'ABLE_11PW')
#     #model.compile(jit_compile=False)
#     return model


def distributed_model(cfg, probe, grid, gpus):
    """ Funtion that splits the beamforming in N grids, to be distributed across multiple GPU's

    Args:
        cfg:
        probe:
        grid:
        gpus:
    """

    N = len(gpus)

    subgrids = np.split(grid, N, axis=1)
    inputs = tf.keras.layers.Input((probe.N_tx, probe.N_el, probe.N_ax, probe.N_ch),
                            batch_size=cfg['batch_size'],
                            name='input_data')

    subgrid_outputs = []
    for gpu, subgrid in zip(gpus, subgrids):
        with tf.device(gpu.name.strip('/physical_device:')):
            subgrid_outputs.append(create_beamformer(probe, subgrid, cfg)(inputs))

    outputs = tf.concat(subgrid_outputs, axis=1)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model_dict = get_models()