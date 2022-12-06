"""
Example script for training deep learning based beamforming
"""
from pathlib import Path
import tensorflow as tf

# make sure you have Pip installed usbmd (see README)
# import usbmd.tensorflow_ultrasound as usbmd_tf
# from usbmd.tensorflow_ultrasound.dataloader import DataLoader, GenerateDataSet
from usbmd.tensorflow_ultrasound.layers.beamformers_v2 import create_beamformer
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage
from usbmd.ui import setup
from usbmd.utils.pixelgrid import get_grid
from usbmd.probes import get_probe
from usbmd.tensorflow_ultrasound.losses import smsle

from examples.deep_beamforming.dataloader import get_dataloader

# def create_sim_data(Nx = 256, Nz = 256, batch_size=1 , fc=1*10**6, c=1540, points=[]):

#   Nx = 256
#   Nz = 256

#   lambda0 = c/fc
#   dx = lambda0/8
#   a = dx*np.arange(0,Nx,4) # array spacing: lambda

#   max_scatterers = 30

#   Nt = 2*(Nz)*dx/c
#   dt = (1/fc)/4 
#   t = dt*np.arange(0,np.round(Nt/dt)) 
#   x  = dx*np.arange(0,Nx) 
#   z  = dx*np.arange(0,Nz) 

#   sig = 1*10**(-6)
#   pulse = lambda tau: np.exp(-0.5*((t-tau)/sig)**2)*np.sin(2*np.pi*fc*(t-tau))
#   sig_x = 1*dx
#   xg,zg = np.meshgrid(x,z)
#   loc =  lambda x0,z0: np.exp(-0.5*(((xg-x0)/sig_x)**2+((zg-z0)/sig_x)**2))

#   if points == []:
#     while 1:
#         inp = []
#         tar = []
#         for i in range(0,batch_size):
#           scatterers = np.random.randint(1,1+max_scatterers)
#           points_x = dx*Nx*np.random.rand(scatterers)
#           points_z = dx*Nz*np.random.rand(scatterers)
#           s_i = 0
#           y_i = 0
#           for j in range(0,scatterers):
#             d_trans = points_z[j]/c
#             tau_j = d_trans + np.sqrt(((points_x[j]-a)/c)**2 + (points_z[j]/c)**2)
#             s_i = s_i + np.array([pulse(tau_j[k]) for k in range(0,len(tau_j))])
#             y_i = y_i + loc(points_x[j],points_z[j])
#           inp.append(s_i.T)
#           tar.append(y_i)

#         yield np.array(inp),np.array(tar)
      
#   else:
#     points = dx*np.array(points)
#     points_x = points[:,0]
#     points_z = points[:,1]
#     scatterers = len(points_x)

#     s_i = 0
#     y_i = 0
#     for j in range(0,scatterers):
#       d_trans = points_z[j]/c
#       tau_j = d_trans + np.sqrt(((points_x[j]-a)/c)**2 + (points_z[j]/c)**2)
#       s_i = s_i + np.array([pulse(tau_j[k]) for k in range(0,len(tau_j))])
#       y_i = y_i + loc(points_x[j],points_z[j])

#     yield s_i.T,y_i,t,x,z,a


def train(config):

    probe = get_probe(config)
    grid = get_grid(config, probe)


    # Will parameterize this
    probe.N_ch = 2 if config.data.get('IQ') else 1
    probe.N_tx = 1

    model = create_beamformer(probe, grid, config, aux_inputs=['grid'])

    # Data loading
    dataloader = get_dataloader(config, probe, grid)
    N_batches = len(dataloader.x_train)

    #Load test data
    x_test, y_test = dataloader.load_test()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                    loss=smsle,
                    metrics=smsle,
                    run_eagerly=False)


    # Create TF dataloader
    tf_train_gen = tf.data.Dataset.from_generator(dataloader.load_batches,
                                                output_types = ((tf.float32, tf.float32), tf.float32),
                                                output_shapes = ((tf.TensorShape(model.inputs[0].shape[1:]), 
                                                                tf.TensorShape(model.inputs[1].shape[1:])),
                                                                tf.TensorShape(model.outputs[0].shape[1:]))
    ).batch(config.data.batch_size)

    model.fit(tf_train_gen,
              steps_per_epoch = N_batches,
              epochs = 10,
              callbacks=[],
              max_queue_size=10,
              workers=1,
              verbose=1)


    # Visualize network output
    pred = model(x_test)
    

    return model



if __name__ == '__main__':

    # choose gpu
    set_gpu_usage(gpu_ids=0)

    # # choose config file
    path_to_config_file = Path.cwd() / 'examples/deep_beamforming/example_config.yaml'
    config = setup(path_to_config_file)

    model = train(config)


















# from logging import raiseExceptions
# import os
# import argparse
# from datetime import datetime, date
# from pathlib import Path
# import tensorflow as tf
# import tensorflow_addons as tfa
# import importlib
# import numpy as np
# import wandb

# from probes import get_probe
# from data_loader import get_dataloader
# from utils.pixelgrid import get_grid
# from utils.config import Config, load_config_from_yaml
# from utils.gpu_config import set_gpu_usage
# from utils.setup import create_savedir
# from common import set_data_paths

# #tf.config.run_functions_eagerly(True)

# def train(cfg, use_wandb=False):
    
#     callbacks = []
    
#     name = cfg.config_path.split('/')[-1].strip('.yaml')

#     # Compatibility for wandb
#     if use_wandb:
#         wandb.init(project='sparse_beamforming', 
#                    entity='aiteam-tue',
#                    config=cfg)
        
#         callbacks.append(wandb.keras.WandbCallback())
#         cfg = Config(wandb.config)
#         name = name+'_'+wandb.run.name
#         cfg.wandb = True

    
#     # Create unique version name


#     savedir = create_savedir(cfg, name)

#     # Initialize probe and beamforming grid
#     probe = get_probe(cfg)
#     grid = get_grid(cfg, probe)
        
#     """
#     =============================================================================
#         Data loading
#     =============================================================================
#     """    
#     #Initialize dataloaders
#     dataloader = get_dataloader(cfg, probe, grid)
#     N_batches = len(dataloader.x_train)

#     #Load test data
#     x_test, y_test = dataloader.load_test()
    
    
    
#     """
#     =============================================================================
#         Model definition
#     =============================================================================
#     """
#     from layers.beamformers_v2 import create_beamformer
#     model = create_beamformer(probe, 
#                             grid, 
#                             cfg, 
#                             aux_inputs=cfg.aux_inputs, 
#                             aux_outputs=cfg.aux_outputs)
    
#     #testrun = model(x_test[0])
    
#     # Create separate model for full-image inference to save GPU memory. Should 
#     # be passed to callbacks. 
#     # For small model sizes, this can be turned off
#     if hasattr(cfg, 'patch_size'):
#             cfg_cb = Config(cfg)
#             cfg_cb['patch_shape'] = (cfg.Nx, cfg.Nz)
#             cfg_cb['batch_size'] = 1
#             cfg_cb['aux_inputs'] = ['grid', 'angles', 'tzero'] # Adding angles and tzero as additional input for PICMUS
#             cfg_cb['aux_outputs'] = ['DAS']
#             model_cb = create_beamformer(probe, 
#                                         grid, 
#                                         cfg_cb, 
#                                         aux_inputs=cfg_cb.aux_inputs, 
#                                         aux_outputs=cfg_cb.aux_outputs)
#     else:
#         model_cb = model
    
#     model.summary()
    
    
#     """
#     =============================================================================
#         Initialize and compile model
#     =============================================================================
#     """
    
#     from losses import smsle, smsle2d
    
# # =============================================================================
# #     optimizers = [
# #         tf.keras.optimizers.Adam(learning_rate=1e-2),
# #         tf.keras.optimizers.Adam(learning_rate=1e-4)]
# #     
# #     optimizers_and_layers = [(optimizers[0], model.layers[0:4]), (optimizers[1], model.layers[4:])]
# #     optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
# # =============================================================================
    
#     optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

#     # If there are multiple outputs, apply weighed loss scheme
#     if len(model.outputs) > 1: 
#         loss_weights = {}
        
#         weights = np.exp(np.linspace(-1., 0., len(model.outputs)))
#         weights /= weights.sum()

#         for i, output in enumerate(model.outputs):
#             loss_weights[output.name] = weights[i]
        
#         model.compile(optimizer=optimizer,
#                     loss=smsle,
#                     loss_weights=loss_weights,
#                     metrics=smsle)
    
#     else:
#         model.compile(optimizer=optimizer,
#                     loss=smsle,
#                     metrics=smsle,
#                     run_eagerly=False)
    

#     """
#     =============================================================================
#         Training
#     =============================================================================
#     """
    
#     #Initialize callbacks
#     from callbacks import show_test_images, training_callback, quantization_callback, analogcombiner_callback, evaluate_PSNR
    
#     #Save model weights
#     if not (savedir/"savedWeights_interval").is_dir():
#             os.makedirs(savedir/"savedWeights_interval")
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(savedir/"savedWeights_interval",'weights_epoch_{epoch:02d}.hdf5'),
#             monitor='loss',
#             verbose=1,
#             save_best_only=True,
#             save_weights_only=True,
#             mode='min',
#             save_freq='epoch')
    
#     callbacks.append(show_test_images(savedir, cfg_cb, probe, grid, x_test, y_test, evalmodel=model_cb, show_set=cfg.show_set))
#     callbacks.append(training_callback(savedir))
#     callbacks.append(cp_callback)
#     #callbacks.append(evaluate_PSNR(savedir, cfg_cb, probe, grid, x_test, y_test, evalmodel=model_cb))
    
#     if cfg.num_code_words:
#         callbacks.append(quantization_callback(savedir, cfg))
    
#     if cfg.analog_combine:
#         callbacks.append(analogcombiner_callback(savedir, cfg))

#     tf_train_gen = tf.data.Dataset.from_generator(dataloader.load_batches,
#                                                   output_types = ((tf.float32, tf.float32), tf.float32),
#                                                   output_shapes = ((tf.TensorShape(model.inputs[0].shape[1:]), 
#                                                                    tf.TensorShape(model.inputs[1].shape[1:])),
#                                                                    tf.TensorShape(model.outputs[0].shape[1:]))
#     ).batch(cfg.batch_size)


#     # Save config
#     cfg.save_to_yaml((savedir/'config.yaml'))
    
#     model.fit(tf_train_gen,
#               steps_per_epoch = N_batches,
#               epochs = cfg.epochs,
#               callbacks=callbacks,
#               max_queue_size=10,
#               workers=1,
#               verbose=1)
              
#     model.save(savedir/'model')



# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, default='python/configs/training/realtime_ABLE.yaml', help='path to config file.')
#     parser.add_argument('--wandb', default=True, action='store_true', help='add this flag to use wandb')
#     parser.add_argument('--sweepid', default=None, type=str, help='When using wandb, set a sweepid')
#     parser.add_argument('--api', default=None, type=str, help='(Optional) Manually provide API key for wandb')
#     parser.add_argument('--comment', default='default', type=str, help='Provide additional description to the run')
#     parser.add_argument('--gpu', default=None, type=int, help='(Optional) Choose a specific GPU')    
   
#     args = parser.parse_args()
#     return args


# if __name__ == "__main__":
#     args = parse_args()
#     cfg = load_config_from_yaml(args.config)
#     cfg.config_path = str(args.config)
#     set_gpu_usage(args.gpu)
    
#     cfg.data_path = set_data_paths()
  
#     if not args.comment:
#         if args.sweepid:
#             cfg.comment = str(args.sweepid)

#     # compatibility for wandb
#     if args.wandb:
#         if args.api:
#             wandb.login(key=args.api)

#         if args.sweepid:  
#             def train_with_sweep():
#                 return train(cfg)
            
#             wandb.agent(
#                 args.sweepid,
#                 function=train_with_sweep,
#                 project='sparse_beamforming', 
#                 entity='aiteam-tue',
#             )
#         else:
#             model = train(cfg, use_wandb=args.wandb)
#     else:  
#         model = train(cfg)