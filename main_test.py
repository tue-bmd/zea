import tensorflow as tf
import numpy as np

import importlib
import tensorflow_ultrasound as tfu




"""
=============================================================================
    Initialization
=============================================================================
"""


from utils.init_GPU import initGPU
initGPU() #initialize GPU

#load config
cfg = tfu.dataloader.setup()




"""
=============================================================================
    Data loading
=============================================================================
"""

ui = tfu.dataloader.DataLoaderUI(cfg)




"""
=============================================================================
    Model
=============================================================================
"""


#create GPU model for training
model = tfu.models.DAS.get_model(cfg)

# Create separate cpu model for "full size" inference, to save GPU memory
with tf.device():
    cfg_cpu = cfg 
    #adjust cpu config here
    model_cpu = tfu.models.ABLE.get_model(cfg_cpu)
    
    
    
"""
=============================================================================
    Callbacks
=============================================================================
"""
    



"""
=============================================================================
    Training
=============================================================================
"""