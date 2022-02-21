import numpy as np

# make sure you have Pip installed usbmd (see README)
import usbmd.tensorflow_ultrasound as usbmd_tf
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage
from usbmd.tensorflow_ultrasound.dataloader import (
    setup, get_dataset, get_probe, DataLoader, GenerateDataSet,
)

# choose gpu
set_gpu_usage(gpu_ids=0)

# choose config file
path_to_config_file = 'configs/config_picmus.yml'
config = setup(path_to_config_file)

# generate image dataset from raw data
destination_folder = 'D:/data/ultrasound/PICMUS/picmus_image'
try:
    gen = GenerateDataSet(
        config, 
        destination_folder=destination_folder, 
        retain_folder_structure=False,
    )
    gen.generate()
except ValueError:
    print(f'Dataset already exists in {destination_folder}')

# initiate dataloader
dataloader = DataLoader(
    destination_folder, 
    batch_size=1, 
    image_shape=(1249, 387), 
    shuffle=True,
)

# get image from batch
batch = next(iter(dataloader))
image = np.squeeze(batch[0])

# get a dataset object for plotting
dataset = get_dataset(config.data.dataset_name)(config=config.data)

# ugly, probe should be defined in init somewhere probably
# but probe is necessary for plot function currently
probe = get_probe(config, dataset)
dataset.probe = probe

# plot image using dataset properties
dataset.plot(image, image_range=dataloader.normalization, save=False)

'''
model = ...
model.fit()

'''