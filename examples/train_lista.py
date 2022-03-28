import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# make sure you have Pip installed usbmd (see README)
import usbmd.tensorflow_ultrasound as usbmd_tf
from usbmd.tensorflow_ultrasound.utils.gpu_config import set_gpu_usage
from usbmd.tensorflow_ultrasound.dataloader import (
    setup, get_dataset, get_probe, DataLoader, GenerateDataSet,
)
from usbmd.tensorflow_ultrasound.models import lista

# choose gpu
set_gpu_usage(gpu_ids=0)

# # choose config file
path_to_config_file = 'configs/config_picmus.yaml'
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

RUN_EAGERLY = True # for debugging set to true
image_shape = (1249, 387)
epochs = 100
learning_rate = 0.001

# initiate dataloader
dataloader = DataLoader(
    destination_folder,
    destination_folder,
    batch_size=1,
    image_shape=image_shape,
    shuffle=True,
)

# get image from batch
batch = next(iter(dataloader))
image = np.squeeze(batch[0])

# plot image
plt.figure()
plt.imshow(image, cmap='gray')

model = lista.Unfolding_model(image_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',
    run_eagerly=RUN_EAGERLY,
)

model.summary()

model.fit(dataloader, epochs=epochs)

if len(dataloader) < 10:
    fig, axs = plt.subplots(len(dataloader), 2)
    for i, batch in enumerate(dataloader):
        X, Y = batch
        out = np.squeeze(model(X))
        axs[i, 0].imshow(np.squeeze(Y), cmap='gray')
        axs[i, 1].imshow(out, cmap='gray')
        
for ax in axs.ravel():
    ax.axis('off')
    
fig.tight_layout()
