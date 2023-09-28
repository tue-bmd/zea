import h5py
from usbmd.utils.utils import print_hdf5_attrs
import cv2

# Define path to the data file
data_path = r"Z:\Ultrasound-BMd\data\USBMD-example-data\planewave_l115v.hdf5"
# data_path = r"C:\Users\s153800\Downloads\point_l115v_0000.hdf5"
# data_path = r"C:\Users\s153800\Downloads\point_l115v_0003.hdf5"


from usbmd.data_format.usbmd_data_format import load_usbmd_file
import numpy as np


# Load the data file and construct a probe and scan object
# We will only load the first two frames of the data file
data, scan, probe = load_usbmd_file(data_path, frames=[0,])


import matplotlib.pyplot as plt

print(data.shape)

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot(np.arange(scan.N_ax)/scan.fs*scan.c, data[0, 3, 64, :, 0])
# ax.set_xlabel('')
# plt.show()
# exit()


SINGLE_FRAME = False
# SINGLE_FRAME = True

if SINGLE_FRAME:

      tx = 3

      data = data[0:1, tx:tx+1]
      # scan.zlims = (0.0, 80.58375e-3)
      # scan.xlims = (-0.031, 0.031)
      scan.N_tx = 1
      scan.t0_delays = scan.t0_delays[tx:tx+1, :]
      scan.initial_times = scan.initial_times[tx:tx+1]


# elements = [n for n in range(20)] + [n for n in range(28, 128)]

# data[:, :, elements] = 0


print(scan.initial_times)

scan.Nx = 512
scan.Nz = 512

# Hack that is needed because the Scan base class does not contain the angles
# attribute and the beamformer assumes that it does
scan.angles = np.linspace(0, 2, scan.N_tx, endpoint=False)

# Print some info about the data
print('Data file loaded successfully')
print('The data tensor has shape: {}'.format(data.shape))
print('The dimensions of the data are (n_frames, n_transmits, n_elements, '
      'n_axial_samples, n_rf_iq_channels)')



from pathlib import Path
from usbmd.pytorch_ultrasound.layers.beamformers import get_beamformer
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.config_validation import check_config

# Load the config file
config_path = Path('configs', 'config_usbmd_rf.yaml')
config = load_config_from_yaml(config_path)

# Check the config file for errors
check_config(config)

# Create the beamformer
beamformer = get_beamformer(probe=probe, scan=scan, config=config)



import torch
from usbmd.processing import rf2iq, log_compress


# Swap the 3th and 4th axes
data = np.swapaxes(data, 3, 4)

# Transform the data to IQ data
iq_data = rf2iq(data,
                fs=scan.fs,
                fc=scan.fc,
                bandwidth=probe.bandwidth,
                separate_channels=False)

# Swap the 3th and 4th axes back
iq_data = np.swapaxes(iq_data, 3, 4)

# Turn the data into a torch tensor
iq_data = torch.from_numpy(iq_data)
# iq_data = torch.from_numpy(data)

# Beamform the data
beamformer_output = beamformer(iq_data)

image = beamformer_output['beamformed'].numpy()[0, :, :, 0]

image = np.abs(image)

image = image/image.max()
image = log_compress(image)

print(f'Image shape: {image.shape}')

import matplotlib.pyplot as plt

# Plot the image
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image,
          cmap='gray',
          extent=[scan.xlims[0], scan.xlims[1], scan.zlims[1], scan.zlims[0]],
          vmin=-50,
          vmax=0,
          )

# Add labels
ax.set_xlabel('Lateral distance [m]')
ax.set_ylabel('Depth [m]')

# Turn the figure black
ax.set_facecolor((0, 0, 0))
fig.patch.set_facecolor((0, 0, 0))

# Turn the ticks white
ax.tick_params(axis='both', colors='white')

# Turn the labels white
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

plt.show()
