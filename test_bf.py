import matplotlib.pyplot as plt
import torch

from usbmd.pytorch_ultrasound.layers.beamformers import create_beamformer
from usbmd.probes import get_probe, Verasonics_l11_4v
from usbmd.datasets import PICMUS, DummyDataset
from usbmd.utils.config import load_config_from_yaml, Config
from usbmd.utils.pixelgrid import make_pixel_grid_v2
from usbmd.processing import Process
from usbmd.common import set_data_paths


config = load_config_from_yaml(r'configs\config_picmus.yaml')

datapaths = set_data_paths(local=True)

config.data.file_path = r'C:\Users\s153800\Documents\3 resources\ml-data\PICMUS'

dataset = PICMUS(config.data)

# probe = get_probe('verasonics_l11_4v')#config, dataset)
probe = Verasonics_l11_4v(config)

dataset.probe = probe

grid = make_pixel_grid_v2(config.scan.xlims, config.scan.zlims, config.get('Nx', 128), config.get('Nz', 160))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

beamformer = create_beamformer(probe, grid, config).to(device)

# Get data and bring to device
input_data = torch.from_numpy(dataset[0]).to(device)

# Add batch dimension to data
input_data = input_data[None]

# Perform beamforming and convert to numpy array
beamformed = beamformer(input_data)['beamformed'].cpu().numpy()


process_obj = Process(config, probe)
image = process_obj.to_image(beamformed[0])

plt.imshow(image, cmap='gray')
plt.show()