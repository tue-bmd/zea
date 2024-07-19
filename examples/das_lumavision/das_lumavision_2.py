import matplotlib.pyplot as plt
plt.switch_backend("agg")

import usbmd
from usbmd import setup
from usbmd.data import USBMDDataSet
from usbmd.probes import Probe
from usbmd.processing import Process
from usbmd.scan import Scan
from usbmd.utils import update_dictionary
from usbmd.utils.device import init_device

device = init_device("torch", "auto:1")

# let's check if your usbmd version is up to date
assert usbmd.__version__ >= "2.0", "Please update usbmd to version 2.0 or higher"

# choose your config file with all your settings
# config_path = "configs/config_picmus_rf.yaml"
config_path = "configs/config_opsf.yaml"

# setup function handles local data paths, default config settings and GPU usage
# make sure to create your own users.yaml using usbmd/datapaths.py
users_paths = "users.yaml"
config = setup(config_path, users_paths, create_user=True)

# intialize the dataset
dataset = USBMDDataSet(config.data)

# get scan and probe parameters from the dataset and config
file_scan_params = dataset.get_scan_parameters_from_file()
file_probe_params = dataset.get_probe_parameters_from_file()
config_scan_params = config.scan

# merging of manual config and dataset scan parameters
scan_params = update_dictionary(file_scan_params, config_scan_params)
scan = Scan(**scan_params)
probe = Probe(**file_probe_params)
process = Process(config=config, scan=scan, probe=probe)

# initialize the processing pipeline
process.set_pipeline(
    operation_chain=[
        {"name": "beamform"},
        {"name": "demodulate"},
        {"name": "envelope_detect"},
        {"name": "downsample"},
        {"name": "normalize"},
        {"name": "log_compress"},
        {"name": "scan_convert"},
    ],
    device=device,
)

# pick a frame from the dataset
file_idx = 0
frame_idx = 10
data = dataset[(file_idx, frame_idx)]

# process the data
image = process.run(data)

# plot the image
plt.figure()
plt.imshow(image, cmap="gray")
plt.savefig("test2.png")
# plt.show()
