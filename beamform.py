import os

# Set keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt

from usbmd import setup
from usbmd.data import load_usbmd_file
from usbmd.processing import Process
from plotlib import *

use_style(STYLE_DARK)

# choose your config file
# all necessary settings should be in the config file
config_path = "configs/config_picmus_rf.yaml"

# setup function handles local data paths, default config settings and GPU usage
# make sure to create your own users.yaml using usbmd/datapaths.py
users_paths = "users.yaml"
config = setup(config_path, users_paths, create_user=True)

# we now manually point to our data
data_root = config.data.user.data_root
user = config.data.user.username

print(f"\n🔔 Hi {user}! You are using data from {data_root}\n")

# data_path = "/home/vincent/Desktop/cardiac/20241021_P10_A2CH_0000_usbmd.hdf5"
data_dir = Path(
    "/home/vincent/mnt/USBMDNAS/Ultrasound-BMd/data/USBMD_datasets/2023_USBMD_carotid/"
)
data_path = "4_cross_bifur_R_0000.hdf5"

from pathlib import Path


import h5py
import numpy as np

idx = np.array([128])
# config.transmits = idx

print(data_path)

# data_paths = list(data_dir.iterdir())
# data_paths.reverse()
data_paths = [data_dir / data_path]
# print(data_paths)
# exit()

for data_path in data_paths:
    if not data_path.name.endswith(".hdf5"):
        continue
    print(data_path.name)

    with h5py.File(data_path, "r") as f:
        print(f["data"]["raw_data"].shape)
        raw_data = []
        for tx in idx:
            raw_data.append(f["data"]["raw_data"][0, tx][None, None])

        raw_data = np.concatenate(raw_data, axis=1)

        print(raw_data.shape)
        raw_data = raw_data.reshape(
            (
                raw_data.shape[0],
                raw_data.shape[1],
                raw_data.shape[3],
                raw_data.shape[2],
                raw_data.shape[4],
            )
        )
        raw_data = raw_data.transpose(0, 1, 3, 2, 4)
    # continue
    tx = 64

    # vmax = 100
    # n_tx = len(idx)
    # fig, axes = plt.subplots(1, n_tx, figsize=(12, 4))
    # axes = np.atleast_1d(axes)
    # for i in range(n_tx):
    #     axes[i].imshow(raw_data[0, i, :, :, 0], aspect="auto", vmin=-vmax, vmax=vmax)
    # # plt.show()
    # continue

    # only 1 frame in PICMUS to be selected
    selected_frames = [0]

    # loading a file manually using `load_usbmd_file`
    data, scan, probe = load_usbmd_file(
        data_path, frames=selected_frames, config=config, data_type="raw_data"
    )

    # initialize the Process class
    process = Process(config=config, scan=scan, probe=probe)

    # initialize the processing pipeline so it know what kind
    # of data it is processing and what it should output
    process.set_pipeline(dtype="raw_data", to_dtype="image")

    # index the first frame
    data_frame = data[0]
    data_frame = (
        data_frame
        if data_frame.shape[1] > data_frame.shape[2]
        else data_frame.transpose(0, 2, 1, 3)
    )
    print(data_frame.shape)
    plt.figure()
    plt.imshow(data_frame[128, :, :, 0], vmin=-200, vmax=200)

    # lastly instead of setting the pipeline with `dtype` and `to_dtype`
    # we can also opt for passing a custom operation chain as follows

    # initialize the processing pipeline
    process.set_pipeline(
        operation_chain=[
            {"name": "tof_correction"},
            {"name": "delay_and_sum"},
            {"name": "demodulate"},
            {"name": "envelope_detect"},
            {"name": "downsample"},
            {"name": "normalize"},
            # we now only set log_compress parameters to show how it can be done
            # if you don't pass any parameters it will use default or
            # params from config / scan / probe
            {"name": "log_compress", "params": {"dynamic_range": (-60, 0)}},
        ],
    )

    image = process.run(data_frame)

    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(data_path.name)
    plt.savefig("out.png")
    plt.show()
