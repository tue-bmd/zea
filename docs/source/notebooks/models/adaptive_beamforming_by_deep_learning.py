"""test file to be converted to notebook"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from IPython.display import display

import zea
from zea import init_device, load_file
from zea.models.able import ABLE
from zea.ops import (
    DelayAndSum,
    EnvelopeDetect,
    Lambda,
    LogCompress,
    Normalize,
    PatchedGrid,
    Pipeline,
    ReshapeToGrid,
    TOFCorrection,
)
from zea.visualize import set_mpl_style

grid_size_x = 100
grid_size_z = 50
device = init_device(verbose=False)
set_mpl_style()


# Load a single frame of raw RF data from PICMUS
path = "hf://zeahub/picmus/database/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
data, scan, probe = load_file(
    path=path,
    indices=[0],
    data_type="raw_data",
)
scan.set_transmits(1)
data = data[:, scan.selected_transmits]
zlims = (0, 0.06)
xlims = (-0.019, 0.019)
dynamic_range = (-50, 0)

scan.n_ch = data.shape[-1]  # iq data
scan.zlims = zlims
scan.xlims = xlims
scan.grid_size_x = grid_size_x
scan.grid_size_z = grid_size_z


# Build the beamforming pipeline with ABLE
pipeline = Pipeline(
    operations=[
        # PatchedGrid(
        #     operations=[
        #         TOFCorrection(),
        #         #Lambda(ABLE().call, name="ABLE Reconstruction"),
        #         DelayAndSum(),
        #     ],
        #     num_patches=10,
        #     jit_options=None,
        # ),
        TOFCorrection(),
        ReshapeToGrid(),
        Lambda(ABLE().call, name="ABLE Reconstruction"),
        DelayAndSum(reshape_grid=False),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
    jit_options=None,
)


parameters = pipeline.prepare_parameters(probe, scan, dynamic_range=dynamic_range)
parameters["demodulation_frequency"] = parameters["sampling_frequency"]

inputs = {pipeline.key: keras.ops.convert_to_tensor(data)}
outputs = pipeline(**inputs, **parameters)
bmode = outputs[pipeline.output_key]

bmode_img = zea.display.to_8bit(bmode[0], dynamic_range=dynamic_range)
display(bmode_img)
