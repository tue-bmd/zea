import os

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Config, File, Pipeline

path = (
    "hf://zeahub/picmus/database/experiments/resolution_distorsion/"
    "resolution_distorsion_expe_dataset_rf/resolution_distorsion_expe_dataset_rf.hdf5"
)
config = Config.from_path("hf://zeahub/picmus/config_rf.yaml")

with File(path) as f:
    probe_geometry = f.probe.get_parameters()["probe_geometry"]

# Force the top row (z=0) of the grid to land exactly on the element x-positions --
# this is the 0/0 case in compute_xl's Newton-Raphson seed. With the default grid
# (400 pixels over +/-20mm) none of the pixel x's exactly match one of the 128
# element x's, so the bug goes unnoticed; it happens whenever grid_size_x
# is chosen to match the element pitch, or simply for the on-axis pixel at x=0 when
# the array has an odd number of elements.
config.parameters.apply_lens_correction = True
config.parameters.zlims = [0.0, 0.055]  # grid starts exactly at the transducer face
config.parameters.xlims = [float(probe_geometry[:, 0].min()), float(probe_geometry[:, 0].max())]
config.parameters.grid_size_x = probe_geometry.shape[0]
config.parameters.dynamic_range = [-60, 0]

with File(path) as f:
    parameters = f.load_parameters(**config.parameters)
    raw = f.data.raw_data[:1]

pipeline = Pipeline.from_config(config)
inputs = pipeline.prepare_parameters(parameters)
outputs = pipeline(data=raw, **inputs, return_numpy=True)
recon = outputs["data"][0]

print("any nan:", np.isnan(recon).any())  # False -- wire targets reconstruct cleanly

image = zea.display.to_8bit(recon, dynamic_range=(-60, 0), pillow=False)
extent_mm = [v * 1e3 for v in parameters.extent_imshow]

zea.visualize.set_mpl_style()
plt.imshow(image, extent=extent_mm, cmap="gray", vmin=0, vmax=255)
plt.xlabel("X (mm)")
plt.ylabel("Z (mm)")
plt.title("PICMUS resolution_distorsion, lens correction on, grid z starts at 0")
out_path = "snippet_reconstruction_main.png"
plt.savefig(out_path, bbox_inches="tight", dpi=100)
print("saved", out_path)
