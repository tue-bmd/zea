"""
Plane Wave Compounding
===================================================

This example demonstrates plane wave compounding of data from the
PICMUS challenge (Plane-wave Imaging Challenge in Medical Ultrasound). PICMUS
provides standardized datasets for evaluating plane wave imaging algorithms.

**Example overview:**

- Load ultrasound data from UFF format files
- Beamform data from multiple plane-wave transmits
- Coherently compound the results
- Visualize the results

Attribution:

- Example inspired by vbeam examples (https://github.com/magnusdk/vbeam/)
- Dataset from PICMUS challenge (https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/)
"""

# %%
# .. note::
#    To run this example, we recommend installing mach with all optional dependencies:
#
#    .. code-block:: bash
#
#       pip install mach-beamform[all]
#
#    This ensures all dependencies for the examples are available.

# %%
# Import Required Libraries
# -------------------------
# Let's start by importing the necessary libraries.

import hashlib
import time

import matplotlib.pyplot as plt
import numpy as np

# Import mach modules
from mach import experimental
from mach._vis import db_zero
from mach.io.uff import create_beamforming_setup
from mach.io.utils import cached_download

try:
    from pyuff_ustb import Uff
except ImportError as err:
    raise ImportError(
        "⚠️  pyuff_ustb is required for UFF data loading. Install with: pip install pyuff-ustb"
    ) from err

# Convenience constants
MM_PER_METER = 1000

# %%
# Download PICMUS Challenge Dataset
# ---------------------------------
#
# The PICMUS challenge contains multiple datasets with multi-angle plane-wave
# transmits. This example uses the resolution dataset, which features:
#
# - **128-element linear array** at 5.2 MHz center frequency
# - **75 plane wave transmits** at angles from -16° to +16°
# - **Point targets and cysts** for resolution and contrast assessment

print("📂 Downloading PICMUS challenge dataset...")

# Download the UFF data file (cached locally after first download)
url = "http://www.ustb.no/datasets/PICMUS_experiment_resolution_distortion.uff"
uff_path = cached_download(
    url,
    expected_size=145_518_524,
    expected_hash="c93af0781daeebf771e53a42a629d4f311407410166ef2f1d227e9d2a1b8c641",
    digest=hashlib.sha256,
    filename="PICMUS_experiment_resolution_distortion.uff",
)

print(f"✓ Dataset downloaded to: {uff_path}")
print(f"  File size: {uff_path.stat().st_size / 1e6:.1f} MB")

# %%
# Load and Inspect PICMUS Data
# ----------------------------
#
# UFF files contain structured ultrasound data including channel data (RF signals),
# probe geometry, and scan parameters. Let's examine the PICMUS dataset structure.

print("\n📋 Loading PICMUS data structure...")

# Open UFF file and extract components
uff_file = Uff(str(uff_path))
channel_data = uff_file.read("/channel_data")
scan = uff_file.read("/scan")

# Display challenge dataset information
print("\n📊 PICMUS Challenge Dataset:")
print(f"   Plane wave transmits: {len(channel_data.sequence)}")
print(f"   Array elements: {channel_data.probe.N}")
print(f"   Samples per acquisition: {channel_data.data.shape[0]}")
print(f"   Frames: {channel_data.data.shape[-1] if channel_data.data.ndim > 3 else 1}")
print(f"   Sampling frequency: {channel_data.sampling_frequency / 1e6:.1f} MHz")
print(f"   Center frequency: {channel_data.modulation_frequency / 1e6:.1f} MHz")
print(f"   Speed of sound: {channel_data.sound_speed} m/s")

# Display plane wave transmit angles
angles_deg = [np.rad2deg(wave.source.azimuth) for wave in channel_data.sequence]
print(f"   Plane wave angles: {min(angles_deg):.1f}° to {max(angles_deg):.1f}°")
print(f"   Angular step: {np.diff(angles_deg)[0]:.1f}°")

# Display imaging region
print("\n🎯 Imaging Region:")
print(f"   Lateral samples: {scan.x_axis.size}")
print(f"   Depth samples: {scan.z_axis.size}")
print(
    f"   Lateral extent: {scan.x_axis.min() * MM_PER_METER:.1f} to {scan.x_axis.max() * MM_PER_METER:.1f} mm"
)
print(
    f"   Depth extent: {scan.z_axis.min() * MM_PER_METER:.1f} to {scan.z_axis.max() * MM_PER_METER:.1f} mm"
)

# %%
# Extract metadata for mach
# ----------------------------------------

print("\n🔄 Preparing data for beamforming...")

# Create beamforming setup for all plane wave angles
beamform_kwargs = create_beamforming_setup(
    channel_data=channel_data,
    scan=scan,
    f_number=1.7,
)

print("📊 Beamforming setup:")
print(f"   Sensor data shape: {beamform_kwargs['channel_data'].shape}")
print("   (transmits, elements, samples, frames)")
print(f"   Output points: {beamform_kwargs['scan_coords_m'].shape[0]:,}")
print(f"   Transmit arrivals shape: {beamform_kwargs['tx_wave_arrivals_s'].shape}")
print(f"   F-number: {beamform_kwargs['f_number']}")

# Extract number of plane-wave transmits
n_transmits = beamform_kwargs["channel_data"].shape[0]
print(f"   Beamforming {n_transmits} plane-wave transmits")

# %%
# Beamform and compound
# -------------------------------------------
#
# Now we beamform and compound the data:
#
# 1. **Individual beamforming**: Apply delay-and-sum beamforming to each plane-wave transmit
# 2. **Coherent compounding**: Sum the results to form the final image
#
# Note: The mach.experimental API is subject to change.

print("\n🚀 Beamforming and compounding...")

result = experimental.beamform(**beamform_kwargs)

print("✓ Beamforming and compounding completed!")
print(f"  Output shape: {result.shape} (points, frames)")
print(f"  Data type: {result.dtype}")
print(f"  Coherently compounded {n_transmits} plane-wave transmits")

# %%
# Reshape results
# ---------------------------
#
# The beamformed data needs to be reshaped to the 2D imaging grid.

print("\n📊 Processing compounded results...")

# Reshape from flattened points back to 2D imaging grid
grid_shape = (scan.x_axis.size, scan.z_axis.size)
beamformed_image = result.reshape(grid_shape)

# Extract magnitude for B-mode display (envelope detection)
bmode_image = np.abs(beamformed_image)

print("✓ Image processing complete")
print(f"  Image shape: {bmode_image.shape} (lateral, depth)")
print(f"  Dynamic range: {bmode_image.min():.2e} to {bmode_image.max():.2e}")

# %%
# Visualize B-Mode
# -----------------------------------------

print("\n📊 Visualizing B-mode...")

# Convert to logarithmic (dB) scale for display
bmode_db = db_zero(beamformed_image)

# Create high-quality visualization
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

# Set up coordinate system for proper display
extent = [
    scan.x_axis.min() * MM_PER_METER,
    scan.x_axis.max() * MM_PER_METER,
    scan.z_axis.max() * MM_PER_METER,
    scan.z_axis.min() * MM_PER_METER,
]

# Display the compounded image
im = ax.imshow(
    bmode_db.T,
    cmap="gray",  # Clinical grayscale colormap
    vmin=-40,  # 40 dB dynamic range
    vmax=0,  # Normalized to maximum
    extent=extent,  # Physical coordinates in mm
    aspect="equal",  # Preserve spatial relationships
    origin="upper",  # Depth increases downward (standard)
    interpolation="nearest",  # Preserve sharp phantom features
)

# Add comprehensive labeling
ax.set_title(
    f"PICMUS Challenge: Plane Wave Compounding\n"
    f"Coherent Compounding of {n_transmits} Plane Wave Angles "
    f"({min(angles_deg):.0f}° to {max(angles_deg):.0f}°)",
    fontsize=14,
)
ax.set_xlabel("Lateral Distance [mm]", fontsize=12)
ax.set_ylabel("Depth [mm]", fontsize=12)

# Add colorbar with proper formatting
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Magnitude [dB]", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Add subtle grid for better readability
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# save the figure with high quality
plt.savefig("plane_wave_compounding_picmus.png", bbox_inches="tight", dpi=300)

# %%
# Timing loop
# ---------------------------
#
# Run a simple timing loop after visualization to benchmark a repeated operation.

print("\n⏱️  Running timing loop (1000 iterations)...")

iterations = 1000
start = time.perf_counter()
for _ in range(iterations):
    _ = db_zero(beamformed_image)
elapsed = time.perf_counter() - start

print(f"✓ Timing loop complete")
print(f"  Total time: {elapsed:.3f} s")
print(f"  Avg per iteration: {elapsed / iterations * 1e3:.3f} ms")

# %%
# Expected results
# ----------------
#
# The plane wave compounded image should clearly resolve:
#
# - **Point targets**: Sharp, well-defined spots for lateral/axial resolution measurement
# - **Hyperechoic lesion**: Bright circle to test for geometric distortion
# - **Uniform speckle**: Consistent background texture in tissue-mimicking regions
