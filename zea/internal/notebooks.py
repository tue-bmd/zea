import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from zea import Scan
from zea.io_lib import save_to_gif


def animate_images(
    images, path, scan: Scan = None, interval=100, cmap="gray", figsize=(5, 4), dpi=80
):
    """Helper function to animate a list of images."""
    if interval <= 0:
        raise ValueError("interval must be a positive integer (milliseconds).")
    if len(images) == 0:
        raise ValueError("images must be a non-empty sequence.")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if scan is not None:
        extent = scan.extent * 1e3 if getattr(scan, "extent") is not None else None
    else:
        extent = None
    im = ax.imshow(np.array(images[0]), animated=True, cmap=cmap, extent=extent)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")

    def update(frame):
        im.set_array(np.array(images[frame]))
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(images),
        blit=True,
        interval=interval,
    )
    plt.close(fig)
    fps = max(1, 1000 // interval)

    ani.save(path, writer="imagemagick", fps=fps)


def animate_volume_mip(
    volume,
    path,
    n_frames=60,
    interval=50,
    cmap="gray",
    axis=0,
    zoom=1.0,
):
    """Create an animated 3D MIP visualization rotating around the volume.

    Args:
        volume (tensor): 3D or 4D volume array.
            If 4D with shape (1, D, H, W), the first dim is squeezed.
        path (str): Output path for the gif file.
        n_frames (int): Number of frames for the 360-degree rotation.
        interval (int): Milliseconds between frames.
        cmap (str): Colormap name.
        axis (int): Axis of rotation: 0 (D/depth), 1 (H/vertical), or 2 (W/horizontal).
        zoom (float): Zoom factor. Values > 1 zoom in, < 1 zoom out.
    """
    # Squeeze batch dimension if present
    vol = np.squeeze(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeezing, got shape {vol.shape}")

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    # Normalize volume to 0-255 uint8
    vol = vol.astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol = (vol * 255).astype(np.uint8)

    # Transpose so rotation is always around axis 1 (vertical in output)
    # We rotate in the (axis0, axis2) plane, axis1 is the vertical output axis
    if axis == 0:
        # Rotate around original axis 0: (D, H, W) -> (H, D, W), rotate in (D, W) plane
        vol = np.transpose(vol, (1, 0, 2))
    elif axis == 1:
        # Rotate around original axis 1: (D, H, W) -> (D, H, W), rotate in (D, W) plane
        pass
    elif axis == 2:
        # Rotate around original axis 2: (D, H, W) -> (D, W, H), rotate in (D, H) plane
        vol = np.transpose(vol, (0, 2, 1))

    d, h, w = vol.shape

    # Generate rotation angles
    angles = np.linspace(0, 360, n_frames, endpoint=False)

    # Output size to fit rotated volume
    diag = int(np.ceil(np.sqrt(w**2 + d**2)))
    out_w = int(np.ceil(diag / zoom))
    out_h = int(np.ceil(h / zoom))

    cx, cz = w / 2, d / 2
    out_cx = out_w / 2

    # Pre-compute output coordinate grid for rotation plane
    x_out, z_out = np.meshgrid(
        (np.arange(out_w) - out_cx) / zoom,
        (np.arange(diag) - diag / 2) / zoom,
        indexing="xy",
    )

    # Get colormap LUT
    cmap_fn = plt.get_cmap(cmap)
    lut = (cmap_fn(np.arange(256))[:, :3] * 255).astype(np.uint8)

    def compute_rotated_mip(angle_deg):
        """Compute MIP by rotating in 2D plane then projecting."""
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Inverse rotation to find source coordinates
        src_x = cos_a * x_out + sin_a * z_out + cx
        src_z = -sin_a * x_out + cos_a * z_out + cz

        # Validity mask
        valid = (src_x >= 0) & (src_x < w) & (src_z >= 0) & (src_z < d)

        # Nearest-neighbor indices
        xi = np.clip(np.rint(src_x).astype(np.int32), 0, w - 1)
        zi = np.clip(np.rint(src_z).astype(np.int32), 0, d - 1)

        # Gather all y slices at once: (diag, out_w, h)
        vals = vol[zi, :, xi]
        vals = np.where(valid[:, :, None], vals, 0)

        # MIP along depth (axis 0), result: (out_w, h)
        mip = np.max(vals, axis=0).T

        # Crop/pad to output height with zoom
        if out_h < h:
            start = (h - out_h) // 2
            mip = mip[start : start + out_h, :]
        elif out_h > h:
            pad_top = (out_h - h) // 2
            pad_bot = out_h - h - pad_top
            mip = np.pad(mip, ((pad_top, pad_bot), (0, 0)), mode="constant")

        return lut[mip]

    # Compute all frames
    frames = np.stack([compute_rotated_mip(angle) for angle in angles])

    # Save using zea's save_to_gif
    fps = max(1, 1000 // interval)
    save_to_gif(frames, path, fps=fps)
