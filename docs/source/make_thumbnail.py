"""Make a thumbnail for the examples page.

Every example on the examples page has a thumbnail in
``docs/_static/thumbnails/<notebook name>.webp``: the image only, with no
axes, ticks, titles, text or colorbars. Make it from the image array in the
notebook, not from a screenshot of a figure. There are two kinds:

``mode="fill"`` (default)
    For rectangular images (linear-array B-mode, Doppler, speed-of-sound
    maps, ...). The image is cut to 4:3 around its centre, so it fills the card
    without black bars. Pass ``box`` to choose the region first.

``mode="cone"``
    For images in a scan cone (cardiac sector scans). The area outside the
    cone becomes transparent, so the thumbnail works in both the light and the
    dark theme. The outside is found as the region, connected to the border,
    that holds exactly the fill value of the scan conversion, so make the
    thumbnail from the scan-converted array before any 8-bit conversion or
    plotting.

From a scratch cell at the end of the notebook (do not commit the cell)::

    import sys

    sys.path.insert(0, "../..")  # docs/source, seen from notebooks/<group>/
    from make_thumbnail import make_thumbnail

    make_thumbnail(image, "my_example", mode="cone", vmin=-60, vmax=0)

``image`` is a 2-D array (shown with ``cmap``, ``vmin`` and ``vmax``), an RGB
or RGBA array, a PIL image or a path. Overlays such as segmentation masks are
best blended into an RGB array first.

To crop an image file instead (``--frame`` picks a frame of an animated GIF)::

    python docs/source/make_thumbnail.py figure.png my_example --box 66 19 325 383

Check the result on the examples page before committing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

THUMBNAIL_DIR = Path(__file__).resolve().parents[1] / "_static" / "thumbnails"
ASPECT = 4 / 3
MAX_SIZE = 640


def load(path: str | Path, frame: int | None = None) -> Image.Image:
    """Load an image file, or one frame (default: the middle) of an animated one."""
    image = Image.open(path)
    n_frames = getattr(image, "n_frames", 1)
    if n_frames > 1:
        image.seek(n_frames // 2 if frame is None else frame)
    return image.convert("RGB")


def _to_rgba(image, cmap: str, vmin, vmax) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGBA"))
    image = np.asarray(image)
    if image.ndim == 2:
        import matplotlib

        vmin = np.nanmin(image) if vmin is None else vmin
        vmax = np.nanmax(image) if vmax is None else vmax
        scaled = np.clip((image - vmin) / (vmax - vmin), 0, 1)
        return (matplotlib.colormaps[cmap](scaled) * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if image.shape[-1] == 3:
        image = np.concatenate([image, np.full(image.shape[:2] + (1,), 255, np.uint8)], -1)
    return image


def _outside_cone(image) -> np.ndarray:
    """The border-connected region that holds exactly the corner (fill) value."""
    image = np.asarray(image)
    corner = image[0, 0]
    if image.ndim == 3:
        same = np.all(image == corner, axis=-1)
    elif np.isnan(corner):
        same = np.isnan(image)
    else:
        same = image == corner
    labels, _ = ndimage.label(same)
    edges = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    border = np.unique(edges)
    return np.isin(labels, border[border > 0])


def fill(image: Image.Image) -> Image.Image:
    """Cut the largest centred 4:3 region out of ``image``."""
    w, h = image.size
    if w / h > ASPECT:
        new_w = round(h * ASPECT)
        left = (w - new_w) // 2
        return image.crop((left, 0, left + new_w, h))
    new_h = round(w / ASPECT)
    top = (h - new_h) // 2
    return image.crop((0, top, w, top + new_h))


def make_thumbnail(
    image,
    name: str,
    *,
    mode: str = "fill",
    box: tuple[int, int, int, int] | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    out_dir: Path = THUMBNAIL_DIR,
) -> Path:
    """Save ``image`` as the thumbnail ``<out_dir>/<name>.webp``.

    Args:
        image: 2-D array, RGB(A) array, PIL image or path to an image file.
        name: the notebook file name without ``.ipynb``.
        mode: ``"fill"`` to cut to 4:3, ``"cone"`` to make the area outside a
            scan cone transparent.
        box: optional crop (left, top, right, bottom) in pixels, applied first.
        cmap, vmin, vmax: how to show a 2-D array.
        out_dir: where to save, by default ``docs/_static/thumbnails``.
    """
    if mode not in ("fill", "cone"):
        raise ValueError(f"mode must be 'fill' or 'cone', not {mode!r}")
    if isinstance(image, (str, Path)):
        image = load(image)
    if box is not None:
        left, top, right, bottom = box
        if isinstance(image, Image.Image):
            image = image.crop(box)
        else:
            image = np.asarray(image)[top:bottom, left:right]

    rgba = _to_rgba(image, cmap, vmin, vmax)
    if mode == "cone":
        source = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else image
        rgba = rgba.copy()
        rgba[_outside_cone(source), 3] = 0
        thumb = Image.fromarray(rgba)
        thumb = thumb.crop(thumb.getbbox())
    else:
        thumb = fill(Image.fromarray(rgba).convert("RGB"))

    thumb.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.webp"
    thumb.save(path, quality=88)
    return path


def main():
    parser = argparse.ArgumentParser(description="Make an examples-page thumbnail.")
    parser.add_argument("image", help="source image (png, jpg, gif, ...)")
    parser.add_argument("name", help="notebook file name without .ipynb")
    parser.add_argument("--box", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--frame", type=int, help="frame of an animated image")
    parser.add_argument("--mode", choices=["fill", "cone"], default="fill")
    args = parser.parse_args()
    box = tuple(args.box) if args.box else None
    path = make_thumbnail(load(args.image, args.frame), args.name, mode=args.mode, box=box)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
