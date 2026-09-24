"""Gradio visualiser for zea datasets.

Usage:
    python -m zea.data.app
    python -m zea.data.app --share
    python -m zea.data.app --server-port 7861
"""

import base64
import contextlib
import html
import io
import os
import re
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tyro
from keras import ops

from zea import display, io_lib
from zea.cli_args import AppArgs
from zea.config import Config
from zea.data.dataloader import Dataloader
from zea.data.chunk_cache import network_bytes
from zea.data.datasets import FILE_TYPES, Dataset
from zea.data.file import File
from zea.data.process import (
    _axis_selections_from_params,
    _get_config_parameters,
    _key_requires_pipeline,
)
from zea.internal.device import init_device
from zea.internal.preset_utils import HF_PREFIX, _hf_list_files, _hf_parse_path
from zea.ops.pipeline import Pipeline

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "gradio is required for the zea app. Install with: pip install 'zea[app]'"
    ) from exc

# Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY → HTTP_422_UNPROCESSABLE_CONTENT;
# gradio hasn't updated yet, so filter the noise until a gradio release catches up.
warnings.filterwarnings(
    "ignore",
    message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated",
)


def _bind_gradio_event(component: Any, method: str, *args, **kwargs):
    # Gradio's built-in indicator says "processing" even for a quick lookup; slow
    # steps show their own status message instead (see _html_busy).
    kwargs.setdefault("show_progress", "hidden")
    return getattr(component, method)(*args, **kwargs)


# ── Logo ───────────────────────────────────────────────────────────────────────

_LOGO_PATH = Path(__file__).parent.parent.parent / "docs/_static/zea-logo.png"


def _logo_html(height: int = 36) -> str:
    try:
        with open(_LOGO_PATH, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="height:{height}px;width:auto;max-height:{height}px;'
            'vertical-align:middle;margin-right:8px;display:inline-block" />'
        )
    except Exception:
        return ""


# ── Colours ───────────────────────────────────────────────────────────────────

# One accent colour (zea yellow), used sparingly; everything else stays neutral.
_YELLOW = "#f5c518"
_MUTED = "#9ca3af"
_AMBER = "#f59e0b"


def build_theme() -> "gr.themes.Base":
    """Calm, neutral theme with zea yellow as the only accent."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.yellow,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    ).set(
        slider_color="*primary_400",
        # Primary buttons in zea yellow with dark text, like Run.
        button_primary_background_fill=_YELLOW,
        button_primary_background_fill_dark=_YELLOW,
        button_primary_background_fill_hover="#e6b800",
        button_primary_background_fill_hover_dark="#e6b800",
        button_primary_text_color="#111111",
        button_primary_text_color_dark="#111111",
        button_primary_border_color=_YELLOW,
        button_primary_border_color_dark=_YELLOW,
        checkbox_label_background_fill_selected="*primary_50",
        checkbox_label_background_fill_selected_dark="*neutral_800",
        checkbox_label_border_color_selected="*primary_400",
        checkbox_label_border_color_selected_dark="*primary_400",
    )


# ── Data key choices ──────────────────────────────────────────────────────────

_DATA_KEYS = [
    "data/raw_data",
    "data/aligned_data/values",
    "data/beamformed_data/values",
    "data/envelope_data/values",
    "data/image/values",
    "data/segmentation/values",
    "data/sos_map/values",
    "data/attenuation_map/values",
]

# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "PICMUS › Contrast speckle (RF)": {
        "dataset": (
            "hf://zeahub/picmus/database/experiments/contrast_speckle/"
            "contrast_speckle_expe_dataset_rf"
        ),
        "config": "hf://zeahub/picmus/config_rf.yaml",
        "key": "data/raw_data",
    },
    "PICMUS › Resolution distortion (IQ)": {
        "dataset": (
            "hf://zeahub/picmus/database/experiments/resolution_distorsion/"
            "resolution_distorsion_expe_dataset_iq"
        ),
        "config": "hf://zeahub/picmus/config_iq.yaml",
        "key": "data/raw_data",
    },
    "zea › Cardiac 2026": {
        "dataset": "hf://zeahub/zea-cardiac-2026",
        "config": "hf://zeahub/zea-cardiac-2026/config.yaml",
        "key": "data/raw_data",
    },
    "zea › Carotid 2023": {
        "dataset": "hf://zeahub/zea-carotid-2023",
        "config": "hf://zeahub/zea-carotid-2023/config.yaml",
        "key": "data/raw_data",
    },
    "CAMUS › Cardiac echo": {
        "dataset": "hf://zeahub/camus",
        "config": "hf://zeahub/configs/config_camus.yaml",
        "key": "data/image/values",
    },
}


_OPENH_RF = "hf://nvidia/OpenH-RF"


def _openh_rf(
    subset: str,
    file: str,
    n_frames: int = 1,
    data_dir: str = "data",
    revision: str | None = None,
) -> dict:
    """Preset for an OpenH-RF subset: opens the subset folder with *file* pre-selected
    and the subset's ``pipeline.yaml`` as config, optionally at a *revision*."""
    root = f"{_OPENH_RF}/{subset}"
    folder = f"{root}/{data_dir}" if data_dir else root
    preset = {
        "dataset": root,
        "config": f"{root}/pipeline.yaml",
        "key": "data/raw_data",
        "file": f"{folder}/{file}",
        "n_frames": n_frames,
    }
    if revision:
        preset["revision"] = revision
    return preset


# The per-subset Oslo pipelines live in an open pull request for now; drop the
# revision once https://huggingface.co/datasets/nvidia/OpenH-RF/discussions/66 is merged.
_OSLO_REVISION = "refs/pr/66"


def _oslo(subset: str, file: str, n_frames: int = 1) -> dict:
    """Oslo keeps its files directly in each subset folder (no ``data/``)."""
    return _openh_rf(f"oslo/{subset}", file, n_frames, data_dir="", revision=_OSLO_REVISION)


# Optional preset keys: "file" pre-selects a file, "n_frames" sets the frame count for it.
PRESETS.update(
    {
        "OpenH-RF › Concordia": _openh_rf("concordia", "image_0005.hdf5"),
        "OpenH-RF › KAIST-SNUBH › Barreleye": _openh_rf("kaist-snubh-barreleye", "S01_D1.hdf5"),
        "OpenH-RF › Technion › Bladder": _openh_rf("technion/bladder", "a1.hdf5", 10),
        "OpenH-RF › Technion › Cardiac": _openh_rf("technion/cardiac", "c1.hdf5", 32),
        "OpenH-RF › Technion › Phantom": _openh_rf("technion/phantom", "ph.hdf5"),
        "OpenH-RF › TU/e › AAA": _openh_rf("tue-aaa", "AAA_subject11.hdf5"),
        "OpenH-RF › TU/e › Carotid": _openh_rf("tue-carotid", "5_long_bifur_R_0000.hdf5"),
        "OpenH-RF › Vanderbilt": _openh_rf(
            "vanderbilt", "Fundamental/118420_1_Focused_Uncoded_TX.hdf5", 10
        ),
        # Oslo: the example acquisition from each sub-dataset's README; its pipeline.yaml
        # is tuned for that file, so other files in the subset may need another one.
        "OpenH-RF › Oslo › Cardiac": _oslo(
            "A_cardiac", "Verasonics_P2-4_parasternal_long_subject_1.hdf5"
        ),
        "OpenH-RF › Oslo › Carotid": _oslo("B_carotid", "L7_FI_carotid_cross_1.hdf5"),
        "OpenH-RF › Oslo › Verasonics phantom": _oslo(
            "C_verasonics_phantom", "FI_P4_cysts_center.hdf5"
        ),
        "OpenH-RF › Oslo › Alpinion phantom": _oslo(
            "D_alpinion_phantom", "Alpinion_L3-8_CPWC_hypoechoic.hdf5"
        ),
        "OpenH-RF › Oslo › Simulation": _oslo("E_simulation", "PICMUS_numerical_calib_v2.hdf5"),
        "OpenH-RF › Oslo › Motion": _oslo("F_motion", "SWE_L7_type_III.hdf5", 20),
    }
)

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
:root { color-scheme: dark; }
footer { display: none !important; }
.run-status { min-height: 1.6em; font-size: 0.9em; margin-top: -8px; }
.status-box { max-height: 320px; overflow-y: auto; scroll-behavior: smooth; }
.revision-dropdown .wrap select { padding-right: 2.2em !important; }
.run-btn { background: #f5c518 !important; border-color: #f5c518 !important;
  color: #111 !important; }
.run-btn:hover { background: #e6b800 !important; border-color: #e6b800 !important; }
.run-btn:disabled { background: #5a4a00 !important; border-color: #5a4a00 !important;
  color: #888 !important; opacity: 0.5 !important; }
.frame-slider input[type=number] {
  pointer-events: none !important; background: transparent !important;
  border: none !important; box-shadow: none !important; cursor: default !important; }
.frame-slider button { display: none !important; }
.zea-frames-info { color: #9ca3af; font-size: 0.85em; margin: 6px 0 0; }
.zea-divider { display: flex; align-items: center; gap: 8px; color: #9ca3af;
  font-size: 0.8em; margin: 2px 0; }
.zea-divider::before, .zea-divider::after { content: ""; flex: 1;
  border-top: 1px solid #44403c; }
.zea-indeterminate { background-color: #44403c; border-radius: 3px; height: 5px;
  background-image: linear-gradient(90deg, transparent, #f5c518, transparent);
  background-size: 40% 100%; background-repeat: no-repeat;
  animation: zea-slide 1.2s linear infinite; }
.zea-config-badge { font-size: 0.8em; color: #9ca3af; margin: 0 0 -6px; }
.zea-badge-accent { color: #f5c518; }
.zea-links { font-size: 0.8em; color: #9ca3af; margin: -2px 0 4px; }
.zea-links a { color: inherit; text-decoration: none; border-bottom: 1px dotted #9ca3af; }
.zea-links a:hover { color: #f5c518; border-bottom-color: #f5c518; }
.zea-path input, .zea-path textarea { font-family: var(--font-mono) !important;
  font-size: 0.9em !important; }
@keyframes zea-slide { from { background-position: -40% 0; } to { background-position: 140% 0; } }
"""

# Dark mode only: the colours above assume a dark background. Gradio toggles a `dark`
# class to follow the OS preference, so pin it on and put it back if Gradio removes it.
# Gradio runs launch(js=...) as a <script>, not as a function, hence the IIFE.
JS = """
(() => {
    const container = document.querySelector('.gradio-container');
    const targets = [document.body, container && container.parentElement].filter(Boolean);
    const pin = () => targets.forEach((el) => {
        if (!el.classList.contains('dark')) el.classList.add('dark');
    });
    pin();
    targets.forEach((el) => new MutationObserver(pin).observe(el, { attributeFilter: ['class'] }));
})();
"""

_SCROLL_JS = """
() => {
    requestAnimationFrame(() => {
        const el = document.querySelector('.status-box');
        if (el) el.scrollTop = el.scrollHeight;
    });
}
"""

# ── Stop signal ───────────────────────────────────────────────────────────────

_stop_event = threading.Event()

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)


def _is_hf(path: str) -> bool:
    return str(path).strip().startswith(HF_PREFIX)


# Matches the scheme of mistyped HF paths such as ``hf:://``, ``hf:/`` or ``HF://``.
_HF_SCHEME_RE = re.compile(r"^hf:[:/]*", re.IGNORECASE)


def _normalize_path(path: str | None) -> str:
    """Clean up a user-typed dataset/config path.

    Strips whitespace and surrounding quotes, repairs mistyped ``hf://`` schemes
    (``hf:://``, ``hf:/``, ``HF://``) and drops trailing slashes from HF paths.
    """
    path = str(path or "").strip().strip("'\"").strip()
    if _HF_SCHEME_RE.match(path):
        path = HF_PREFIX + _HF_SCHEME_RE.sub("", path, count=1).rstrip("/")
    return path


def _hf_web_url(path: str, revision: str | None = None) -> str | None:
    """Browser URL on huggingface.co for an ``hf://`` dataset path at *revision*."""
    from urllib.parse import quote

    path = _normalize_path(path)
    if not _is_hf(path):
        return None
    repo_id, subpath = _hf_parse_path(path)
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        return None
    rev = quote((revision or "").strip() or "main", safe="")
    base = f"https://huggingface.co/datasets/{repo_id}"
    if not subpath:
        return base if rev == "main" else f"{base}/tree/{rev}"
    kind = "blob" if Path(subpath).suffix.lower() in (*FILE_TYPES, ".yaml", ".yml") else "tree"
    return f"{base}/{kind}/{rev}/{quote(subpath)}"


def _hf_links_html(
    dataset: str, dataset_rev: str | None, config: str, config_rev: str | None
) -> str:
    """'View on Hugging Face' links for the dataset and config, or '' for local paths."""
    links = []
    for label, path, rev in (("Dataset", dataset, dataset_rev), ("Config", config, config_rev)):
        url = _hf_web_url(path, rev)
        if url:
            rev_note = f" @ {html.escape(rev)}" if rev and rev != "main" else ""
            links.append(
                f'<a href="{html.escape(url)}" target="_blank" rel="noopener">'
                f"{label} on Hugging Face{rev_note} &#8599;</a>"
            )
    return f'<div class="zea-links">{" &nbsp;·&nbsp; ".join(links)}</div>' if links else ""


def _display_names(root: str, file_paths: list[str]) -> list[str]:
    """Label each file by its path relative to *root*, so same-named files in
    different subdirectories stay distinguishable. Falls back to the basename."""
    root = root.rstrip("/")
    names = []
    for fp in file_paths:
        fp = str(fp)
        rel = None
        if _is_hf(fp):
            if fp.startswith(root + "/"):
                rel = fp[len(root) + 1 :]
        else:
            try:
                rel = str(Path(fp).resolve().relative_to(Path(root).resolve()))
            except ValueError:
                pass
        names.append(rel if rel and rel != "." else Path(fp).name)
    return names


_CONFIG_CANDIDATES = ("config.yaml", "pipeline.yaml", "config.yml", "pipeline.yml")


def _find_config(dataset_path: str, revision: str | None = None) -> str | None:
    """Look for a config next to the dataset, walking up to the repo root (HF) or
    one parent directory (local). Returns the config path, or ``None``."""
    dataset_path = _normalize_path(dataset_path)
    if not dataset_path:
        return None
    try:
        if _is_hf(dataset_path):
            repo_id, subpath = _hf_parse_path(dataset_path)
            kwargs = {"revision": revision} if revision else {}
            files = set(_hf_list_files(repo_id, **kwargs))
            parts = subpath.split("/") if subpath else []
            if parts and Path(parts[-1]).suffix.lower() in FILE_TYPES:
                parts = parts[:-1]
            while True:
                prefix = "/".join(parts)
                for name in _CONFIG_CANDIDATES:
                    rel = f"{prefix}/{name}" if prefix else name
                    if rel in files:
                        return f"{HF_PREFIX}{repo_id}/{rel}"
                if not parts:
                    return None
                parts.pop()
        p = Path(dataset_path)
        if p.is_file():
            p = p.parent
        for d in (p, p.parent):
            for name in _CONFIG_CANDIDATES:
                if (d / name).is_file():
                    return str(d / name)
    except Exception:
        pass
    return None


def _enrich_error(exc: Exception) -> str:
    try:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
        )
        from huggingface_hub.utils import HFValidationError

        if isinstance(exc, GatedRepoError):
            return (
                str(exc) + "\n\nThis repository is gated. Accept the terms on Hugging Face "
                "and set the HF_TOKEN environment variable."
            )
        if isinstance(exc, RepositoryNotFoundError):
            return (
                str(exc) + "\n\nRepository not found. Check the path. "
                "If the repo is private, set the HF_TOKEN environment variable."
            )
        if isinstance(exc, EntryNotFoundError):
            return str(exc) + "\n\nFile not found. Check the path."
        if isinstance(exc, HFValidationError):
            return str(exc) + "\n\nInvalid Hugging Face repository ID format."
    except ImportError:
        pass
    return str(exc)


def _html_pass(msg: str) -> str:
    return f'<p style="margin:2px 0;color:#22c55e">&#10004; {html.escape(msg)}</p>'


def _html_fail(msg: str, err: Exception | str | None = None) -> str:
    out = f'<p style="margin:2px 0;color:#ef4444">&#10008; {html.escape(msg)}</p>'
    if err is not None:
        detail = _enrich_error(err) if isinstance(err, Exception) else str(err)
        escaped = html.escape(detail).replace("\n", "<br>")
        out += f'<p style="margin:2px 0 2px 1.5em;font-size:0.85em;color:#ef4444">{escaped}</p>'
    return out


def _html_warn(msg: str) -> str:
    return f'<p style="margin:2px 0;color:{_AMBER}">&#9888; {html.escape(msg)}</p>'


def _html_info(msg: str) -> str:
    return f'<p style="margin:2px 0;color:{_MUTED}">&#8250; {html.escape(msg)}</p>'


def _html_busy(msg: str) -> str:
    """A status line with an indeterminate bar, for steps of unknown duration."""
    return (
        f'<div style="margin:4px 0"><span style="color:{_MUTED};font-size:0.85em">'
        f'{html.escape(msg)}</span><div class="zea-indeterminate" style="margin-top:3px">'
        "</div></div>"
    )


def _busy_lookup(path: str) -> str:
    path = _normalize_path(path)
    if not path:
        return ""
    return _html_busy("Looking up files on Hugging Face…" if _is_hf(path) else "Looking up files…")


def _busy_revision(revision: str) -> str:
    return _html_busy(f"Looking up files at revision {revision}…")


def _busy_example(name: str) -> str:
    return _html_busy("Loading example…") if name else ""


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def _html_progress(current: int, total: int, streamed_bytes: int | None = None) -> str:
    pct = int(current / total * 100)
    streamed = f" · {_fmt_bytes(streamed_bytes)} streamed" if streamed_bytes else ""
    return (
        f'<div class="zea-progress" style="margin:4px 0">'
        f'<span style="color:{_MUTED};font-size:0.9em">'
        f"Processing frame {current}/{total}{streamed}</span>"
        f'<div style="background:#44403c;border-radius:3px;height:5px;margin-top:3px">'
        f'<div style="background:{_YELLOW};border-radius:3px;height:5px;width:{pct}%"></div>'
        f"</div></div>"
    )


# ── HF / file listing ─────────────────────────────────────────────────────────


def _fetch_hf_revisions(path: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_refs

        repo_id, _ = _hf_parse_path(_normalize_path(path))
        if "/" not in repo_id or not repo_id.split("/")[1]:
            return ["main"]
        refs = list_repo_refs(repo_id, repo_type="dataset")
        branches = [b.name for b in refs.branches]
        tags = [t.name for t in refs.tags]
        all_revs = branches + tags
        return all_revs if all_revs else ["main"]
    except Exception:
        return ["main"]


def _list_dataset_files(
    path: str,
    revision: str | None = None,
    _errors: list | None = None,
) -> tuple[list[str], list[str]]:
    """List HDF5 files in a dataset without downloading any data.

    Uses Dataset with lazy=True so HF files are listed via the API but not
    downloaded. For local paths it scans the directory tree.
    Returns (display_names, full_paths).

    If *_errors* is provided (a list), any exception encountered is appended to
    it instead of being silently dropped, so callers can surface the problem.
    """
    path = _normalize_path(path)
    if not path:
        return [], []
    try:
        if not _is_hf(path) and not Path(path).exists():
            raise FileNotFoundError(
                f"Path not found: {path}. Use a local path or hf://owner/repo[/subdir]."
            )
        ds = Dataset(path, lazy=True, revision=revision, _suggest_lazy=False)
        file_paths = sorted(ds.file_paths)
        ds.close()
        return _display_names(path, file_paths), file_paths
    except Exception as exc:
        if _errors is not None:
            _errors.append(exc)
        return [], []


# ── File metadata ─────────────────────────────────────────────────────────────


def _read_file_info(file_path: str, revision: str | None = None) -> dict:
    """Open an HDF5 file and read metadata/shape without loading data arrays."""
    info: dict = {}
    hf_kwargs = {"revision": revision} if revision and _is_hf(file_path) else {}
    try:
        with File(file_path, **hf_kwargs) as f:
            # zea version
            info["zea_version"] = f.zea_version

            # File-level attributes
            for attr in ("us_machine", "description"):
                val = f.attrs.get(attr)
                if val:
                    info[attr] = str(val)

            # Probe group
            try:
                info["probe_name"] = f.probe_name
            except Exception:
                pass
            try:
                if "probe" in f:
                    pg = f["probe"]
                    if "type" in pg:
                        raw = pg["type"][()]
                        info["probe_type"] = raw.decode() if isinstance(raw, bytes) else str(raw)
                    if "probe_center_frequency" in pg:
                        info["probe_fc_hz"] = float(pg["probe_center_frequency"][()])
                    if "probe_bandwidth_percent" in pg:
                        info["probe_bw_pct"] = float(pg["probe_bandwidth_percent"][()])
                    if "probe_geometry" in pg:
                        info["n_el_probe"] = int(pg["probe_geometry"].shape[0])
            except Exception:
                pass

            # Tracks
            n_tracks = f._n_tracks
            info["n_tracks"] = n_tracks
            try:
                if n_tracks > 1:
                    tracks = f.tracks
                    info["track_labels"] = [t.label or f"track {i}" for i, t in enumerate(tracks)]
                    info["n_frames_per_track"] = [t.n_frames for t in tracks]
                else:
                    info["track_labels"] = []
                    info["n_frames_per_track"] = [f.n_frames]
            except Exception:
                info.setdefault("track_labels", [])
                info.setdefault("n_frames_per_track", [])

            # Scan parameters: read only the few values shown. Loading the whole scan
            # group costs one round trip per dataset on a streamed file.
            try:
                scan_key = "tracks/track_0/scan" if n_tracks > 1 else "scan"
                if scan_key in f:
                    sg = f[scan_key]
                    for name, field in (
                        ("sampling_frequency", "fs_hz"),
                        ("center_frequency", "fc_hz"),
                        ("sound_speed", "sound_speed"),
                    ):
                        if name in sg:
                            info[field] = float(np.asarray(sg[name][()]).flat[0])
                    if "t0_delays" in sg:
                        shp = sg["t0_delays"].shape
                        if len(shp) >= 2:
                            info["n_tx"] = int(shp[0])
                            info["n_el"] = int(shp[1])
            except Exception:
                pass

            # For multi-track files the data lives at tracks/track_0/data/{bare}.
            # Use that path directly to avoid triggering "Multiple tracks found"
            # warnings on every format_key call.
            _data_root = "tracks/track_0/data" if n_tracks > 1 else None

            # n_ax from raw_data shape
            try:
                fkey = f"{_data_root}/raw_data" if _data_root else f.format_key("data/raw_data")
                shp = f[fkey].shape
                if len(shp) >= 3:
                    info["n_ax"] = int(shp[2])
            except Exception:
                pass

            # Discover data keys: only data/<flat> (no slash) and data/<map>/values
            available = []
            try:
                data_prefix = _data_root if _data_root else f.format_key("data")
                if data_prefix in f:
                    data_grp = f[data_prefix]
                    # Walk links rather than visititems(): visiting opens every dataset's
                    # header, which on a streamed file costs seconds per dataset.
                    for name in data_grp:
                        if data_grp.get(name, getclass=True) is h5py.Group:
                            if "values" in data_grp[name]:
                                available.append(f"data/{name}/values")
                        else:
                            available.append(f"data/{name}")
            except Exception:
                pass
            # Fall back to checking known keys if discovery failed
            if not available:
                for k in _DATA_KEYS:
                    try:
                        bare = k.removeprefix("data/")
                        fk = f"{_data_root}/{bare}" if _data_root else f.format_key(k)
                        if fk in f:
                            available.append(k)
                    except Exception:
                        pass
            if available:
                info["available_keys"] = available

            # Metadata group: credit, subject, annotations
            try:
                if "metadata" in f:
                    mg = f["metadata"]
                    if "credit" in mg:
                        raw = mg["credit"][()]
                        s = raw.decode() if isinstance(raw, bytes) else str(raw)
                        if s:
                            info["credit"] = s
                    if "subject" in mg:
                        sg = mg["subject"]
                        for field in ("id", "type"):
                            if field in sg:
                                raw = sg[field][()]
                                s = raw.decode() if isinstance(raw, bytes) else str(raw)
                                if s:
                                    info[f"subject_{field}"] = s
                    if "annotations" in mg:
                        ag = mg["annotations"]
                        for field in ("anatomy", "view"):
                            if field in ag:
                                raw = ag[field][()]
                                if isinstance(raw, (bytes, np.bytes_)):
                                    info[f"annot_{field}"] = raw.decode()
                                elif isinstance(raw, np.ndarray):
                                    unique = np.unique(raw)
                                    if len(unique) == 1:
                                        v = unique[0]
                                        val = v.decode() if isinstance(v, bytes) else str(v)
                                        if val:
                                            info[f"annot_{field}"] = val
                                elif raw:
                                    info[f"annot_{field}"] = str(raw)
            except Exception:
                pass
    except Exception as exc:
        info["error"] = _enrich_error(exc)

    return info


_SEP = '&nbsp;<span style="color:#4b5563">·</span>&nbsp;'


def _build_meta_card_html(info: dict) -> str:
    """Build a sectioned HTML info card from a _read_file_info dict."""
    if not info:
        return ""

    def _badge(label: str, value: str, color: str = "#9ca3af") -> str:
        return (
            f'<span style="display:inline-block;margin:1px 3px 1px 0;'
            f"padding:1px 6px;border-radius:3px;background:rgba(255,255,255,0.06);"
            f'color:{color};white-space:nowrap">'
            f'<span style="color:#6b7280;font-size:0.88em">{label}&nbsp;</span>{value}</span>'
        )

    def _section(title: str, badges: list[str]) -> str:
        if not badges:
            return ""
        joined = "".join(badges)
        return (
            f'<div style="margin-top:5px">'
            f'<div style="color:#6b7280;font-size:0.78em;text-transform:uppercase;'
            f'letter-spacing:0.05em;margin-bottom:2px">{title}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:2px">{joined}</div>'
            f"</div>"
        )

    sections = []

    # ── File / version ──────────────────────────────────────────────────────
    file_badges = []
    zv = info.get("zea_version")
    if zv:
        file_badges.append(_badge("zea", zv, _YELLOW))
    else:
        file_badges.append(
            '<span style="display:inline-block;margin:1px 3px 1px 0;padding:1px 6px;'
            "border-radius:3px;background:rgba(255,255,255,0.06);"
            'color:#6b7280;font-size:0.88em;white-space:nowrap">legacy format</span>'
        )
    n_frames_list = info.get("n_frames_per_track", [])
    n_tracks = info.get("n_tracks", 1)
    if n_frames_list:
        total = sum(n_frames_list)
        file_badges.append(_badge("frames", str(total)))
        if n_tracks > 1:
            file_badges.append(_badge("tracks", str(n_tracks)))
    sections.append(_section("File", file_badges))

    # ── Probe ───────────────────────────────────────────────────────────────
    probe_badges = []
    if info.get("probe_name"):
        probe_badges.append(
            f'<span style="display:inline-block;margin:1px 3px 1px 0;padding:1px 6px;'
            f"border-radius:3px;background:rgba(255,255,255,0.06);"
            f'color:#e5e7eb;font-weight:600;white-space:nowrap">{info["probe_name"]}</span>'
        )
    if info.get("probe_type"):
        probe_badges.append(_badge("type", info["probe_type"], "#d1d5db"))
    n_el = info.get("n_el_probe") or info.get("n_el")
    if n_el:
        probe_badges.append(_badge("el", str(n_el)))
    p_fc = info.get("probe_fc_hz")
    if p_fc:
        probe_badges.append(_badge("fc", f"{p_fc / 1e6:.1f}&nbsp;MHz"))
    if info.get("probe_bw_pct"):
        probe_badges.append(_badge("BW", f"{info['probe_bw_pct']:.0f}%"))
    if probe_badges:
        sections.append(_section("Probe", probe_badges))

    # ── Scan ────────────────────────────────────────────────────────────────
    scan_badges = []
    if info.get("us_machine"):
        scan_badges.append(_badge("system", info["us_machine"], "#d1d5db"))
    if info.get("fs_hz"):
        scan_badges.append(_badge("fs", f"{info['fs_hz'] / 1e6:.1f}&nbsp;MHz"))
    tx_fc = info.get("fc_hz")
    if tx_fc and (not p_fc or abs(p_fc - tx_fc) > 0.5e6):
        scan_badges.append(_badge("tx&nbsp;fc", f"{tx_fc / 1e6:.1f}&nbsp;MHz"))
    if info.get("sound_speed"):
        scan_badges.append(_badge("c", f"{info['sound_speed']:.0f}&nbsp;m/s"))
    if info.get("n_tx"):
        scan_badges.append(_badge("tx", str(info["n_tx"])))
    if info.get("n_ax"):
        scan_badges.append(_badge("ax", str(info["n_ax"])))
    if scan_badges:
        sections.append(_section("Scan", scan_badges))

    # ── Metadata ────────────────────────────────────────────────────────────
    meta_badges = []
    if info.get("subject_type"):
        meta_badges.append(_badge("subject", info["subject_type"], "#d1d5db"))
    if info.get("subject_id"):
        meta_badges.append(_badge("id", info["subject_id"]))
    if info.get("annot_anatomy"):
        meta_badges.append(_badge("anatomy", info["annot_anatomy"], "#d1d5db"))
    if info.get("annot_view"):
        meta_badges.append(_badge("view", info["annot_view"]))
    if info.get("credit"):
        meta_badges.append(
            '<div style="width:100%;margin:2px 0;color:#9ca3af;font-size:0.88em;'
            'word-break:break-word;line-height:1.5">'
            f'<span style="color:#6b7280">credit&nbsp;</span>{info["credit"]}</div>'
        )
    if info.get("description"):
        meta_badges.append(
            '<div style="width:100%;margin:2px 0;color:#9ca3af;font-size:0.88em;'
            'word-break:break-word;line-height:1.5">'
            f'<span style="color:#6b7280">desc&nbsp;</span>{info["description"]}</div>'
        )
    if meta_badges:
        sections.append(_section("Metadata", meta_badges))

    if not sections:
        return ""

    return (
        f'<div style="border-left:3px solid {_YELLOW};border-radius:4px;'
        f"background:rgba(245,197,24,0.05);padding:6px 10px;margin-bottom:4px;"
        f'font-size:0.83em">' + "".join(sections) + "</div>"
    )


def _file_load_updates(fpath: str, revision: str | None, key: str, n_frames: int = 1) -> tuple:
    """Download (if HF) and read a file; return the 7 gr.update() values for file-select outputs.

    Returns: (start_frame_upd, n_frames_upd, meta_html, track_upd, track_labels,
               run_btn_upd, key_input_upd, frame_state)
    """
    info = _read_file_info(fpath, revision)

    n_frames_list = info.get("n_frames_per_track", [])
    n_tracks = info.get("n_tracks", 1)
    track_labels = info.get("track_labels", [])
    n = n_frames_list[0] if n_frames_list else 0
    available_keys = info.get("available_keys", _DATA_KEYS)
    current_key = (key or "").strip()
    if current_key in available_keys:
        new_key = current_key
    elif "data/raw_data" in available_keys:
        new_key = "data/raw_data"
    else:
        new_key = None  # user must choose

    if "error" in info and "zea_version" not in info:
        meta_html = _html_fail("Cannot read file", info["error"])
    else:
        meta_html = _build_meta_card_html(info)

    sf_upd, nf_upd = _frame_sliders(n, n_frames)

    if n_tracks > 1 and track_labels:
        # Use numeric indices as values so duplicate labels don't break selection.
        choices = [(label, i) for i, label in enumerate(track_labels)]
        track_upd = gr.update(choices=choices, value=0, visible=True, interactive=True)
    else:
        track_upd = gr.update(choices=[("track 0", 0)], value=0, visible=False, interactive=False)

    return (
        sf_upd,
        nf_upd,
        meta_html,
        track_upd,
        track_labels,
        gr.update(interactive=new_key is not None),  # run_btn: only if key auto-resolved
        gr.update(choices=available_keys, value=new_key, interactive=True),
        {"file": fpath, "n_per_track": n_frames_list, "track": 0, "clip": int(n_frames) > 1},
    )


def _frame_sliders(n: int, n_frames: int = 1) -> tuple:
    """Start-frame / frame-count slider updates for a track with *n* frames."""
    if n > 1:
        return (
            gr.update(maximum=n - 1, value=0, interactive=True),
            gr.update(maximum=n, value=max(1, min(int(n_frames), n)), interactive=True),
        )
    # Single or zero-frame track: keep maximum > minimum (Gradio requires it strictly).
    return gr.update(value=0, interactive=False), gr.update(value=1, interactive=False)


def _frames_info_html(n: int | None, clip: bool = False) -> str:
    if n is None:
        return '<div class="zea-frames-info">Select a file to see its frames.</div>'
    if n == 0:
        return '<div class="zea-frames-info">No frames found for this data.</div>'
    if n == 1:
        return (
            '<div class="zea-frames-info">&#9432;&nbsp;This file contains a '
            "<b>single frame</b>. Press Run to show it.</div>"
        )
    hint = "rendered as a GIF" if clip else "pick one, or switch to Clip for a GIF"
    return f'<div class="zea-frames-info"><b>{n:,}</b> frames available · {hint}</div>'


def _loading_meta_html(streamed_bytes: int = 0, size_bytes: int | None = None) -> str:
    """Progress card shown while a file's metadata is streamed. The amount h5py needs to
    read is not known up front, so the bar is indeterminate and the byte count is live."""
    of_file = f" of a {_fmt_bytes(size_bytes)} file" if size_bytes else ""
    return (
        f'<div style="margin:4px 0">'
        f'<span style="color:{_MUTED};font-size:0.9em">Streaming file metadata · '
        f"{_fmt_bytes(streamed_bytes)} fetched{of_file}</span>"
        f'<div class="zea-indeterminate" style="margin-top:3px"></div>'
        f'<span style="color:#6b7280;font-size:0.8em">Only the parts that are needed are read'
        f"; the file is not downloaded.</span></div>"
    )


_NO_DATASET_INFO = "Choose a dataset in ① Data first."
_FILE_INFO = "Type to filter."


# ── Config loader ─────────────────────────────────────────────────────────────


def _load_config_text(path: str, revision: str | None = None) -> str:
    path = _normalize_path(path)
    revision = (revision or "").strip() or None
    if not path:
        return "# No config path specified."
    try:
        if _is_hf(path):
            from huggingface_hub import hf_hub_download

            repo_id, filepath = _hf_parse_path(path)
            if not filepath:
                return "# Config path must point to a file: hf://owner/repo/path/config.yaml"
            local = hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type="dataset",
                revision=revision,
            )
            with open(local) as fh:
                return fh.read()
        else:
            with open(path) as fh:
                return fh.read()
    except Exception as exc:
        return f"# Failed to load config:\n# {exc}"


# ── Core check / run pipeline ──────────────────────────────────────────────────


def run_checks(
    dataset_path: str,
    config_path: str,
    dataset_revision: str | None = None,
    config_revision: str | None = None,
    key: str = "data/raw_data",
    file_index: int = 0,
    start_frame: int = 0,
    n_frames: int = 1,
    keep_keys: tuple = ("maxval",),
    stop_check=None,
    track_index: int = 0,
    status_lines: list[str] | None = None,
):
    """Validate and beamform frame(s) from a zea dataset; yields ``(html, image)`` pairs.

    Pass a list as *status_lines* to have the individual status lines collected into it
    (the latest step is ``status_lines[-1]``).
    """
    dataset_path = _normalize_path(dataset_path)
    config_path = _normalize_path(config_path)
    file_index = int(file_index)
    start_frame = int(start_frame)
    n_frames = max(1, int(n_frames))
    track_index = int(track_index)
    lines: list[str] = status_lines if status_lines is not None else []

    def _stopped():
        return stop_check is not None and stop_check()

    def _emit(line, image=None):
        lines.append(line)
        return "".join(lines), image

    def _replace_last(line, image=None):
        if lines:
            lines[-1] = line
        else:
            lines.append(line)
        return "".join(lines), image

    eff_config_rev = config_revision if config_revision is not None else dataset_revision
    config_hf_kwargs = {"revision": eff_config_rev} if eff_config_rev else {}

    # HF token check
    if _is_hf(dataset_path) or _is_hf(config_path):
        has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if not has_token:
            try:
                from huggingface_hub import get_token as _get_hf_token

                has_token = bool(_get_hf_token())
            except Exception:
                pass
        if not has_token:
            gr.Warning(
                "No HF token found. Set HF_TOKEN or run 'huggingface-cli login'. "
                "Private repos will fail and downloads may be rate-limited."
            )

    # 1. List dataset files (lazy — no download) and resolve selected file
    _src = "from HF" if _is_hf(dataset_path) else "from disk"
    yield _emit(_html_info(f"Opening dataset {_src}…"))
    try:
        ds = Dataset(
            dataset_path, lazy=True, revision=dataset_revision or None, _suggest_lazy=False
        )
        num_files = len(ds)
        if not num_files:
            yield _replace_last(_html_fail("Open dataset", "No HDF5 files found."))
            return
        if file_index >= num_files:
            yield _replace_last(
                _html_fail(
                    "File index out of range",
                    f"File index {file_index} >= {num_files} files.",
                )
            )
            return
        file_path = ds.file_paths[file_index]
        ds.close()
    except Exception as exc:
        yield _replace_last(_html_fail("Open dataset", exc))
        return
    yield _replace_last(_html_pass(f"Dataset opened · {num_files} file(s)"))
    if _stopped():
        return

    # 2. Load config (non-fatal — fall back to raw display on failure)
    config_params: dict = {}
    pipeline = None
    if not config_path:
        yield _emit(_html_warn("No config path set. Will display data without processing."))
    else:
        _src = "from HF" if _is_hf(config_path) else "from disk"
        yield _emit(_html_info(f"Loading config {_src}…"))
        config_loaded = False
        try:
            config = Config.from_path(config_path, **config_hf_kwargs)
            config_params = _get_config_parameters(config)
            config_loaded = True
        except Exception as exc:
            yield _replace_last(
                _html_warn(f"Config unavailable ({exc}) · will display data without processing.")
            )
        if config_loaded:
            yield _replace_last(_html_pass("Config loaded"))
            if _stopped():
                return

            # 3. Build pipeline
            yield _emit(_html_info(f"Building pipeline {_src}…"))
            try:
                pipeline = Pipeline.from_path(config_path, with_batch_dim=False, **config_hf_kwargs)
            except Exception as exc:
                yield _replace_last(
                    _html_warn(
                        f"Pipeline build failed ({exc}) · will display data without processing."
                    )
                )
            if pipeline is not None:
                if not _key_requires_pipeline(key):
                    # Key doesn't need beamforming — skip pipeline, use raw display
                    pipeline = None
                    yield _replace_last(
                        _html_warn("Pipeline ignored: this key does not need beamforming.")
                    )
                else:
                    yield _replace_last(_html_pass("Pipeline built"))
                if _stopped():
                    return

    # A pipeline-required key cannot fall back to raw display. Reject here so the
    # missing-config, unreadable-config and failed-build cases are all covered,
    # not only the failed-build case inside the config block above.
    if pipeline is None and _key_requires_pipeline(key):
        yield _emit(
            _html_fail(
                "Pipeline required",
                f"Key '{key}' contains raw RF data, so a valid pipeline config is needed. "
                "Provide a config with a 'pipeline:' section or select a different data key.",
            )
        )
        return

    # 4 – 5: Open file once — resolve key and load parameters
    _data_key: str | None = None
    parameters = None
    params: dict = {}
    end_frame = start_frame + n_frames
    actual_n = n_frames
    processed_frames: list[np.ndarray] = []

    hf_kwargs = {"revision": dataset_revision} if dataset_revision and _is_hf(file_path) else {}
    try:
        with File(file_path, **hf_kwargs) as f:
            # 4. Resolve data key and load parameters
            if pipeline is not None:
                n_tracks = f._n_tracks
                if n_tracks > 1:
                    track = f.tracks[track_index]
                    parameters = _run_quiet(track.load_parameters)
                    parameters.update(config_params)
                    bare = key.removeprefix("data/")
                    _data_key = f"tracks/track_{track_index}/data/{bare}"
                else:
                    parameters = _run_quiet(f.load_parameters)
                    parameters.update(config_params)
                    _data_key = f.format_key(key)
            else:
                _data_key = f.format_key(key)

            # Stem for output file naming; fps for GIF output
            _file_stem = f.stem
            _fps_params = parameters  # already loaded for pipeline path
            if _fps_params is None:
                try:
                    _fps_params = _run_quiet(f.load_parameters)
                except Exception:
                    pass
            fps = 20
            if _fps_params is not None:
                try:
                    fps = int(round(_fps_params.frames_per_second))
                except (ValueError, AttributeError):
                    pass

            total_frames = f[_data_key].shape[0]

            if start_frame >= total_frames:
                yield _emit(
                    _html_fail(
                        "Frame index out of range",
                        f"Start frame {start_frame} >= {total_frames} frames in file.",
                    )
                )
                return

            end_frame = min(start_frame + n_frames, total_frames)
            actual_n = end_frame - start_frame
            if actual_n < n_frames:
                yield _emit(
                    _html_warn(
                        f"Requested {n_frames} frames but only {actual_n} available "
                        f"(frames {start_frame}–{end_frame - 1})."
                    )
                )

            if pipeline is not None:
                # Show frame + transmit counts; transmits only exist for raw/aligned data.
                _tx = getattr(parameters, "selected_transmits", None)
                if _tx is not None:
                    yield _emit(
                        _html_pass(f"Data loaded · {total_frames} frame(s), {len(_tx)} transmit(s)")
                    )
                else:
                    yield _emit(_html_pass(f"Data loaded · {total_frames} frame(s)"))
                if _stopped():
                    return

                yield _emit(_html_pass("Parameters loaded"))
                if _stopped():
                    return

                # 5. Prepare pipeline parameters
                try:
                    params = _run_quiet(pipeline.prepare_parameters, parameters)
                except Exception as exc:
                    yield _emit(_html_fail("Prepare parameters", exc))
                    return
                if _stopped():
                    return

    except Exception as exc:
        yield _emit(_html_fail("Open file", exc))
        return

    # 6. Process frames — Dataloader provides prefetching and HDF5-level transmit
    # pre-filtering (axis_selections), matching the optimisations in zea process.
    _axis_sel = _axis_selections_from_params(parameters) if pipeline is not None else None
    _dl_revision = dataset_revision if _is_hf(str(file_path)) else None
    try:
        _dataloader = Dataloader(
            str(file_path),
            key=_data_key,
            batch_size=None,
            shuffle=False,
            offset_n_frames=start_frame,
            limit_n_frames=actual_n,
            n_frames=None,
            num_threads=4,
            sort_files=False,
            dtype="float32",
            axis_selections=_axis_sel,
            # Already validated by the File open in step 4 above.
            validate=False,
            revision=_dl_revision,
        )
    except Exception as exc:
        yield _emit(_html_fail("Open file", exc))
        return

    # try/finally so the loop's early returns and generator abandonment both close
    # the dataloader's file handles.
    bytes_at_start = network_bytes()
    try:
        for i, frame in enumerate(_dataloader):
            try:
                frame = np.asarray(frame)
                if pipeline is not None:
                    output = _run_quiet(pipeline, data=frame, **params)
                    processed = ops.convert_to_numpy(output["data"])
                    for k in keep_keys:
                        if k in output:
                            params[k] = output[k]
                else:
                    # Raw fallback: reduce to 2D
                    while frame.ndim > 2:
                        if frame.shape[0] == 1:
                            frame = frame[0]
                        elif frame.shape[-1] == 1:
                            frame = frame[..., 0]
                        elif frame.ndim == 3:
                            # Multi-channel last dim (e.g. segmentation one-hot)
                            frame = np.argmax(frame, axis=-1)
                        else:
                            frame = frame[0]
                    if frame.ndim < 2:
                        yield _emit(
                            _html_fail(
                                "Cannot display",
                                f"Data shape {frame.shape} after indexing; need at least 2D.",
                            )
                        )
                        return
                    processed = frame
            except Exception as exc:
                yield _emit(_html_fail(f"Process frame {start_frame + i}", exc))
                return

            processed_frames.append(processed)

            pbar = _html_progress(i + 1, actual_n, network_bytes() - bytes_at_start)
            if i == 0:
                yield _emit(pbar)
            else:
                yield _replace_last(pbar)

            if _stopped():
                return
    finally:
        _dataloader.close()

    # 7. Convert to image / GIF
    try:
        if pipeline is not None:
            dr = getattr(parameters, "dynamic_range", None)
            dynamic_range = tuple(dr) if dr is not None else (-60, 0)
            to_u8 = lambda arr: display.to_8bit(arr, dynamic_range, pillow=False)
        else:
            # Normalise each frame independently (min-max → uint8)
            def to_u8(arr):
                arr = np.asarray(arr, dtype=np.float32)
                lo, hi = float(arr.min()), float(arr.max())
                if hi > lo:
                    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
                return np.zeros(arr.shape, dtype=np.uint8)

        if actual_n == 1:
            u8 = to_u8(processed_frames[0])
            from PIL import Image as _PILImage

            result_image = _PILImage.fromarray(u8)
        else:
            frames_u8 = [to_u8(f) for f in processed_frames]
            video = np.stack(frames_u8, axis=0)
            tmp = tempfile.NamedTemporaryFile(prefix=f"{_file_stem}_", suffix=".gif", delete=False)
            io_lib.save_video(video, Path(tmp.name), fps=fps)
            result_image = tmp.name
    except Exception as exc:
        yield _emit(_html_fail("Convert to image", exc))
        return

    frame_label = (
        f"frame {start_frame}" if actual_n == 1 else f"frames {start_frame}–{end_frame - 1}"
    )
    done_html = (
        f'<hr style="margin:6px 0;border-color:#44403c">'
        f'<p style="margin:4px 0;color:{_YELLOW}"><b>&#10004; Processing done</b>'
        f' <span style="color:#6b7280">· file {file_index + 1}/{num_files}'
        f" &middot; {frame_label}</span></p>"
    )
    yield _replace_last(done_html, result_image)

    if not _is_hf(dataset_path):
        yield _emit(_html_warn("Local dataset path, not yet on Hugging Face."), result_image)
    if not _is_hf(config_path):
        yield _emit(_html_warn("Local config path, not yet on Hugging Face."), result_image)
    if _is_hf(dataset_path) and _is_hf(config_path):
        rp = _hf_parse_path(dataset_path)[0].lower()
        cp = _hf_parse_path(config_path)[0].lower()
        if rp != cp:
            yield _emit(
                _html_warn("Dataset and config are on different HF repositories."),
                result_image,
            )


# ── Gradio interface ───────────────────────────────────────────────────────────


def _config_badge_html(config: str, revision: str | None, applied: str, dirty: bool) -> str:
    """One-line summary of which config a run will use, shown above the tabs."""
    config = _normalize_path(config)
    name = html.escape(Path(config).name) if config else ""
    rev = f" @ {html.escape(revision)}" if revision and _is_hf(config) else ""
    if applied:
        source = f" (based on {name})" if name else ""
        text = f'<span class="zea-badge-accent">Config · edited in the editor</span>{source}'
    elif name:
        text = f"Config · <b>{name}</b>{rev}"
    else:
        text = "Config · none yet"
    if dirty:
        text += (
            ' <span class="zea-badge-accent">· unsaved edits</span> '
            "(press <b>Use edited config</b> to apply)"
        )
    return f'<div class="zea-config-badge">{text}</div>'


def build_interface() -> "gr.Blocks":
    """Build and return the Gradio Blocks interface."""

    logo = _logo_html(height=54)

    with gr.Blocks(title="zea visualizer") as demo:
        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(
            f'<div style="display:flex;align-items:flex-end;padding:8px 0 4px;'
            f'margin-bottom:6px">'
            f'<a href="https://github.com/tue-bmd/zea" target="_blank" rel="noopener" '
            f'title="zea on GitHub" style="flex-shrink:0;margin-right:10px">{logo}</a>'
            f'<div style="display:flex;align-items:center;'
            f'border-bottom:1px solid #44403c;flex:1;padding-bottom:5px">'
            f'<span style="font-size:1.35em;font-weight:700;color:{_YELLOW}">zea</span>'
            f'<span style="font-size:1.35em;font-weight:400;margin-left:6px;color:{_MUTED}">'
            f"dataset visualizer</span>"
            f"</div>"
            f"</div>"
        )

        # ── Hidden state ────────────────────────────────────────────────────
        config_rev_decoupled = gr.State(False)
        file_paths_state = gr.State([])
        # Last config path filled in automatically (dataset discovery or preset). A
        # config the user typed differs from it and is never overwritten.
        config_auto_state = gr.State("")
        track_labels_state = gr.State([])
        # Config YAML applied from the editor ("" = use the config path). Runs use this
        # snapshot, never the live editor text, so what is in use is always explicit.
        editor_applied_state = gr.State("")
        # True while the editor holds edits that have not been applied.
        editor_dirty_state = gr.State(False)
        # {file_path: n_frames} requested by the active preset for its pre-selected file.
        preset_frames_state = gr.State({})
        # Frame layout of the loaded file: {"file", "n_per_track", "track", "clip"}.
        frame_state = gr.State({})

        # ── Main row ────────────────────────────────────────────────────────
        with gr.Row():
            # Left: tabbed controls ─────────────────────────────────────────
            with gr.Column(scale=2, min_width=420):
                config_badge = gr.HTML(_config_badge_html("", None, "", False))
                with gr.Tabs(selected="data") as tabs:
                    with gr.Tab("① Data", id="data"):
                        preset_selector = gr.Dropdown(
                            label="Example presets (optional)",
                            choices=list(PRESETS.keys()),
                            value=None,
                            interactive=True,
                            info="Fills in the fields below. Skip it if you know your paths.",
                        )
                        gr.HTML('<div class="zea-divider">or enter your own paths</div>')

                        with gr.Row():
                            dataset_input = gr.Textbox(
                                label="Dataset path",
                                placeholder="hf://owner/repo[/subfolder] or /local/path",
                                max_lines=1,
                                scale=4,
                                elem_classes=["zea-path"],
                            )
                            dataset_rev_input = gr.Dropdown(
                                label="Revision",
                                choices=["main"],
                                value=None,
                                allow_custom_value=True,
                                interactive=False,
                                scale=1,
                                min_width=140,
                                elem_classes=["revision-dropdown"],
                            )
                        with gr.Row():
                            config_input = gr.Textbox(
                                label="Config path",
                                placeholder="Auto-detected from the dataset, or enter a .yaml path",
                                max_lines=1,
                                scale=4,
                                elem_classes=["zea-path"],
                            )
                            config_rev_input = gr.Dropdown(
                                label="Revision (auto)",
                                choices=["main"],
                                value=None,
                                allow_custom_value=True,
                                interactive=False,
                                scale=1,
                                min_width=140,
                                elem_classes=["revision-dropdown"],
                            )

                        hf_links = gr.HTML("")

                        # Status of slow lookups (dataset path, example, revision).
                        data_status = gr.HTML("")
                        next_btn = gr.Button("Next: select a file  →", interactive=False)

                    with gr.Tab("② File & run", id="file"):
                        file_selector = gr.Dropdown(
                            label="File",
                            choices=[],
                            value=None,
                            interactive=False,
                            filterable=True,
                            info=_NO_DATASET_INFO,
                        )

                        with gr.Row():
                            key_input = gr.Dropdown(
                                label="Data key",
                                choices=_DATA_KEYS,
                                value=None,
                                allow_custom_value=True,
                                interactive=False,
                                scale=2,
                            )
                            # Only shown for multi-track files.
                            track_selector = gr.Dropdown(
                                label="Track",
                                choices=[("Track 0", 0)],
                                value=0,
                                interactive=False,
                                visible=False,
                                scale=1,
                            )

                        frames_info = gr.HTML(_frames_info_html(None))
                        frame_mode = gr.Radio(
                            ["Single frame", "Clip (GIF)"],
                            value="Single frame",
                            show_label=False,
                            container=False,
                            visible=False,
                        )

                        with gr.Row():
                            start_frame_input = gr.Slider(
                                label="Frame",
                                visible=False,
                                minimum=0,
                                maximum=999,
                                value=0,
                                step=1,
                                interactive=False,
                                elem_classes=["frame-slider"],
                            )
                            n_frames_input = gr.Slider(
                                label="Frame count",
                                visible=False,
                                minimum=1,
                                maximum=999,
                                value=1,
                                step=1,
                                interactive=False,
                                elem_classes=["frame-slider"],
                            )

                        with gr.Row():
                            run_btn = gr.Button(
                                "Run",
                                variant="primary",
                                scale=3,
                                interactive=False,
                                elem_classes=["run-btn"],
                            )
                            stop_btn = gr.Button("Stop", variant="stop", scale=1, interactive=False)
                        # Latest step / progress of a run, right under the button.
                        run_status = gr.HTML("", elem_classes=["run-status"])

                    with gr.Tab("Config editor"):
                        config_editor = gr.Code(
                            label="Config YAML",
                            language="yaml",
                            lines=20,
                        )
                        with gr.Row():
                            apply_config_btn = gr.Button(
                                "Use edited config", variant="primary", interactive=False
                            )
                            revert_config_btn = gr.Button("Revert to file", interactive=False)

            # Right: metadata card + image + status ─────────────────────────
            with gr.Column(scale=3):
                meta_card = gr.HTML("")
                image_output = gr.Image(
                    label="Output",
                    type="filepath",
                    height=400,
                )
                status_output = gr.HTML(
                    label="Status",
                    elem_classes=["status-box"],
                )

        # ── Event wiring ────────────────────────────────────────────────────

        # Hugging Face links follow the paths and the selected revisions.
        _link_inputs = [dataset_input, dataset_rev_input, config_input, config_rev_input]
        for _component in _link_inputs:
            _bind_gradio_event(_component, "change", _hf_links_html, _link_inputs, [hf_links])

        # Revision toggle (fast, no network)
        def _rev_toggle(path):
            return gr.update(interactive=_is_hf(path))

        _bind_gradio_event(
            dataset_input, "change", _rev_toggle, [dataset_input], [dataset_rev_input]
        )
        _bind_gradio_event(config_input, "change", _rev_toggle, [config_input], [config_rev_input])

        _TRACK_RESET = gr.update(
            choices=[("Track 0", 0)], value=0, visible=False, interactive=False
        )

        # Dataset blur → fetch revisions + file list (no download)
        def _on_dataset_blur(path, config, config_auto):
            raw = path or ""
            path = _normalize_path(raw)
            path_upd = gr.update(value=path) if path != raw.strip() else gr.update()
            config = _normalize_path(config)
            _disable_run = gr.update(interactive=False)
            if not path:
                return (
                    path_upd,
                    gr.update(),
                    config_auto,
                    gr.update(),
                    gr.update(),
                    [],
                    _TRACK_RESET,
                    [],
                    "",
                    _disable_run,
                    gr.update(choices=_DATA_KEYS),
                )
            errors: list[Exception] = []
            names, paths = _list_dataset_files(path, _errors=errors)
            auto_val = paths[0] if len(paths) == 1 else None
            file_update = gr.update(
                choices=list(zip(names, paths)), value=auto_val, interactive=bool(paths)
            )
            _reset_key = gr.update(choices=_DATA_KEYS, value=None, interactive=False)

            if not names:
                if errors:
                    short = _enrich_error(errors[0]).split("\n\n")[0]
                    gr.Warning(f"Cannot open dataset: {short}")
                    meta_html = _html_fail("Cannot open dataset", errors[0])
                else:
                    gr.Warning("No HDF5 files found at this path.")
                    meta_html = _html_warn("No HDF5 files found at this path.")
            else:
                meta_html = ""

            # Auto-fill the config only when the user has not typed their own.
            config_upd, new_auto = gr.update(), config_auto
            if not errors and (not config or config == config_auto):
                found = _find_config(path) or ""
                config_upd, new_auto = gr.update(value=found), found
                if not found and paths:
                    meta_html += _html_warn(
                        "No config.yaml or pipeline.yaml found next to the dataset. "
                        "Set a config path to process raw data."
                    )

            if not _is_hf(path) or errors:
                # Local path, or repo inaccessible — no revisions to fetch.
                rev_upd = gr.update(interactive=False, choices=["main"], value=None)
            else:
                revisions = _fetch_hf_revisions(path)
                default = "main" if "main" in revisions else revisions[0]
                rev_upd = gr.update(interactive=True, choices=revisions, value=default)
            return (
                path_upd,
                config_upd,
                new_auto,
                rev_upd,
                file_update,
                paths,
                _TRACK_RESET,
                [],
                meta_html,
                _disable_run,
                _reset_key,
            )

        _bind_gradio_event(
            dataset_input, "blur", _busy_lookup, [dataset_input], [data_status]
        ).then(
            _on_dataset_blur,
            inputs=[dataset_input, config_input, config_auto_state],
            outputs=[
                dataset_input,
                config_input,
                config_auto_state,
                dataset_rev_input,
                file_selector,
                file_paths_state,
                track_selector,
                track_labels_state,
                meta_card,
                run_btn,
                key_input,
            ],
            show_progress="hidden",
        ).then(lambda: "", None, [data_status], show_progress="hidden")

        # Config blur → validate path + fetch revisions
        def _on_config_blur(path):
            raw = path or ""
            path = _normalize_path(raw)
            path_upd = gr.update(value=path) if path != raw.strip() else gr.update()
            return path_upd, _check_config_path(path)

        def _check_config_path(path):
            if not path:
                return gr.update(interactive=False, choices=["main"], value=None)
            if not _is_hf(path):
                p = Path(path)
                if not p.exists():
                    gr.Warning(f"Config file not found: {path}")
                elif p.suffix.lower() not in (".yaml", ".yml"):
                    suffix = p.suffix or "(no extension)"
                    gr.Warning(f"Config path should be a .yaml file, got: {suffix}")
                return gr.update(interactive=False, choices=["main"], value=None)
            # HF path — check repo + specific file, then fetch revisions
            try:
                from huggingface_hub import file_exists, list_repo_refs

                repo_id, filepath = _hf_parse_path(path)
                if "/" not in repo_id or not repo_id.split("/")[1]:
                    gr.Warning("Invalid Hugging Face path: expected hf://owner/repo-name/…")
                    return gr.update(interactive=False, choices=["main"], value=None)
                if not filepath:
                    gr.Warning("Config path must point to a .yaml file inside the repo.")
                refs = list_repo_refs(repo_id, repo_type="dataset")
                branches = [b.name for b in refs.branches]
                tags = [t.name for t in refs.tags]
                revisions = branches + tags or ["main"]
                default = "main" if "main" in revisions else revisions[0]
                if filepath and not file_exists(
                    repo_id, filepath, repo_type="dataset", revision=default
                ):
                    gr.Warning(f"Config file not found in repo: {filepath}")
                return gr.update(interactive=True, choices=revisions, value=default)
            except Exception as exc:
                short = _enrich_error(exc).split("\n\n")[0]
                gr.Warning(f"Cannot access config: {short}")
                return gr.update(interactive=False, choices=["main"], value=None)

        _bind_gradio_event(
            config_input,
            "blur",
            _on_config_blur,
            [config_input],
            [config_input, config_rev_input],
        )

        # Dataset revision change → refresh file list; auto-reload selected file at new revision
        def _on_dataset_rev_change_gen(rev, path, decoupled, current_file, key):
            cfg_upd = gr.update() if decoupled else gr.update(value=rev)
            path = (path or "").strip()
            _reset_key = gr.update(choices=_DATA_KEYS, value=None, interactive=False)
            _clear = (
                cfg_upd,
                gr.update(),
                [],
                "",
                _TRACK_RESET,
                [],
                gr.update(interactive=False),
                _reset_key,
                gr.update(value=0, interactive=False),
                gr.update(value=1, interactive=False),
                {},
            )

            if not path:
                yield _clear
                return

            errors: list[Exception] = []
            names, fpaths = _list_dataset_files(path, rev or None, _errors=errors)
            new_val = current_file if (current_file and current_file in fpaths) else None
            file_upd = gr.update(
                choices=list(zip(names, fpaths)), value=new_val, interactive=bool(fpaths)
            )

            if not new_val:
                if not fpaths and errors:
                    meta_html = _html_fail("Cannot open dataset", errors[0])
                elif not fpaths:
                    meta_html = _html_warn("No HDF5 files found at this path.")
                else:
                    meta_html = ""
                yield (
                    cfg_upd,
                    file_upd,
                    fpaths,
                    meta_html,
                    _TRACK_RESET,
                    [],
                    gr.update(interactive=False),
                    _reset_key,
                    gr.update(value=0, interactive=False),
                    gr.update(value=1, interactive=False),
                    {},
                )
                return

            # Same file still exists — show loading then reload at new revision
            yield (
                cfg_upd,
                file_upd,
                fpaths,
                _loading_meta_html(),
                _TRACK_RESET,
                [],
                gr.update(interactive=False),
                _reset_key,
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(),
            )

            sf, nf, meta, trk, tlbls, run_upd, key_upd, fstate = _file_load_updates(
                new_val, rev or None, key
            )
            yield cfg_upd, gr.update(), fpaths, meta, trk, tlbls, run_upd, key_upd, sf, nf, fstate

        _bind_gradio_event(
            dataset_rev_input, "input", _busy_revision, [dataset_rev_input], [data_status]
        ).then(
            _on_dataset_rev_change_gen,
            [dataset_rev_input, dataset_input, config_rev_decoupled, file_selector, key_input],
            [
                config_rev_input,
                file_selector,
                file_paths_state,
                meta_card,
                track_selector,
                track_labels_state,
                run_btn,
                key_input,
                start_frame_input,
                n_frames_input,
                frame_state,
            ],
            show_progress="hidden",
        ).then(lambda: "", None, [data_status], show_progress="hidden")

        # User manually picks a config revision → decouple
        def _on_config_rev_input():
            return True, gr.update(label="Revision")

        _bind_gradio_event(
            config_rev_input,
            "input",
            _on_config_rev_input,
            [],
            [config_rev_decoupled, config_rev_input],
        )

        # Preset → fill all fields + reset sync state (no file auto-load)
        def _apply_preset(name):
            if name not in PRESETS:
                return (gr.update(),) * 15
            p = PRESETS[name]
            ds = p.get("dataset", "")
            cfg = p.get("config", "")
            key = p.get("key", "data/raw_data")
            ds_revs = _fetch_hf_revisions(ds) if _is_hf(ds) else ["main"]
            cfg_revs = _fetch_hf_revisions(cfg) if _is_hf(cfg) else ["main"]
            ds_def = "main" if "main" in ds_revs else (ds_revs[0] if ds_revs else "main")
            cfg_def = "main" if "main" in cfg_revs else (cfg_revs[0] if cfg_revs else "main")
            rev = p.get("revision")
            if rev:
                # PR refs (refs/pr/N) are not among a repo's branches and tags.
                ds_revs = ds_revs if rev in ds_revs else [*ds_revs, rev]
                cfg_revs = cfg_revs if rev in cfg_revs else [*cfg_revs, rev]
                ds_def = cfg_def = rev
            names, paths = _list_dataset_files(ds, ds_def)
            # Pre-select the preset's file (this triggers the file load); otherwise let
            # the user pick, or auto-pick when there is only one.
            file_val = p.get("file") if p.get("file") in paths else None
            if file_val is None and len(paths) == 1:
                file_val = paths[0]
            frames = {file_val: p["n_frames"]} if file_val and "n_frames" in p else {}
            return (
                gr.update(value=ds),
                gr.update(value=cfg),
                gr.update(interactive=_is_hf(ds), choices=ds_revs, value=ds_def),
                gr.update(
                    interactive=_is_hf(cfg),
                    choices=cfg_revs,
                    value=cfg_def,
                    label="Revision (auto)",
                ),
                False,  # config_rev_decoupled → reset
                gr.update(
                    choices=_DATA_KEYS, value=key, interactive=False
                ),  # key_input — pre-filled from preset but locked until file is loaded
                gr.update(choices=list(zip(names, paths)), value=file_val, interactive=bool(paths)),
                paths,
                _TRACK_RESET,  # track_selector
                [],  # track_labels_state
                gr.update(value=None),  # image_output clear
                "",  # meta_card
                gr.update(interactive=False),  # run_btn — re-enabled after file is picked
                cfg,  # config_auto_state — preset config may be replaced by discovery
                frames,  # preset_frames_state
            )

        _bind_gradio_event(
            preset_selector, "change", _busy_example, [preset_selector], [data_status]
        ).then(
            _apply_preset,
            [preset_selector],
            [
                dataset_input,
                config_input,
                dataset_rev_input,
                config_rev_input,
                config_rev_decoupled,
                key_input,
                file_selector,
                file_paths_state,
                track_selector,
                track_labels_state,
                image_output,
                meta_card,
                run_btn,
                config_auto_state,
                preset_frames_state,
            ],
            show_progress="hidden",
        ).then(lambda: "", None, [data_status], show_progress="hidden")

        # File selected → load file (may download HF), show metadata + update sliders
        # 8 primary outputs + 7 lock outputs (stop_btn, file_selector, preset_selector,
        # dataset_input, dataset_rev_input, config_input, config_rev_input)
        _NO_FILE = (
            gr.update(interactive=False),  # start_frame_input
            gr.update(interactive=False),  # n_frames_input
            "",  # meta_card
            _TRACK_RESET,  # track_selector
            [],  # track_labels_state
            gr.update(interactive=False),  # run_btn
            gr.update(choices=_DATA_KEYS, value=None, interactive=False),  # key_input
            gr.update(value=None),  # image_output
            gr.update(),  # stop_btn — no change
            gr.update(),  # file_selector — no change
            gr.update(),  # preset_selector — no change
            gr.update(),  # dataset_input — no change
            gr.update(),  # dataset_rev_input — no change
            gr.update(),  # config_input — no change
            gr.update(),  # config_rev_input — no change
            {},  # frame_state
        )

        def _on_file_select_gen(
            selected_name, file_paths, key, ds_revision, config_path, preset_frames
        ):
            # selected_name is the full path (dropdown value), not the basename.
            if not selected_name or not file_paths or selected_name not in file_paths:
                yield _NO_FILE
                return
            fpath = selected_name

            # For HF paths, look up the size for context in the progress card.
            # list_repo_tree is cached by HF Hub, so this is usually a fast local hit.
            size_bytes: int | None = None
            if _is_hf(fpath):
                try:
                    from zea.internal.preset_utils import _hf_list_h5_files

                    hf_size_kwargs = {"revision": ds_revision} if ds_revision else {}
                    _hits = _hf_list_h5_files(fpath, **hf_size_kwargs)
                    size_bytes = _hits[0][1] if _hits else None
                except Exception:
                    pass

            def _progress(streamed: int, lock: bool) -> tuple:
                upd = gr.update(interactive=False) if lock else gr.update()
                return (
                    upd,  # start_frame_input
                    upd,  # n_frames_input
                    _loading_meta_html(streamed, size_bytes),  # meta_card
                    _TRACK_RESET if lock else gr.update(),  # track_selector
                    [] if lock else gr.update(),  # track_labels_state
                    upd,  # run_btn
                    upd,  # key_input
                    gr.update(value=None) if lock else gr.update(),  # image_output
                    upd,  # stop_btn — reading metadata cannot be interrupted
                    upd,  # file_selector — prevent switching files
                    upd,  # preset_selector
                    upd,  # dataset_input
                    upd,  # dataset_rev_input
                    upd,  # config_input
                    upd,  # config_rev_input
                    gr.update(),  # frame_state
                )

            # Step 1: lock the inputs while the metadata is read.
            yield _progress(0, lock=True)

            # Step 2: read the metadata in a worker so the streamed byte count can be
            # shown live (the amount h5py needs to read is not known in advance).
            n_frames = (preset_frames or {}).get(fpath, 1)
            result: dict = {}

            def _work():
                result["out"] = _file_load_updates(fpath, ds_revision or None, key, n_frames)

            bytes_at_start = network_bytes()
            worker = threading.Thread(target=_work, daemon=True)
            worker.start()
            while worker.is_alive():
                worker.join(0.3)
                if worker.is_alive():
                    yield _progress(network_bytes() - bytes_at_start, lock=False)

            sf, nf, meta, trk, tlbls, run_upd, key_upd, fstate = result["out"]
            yield (
                sf,
                nf,
                meta,
                trk,
                tlbls,
                run_upd,
                key_upd,
                gr.update(),  # image_output — no change
                gr.update(interactive=False),  # stop_btn — disable again
                gr.update(interactive=True),  # file_selector — re-enable
                gr.update(interactive=True),  # preset_selector
                gr.update(interactive=True),  # dataset_input
                gr.update(interactive=_is_hf(fpath)),  # dataset_rev_input
                gr.update(interactive=True),  # config_input
                gr.update(interactive=_is_hf(config_path or "")),  # config_rev_input
                fstate,
            )

        file_select_event = _bind_gradio_event(
            file_selector,
            "change",
            _on_file_select_gen,
            show_progress="hidden",
            inputs=[
                file_selector,
                file_paths_state,
                key_input,
                dataset_rev_input,
                config_input,
                preset_frames_state,
            ],
            outputs=[
                start_frame_input,
                n_frames_input,
                meta_card,
                track_selector,
                track_labels_state,
                run_btn,
                key_input,
                image_output,
                stop_btn,
                file_selector,
                preset_selector,
                dataset_input,
                dataset_rev_input,
                config_input,
                config_rev_input,
                frame_state,
            ],
        )

        # Tab navigation is always the user's choice: once the dataset resolves, the
        # Next button turns yellow instead of switching tabs automatically.
        _bind_gradio_event(next_btn, "click", lambda: gr.Tabs(selected="file"), [], [tabs])
        _bind_gradio_event(
            file_paths_state,
            "change",
            lambda paths: (
                gr.update(interactive=bool(paths), variant="primary" if paths else "secondary"),
                gr.update(info=_FILE_INFO if paths else _NO_DATASET_INFO),
            ),
            [file_paths_state],
            [next_btn, file_selector],
        )

        # A new file makes the previous run's summary stale.
        _bind_gradio_event(file_selector, "change", lambda: "", [], [run_status])

        # Key chosen → enable run button (file is already loaded at this point)
        _bind_gradio_event(
            key_input,
            "change",
            lambda key, fname: gr.update(interactive=bool(key and fname)),
            inputs=[key_input, file_selector],
            outputs=[run_btn],
        )

        # Track changed → update frame sliders for that track's n_frames
        def _on_track_change(track_id, fstate):
            # track_id is the numeric index emitted by the dropdown (None when unset).
            n_per_track = (fstate or {}).get("n_per_track") or []
            if track_id is None or int(track_id) >= len(n_per_track):
                return gr.update(), gr.update(), gr.update()
            sf, nf = _frame_sliders(n_per_track[int(track_id)])
            return sf, nf, {**fstate, "track": int(track_id), "clip": False}

        _bind_gradio_event(
            track_selector,
            "change",
            _on_track_change,
            [track_selector, frame_state],
            [start_frame_input, n_frames_input, frame_state],
        )

        # Frame controls: one slider for a single frame, start + count for a clip.
        # Hidden entirely when there is nothing to choose.
        def _frame_mode_updates(mode, fstate):
            fstate = fstate or {}
            n_per_track = fstate.get("n_per_track") or []
            track = fstate.get("track", 0)
            n = n_per_track[track] if track < len(n_per_track) else None
            choosable = n is not None and n > 1
            clip = choosable and mode == "Clip (GIF)"
            return (
                _frames_info_html(n, clip),
                gr.update(label="Start frame" if clip else "Frame", visible=choosable),
                gr.update(visible=clip),
            )

        def _on_frame_state(fstate):
            mode = "Clip (GIF)" if (fstate or {}).get("clip") else "Single frame"
            n_per_track = (fstate or {}).get("n_per_track") or []
            track = (fstate or {}).get("track", 0)
            choosable = track < len(n_per_track) and n_per_track[track] > 1
            return (
                gr.update(value=mode, visible=choosable),
                *_frame_mode_updates(mode, fstate),
            )

        _bind_gradio_event(
            frame_state,
            "change",
            _on_frame_state,
            [frame_state],
            [frame_mode, frames_info, start_frame_input, n_frames_input],
        )
        _bind_gradio_event(
            frame_mode,
            "input",
            _frame_mode_updates,
            [frame_mode, frame_state],
            [frames_info, start_frame_input, n_frames_input],
        )

        # ── Config editor ───────────────────────────────────────────────────
        # The editor mirrors the config path and reloads whenever it changes; edits
        # only take effect after "Use edited config", and "Revert to file" undoes that.
        def _reload_editor(path, revision, applied):
            path = _normalize_path(path)
            if path and not path.lower().endswith((".yaml", ".yml")):
                # Still being typed; keep the editor as is.
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            if applied:
                gr.Info("Config path changed: the edited config was replaced by the file.")
            text = _load_config_text(path, revision) if path else ""
            return (
                text,
                "",  # editor_applied_state
                False,  # editor_dirty_state
                gr.update(interactive=False),  # apply
                gr.update(interactive=False),  # revert
            )

        _editor_outputs = [
            config_editor,
            editor_applied_state,
            editor_dirty_state,
            apply_config_btn,
            revert_config_btn,
        ]
        for _component in (config_input, config_rev_input):
            _bind_gradio_event(
                _component,
                "change",
                _reload_editor,
                [config_input, config_rev_input, editor_applied_state],
                _editor_outputs,
            )

        def _on_editor_input():
            return True, gr.update(interactive=True), gr.update(interactive=True)

        _bind_gradio_event(
            config_editor,
            "input",
            _on_editor_input,
            [],
            [editor_dirty_state, apply_config_btn, revert_config_btn],
        )

        def _apply_editor(text):
            text = text or ""
            try:
                import yaml

                parsed = yaml.safe_load(text)
                if not isinstance(parsed, dict):
                    raise ValueError("expected a YAML mapping")
            except Exception as exc:
                raise gr.Error(f"Config is not valid YAML: {exc}")
            gr.Info("Using the edited config for the next run.")
            return text, False, gr.update(interactive=False), gr.update(interactive=True)

        _bind_gradio_event(
            apply_config_btn,
            "click",
            _apply_editor,
            [config_editor],
            [editor_applied_state, editor_dirty_state, apply_config_btn, revert_config_btn],
        )

        def _revert_editor(path, revision):
            path = _normalize_path(path)
            text = _load_config_text(path, revision) if path else ""
            return text, "", False, gr.update(interactive=False), gr.update(interactive=False)

        _bind_gradio_event(
            revert_config_btn,
            "click",
            _revert_editor,
            [config_input, config_rev_input],
            _editor_outputs,
        )

        # The badge above the tabs always says which config a run will use.
        _badge_inputs = [config_input, config_rev_input, editor_applied_state, editor_dirty_state]
        for _component in _badge_inputs:
            _bind_gradio_event(
                _component, "change", _config_badge_html, _badge_inputs, [config_badge]
            )

        # Run generator
        def _on_run(
            dataset,
            config,
            ds_rev,
            cfg_rev,
            key,
            file_name,
            file_paths,
            track_name,
            track_labels,
            start_f,
            n_f,
            applied_yaml,
            editor_dirty,
            mode,
        ):
            _stop_event.clear()
            if mode != "Clip (GIF)":
                n_f = 1
            dataset = _normalize_path(dataset)
            config = _normalize_path(config)
            if not dataset:
                raise gr.Warning("Please enter a dataset path.")
            if not file_name:
                raise gr.Warning("Please select a file from the dropdown first.")
            if not key:
                raise gr.Warning("Please select a data key from the dropdown first.")

            config_resolved = config or ""
            tmp_cfg = None
            if editor_dirty:
                gr.Info("The config editor has unapplied edits; this run ignores them.")
            if applied_yaml and applied_yaml.strip():
                tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
                tmp_cfg.write(applied_yaml)
                tmp_cfg.close()
                config_resolved = tmp_cfg.name
                cfg_rev = None

            # Resolve file index from selected path (dropdown value is the full path)
            if not file_paths or file_name not in file_paths:
                raise gr.Warning("Selected file is no longer available. Please reselect a file.")
            file_index = file_paths.index(file_name)

            # Resolve track index — track_name is the numeric ID emitted by the dropdown.
            track_index = 0
            if track_name is not None and track_labels:
                track_index = int(track_name)

            # Shared restore tuple for run_event extra outputs (12 components after
            # status_output and image_output). Intermediate yields use no-ops; the
            # first/last yields use explicit enable/disable values.
            _noop_extras = (gr.update(),) * 12
            _lock_extras = (
                gr.update(interactive=True),  # stop_btn — enable
                gr.update(interactive=False),  # run_btn
                gr.update(interactive=False),  # file_selector
                gr.update(interactive=False),  # preset_selector
                gr.update(interactive=False),  # dataset_input
                gr.update(interactive=False),  # dataset_rev_input
                gr.update(interactive=False),  # config_input
                gr.update(interactive=False),  # config_rev_input
                gr.update(interactive=False),  # key_input
                gr.update(interactive=False),  # start_frame_input
                gr.update(interactive=False),  # n_frames_input
                gr.update(interactive=False),  # track_selector
            )
            _unlock_extras = (
                gr.update(interactive=False),  # stop_btn — disable
                gr.update(interactive=True),  # run_btn
                gr.update(interactive=True),  # file_selector
                gr.update(interactive=True),  # preset_selector
                gr.update(interactive=True),  # dataset_input
                gr.update(interactive=_is_hf(dataset)),  # dataset_rev_input
                gr.update(interactive=True),  # config_input
                gr.update(interactive=_is_hf(config)),  # config_rev_input
                gr.update(interactive=True),  # key_input
                gr.update(interactive=True),  # start_frame_input
                gr.update(interactive=True),  # n_frames_input
                gr.update(),  # track_selector — keep
            )

            # Lock all inputs before starting; enable stop button.
            yield gr.update(), gr.update(), *_lock_extras, _html_info("Starting…")

            status_lines: list[str] = []

            def _latest() -> str:
                return status_lines[-1] if status_lines else ""

            def _log() -> str:
                # The progress bar is shown under the Run button only, not in the log too.
                return "".join(ln for ln in status_lines if "zea-progress" not in ln)

            try:
                for html, img in run_checks(
                    dataset,
                    config_resolved,
                    ds_rev if ds_rev else None,
                    cfg_rev if cfg_rev else None,
                    (key or "data/raw_data").strip() or "data/raw_data",
                    file_index,
                    int(start_f or 0),
                    int(n_f or 1),
                    stop_check=_stop_event.is_set,
                    track_index=track_index,
                    status_lines=status_lines,
                ):
                    if img is None:
                        yield _log(), None, *_noop_extras, _latest()
                    elif isinstance(img, str):
                        yield _log(), img, *_noop_extras, gr.update()
                    else:
                        tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        img.save(tmp_png.name)
                        yield _log(), tmp_png.name, *_noop_extras, gr.update()
                # Normal completion: summarise next to the button and restore the UI.
                done = next((ln for ln in status_lines if "Processing done" in ln), None)
                if done:
                    summary = _html_pass("Done")
                elif _stop_event.is_set():
                    summary = _html_warn("Stopped")
                else:
                    summary = _latest() + _html_info("See the status log for details.")
                yield gr.update(), gr.update(), *_unlock_extras, summary
            except Exception as exc:
                import traceback as _tb

                yield (
                    _html_fail("Unexpected error", exc)
                    + f'<pre style="font-size:0.75em;color:#6b7280;white-space:pre-wrap">'
                    f"{_tb.format_exc()}</pre>",
                    None,
                    *_noop_extras,
                    _html_fail("Unexpected error", exc),
                )
                # Restore UI after error.
                yield gr.update(), gr.update(), *_unlock_extras, gr.update()
            finally:
                # Cleanup only — no yield here; yielding after GeneratorExit raises RuntimeError.
                if tmp_cfg is not None:
                    try:
                        os.unlink(tmp_cfg.name)
                    except OSError:
                        pass

        run_event = _bind_gradio_event(
            run_btn,
            "click",
            _on_run,
            show_progress="hidden",
            inputs=[
                dataset_input,
                config_input,
                dataset_rev_input,
                config_rev_input,
                key_input,
                file_selector,
                file_paths_state,
                track_selector,
                track_labels_state,
                start_frame_input,
                n_frames_input,
                editor_applied_state,
                editor_dirty_state,
                frame_mode,
            ],
            outputs=[
                status_output,
                image_output,
                stop_btn,
                run_btn,
                file_selector,
                preset_selector,
                dataset_input,
                dataset_rev_input,
                config_input,
                config_rev_input,
                key_input,
                start_frame_input,
                n_frames_input,
                track_selector,
                run_status,
            ],
        )

        def _on_stop(current_key):
            _stop_event.set()
            # Explicitly restore the UI since cancelled generators' finally yields
            # may not reach the client.
            return (
                gr.update(interactive=False),  # stop_btn
                gr.update(interactive=bool(current_key)),  # run_btn
                gr.update(interactive=True),  # file_selector
                gr.update(interactive=True),  # preset_selector
                gr.update(interactive=True),  # dataset_input
                gr.update(),  # dataset_rev_input — keep
                gr.update(interactive=True),  # config_input
                gr.update(),  # config_rev_input — keep
                gr.update(interactive=bool(current_key)),  # key_input
                gr.update(interactive=True),  # start_frame_input
                gr.update(interactive=True),  # n_frames_input
                gr.update(),  # track_selector: keep (a file is still loaded)
                gr.update(),  # meta_card
                _html_warn("Stopped"),  # run_status
            )

        # Stop cancels both run and file-loading events, and restores UI directly.
        _bind_gradio_event(
            stop_btn,
            "click",
            _on_stop,
            inputs=[key_input],
            outputs=[
                stop_btn,
                run_btn,
                file_selector,
                preset_selector,
                dataset_input,
                dataset_rev_input,
                config_input,
                config_rev_input,
                key_input,
                start_frame_input,
                n_frames_input,
                track_selector,
                meta_card,
                run_status,
            ],
            cancels=[run_event, file_select_event],
        )

        _bind_gradio_event(status_output, "change", fn=None, js=_SCROLL_JS, show_progress="hidden")
        demo.load(
            _load_config_text,
            inputs=[config_input, config_rev_input],
            outputs=[config_editor],
            show_progress="hidden",
        )

    return demo


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for ``python -m zea.data.app``, equivalent to ``zea app``."""
    args = tyro.cli(AppArgs)
    init_device()
    args.run()


if __name__ == "__main__":
    main()
