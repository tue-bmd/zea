"""Gradio visualiser for zea datasets.

Usage:
    python -m zea.data.app
    python -m zea.data.app --share
    python -m zea.data.app --server-port 7861
"""

import argparse
import base64
import contextlib
import html
import io
import os
import tempfile
import threading
import warnings
from pathlib import Path

import numpy as np
from keras import ops

from zea import display, io_lib
from zea.config import Config
from zea.data.dataloader import Dataloader
from zea.data.datasets import Dataset
from zea.data.file import File
from zea.data.process import (
    _axis_selections_from_params,
    _get_config_parameters,
    _key_requires_pipeline,
)
from zea.internal.device import init_device
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

_YELLOW = "#f5c518"
_PURPLE = "#9333ea"

# ── Data key choices ──────────────────────────────────────────────────────────

_DATA_KEYS = [
    "data/raw_data",
    "data/aligned_data/values",
    "data/beamformed_data/values",
    "data/envelope_data/values",
    "data/image/values",
    "data/segmentation/values",
    "data/sos_map/values",
]

# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "PICMUS — experiment contrast speckle RF": {
        "dataset": (
            "hf://zeahub/picmus/database/experiments/contrast_speckle/"
            "contrast_speckle_expe_dataset_rf"
        ),
        "config": "hf://zeahub/picmus/config_rf.yaml",
        "key": "data/raw_data",
    },
    "PICMUS — experiment resolution distortion IQ": {
        "dataset": (
            "hf://zeahub/picmus/database/experiments/resolution_distorsion/"
            "resolution_distorsion_expe_dataset_iq"
        ),
        "config": "hf://zeahub/picmus/config_iq.yaml",
        "key": "data/raw_data",
    },
    "zea cardiac 2026": {
        "dataset": "hf://zeahub/zea-cardiac-2026",
        "config": "hf://zeahub/zea-cardiac-2026/config.yaml",
        "key": "data/raw_data",
    },
    "zea carotid 2023": {
        "dataset": "hf://zeahub/zea-carotid-2023",
        "config": "hf://zeahub/zea-carotid-2023/config.yaml",
        "key": "data/raw_data",
    },
    "CAMUS — cardiac echo (sample)": {
        "dataset": "hf://zeahub/camus-sample",
        "config": "hf://zeahub/configs/config_camus.yaml",
        "key": "data/image/values",
    },
}

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
footer { display: none !important; }
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
    return str(path).strip().startswith("hf://")


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
    return f'<p style="margin:2px 0;color:{_YELLOW}">&#9888; {html.escape(msg)}</p>'


def _html_info(msg: str) -> str:
    return f'<p style="margin:2px 0;color:{_YELLOW}">&#8250; {html.escape(msg)}</p>'


def _html_progress(current: int, total: int) -> str:
    pct = int(current / total * 100)
    return (
        f'<div style="margin:4px 0">'
        f'<span style="color:{_YELLOW};font-size:0.9em">Processing frame {current}/{total}</span>'
        f'<div style="background:#374151;border-radius:3px;height:5px;margin-top:3px">'
        f'<div style="background:{_PURPLE};border-radius:3px;height:5px;width:{pct}%"></div>'
        f"</div></div>"
    )


# ── HF / file listing ─────────────────────────────────────────────────────────


def _fetch_hf_revisions(path: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_refs

        parts = path.removeprefix("hf://").strip("/").split("/")
        if len(parts) < 2 or not parts[1]:
            return ["main"]
        repo_id = "/".join(parts[:2])
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
    path = (path or "").strip()
    if not path:
        return [], []
    try:
        ds = Dataset(path, lazy=True, revision=revision, _suggest_lazy=False)
        file_paths = sorted(ds.file_paths)
        ds.close()
        names = [Path(p).name for p in file_paths]
        return names, file_paths
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

            # Scan parameters (lightweight — only small arrays)
            try:
                sp = f.get_scan_parameters()
                if "sampling_frequency" in sp:
                    info["fs_hz"] = float(np.asarray(sp["sampling_frequency"]).flat[0])
                if "center_frequency" in sp:
                    info["fc_hz"] = float(np.asarray(sp["center_frequency"]).flat[0])
                if "sound_speed" in sp:
                    info["sound_speed"] = float(np.asarray(sp["sound_speed"]).flat[0])
                if "t0_delays" in sp:
                    d = sp["t0_delays"]
                    if hasattr(d, "shape") and len(d.shape) >= 2:
                        info["n_tx"] = int(d.shape[0])
                        info["n_el"] = int(d.shape[1])
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

                    def _collect(name, obj):
                        if not hasattr(obj, "shape"):
                            return  # skip groups
                        parts = name.split("/")
                        # Accept: bare name (data/raw_data) or <map>/values
                        if len(parts) == 1 or (len(parts) == 2 and parts[1] == "values"):
                            available.append("data/" + name)

                    data_grp.visititems(_collect)
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
    except Exception:
        pass

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
        f'<div style="border-left:3px solid {_PURPLE};border-radius:4px;'
        f"background:rgba(147,51,234,0.07);padding:6px 10px;margin-bottom:4px;"
        f'font-size:0.83em">' + "".join(sections) + "</div>"
    )


def _file_load_updates(fpath: str, revision: str | None, key: str) -> tuple:
    """Download (if HF) and read a file; return the 7 gr.update() values for file-select outputs.

    Returns: (start_frame_upd, n_frames_upd, meta_html, track_upd, track_labels,
               run_btn_upd, key_input_upd)
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

    meta_html = _build_meta_card_html(info)

    if n > 1:
        sf_upd = gr.update(maximum=n - 1, value=0, interactive=True)
        nf_upd = gr.update(maximum=n, value=1, interactive=True)
    elif n == 1:
        sf_upd = gr.update(value=0, interactive=False)
        nf_upd = gr.update(value=1, interactive=False)
    else:
        sf_upd = gr.update(value=0, interactive=False)
        nf_upd = gr.update(value=1, interactive=False)

    if n_tracks > 1 and track_labels:
        # Use numeric indices as values so duplicate labels don't break selection.
        choices = [(label, i) for i, label in enumerate(track_labels)]
        track_upd = gr.update(choices=choices, value=0, visible=True, interactive=True)
    else:
        track_upd = gr.update(choices=[("track 0", 0)], value=0, visible=True, interactive=False)

    return (
        sf_upd,
        nf_upd,
        meta_html,
        track_upd,
        track_labels,
        gr.update(interactive=new_key is not None),  # run_btn: only if key auto-resolved
        gr.update(choices=available_keys, value=new_key, interactive=True),
    )


def _loading_meta_html(size_bytes: int | None = None) -> str:
    size_str = f" · {size_bytes / 1e9:.2f} GB" if size_bytes else ""
    return (
        f'<p style="margin:2px 0;color:{_YELLOW}">&#8987;&nbsp;'
        f"<b>Downloading file{size_str}…</b> This may take a while (see terminal).</p>"
    )


# ── Config loader ─────────────────────────────────────────────────────────────


def _load_config_text(path: str, revision: str | None = None) -> str:
    path = (path or "").strip()
    revision = (revision or "").strip() or None
    if not path:
        return "# No config path specified."
    try:
        if path.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            parts = path.removeprefix("hf://").split("/")
            repo_id = "/".join(parts[:2])
            filepath = "/".join(parts[2:])
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
):
    """Validate and beamform frame(s) from a zea dataset; yields ``(html, image)`` pairs."""
    file_index = int(file_index)
    start_frame = int(start_frame)
    n_frames = max(1, int(n_frames))
    track_index = int(track_index)
    lines: list[str] = []

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
                "No HF token found — set HF_TOKEN or run 'huggingface-cli login'. "
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
    yield _replace_last(_html_pass(f"Dataset opened — {num_files} file(s)"))
    if _stopped():
        return

    # 2. Load config (non-fatal — fall back to raw display on failure)
    config_params: dict = {}
    pipeline = None
    if not config_path:
        yield _emit(_html_warn("No config path set — will display data without processing."))
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
                _html_warn(f"Config unavailable ({exc}) — will display data without processing.")
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
                        f"Pipeline build failed ({exc}) — will display data without processing."
                    )
                )
            if pipeline is not None:
                if not _key_requires_pipeline(key):
                    # Key doesn't need beamforming — skip pipeline, use raw display
                    pipeline = None
                    yield _replace_last(
                        _html_warn("Pipeline ignored — key does not need beamforming.")
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
                f"Key '{key}' contains raw RF data — a valid pipeline config is needed. "
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
                        _html_pass(f"Data loaded — {total_frames} frame(s), {len(_tx)} transmit(s)")
                    )
                else:
                    yield _emit(_html_pass(f"Data loaded — {total_frames} frame(s)"))
                if _stopped():
                    return

                yield _emit(_html_pass("Parameters loaded"))
                if _stopped():
                    return

                # 5. Prepare pipeline parameters
                try:
                    params = _run_quiet(pipeline.prepare_parameters, parameters, **config_params)
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
            return_filename=False,
            offset_n_frames=start_frame,
            limit_n_frames=actual_n,
            n_frames=1,
            num_threads=4,
            insert_frame_axis=False,
            sort_files=False,
            axis_selections=_axis_sel,
            validate=False,
            revision=_dl_revision,
        )
    except Exception as exc:
        yield _emit(_html_fail("Open file", exc))
        return

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
                            f"Data shape {frame.shape} after indexing — need at least 2D.",
                        )
                    )
                    return
                processed = frame
        except Exception as exc:
            yield _emit(_html_fail(f"Process frame {start_frame + i}", exc))
            return

        processed_frames.append(processed)

        pbar = _html_progress(i + 1, actual_n)
        if i == 0:
            yield _emit(pbar)
        else:
            yield _replace_last(pbar)

        if _stopped():
            return

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
        f'<hr style="margin:6px 0;border-color:#374151">'
        f'<p style="margin:4px 0;color:{_YELLOW}"><b>&#10004; Processing done</b>'
        f' <span style="color:#6b7280">— file {file_index + 1}/{num_files}'
        f" &middot; {frame_label}</span></p>"
    )
    yield _replace_last(done_html, result_image)

    if not _is_hf(dataset_path):
        yield _emit(_html_warn("Local dataset path — not yet on Hugging Face."), result_image)
    if not _is_hf(config_path):
        yield _emit(_html_warn("Local config path — not yet on Hugging Face."), result_image)
    if _is_hf(dataset_path) and _is_hf(config_path):
        rp = str(dataset_path).removeprefix("hf://").rstrip("/").split("/")[:2]
        cp = str(config_path).removeprefix("hf://").rstrip("/").split("/")[:2]
        if rp != cp:
            yield _emit(
                _html_warn("Dataset and config are on different HF repositories."),
                result_image,
            )


# ── Gradio interface ───────────────────────────────────────────────────────────

_EDITOR_ACTIVE_HTML = (
    '<div style="background:rgba(245,197,24,0.12);border:1px solid #f5c518;'
    'border-radius:4px;padding:5px 10px;margin:3px 0;font-size:0.8em;color:#f5c518">'
    "&#9888;&nbsp;<b>Editor config active</b> — config path &amp; revision above are "
    "ignored. Click <b>Load config from path</b> in the Config editor tab to revert."
    "</div>"
)


def build_interface() -> "gr.Blocks":
    """Build and return the Gradio Blocks interface."""

    logo = _logo_html(height=54)

    with gr.Blocks(title="zea visualizer") as demo:
        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(
            f'<div style="display:flex;align-items:flex-end;padding:8px 0 4px;'
            f'margin-bottom:6px">'
            f'<div style="flex-shrink:0;margin-right:10px">{logo}</div>'
            f'<div style="display:flex;align-items:center;'
            f'border-bottom:2px solid {_PURPLE};flex:1;padding-bottom:5px">'
            f'<span style="font-size:1.35em;font-weight:700;color:{_PURPLE}">zea</span>'
            f'<span style="font-size:1.35em;font-weight:400;margin-left:5px">'
            f"dataset visualizer</span>"
            f"</div>"
            f"</div>"
        )

        # ── Hidden state ────────────────────────────────────────────────────
        config_rev_decoupled = gr.State(False)
        file_paths_state = gr.State([])
        track_labels_state = gr.State([])
        # True only while the user's manual edits to the config editor should
        # override the config path. Reset whenever the config is (re)loaded or a
        # dataset/config/preset path changes, so stale editor contents are not used.
        editor_override_active = gr.State(False)

        # ── Main row ────────────────────────────────────────────────────────
        with gr.Row():
            # Left: tabbed controls ─────────────────────────────────────────
            with gr.Column(scale=1, min_width=380):
                with gr.Tabs():
                    with gr.Tab("Settings"):
                        preset_selector = gr.Dropdown(
                            label="Preset",
                            choices=list(PRESETS.keys()),
                            value=None,
                            interactive=True,
                            info="Select a preset to auto-fill fields below.",
                        )
                        gr.HTML('<hr style="border-color:#374151;margin:4px 0">')

                        with gr.Row():
                            dataset_input = gr.Textbox(
                                label="Dataset path",
                                placeholder="hf://zeahub/… or /local/path",
                                info="Local path or hf://owner/dataset-name",
                                scale=4,
                            )
                            dataset_rev_input = gr.Dropdown(
                                label="Revision",
                                choices=["main"],
                                value=None,
                                allow_custom_value=True,
                                interactive=False,
                                scale=1,
                                min_width=115,
                                info=" ",
                                elem_classes=["revision-dropdown"],
                            )
                        with gr.Row():
                            config_input = gr.Textbox(
                                label="Config path",
                                placeholder="hf://… or /local/config.yaml",
                                scale=4,
                            )
                            config_rev_input = gr.Dropdown(
                                label="Revision (auto)",
                                choices=["main"],
                                value=None,
                                allow_custom_value=True,
                                interactive=False,
                                scale=1,
                                min_width=115,
                                info=" ",
                                elem_classes=["revision-dropdown"],
                            )

                        # Shows when config editor overrides the config path
                        editor_indicator = gr.HTML("", visible=False)

                        file_selector = gr.Dropdown(
                            label="File",
                            choices=[],
                            value=None,
                            interactive=False,
                            info="Select a file to load its metadata and set frame range.",
                        )

                        key_input = gr.Dropdown(
                            label="Data key",
                            choices=_DATA_KEYS,
                            value=None,
                            allow_custom_value=True,
                            interactive=False,
                        )

                        track_selector = gr.Dropdown(
                            label="Track",
                            choices=[("Track 0", 0)],
                            value=0,
                            interactive=False,
                            visible=True,
                        )

                        with gr.Row():
                            start_frame_input = gr.Slider(
                                label="Start frame",
                                minimum=0,
                                maximum=999,
                                value=0,
                                step=1,
                                interactive=False,
                                elem_classes=["frame-slider"],
                            )
                            n_frames_input = gr.Slider(
                                label="# frames",
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

                    with gr.Tab("Config editor"):
                        load_config_btn = gr.Button("Load config from path", size="sm")
                        config_editor = gr.Code(
                            label="Config YAML",
                            language="yaml",
                            lines=22,
                        )

            # Right: metadata card + image + status ─────────────────────────
            with gr.Column(scale=2):
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

        # Revision toggle (fast, no network)
        def _rev_toggle(path):
            return gr.update(interactive=_is_hf(path))

        dataset_input.change(_rev_toggle, [dataset_input], [dataset_rev_input])
        config_input.change(_rev_toggle, [config_input], [config_rev_input])

        _TRACK_RESET = gr.update(choices=[("Track 0", 0)], value=0, visible=True, interactive=False)

        # Dataset blur → fetch revisions + file list (no download)
        def _on_dataset_blur(path):
            path = (path or "").strip()
            _disable_run = gr.update(interactive=False)
            if not path:
                return (
                    gr.update(),
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

            config_prefill = f"{path.rstrip('/')}/config.yaml" if _is_hf(path) else None

            if not _is_hf(path):
                return (
                    gr.update(),
                    gr.update(interactive=False, choices=["main"], value=None),
                    file_update,
                    paths,
                    _TRACK_RESET,
                    [],
                    meta_html,
                    _disable_run,
                    _reset_key,
                )

            if errors:
                # Repo inaccessible — don't waste a network call on revisions.
                return (
                    gr.update(value=config_prefill),
                    gr.update(interactive=False, choices=["main"], value=None),
                    file_update,
                    paths,
                    _TRACK_RESET,
                    [],
                    meta_html,
                    _disable_run,
                    _reset_key,
                )

            revisions = _fetch_hf_revisions(path)
            default = "main" if "main" in revisions else (revisions[0] if revisions else "main")
            return (
                gr.update(value=config_prefill),
                gr.update(interactive=True, choices=revisions, value=default),
                file_update,
                paths,
                _TRACK_RESET,
                [],
                meta_html,
                _disable_run,
                _reset_key,
            )

        dataset_input.blur(
            _on_dataset_blur,
            inputs=[dataset_input],
            outputs=[
                config_input,
                dataset_rev_input,
                file_selector,
                file_paths_state,
                track_selector,
                track_labels_state,
                meta_card,
                run_btn,
                key_input,
            ],
        )

        # Config blur → validate path + fetch revisions
        def _on_config_blur(path):
            path = (path or "").strip()
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

                parts = path.removeprefix("hf://").strip("/").split("/")
                if len(parts) < 2 or not parts[1]:
                    gr.Warning("Invalid Hugging Face path: expected hf://owner/repo-name/…")
                    return gr.update(interactive=False, choices=["main"], value=None)
                repo_id = "/".join(parts[:2])
                filepath = "/".join(parts[2:]) if len(parts) > 2 else ""
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

        config_input.blur(_on_config_blur, [config_input], [config_rev_input])

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
            )

            sf, nf, meta, trk, tlbls, run_upd, key_upd = _file_load_updates(
                new_val, rev or None, key
            )
            yield cfg_upd, gr.update(), fpaths, meta, trk, tlbls, run_upd, key_upd, sf, nf

        dataset_rev_input.input(
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
            ],
        )

        # User manually picks a config revision → decouple
        def _on_config_rev_input():
            return True, gr.update(label="Revision")

        config_rev_input.input(_on_config_rev_input, [], [config_rev_decoupled, config_rev_input])

        # Preset → fill all fields + reset sync state (no file auto-load)
        def _apply_preset(name):
            if name not in PRESETS:
                return (gr.update(),) * 13
            p = PRESETS[name]
            ds = p.get("dataset", "")
            cfg = p.get("config", "")
            key = p.get("key", "data/raw_data")
            ds_revs = _fetch_hf_revisions(ds) if _is_hf(ds) else ["main"]
            cfg_revs = _fetch_hf_revisions(cfg) if _is_hf(cfg) else ["main"]
            ds_def = "main" if "main" in ds_revs else (ds_revs[0] if ds_revs else "main")
            cfg_def = "main" if "main" in cfg_revs else (cfg_revs[0] if cfg_revs else "main")
            names, paths = _list_dataset_files(ds, ds_def)
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
                gr.update(choices=list(zip(names, paths)), value=None, interactive=bool(paths)),
                paths,
                _TRACK_RESET,  # track_selector
                [],  # track_labels_state
                gr.update(value=None),  # image_output clear
                "",  # meta_card clear
                gr.update(interactive=False),  # run_btn — re-enabled after file is picked
            )

        preset_selector.change(
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
            ],
        )

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
        )

        def _on_file_select_gen(selected_name, file_paths, key, ds_revision, config_path):
            # selected_name is the full path (dropdown value), not the basename.
            if not selected_name or not file_paths or selected_name not in file_paths:
                yield _NO_FILE
                return
            fpath = selected_name

            # For HF paths, look up size so the loading indicator is informative.
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

            # Step 1: disable ALL inputs including stop — download cannot be interrupted.
            yield (
                gr.update(interactive=False),  # start_frame_input
                gr.update(interactive=False),  # n_frames_input
                _loading_meta_html(size_bytes),  # meta_card
                _TRACK_RESET,  # track_selector
                [],  # track_labels_state
                gr.update(interactive=False),  # run_btn
                gr.update(interactive=False),  # key_input
                gr.update(value=None),  # image_output — clear previous result
                gr.update(interactive=False),  # stop_btn — keep disabled; can't cancel download
                gr.update(interactive=False),  # file_selector — prevent switching files
                gr.update(interactive=False),  # preset_selector
                gr.update(interactive=False),  # dataset_input
                gr.update(interactive=False),  # dataset_rev_input
                gr.update(interactive=False),  # config_input
                gr.update(interactive=False),  # config_rev_input
            )

            # Step 2: download at correct revision + read metadata
            sf, nf, meta, trk, tlbls, run_upd, key_upd = _file_load_updates(
                fpath, ds_revision or None, key
            )
            if _is_hf(fpath) and meta:
                meta = _html_info("Download complete · press Run to display") + meta
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
            )

        file_select_event = file_selector.change(
            _on_file_select_gen,
            inputs=[file_selector, file_paths_state, key_input, dataset_rev_input, config_input],
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
            ],
        )

        # Key chosen → enable run button (file is already loaded at this point)
        key_input.change(
            lambda key, fname: gr.update(interactive=bool(key and fname)),
            inputs=[key_input, file_selector],
            outputs=[run_btn],
        )

        # Track changed → update frame sliders for that track's n_frames
        def _on_track_change(track_id, track_labels, selected_file, file_paths, ds_revision):
            # track_id is the numeric index emitted by the dropdown (None when unset).
            if (
                track_id is None
                or not track_labels
                or not selected_file
                or not file_paths
                or selected_file not in file_paths
            ):
                return gr.update(), gr.update()
            try:
                is_hf = _is_hf(selected_file)
                hf_kwargs = {"revision": ds_revision} if ds_revision and is_hf else {}
                with File(selected_file, **hf_kwargs) as f:
                    n = f.tracks[int(track_id)].n_frames
                if n > 1:
                    return (
                        gr.update(maximum=n - 1, value=0, interactive=True),
                        gr.update(maximum=n, value=1, interactive=True),
                    )
                # Single or zero-frame track — disable sliders without setting
                # maximum=0: Gradio requires minimum < maximum strictly, and
                # start_frame_input has minimum=0, so maximum=0 would crash.
                return (
                    gr.update(value=0, interactive=False),
                    gr.update(value=1, interactive=False),
                )
            except Exception:
                return gr.update(), gr.update()

        track_selector.change(
            _on_track_change,
            [
                track_selector,
                track_labels_state,
                file_selector,
                file_paths_state,
                dataset_rev_input,
            ],
            [start_frame_input, n_frames_input],
        )

        # Config editor: mark when user types → editor contents now override the path
        config_editor.input(
            lambda: (gr.update(visible=True, value=_EDITOR_ACTIVE_HTML), True),
            [],
            [editor_indicator, editor_override_active],
        )

        # Load from path → clear editor indicator and override flag
        def _load_and_clear(path, revision):
            return _load_config_text(path, revision), gr.update(visible=False, value=""), False

        load_config_btn.click(
            _load_and_clear,
            [config_input, config_rev_input],
            [config_editor, editor_indicator, editor_override_active],
        )

        # Changing a dataset/config/preset path invalidates any manual editor
        # override so the (new) config path is used on the next run.
        def _clear_editor_override():
            return gr.update(visible=False, value=""), False

        for _component in (dataset_input, config_input):
            _component.change(
                _clear_editor_override, [], [editor_indicator, editor_override_active]
            )
        preset_selector.change(
            _clear_editor_override, [], [editor_indicator, editor_override_active]
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
            editor_yaml,
            editor_override,
        ):
            _stop_event.clear()
            dataset = (dataset or "").strip()
            config = (config or "").strip()
            if not dataset:
                raise gr.Warning("Please enter a dataset path.")
            if not file_name:
                raise gr.Warning("Please select a file from the dropdown first.")
            if not key:
                raise gr.Warning("Please select a data key from the dropdown first.")

            config_resolved = config or ""
            tmp_cfg = None
            if (
                editor_override
                and editor_yaml
                and editor_yaml.strip()
                and not editor_yaml.strip().startswith("#")
            ):
                tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
                tmp_cfg.write(editor_yaml)
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
            yield gr.update(), gr.update(), *_lock_extras

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
                ):
                    if img is None:
                        yield html, None, *_noop_extras
                    elif isinstance(img, str):
                        yield html, img, *_noop_extras
                    else:
                        tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        img.save(tmp_png.name)
                        yield html, tmp_png.name, *_noop_extras
                # Normal completion — restore UI.
                yield gr.update(), gr.update(), *_unlock_extras
            except Exception as exc:
                import traceback as _tb

                yield (
                    _html_fail("Unexpected error", exc)
                    + f'<pre style="font-size:0.75em;color:#6b7280;white-space:pre-wrap">'
                    f"{_tb.format_exc()}</pre>",
                    None,
                    *_noop_extras,
                )
                # Restore UI after error.
                yield gr.update(), gr.update(), *_unlock_extras
            finally:
                # Cleanup only — no yield here; yielding after GeneratorExit raises RuntimeError.
                if tmp_cfg is not None:
                    try:
                        os.unlink(tmp_cfg.name)
                    except OSError:
                        pass

        run_event = run_btn.click(
            _on_run,
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
                config_editor,
                editor_override_active,
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
                _TRACK_RESET,  # track_selector
                "",  # meta_card — clear loading msg
            )

        # Stop cancels both run and file-loading events, and restores UI directly.
        stop_btn.click(
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
            ],
            cancels=[run_event, file_select_event],
        )

        status_output.change(fn=None, js=_SCROLL_JS)
        demo.load(_load_config_text, inputs=[config_input], outputs=[config_editor])

    return demo


# ── CLI ────────────────────────────────────────────────────────────────────────


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the zea Gradio visualizer.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument(
        "--server-port", dest="server_port", type=int, default=None, help="Port to listen on."
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    init_device()
    demo = build_interface()
    demo.launch(
        share=args.share,
        server_port=args.server_port,
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="yellow"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
