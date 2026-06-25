"""Tests for zea.data.process — lightweight, no HF downloads."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.data import generate_example_dataset


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_image_file(path: Path, n_frames: int = 2, h: int = 16, w: int = 16) -> Path:
    """Create an HDF5 file containing data/image/values (uint8, no scan needed)."""
    from zea.data.file import File

    path.parent.mkdir(parents=True, exist_ok=True)
    File.create(
        path,
        data={"image": {"values": np.zeros((n_frames, h, w), dtype=np.uint8)}},
        overwrite=True,
    )
    return path


def _minimal_config(path: Path, *, with_pipeline: bool = False) -> Path:
    """Write a minimal YAML config; optionally include a dummy pipeline section."""
    content = "parameters:\n  sound_speed: 1540\n"
    if with_pipeline:
        content += "pipeline:\n  - op: Identity\n    name: passthrough\n"
    path.write_text(content)
    return path


# ── unit tests ────────────────────────────────────────────────────────────────


def test_key_requires_pipeline_true():
    from zea.data.process import _key_requires_pipeline

    assert _key_requires_pipeline("data/raw_data") is True
    assert _key_requires_pipeline("data/aligned_data/values") is True


def test_key_requires_pipeline_false():
    from zea.data.process import _key_requires_pipeline

    assert _key_requires_pipeline("data/image/values") is False
    assert _key_requires_pipeline("data/envelope_data/values") is False
    assert _key_requires_pipeline("data/segmentation/values") is False
    assert _key_requires_pipeline("") is False
    assert _key_requires_pipeline(None) is False


def test_get_parser_defaults():
    from zea.data.process import get_parser

    p = get_parser()
    args = p.parse_args(["--dataset", "data/", "--config", "cfg.yaml"])
    assert args.key == "data/raw_data"
    assert args.n_frames is None
    assert args.save_as == "gif"
    assert args.overwrite is False
    assert args.keep_dynamic_range is False
    assert args.revision is None
    assert args.config_revision is None
    assert args.num_threads == 16
    assert str(args.save_dir) == "output"


# ── _run_passthrough ──────────────────────────────────────────────────────────


def test_run_passthrough_gif(tmp_path):
    """_run_passthrough saves a GIF for each file in the dataset folder."""
    from zea.data.process import _run_passthrough

    ds_dir = tmp_path / "ds"
    _make_image_file(ds_dir / "scan_a.hdf5")
    _make_image_file(ds_dir / "scan_b.hdf5")
    out_dir = tmp_path / "out"

    _run_passthrough(str(ds_dir), "data/image/values", None, out_dir, "gif", False)

    assert (out_dir / "scan_a.gif").exists()
    assert (out_dir / "scan_b.gif").exists()


def test_run_passthrough_hdf5(tmp_path):
    """_run_passthrough saves HDF5 output files."""
    from zea.data.process import _run_passthrough

    ds_dir = tmp_path / "ds"
    _make_image_file(ds_dir / "scan.hdf5", n_frames=1)
    out_dir = tmp_path / "out"

    _run_passthrough(str(ds_dir), "data/image/values", None, out_dir, "hdf5", False)

    assert (out_dir / "scan.hdf5").exists()


def test_run_passthrough_n_frames_limit(tmp_path):
    """_run_passthrough respects the n_frames limit."""
    from zea.data.file import File
    from zea.data.process import _run_passthrough

    ds_dir = tmp_path / "ds"
    _make_image_file(ds_dir / "scan.hdf5", n_frames=4)
    out_dir = tmp_path / "out"

    _run_passthrough(str(ds_dir), "data/image/values", 2, out_dir, "hdf5", False)

    with File(out_dir / "scan.hdf5") as f:
        key = f.format_key("data/image/values")
        assert f[key].shape[0] == 2


def test_run_passthrough_overwrite_false(tmp_path):
    """_run_passthrough skips existing files when overwrite=False."""
    from zea.data.process import _run_passthrough

    ds_dir = tmp_path / "ds"
    _make_image_file(ds_dir / "scan.hdf5")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Pre-create the output with a sentinel marker
    sentinel = out_dir / "scan.gif"
    sentinel.write_bytes(b"sentinel")

    _run_passthrough(str(ds_dir), "data/image/values", None, out_dir, "gif", False)

    # File must NOT be overwritten
    assert sentinel.read_bytes() == b"sentinel"


def test_run_passthrough_overwrite_true(tmp_path):
    """_run_passthrough replaces existing files when overwrite=True."""
    from zea.data.process import _run_passthrough

    ds_dir = tmp_path / "ds"
    _make_image_file(ds_dir / "scan.hdf5")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = out_dir / "scan.gif"
    sentinel.write_bytes(b"sentinel")

    _run_passthrough(str(ds_dir), "data/image/values", None, out_dir, "gif", True)

    assert sentinel.read_bytes() != b"sentinel"


# ── run_processing — passthrough fallback ─────────────────────────────────────


def test_run_processing_passthrough_fallback(tmp_path):
    """run_processing falls back to passthrough for image keys when config has no pipeline."""
    from zea.data.process import run_processing

    ds_dir = tmp_path / "ds"
    _make_image_file(ds_dir / "scan.hdf5")
    cfg = _minimal_config(tmp_path / "cfg.yaml", with_pipeline=False)
    out_dir = tmp_path / "out"

    run_processing(
        str(ds_dir),
        str(cfg),
        key="data/image/values",
        n_frames=None,
        save_dir=out_dir,
        save_as="gif",
    )

    assert (out_dir / "scan.gif").exists()


def test_run_processing_raw_without_pipeline_raises(tmp_path):
    """run_processing raises when key requires a pipeline but config has none."""
    from zea.data.process import run_processing

    ds_dir = tmp_path / "ds"
    generate_example_dataset(ds_dir / "scan.hdf5", n_frames=1, n_ax=8, n_el=4, n_tx=2)
    cfg = _minimal_config(tmp_path / "cfg.yaml", with_pipeline=False)
    out_dir = tmp_path / "out"

    with pytest.raises((ValueError, KeyError)):
        run_processing(
            str(ds_dir),
            str(cfg),
            key="data/raw_data",
            n_frames=1,
            save_dir=out_dir,
            save_as="gif",
        )


def test_run_processing_invalid_save_as(tmp_path):
    """run_processing raises ValueError for unknown save_as format."""
    from zea.data.process import run_processing

    with pytest.raises(ValueError, match="save_as"):
        run_processing(
            str(tmp_path),
            str(tmp_path / "cfg.yaml"),
            key="data/image/values",
            n_frames=None,
            save_dir=tmp_path / "out",
            save_as="jpg",
        )


def test_run_processing_keep_dynamic_range_requires_hdf5(tmp_path):
    """--keep_dynamic_range is only valid with save_as=hdf5."""
    from zea.data.process import run_processing

    with pytest.raises(ValueError, match="keep_dynamic_range"):
        run_processing(
            str(tmp_path),
            str(tmp_path / "cfg.yaml"),
            key="data/raw_data",
            n_frames=None,
            save_dir=tmp_path / "out",
            save_as="gif",
            keep_dynamic_range=True,
        )


# ── main() dispatch ───────────────────────────────────────────────────────────


def test_main_dispatches_to_run_processing(tmp_path, monkeypatch):
    """zea.__main__.main() calls run_processing for the 'process' subcommand."""
    ds_dir = tmp_path / "ds"
    _make_image_file(ds_dir / "scan.hdf5")
    cfg = _minimal_config(tmp_path / "cfg.yaml")

    monkeypatch.setattr(
        "sys.argv",
        [
            "zea",
            "process",
            "--dataset",
            str(ds_dir),
            "--config",
            str(cfg),
            "--key",
            "data/image/values",
            "--save-as",
            "gif",
        ],
    )

    called = {}

    def _fake_run(dataset, config, key, *args, **kwargs):
        called["dataset"] = dataset
        called["config"] = config
        called["key"] = key

    with patch("zea.data.process.run_processing", _fake_run):
        with patch("zea.internal.device.init_device"):
            from zea.__main__ import main

            main()

    assert called["key"] == "data/image/values"
    assert called["dataset"] == str(ds_dir)
    assert called["config"] == str(cfg)
