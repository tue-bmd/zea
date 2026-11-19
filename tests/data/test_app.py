"""Lightweight smoke tests for zea.data.app — no HF downloads, no Gradio server."""

from unittest.mock import patch

import pytest

from . import generate_example_dataset


@pytest.fixture(autouse=True)
def _offline_presets(monkeypatch):
    """Keep the tests offline: fetching presets from the hub fails, so the bundled ones load."""

    def _offline(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr("zea.data.app._hf_download", _offline)


def test_build_interface_does_not_crash():
    """build_interface() must construct the Gradio Blocks without raising."""
    from zea.data.app import build_interface

    demo = build_interface()
    assert demo is not None
    assert hasattr(demo, "launch")


def test_zea_app_main_calls_build_interface(monkeypatch):
    """zea.__main__.main() with 'app' calls build_interface and launch."""
    monkeypatch.setattr("sys.argv", ["zea", "app"])

    launched = {}

    class _FakeDemo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    with patch("zea.data.app.build_interface", return_value=_FakeDemo()):
        with patch("zea.internal.device.init_device"):
            from zea.__main__ import main

            main()

    assert launched.get("share") is False
    assert launched.get("server_name") is None
    assert launched.get("server_port") is None
    assert launched.get("inbrowser") is True
    # Dark mode only: the JS that pins Gradio's dark class must reach launch().
    from zea.data.app import JS

    assert launched.get("js") == JS


def test_zea_app_passes_share_flag(monkeypatch):
    """--share and --server-port flags are forwarded to demo.launch()."""
    monkeypatch.setattr(
        "sys.argv", ["zea", "app", "--share", "--server-port", "7861", "--no-inbrowser"]
    )

    launched = {}

    class _FakeDemo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    with patch("zea.data.app.build_interface", return_value=_FakeDemo()):
        with patch("zea.internal.device.init_device"):
            from zea.__main__ import main

            main()

    assert launched.get("share") is True
    assert launched.get("server_port") == 7861
    assert launched.get("inbrowser") is False


def test_zea_app_passes_ssl_files(monkeypatch):
    """--server-name, --ssl-certfile and --ssl-keyfile are forwarded to demo.launch()."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "zea",
            "app",
            "--server-name",
            "0.0.0.0",
            "--ssl-certfile",
            "cert.pem",
            "--ssl-keyfile",
            "key.pem",
        ],
    )

    launched = {}

    class _FakeDemo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    with patch("zea.data.app.build_interface", return_value=_FakeDemo()):
        with patch("zea.internal.device.init_device"):
            from zea.__main__ import main

            main()

    assert launched.get("server_name") == "0.0.0.0"
    assert launched.get("ssl_certfile") == "cert.pem"
    assert launched.get("ssl_keyfile") == "key.pem"
    assert launched.get("ssl_verify") is False


# ── Helper-function unit tests ────────────────────────────────────────────────


def test_is_hf():
    from zea.data.app import _is_hf

    assert _is_hf("hf://zeahub/dataset") is True
    assert _is_hf("  hf://org/repo  ") is True
    assert _is_hf("/local/path") is False
    assert _is_hf("") is False


def test_html_helpers_contain_message():
    from zea.data.app import _html_fail, _html_info, _html_pass, _html_progress, _html_warn

    assert "success" in _html_pass("success")
    assert "oops" in _html_fail("oops")
    assert "hint" in _html_warn("hint")
    assert "note" in _html_info("note")

    prog = _html_progress(3, 10)
    assert "3/10" in prog
    assert "30%" in prog


def test_html_fail_includes_error_detail():
    from zea.data.app import _html_fail

    assert "extra detail" in _html_fail("label", "extra detail")
    assert "boom" in _html_fail("label", ValueError("boom"))


def test_enrich_error_plain_exception():
    from zea.data.app import _enrich_error

    assert _enrich_error(RuntimeError("plain")) == "plain"


def test_logo_html_returns_string():
    from zea.data.app import _logo_html

    assert isinstance(_logo_html(), str)


# ── _list_dataset_files ───────────────────────────────────────────────────────


def test_list_dataset_files_empty_path():
    from zea.data.app import _list_dataset_files

    names, paths = _list_dataset_files("")
    assert names == [] and paths == []


def test_list_dataset_files_single_local_file(tmp_path):
    from zea.data.app import _list_dataset_files

    f = tmp_path / "scan.hdf5"
    generate_example_dataset(f)
    names, paths = _list_dataset_files(str(f))
    assert names == ["scan.hdf5"]
    assert paths == [str(f)]


def test_list_dataset_files_local_dir(tmp_path):
    from zea.data.app import _list_dataset_files

    generate_example_dataset(tmp_path / "a.hdf5")
    generate_example_dataset(tmp_path / "b.hdf5")
    names, paths = _list_dataset_files(str(tmp_path))
    assert len(names) == 2


def test_list_dataset_files_missing_local_path(tmp_path):
    from zea.data.app import _list_dataset_files

    errors = []
    names, paths = _list_dataset_files(str(tmp_path / "nonexistent"), _errors=errors)
    assert names == [] and paths == []
    assert len(errors) == 1


def test_list_dataset_files_non_h5_file(tmp_path):
    from zea.data.app import _list_dataset_files

    (tmp_path / "notes.txt").write_text("ignore")
    errors = []
    names, paths = _list_dataset_files(str(tmp_path / "notes.txt"), _errors=errors)
    assert names == [] and paths == []
    assert len(errors) == 1


# ── _load_config_text ─────────────────────────────────────────────────────────


def test_load_config_text_empty_path():
    from zea.data.app import _load_config_text

    assert "No config path" in _load_config_text("")


def test_load_config_text_local_file(tmp_path):
    from zea.data.app import _load_config_text

    cfg = tmp_path / "config.yaml"
    cfg.write_text("pipeline:\n  steps: []\n")
    assert "pipeline" in _load_config_text(str(cfg))


def test_load_config_text_missing_file(tmp_path):
    from zea.data.app import _load_config_text

    result = _load_config_text(str(tmp_path / "missing.yaml"))
    assert "Failed to load config" in result


# ── _build_meta_card_html ─────────────────────────────────────────────────────


def test_build_meta_card_html_empty_dict():
    from zea.data.app import _build_meta_card_html

    assert _build_meta_card_html({}) == ""


def test_build_meta_card_html_legacy_format():
    from zea.data.app import _build_meta_card_html

    assert "legacy format" in _build_meta_card_html({"n_frames_per_track": [5]})


def test_build_meta_card_html_full_info():
    from zea.data.app import _build_meta_card_html

    info = {
        "zea_version": "1.2.0",
        "n_frames_per_track": [20],
        "n_tracks": 1,
        "probe_name": "L12-3v",
        "probe_type": "linear",
        "n_el": 128,
        "probe_fc_hz": 7e6,
        "probe_bw_pct": 77,
        "us_machine": "Verasonics",
        "fs_hz": 40e6,
        "sound_speed": 1540.0,
        "n_tx": 11,
        "n_ax": 2048,
        "subject_type": "human",
        "subject_id": "P001",
        "annot_anatomy": "cardiac",
        "annot_view": "PLAX",
        "credit": "Test lab",
        "description": "test dataset",
    }
    out = _build_meta_card_html(info)
    assert "L12-3v" in out
    assert "7.0" in out  # fc in MHz
    assert "Verasonics" in out
    assert "cardiac" in out


def test_build_meta_card_html_multi_track():
    from zea.data.app import _build_meta_card_html

    info = {"zea_version": "1.0", "n_frames_per_track": [10, 8], "n_tracks": 2}
    out = _build_meta_card_html(info)
    assert "18" in out  # total frames (10+8)
    assert "2" in out  # n_tracks badge


# ── _read_file_info ───────────────────────────────────────────────────────────


def test_read_file_info_valid_file(tmp_path):
    from zea.data.app import _read_file_info

    f = tmp_path / "test.hdf5"
    generate_example_dataset(
        f, add_optional_dtypes=True, n_frames=2, grid_size_z=32, grid_size_x=32
    )
    info = _read_file_info(str(f))
    assert info.get("n_tracks") == 1
    assert info.get("n_frames_per_track") == [2]


def test_read_file_info_nonexistent_path():
    from zea.data.app import _read_file_info

    info = _read_file_info("/nonexistent/path.hdf5")
    assert list(info) == ["error"]


def test_file_load_updates_reports_unreadable_file():
    from zea.data.app import _file_load_updates

    meta = _file_load_updates("/nonexistent/path.hdf5", None, "data/raw_data")[2]
    assert "Cannot read file" in meta


def test_loading_meta_html_mentions_streaming():
    from zea.data.app import _html_progress, _loading_meta_html

    card = _loading_meta_html(3_400_000, 5_750_000_000)
    assert "Streaming" in card and "3.4 MB" in card and "5.8 GB" in card
    assert "streamed" in _html_progress(1, 2, 12_000_000)


def test_bundled_presets_are_well_formed():
    import yaml

    from zea.data.app import _BUNDLED_PRESETS, load_presets

    presets = load_presets()  # offline → the bundled file
    names = [p["name"] for p in yaml.safe_load(_BUNDLED_PRESETS.read_text())["presets"]]
    assert len(names) == len(set(names)) == len(presets) > 0, "preset names must be unique"
    allowed = {"dataset", "config", "key", "file", "n_frames", "revision"}
    for name, p in presets.items():
        assert set(p) <= allowed, (name, set(p) - allowed)
        assert p["dataset"].startswith("hf://"), name
        if "file" in p:
            assert p["file"].startswith(p["dataset"] + "/"), name


def test_load_presets_from_hub(monkeypatch, tmp_path):
    from zea.data import app

    remote = tmp_path / "presets.yaml"
    remote.write_text("schema_version: 1\npresets:\n  - {name: A, dataset: hf://o/r}\n")
    calls = {}

    def _download(repo_id, filename, **kwargs):
        calls.update(repo_id=repo_id, filename=filename, **kwargs)
        return str(remote)

    monkeypatch.setattr(app, "_hf_download", _download)
    assert app.load_presets() == {"A": {"dataset": "hf://o/r"}}
    assert (calls["repo_id"], calls["filename"]) == ("zeahub/app", "presets.yaml")
    assert calls["revision"] == "main"  # "latest" is an alias for main
    app.load_presets(revision="v0.1.8")
    assert calls["revision"] == "v0.1.8"


def test_load_presets_falls_back_to_bundled(tmp_path):
    from zea.data.app import _BUNDLED_PRESETS, _parse_presets, load_presets

    bundled = _parse_presets(_BUNDLED_PRESETS.read_text())
    assert load_presets() == bundled  # hub unreachable
    newer = tmp_path / "presets.yaml"
    newer.write_text("schema_version: 99\npresets: []\n")
    assert load_presets(str(newer)) == bundled  # schema this zea cannot read


def test_parse_presets_filters_by_min_zea_version(monkeypatch):
    import zea
    from zea.data.app import _parse_presets

    text = """
schema_version: 1
presets:
  - {name: old, dataset: hf://o/r, min_zea_version: 0.1.0}
  - {name: new, dataset: hf://o/r, min_zea_version: 99.0.0}
  - {name: broken}
"""
    monkeypatch.setattr(zea, "__version__", "0.1.8")
    with pytest.warns(UserWarning, match="malformed"):
        assert list(_parse_presets(text)) == ["old"]
    monkeypatch.setattr(zea, "__version__", "dev")  # a dev install shows everything
    with pytest.warns(UserWarning):
        assert list(_parse_presets(text)) == ["old", "new"]


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_defaults():
    import tyro

    from zea.data.app import AppArgs

    args = tyro.cli(AppArgs, args=[])
    assert args.share is False
    assert args.server_port is None
    assert args.presets == "hf://zeahub/app/presets.yaml"
    assert args.presets_revision == "latest"


def test_cli_with_flags():
    import tyro

    from zea.data.app import AppArgs

    args = tyro.cli(AppArgs, args=["--share", "--server-port", "8080"])
    assert args.share is True
    assert args.server_port == 8080


# ── run_checks ────────────────────────────────────────────────────────────────


def test_run_checks_no_files_found(tmp_path):
    """run_checks emits a failure message when no HDF5 files exist."""
    from zea.data.app import run_checks

    results = list(run_checks(str(tmp_path), "", key="data/image/values"))
    assert "No HDF5 files found" in results[-1][0]


def test_run_checks_pipeline_required_no_config(tmp_path):
    """run_checks reports failure when a pipeline key is used without a config."""
    from zea.data.app import run_checks

    generate_example_dataset(tmp_path / "data.hdf5", n_frames=2, grid_size_z=32, grid_size_x=32)
    results = list(run_checks(str(tmp_path), "", key="data/raw_data"))
    assert "Pipeline required" in results[-1][0]


def test_run_checks_raw_fallback_success(tmp_path):
    """run_checks completes in raw fallback mode (no config, non-pipeline key)."""
    from PIL import Image as _PILImage

    from zea.data.app import run_checks

    generate_example_dataset(
        tmp_path / "data.hdf5", add_optional_dtypes=True, n_frames=2, grid_size_z=32, grid_size_x=32
    )
    results = list(
        run_checks(str(tmp_path), "", key="data/image/values", start_frame=0, n_frames=1)
    )
    images = [img for _, img in results if img is not None]
    assert len(images) > 0
    assert isinstance(images[0], _PILImage.Image)


# ── Path normalisation / config discovery ─────────────────────────────────────


def test_normalize_path_repairs_hf_scheme():
    from zea.data.app import _normalize_path

    for raw in (
        "hf:://owner/repo/sub",
        "hf:/owner/repo/sub",
        "HF://owner/repo/sub/",
        "  'hf://owner/repo/sub'  ",
        '"hf://owner/repo/sub"',
    ):
        assert _normalize_path(raw) == "hf://owner/repo/sub"
    assert _normalize_path(None) == ""
    assert _normalize_path(" /local/data ") == "/local/data"


def test_display_names_relative_to_root(tmp_path):
    from zea.data.app import _display_names

    hf = ["hf://o/r/data/a/x.hdf5", "hf://o/r/data/b/x.hdf5"]
    assert _display_names("hf://o/r", hf) == ["data/a/x.hdf5", "data/b/x.hdf5"]
    local = [str(tmp_path / "a" / "x.hdf5"), str(tmp_path / "b" / "x.hdf5")]
    assert _display_names(str(tmp_path), local) == ["a/x.hdf5", "b/x.hdf5"]
    # A single file is labelled by its basename
    assert _display_names(local[0], local[:1]) == ["x.hdf5"]


def test_list_dataset_files_missing_local_path_message(tmp_path):
    from zea.data.app import _list_dataset_files

    errors = []
    _list_dataset_files(str(tmp_path / "nonexistent"), _errors=errors)
    assert isinstance(errors[0], FileNotFoundError)
    assert "Path not found" in str(errors[0])


def test_find_config_local(tmp_path):
    from zea.data.app import _find_config

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    generate_example_dataset(data_dir / "scan.hdf5")
    assert _find_config(str(data_dir)) is None
    (tmp_path / "pipeline.yaml").write_text("pipeline: []")
    assert _find_config(str(data_dir / "scan.hdf5")) == str(tmp_path / "pipeline.yaml")
    (data_dir / "config.yaml").write_text("pipeline: []")
    assert _find_config(str(data_dir)) == str(data_dir / "config.yaml")


def test_find_config_hf_walks_up_to_repo_root():
    from zea.data.app import _find_config

    files = ["pipeline.yaml", "data/PAT01/a.hdf5", "other/config.yaml"]
    with patch("zea.data.app._hf_list_files", return_value=files):
        assert _find_config("hf:://o/r/data/PAT01/a.hdf5") == "hf://o/r/pipeline.yaml"
        assert _find_config("hf://o/r/other") == "hf://o/r/other/config.yaml"
    with patch("zea.data.app._hf_list_files", return_value=["data/a.hdf5"]):
        assert _find_config("hf://o/r") is None


def test_config_check_html_local(tmp_path):
    from zea.data.app import _config_check_html

    assert _config_check_html("") == ""
    assert "must point to a .yaml" in _config_check_html(str(tmp_path / "config.txt"))
    missing = _config_check_html(str(tmp_path / "config.yaml"))
    assert "not found" in missing and "Did you mean" not in missing
    (tmp_path / "pipeline.yaml").write_text("pipeline: []")
    assert "Did you mean" in _config_check_html(str(tmp_path / "config.yaml"))
    assert "Config found" in _config_check_html(str(tmp_path / "pipeline.yaml"))


def test_config_check_html_hf_suggests_existing_config():
    from zea.data.app import _config_check_html

    files = ["sub.v2/pipeline.yaml", "sub.v2/a.hdf5"]
    with patch("zea.data.app._hf_list_files", return_value=files) as listing:
        out = _config_check_html("hf://o/r/sub.v2/config.yaml", "refs/pr/1")
        assert "not found at revision refs/pr/1" in out
        assert "hf://o/r/sub.v2/pipeline.yaml" in out
        assert listing.call_args.kwargs == {"revision": "refs/pr/1"}
        assert "Config found" in _config_check_html("hf://o/r/sub.v2/pipeline.yaml")


def test_dataset_check_html():
    from zea.data.app import _dataset_check_html

    assert _dataset_check_html("", 0, []) == ""
    assert "3 zea files" in _dataset_check_html("/data", 3, [])
    assert "No HDF5 files" in _dataset_check_html("/data", 0, [])
    out = _dataset_check_html("/data", 0, [FileNotFoundError("Path not found: /data")])
    assert "Cannot open dataset" in out and "Path not found" in out


# ── Frame picker ──────────────────────────────────────────────────────────────


def test_frame_sliders_bounds_clip_to_file():
    from zea.data.app import _DEFAULT_CLIP_FRAMES, _frame_sliders

    single, clip = _frame_sliders(2000)
    assert single["value"] == [0, 0, 1999]
    assert clip["value"] == [0, _DEFAULT_CLIP_FRAMES - 1, 1999]
    assert _frame_sliders(2000, n_frames=20)[1]["value"] == [0, 19, 1999]
    assert _frame_sliders(10, n_frames=20)[1]["value"] == [0, 9, 9]  # capped by the file
    single, clip = _frame_sliders(1)
    assert single["interactive"] is False and clip["value"] == [0, 0, 0]


def test_frame_range_updates_are_plain_props():
    """Prop names quoted in js_on_load would turn into event triggers in gr.update()."""
    import gradio as gr
    from gradio.blocks import postprocess_update_dict

    from zea.data.app import frame_picker

    with gr.Blocks():
        picker = frame_picker(single=True)
    assert "const SINGLE = true;" in picker.js_on_load
    update = postprocess_update_dict(picker, gr.update(value=[1, 1, 9], interactive=False), True)
    assert not any(callable(v) for v in update.values())
    assert update["value"] == [1, 1, 9]


def test_dataset_files_skips_invalid_files(tmp_path):
    """One non-zea HDF5 file must not make the whole dataset unusable."""
    import h5py

    from zea.data.app import _dataset_check_html, _dataset_files, _list_dataset_files

    generate_example_dataset(tmp_path / "b_valid.hdf5", n_frames=2, grid_size_z=8, grid_size_x=8)
    with h5py.File(tmp_path / "a_other.hdf5", "w") as f:
        f.create_dataset("x", data=[1, 2, 3])
    valid, invalid = _dataset_files(str(tmp_path))
    assert valid == [str(tmp_path / "b_valid.hdf5")]
    assert list(invalid) == [str(tmp_path / "a_other.hdf5")]

    skipped = {}
    names, paths = _list_dataset_files(str(tmp_path), _invalid=skipped)
    assert names == ["b_valid.hdf5"] and skipped == invalid
    out = _dataset_check_html(str(tmp_path), len(paths), [], skipped)
    assert "Skipped 1 file" in out and "a_other.hdf5" in out

    (tmp_path / "b_valid.hdf5").unlink()
    with pytest.raises(ValueError, match="No valid zea files"):
        _dataset_files(str(tmp_path))
