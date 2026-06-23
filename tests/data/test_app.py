"""Lightweight smoke tests for zea.data.app — no HF downloads, no Gradio server."""

from unittest.mock import patch

from . import generate_example_dataset


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
    assert launched.get("server_port") is None


def test_zea_app_passes_share_flag(monkeypatch):
    """--share and --server-port flags are forwarded to demo.launch()."""
    monkeypatch.setattr("sys.argv", ["zea", "app", "--share", "--server-port", "7861"])

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

    assert _read_file_info("/nonexistent/path.hdf5") == {}


# ── get_parser ────────────────────────────────────────────────────────────────


def test_get_parser_defaults():
    from zea.data.app import get_parser

    args = get_parser().parse_args([])
    assert args.share is False
    assert args.server_port is None


def test_get_parser_with_flags():
    from zea.data.app import get_parser

    args = get_parser().parse_args(["--share", "--server-port", "8080"])
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
