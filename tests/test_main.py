"""Lightweight tests for the ``zea`` CLI entry point (zea.__main__)."""

import pytest


def _parser():
    from zea.__main__ import get_parser

    return get_parser()


# ── parser structure ──────────────────────────────────────────────────────────


def test_subcommands_exist():
    """Both 'process' and 'app' subcommands must be registered."""
    p = _parser()
    # argparse stores subparser choices on the subparsers action
    subparsers_action = next(a for a in p._actions if hasattr(a, "_name_parser_map"))
    assert "process" in subparsers_action._name_parser_map
    assert "app" in subparsers_action._name_parser_map


def test_no_subcommand_exits_nonzero():
    """Invoking zea with no subcommand should exit with a non-zero status."""
    with pytest.raises(SystemExit) as exc_info:
        _parser().parse_args([])
    assert exc_info.value.code != 0


# ── process subcommand ────────────────────────────────────────────────────────


def test_process_help_exits_zero(capsys):
    """zea process --help should print usage and exit 0."""
    with pytest.raises(SystemExit) as exc_info:
        _parser().parse_args(["process", "--help"])
    assert exc_info.value.code == 0
    assert "dataset" in capsys.readouterr().out


def test_process_parses_required_flags():
    args = _parser().parse_args(
        ["process", "--dataset", "hf://zeahub/data", "--config", "cfg.yaml"]
    )
    assert args.command == "process"
    assert args.dataset == "hf://zeahub/data"
    assert args.config == "cfg.yaml"
    assert str(args.save_dir) == "output"  # default


def test_process_short_flags():
    args = _parser().parse_args(
        [
            "process",
            "-d",
            "hf://zeahub/data",
            "-c",
            "cfg.yaml",
        ]
    )
    assert args.dataset == "hf://zeahub/data"
    assert args.config == "cfg.yaml"


def test_process_optional_args():
    args = _parser().parse_args(
        [
            "process",
            "--dataset",
            "hf://zeahub/data",
            "--config",
            "config.yaml",
            "--save-dir",
            "/tmp/out",
            "--revision",
            "v0.1.0",
            "--config-revision",
            "v0.2.0",
            "--save-as",
            "mp4",
        ]
    )
    assert args.config == "config.yaml"
    assert str(args.save_dir) == "/tmp/out"
    assert args.revision == "v0.1.0"
    assert args.config_revision == "v0.2.0"
    assert args.save_as == "mp4"


def test_process_defaults():
    args = _parser().parse_args(["process", "--dataset", "data/", "--config", "cfg.yaml"])
    assert args.key == "data/raw_data"
    assert args.n_frames is None
    assert args.save_as == "gif"
    assert args.overwrite is False
    assert args.keep_dynamic_range is False
    assert args.revision is None
    assert args.config_revision is None
    assert str(args.save_dir) == "output"


# ── app subcommand ────────────────────────────────────────────────────────────


def test_app_help_exits_zero(capsys):
    """zea app --help should exit 0 without importing gradio."""
    with pytest.raises(SystemExit) as exc_info:
        _parser().parse_args(["app", "--help"])
    assert exc_info.value.code == 0


def test_app_defaults():
    args = _parser().parse_args(["app"])
    assert args.command == "app"
    assert args.share is False
    assert args.server_port is None


def test_app_flags():
    args = _parser().parse_args(["app", "--share", "--server-port", "7861"])
    assert args.share is True
    assert args.server_port == 7861
