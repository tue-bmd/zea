"""Lightweight tests for the ``zea`` CLI entry point (zea.__main__)."""

import contextlib
import io

import pytest
import tyro

from zea.__main__ import CLI
from zea.cli_args import AppArgs, ProcessArgs


def parse_args(argv):
    return tyro.cli(CLI, args=argv)  # ty: ignore[no-matching-overload]


# ── parser structure ──────────────────────────────────────────────────────────


def test_subcommands_exist():
    """Both 'process' and 'app' subcommands must be registered."""
    assert isinstance(
        parse_args(["process", "-d", "data/", "-c", "cfg.yaml"]).subcommand, ProcessArgs
    )
    assert isinstance(parse_args(["app"]).subcommand, AppArgs)


def test_no_subcommand_exits_nonzero():
    """Invoking zea with no subcommand should exit with a non-zero status."""
    with pytest.raises(SystemExit) as exc_info:
        parse_args([])
    assert exc_info.value.code != 0


# ── process subcommand ────────────────────────────────────────────────────────


def test_process_help_exits_zero():
    """zea process --help should print usage and exit 0."""
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stdout(buf):
        parse_args(["process", "--help"])
    assert exc_info.value.code == 0
    assert "dataset" in buf.getvalue()


def test_process_parses_required_flags():
    cli_args = parse_args(["process", "--dataset", "hf://zeahub/data", "--config", "cfg.yaml"])
    args = cli_args.subcommand
    assert isinstance(args, ProcessArgs)
    assert args.dataset == "hf://zeahub/data"
    assert args.config == "cfg.yaml"
    assert str(args.save_dir) == "output"  # default


def test_process_short_flags():
    cli_args = parse_args(
        [
            "process",
            "-d",
            "hf://zeahub/data",
            "-c",
            "cfg.yaml",
        ]
    )
    args = cli_args.subcommand
    assert args.dataset == "hf://zeahub/data"
    assert args.config == "cfg.yaml"


def test_process_optional_args():
    cli_args = parse_args(
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
    args = cli_args.subcommand
    assert args.config == "config.yaml"
    assert str(args.save_dir) == "/tmp/out"
    assert args.revision == "v0.1.0"
    assert args.config_revision == "v0.2.0"
    assert args.save_as == "mp4"


def test_process_defaults():
    cli_args = parse_args(["process", "--dataset", "data/", "--config", "cfg.yaml"])
    args = cli_args.subcommand
    assert args.key == "data/raw_data"
    assert args.n_frames is None
    assert args.save_as == "gif"
    assert args.overwrite is False
    assert args.keep_dynamic_range is False
    assert args.revision is None
    assert args.config_revision is None
    assert str(args.save_dir) == "output"


def test_process_boolean_flags():
    cli_args = parse_args(
        ["process", "-d", "data/", "-c", "cfg.yaml", "--overwrite", "--keep-dynamic-range"]
    )
    args = cli_args.subcommand
    assert args.overwrite is True
    assert args.keep_dynamic_range is True


# ── app subcommand ────────────────────────────────────────────────────────────


def test_app_help_exits_zero():
    """zea app --help should exit 0 without importing gradio."""
    with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stdout(io.StringIO()):
        parse_args(["app", "--help"])
    assert exc_info.value.code == 0


def test_app_defaults():
    cli_args = parse_args(["app"])
    args = cli_args.subcommand
    assert isinstance(args, AppArgs)
    assert args.share is False
    assert args.server_port is None


def test_app_flags():
    cli_args = parse_args(["app", "--share", "--server-port", "7861"])
    args = cli_args.subcommand
    assert args.share is True
    assert args.server_port == 7861
