"""Lightweight tests for the ``python -m zea.data.convert`` CLI (zea.data.convert.__main__)."""

import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import tyro

from zea.data.convert.__main__ import (
    Dataset,
    _Camus,
    _Cetus,
    _Echonet,
    _EchonetLVH,
    _EchoXFlow,
    _Picmus,
    _Verasonics,
)


def parse_args(argv):
    return tyro.cli(Dataset, args=argv)  # ty: ignore[no-matching-overload]


# ── parser structure ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("subcommand", "cls"),
    [
        ("echonet", _Echonet),
        ("echonetlvh", _EchonetLVH),
        ("camus", _Camus),
        ("cetus", _Cetus),
        ("picmus", _Picmus),
        ("verasonics", _Verasonics),
        ("echoxflow", _EchoXFlow),
    ],
)
def test_all_subcommands_exist(subcommand, cls):
    """Every dataset must be registered as a subcommand and parse to its dataclass."""
    args = parse_args([subcommand, "src", "dst"])
    assert isinstance(args, cls)


def test_no_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        parse_args([])
    assert exc_info.value.code != 0


def test_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["not-a-dataset", "src", "dst"])
    assert exc_info.value.code != 0


def test_top_level_help_exits_zero():
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stdout(buf):
        parse_args(["--help"])
    assert exc_info.value.code == 0
    assert "echonet" in buf.getvalue()


# ── positional src/dst (shared across all datasets) ────────────────────────────


def test_positional_src_dst_required():
    with pytest.raises(SystemExit):
        parse_args(["camus"])


def test_positional_src_dst_parsed_as_path():
    args = parse_args(["camus", "raw/", "output/"])
    assert args.src == Path("raw/")
    assert args.dst == Path("output/")


# ── echonet ──────────────────────────────────────────────────────────────────


def test_echonet_defaults():
    args = parse_args(["echonet", "src", "dst"])
    assert args.split_path is None
    assert args.no_hyperthreading is False


def test_echonet_flags():
    args = parse_args(
        ["echonet", "src", "dst", "--split-path", "split.yaml", "--no-hyperthreading"]
    )
    assert args.split_path == Path("split.yaml")
    assert args.no_hyperthreading is True


# ── camus ────────────────────────────────────────────────────────────────────


def test_camus_defaults():
    args = parse_args(["camus", "src", "dst"])
    assert args.download is False
    assert args.no_hyperthreading is False
    assert args.upload is False
    assert args.revision is None
    assert args.reduced_dataset is False


def test_camus_flags():
    args = parse_args(
        ["camus", "src", "dst", "--download", "--upload", "--revision", "v1", "--reduced-dataset"]
    )
    assert args.download is True
    assert args.upload is True
    assert args.revision == "v1"
    assert args.reduced_dataset is True


# ── verasonics ───────────────────────────────────────────────────────────────


def test_verasonics_defaults():
    args = parse_args(["verasonics", "src", "dst"])
    assert args.frames is None
    assert args.allow_accumulate is False
    assert args.device == "cpu"
    assert args.upload is False
    assert args.revision is None
    assert args.hf_repo_id == ""


def test_verasonics_flags():
    args = parse_args(
        [
            "verasonics",
            "src",
            "dst",
            "--frames",
            "0-3",
            "7",
            "--allow-accumulate",
            "--device",
            "gpu:0",
            "--upload",
            "--revision",
            "v1",
            "--hf-repo-id",
            "zeahub/test",
        ]
    )
    assert args.frames == ["0-3", "7"]
    assert args.allow_accumulate is True
    assert args.device == "gpu:0"
    assert args.hf_repo_id == "zeahub/test"


# ── echoxflow ────────────────────────────────────────────────────────────────


def test_echoxflow_defaults():
    args = parse_args(["echoxflow", "src", "dst"])
    assert args.croissant is None
    assert args.min_frames == 10
    assert args.min_fps == 30.0
    assert args.limit is None
    assert args.overwrite is False
    assert args.hf_repo_id == ""


def test_echoxflow_flags():
    args = parse_args(
        [
            "echoxflow",
            "src",
            "dst",
            "--min-frames",
            "5",
            "--min-fps",
            "15",
            "--limit",
            "2",
            "--overwrite",
        ]
    )
    assert args.min_frames == 5
    assert args.min_fps == 15
    assert args.limit == 2
    assert args.overwrite is True


# ── dispatch (.run() forwards the parsed args to the right converter) ──────────


@pytest.mark.parametrize(
    ("subcommand", "converter_path", "cls"),
    [
        ("echonet", "zea.data.convert.echonet.convert_echonet", _Echonet),
        ("camus", "zea.data.convert.camus.convert_camus", _Camus),
        ("cetus", "zea.data.convert.cetus.convert_cetus", _Cetus),
        ("picmus", "zea.data.convert.picmus.convert_picmus", _Picmus),
        ("verasonics", "zea.data.convert.verasonics.convert_verasonics", _Verasonics),
        ("echoxflow", "zea.data.convert.echoxflow.convert_echoxflow", _EchoXFlow),
    ],
)
def test_run_dispatches_to_converter(subcommand, converter_path, cls):
    """Each dataclass's run() imports and calls its converter with itself."""
    args = parse_args([subcommand, "src", "dst"])
    with patch(converter_path) as mock_convert:
        args.run()
    assert mock_convert.call_count == 1
    (called_args,), _ = mock_convert.call_args
    assert called_args is args
    assert isinstance(called_args, cls)


def test_echonetlvh_run_forwards_positional_args():
    """EchonetLVH's run() forwards individual fields (not the dataclass) to its converter."""
    args = parse_args(["echonetlvh", "src", "dst", "--max-workers", "2"])
    with patch("zea.data.convert.echonetlvh.convert_echonetlvh") as mock_convert:
        args.run()
    assert mock_convert.call_count == 1
    call_args, _ = mock_convert.call_args
    assert call_args[0] == args.src
    assert call_args[1] == args.dst
    assert call_args[-1] == 2  # max_workers


def test_main_dispatches_to_run(monkeypatch):
    """zea.data.convert.__main__.main() parses argv and calls .run() on the result."""
    monkeypatch.setattr("sys.argv", ["zea-convert", "camus", "src", "dst", "--download"])

    with patch("zea.data.convert.camus.convert_camus") as mock_convert:
        from zea.data.convert.__main__ import main

        main()

    assert mock_convert.call_count == 1
    (called_args,), _ = mock_convert.call_args
    assert isinstance(called_args, _Camus)
    assert called_args.download is True
