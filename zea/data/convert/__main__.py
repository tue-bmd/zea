"""CLI for converting common open-source ultrasound datasets to the zea format.

Usage::

    python -m zea.data.convert <dataset> <src> <dst> [options]

Examples::

    python -m zea.data.convert camus ./raw ./output --download
    python -m zea.data.convert cetus ./raw ./output --download
    python -m zea.data.convert echonet ./raw ./output
    python -m zea.data.convert echoxflow ./raw ./output

Run ``python -m zea.data.convert --help`` for all options.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union

import tyro

from zea.internal.device import init_device


@dataclass
class _Echonet:
    """Convert Echonet dataset."""

    src: tyro.conf.Positional[Path]
    """Source folder path."""
    dst: tyro.conf.Positional[Path]
    """Destination folder path."""
    split_path: Path | None = None
    """Path to the split.yaml file containing the dataset split if a split should be copied."""
    no_hyperthreading: bool = False
    """Disable hyperthreading for multiprocessing."""

    def run(self):
        from zea.data.convert.echonet import convert_echonet

        convert_echonet(self)


@dataclass
class _EchonetLVH:
    """Convert EchonetLVH dataset."""

    src: tyro.conf.Positional[Path]
    """Source folder path."""
    dst: tyro.conf.Positional[Path]
    """Destination folder path."""
    no_rejection: bool = False
    """Do not reject sequences in `manual_rejections.txt`."""
    rejection_path: Path | None = None
    """Path to custom rejection txt file (defaults to `manual_rejections.txt` from zea)."""
    convert_measurements: bool = False
    """Only convert measurements CSV file."""
    convert_images: bool = False
    """Only convert image files."""
    max_files: int | None = None
    """Maximum number of files to process (for testing)."""
    force: bool = False
    """Force recomputation even if parameters already exist."""
    max_workers: int = 4
    """Maximum number of workers to use for precomputing cone parameters and dataloading."""

    def run(self):
        from zea.data.convert.echonetlvh import convert_echonetlvh

        convert_echonetlvh(
            self.src,
            self.dst,
            self.no_rejection,
            self.rejection_path,
            self.convert_measurements,
            self.convert_images,
            self.max_files,
            self.force,
            self.max_workers,
        )


@dataclass
class _Camus:
    """Convert CAMUS dataset."""

    src: tyro.conf.Positional[Path]
    """Source folder path, should contain either manually downloaded dataset or will be
    target location for automated download with the --download flag."""
    dst: tyro.conf.Positional[Path]
    """Destination folder path."""
    download: bool = False
    """Download the CAMUS dataset from the server, will be saved to the --src path."""
    no_hyperthreading: bool = False
    """Disable hyperthreading for multiprocessing."""
    upload: bool = False
    """Upload the converted dataset to HuggingFace Hub (zeahub/camus or zeahub/camus-sample)."""
    revision: str | None = None
    """Revision branch to upload to on HuggingFace Hub. Required when --upload is set.
    Upload to 'main' is not allowed."""
    reduced_dataset: bool = False
    """Only convert and upload a small hardcoded sample subset (camus-sample)."""

    def run(self):
        from zea.data.convert.camus import convert_camus

        convert_camus(self)


@dataclass
class _Cetus:
    """Convert CETUS dataset."""

    src: tyro.conf.Positional[Path]
    """Source folder path, should contain either manually downloaded dataset or will be
    target location for automated download with the --download flag."""
    dst: tyro.conf.Positional[Path]
    """Destination folder path."""
    download: bool = False
    """Download the CETUS dataset from the server, will be saved to the --src path."""
    no_hyperthreading: bool = False
    """Disable hyperthreading for multiprocessing."""
    upload: bool = False
    """Upload the converted dataset to HuggingFace Hub (zeahub/cetus-miccai-2014)."""
    revision: str | None = None
    """Revision branch to upload to on HuggingFace Hub. Required when --upload is set.
    Upload to 'main' is not allowed."""

    def run(self):
        from zea.data.convert.cetus import convert_cetus

        convert_cetus(self)


@dataclass
class _Picmus:
    """Convert PICMUS dataset."""

    src: tyro.conf.Positional[Path]
    """Source folder path. Should contain either a manually downloaded and extracted
    archive (archive_to_download/ or picmus.zip) or will be used as the download target
    when --download is given. An 'in_vivo/' sub-directory, if present, is automatically
    included."""
    dst: tyro.conf.Positional[Path]
    """Destination folder path."""
    download: bool = False
    """Download both the main PICMUS dataset and the in-vivo partition from the PICMUS
    challenge website before converting."""
    upload: bool = False
    """Upload the converted dataset to HuggingFace Hub (zeahub/picmus)."""
    revision: str | None = None
    """Revision branch to upload to on HuggingFace Hub. Required when --upload is set.
    Upload to 'main' is not allowed."""

    def run(self):
        from zea.data.convert.picmus import convert_picmus

        convert_picmus(self)


@dataclass
class _Verasonics:
    """Convert Verasonics data to zea dataset."""

    src: tyro.conf.Positional[Path]
    """Source folder path."""
    dst: tyro.conf.Positional[Path]
    """Destination folder path."""
    frames: list[str] | None = None
    """The frames to add to the file. This can be a list of integers, a range of integers
    (e.g. 4-8), or 'all'. Defaults to 'all', unless specified in a convert.yaml file."""
    allow_accumulate: bool = False
    """Sometimes, some transmits are already accumulated on the Verasonics system (e.g.
    harmonic imaging through pulse inversion). In this case, the mode in the Receive
    structure is set to 1 (accumulate). If this flag is set, such files will be processed.
    Otherwise, an error is raised when such a mode is detected."""
    device: str = "cpu"
    """Device to use for conversion (e.g., 'cpu' or 'gpu:0')."""
    upload: bool = False
    """Upload the converted dataset to HuggingFace Hub after conversion. Only for zea
    maintainers with push access to the repository."""
    revision: str | None = None
    """Required when --upload is set. Upload to 'main' is not allowed."""
    hf_repo_id: str = ""
    """HuggingFace repo ID for ownership checks and optional upload. Required if --upload
    is set."""

    def run(self):
        from zea.data.convert.verasonics import convert_verasonics

        convert_verasonics(self)


@dataclass
class _EchoXFlow:
    """Convert EchoXFlow dataset."""

    src: tyro.conf.Positional[str]
    """EchoXFlow data root, e.g. /data/EchoXFlow/data"""
    dst: tyro.conf.Positional[str]
    """Destination folder path."""
    croissant: str | None = None
    """Path to croissant.json (default: <src>/croissant.json)."""
    min_frames: int = 10
    """Minimum B-mode frame count."""
    min_fps: float = 30.0
    """Minimum frame rate (Hz)."""
    limit: int | None = None
    """Convert at most N recordings."""
    overwrite: bool = False
    """Overwrite existing output files."""
    upload: bool = False
    """Upload the converted dataset to HuggingFace Hub (zeahub/echoxflow)."""
    revision: str | None = None
    """Target branch on the Hub. Required when --upload is set; upload to 'main' is
    blocked."""
    hf_repo_id: str = ""
    """HuggingFace repo id for ownership checks and optional upload (default:
    zeahub/echoxflow)."""

    def run(self):
        from zea.data.convert.echoxflow import convert_echoxflow

        convert_echoxflow(self)


Dataset = Union[
    Annotated[_Echonet, tyro.conf.subcommand("echonet")],
    Annotated[_EchonetLVH, tyro.conf.subcommand("echonetlvh")],
    Annotated[_Camus, tyro.conf.subcommand("camus")],
    Annotated[_Cetus, tyro.conf.subcommand("cetus")],
    Annotated[_Picmus, tyro.conf.subcommand("picmus")],
    Annotated[_Verasonics, tyro.conf.subcommand("verasonics")],
    Annotated[_EchoXFlow, tyro.conf.subcommand("echoxflow")],
]


def main():
    """Parse command-line arguments and dispatch to the selected dataset conversion routine."""
    args = tyro.cli(Dataset)  # ty: ignore[no-matching-overload]
    args.run()


if __name__ == "__main__":
    init_device(allow_preallocate=False)
    main()
