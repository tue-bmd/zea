"""Lightweight CLI argument definitions for the ``zea`` command line tool.

Kept free of heavy imports (keras, ``zea.data``, …) so that ``zea --help`` and
``zea process --help`` can be rendered without loading an ML backend. This
module lives at the top level of the package (rather than under ``zea.data``)
because importing ``zea.data`` eagerly pulls in keras. The actual processing
code lives in :mod:`zea.data.process`.
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Union

import tyro

SUPPORTED_FORMATS = ["gif", "mp4", "hdf5"]
sitk = importlib.util.find_spec("SimpleITK")
if sitk is not None:
    SUPPORTED_FORMATS += ["nii.gz"]


@dataclass
class AppArgs:
    """Arguments for the interactive Gradio dataset visualizer."""

    share: Annotated[
        bool,
        tyro.conf.arg(help="Create a public Gradio share link."),
    ] = False
    server_port: Annotated[
        int | None,
        tyro.conf.arg(
            help="Port for the Gradio server to listen on. If None, will search for an available "
            "port starting at 7860. Defaults to None."
        ),
    ] = None


@dataclass
class ProcessArgs:
    """Arguments for beamforming a zea dataset."""

    dataset: Annotated[
        str,
        tyro.conf.arg(
            aliases=["-d"],
            help="Path/URI to the zea dataset (folder of HDF5 files or a single HDF5 file).",
        ),
    ]
    config: Annotated[
        str,
        tyro.conf.arg(
            aliases=["-c"],
            help="Path to config.yaml for the beamforming pipeline.",
        ),
    ]
    save_dir: Annotated[
        Path,
        tyro.conf.arg(
            aliases=["-o"],
            help="Directory where output files are written. Default: output/",
        ),
    ] = Path("output")
    key: Annotated[
        str,
        tyro.conf.arg(
            help="Data key to load from each file (e.g. data/raw_data, data/image/values).",
        ),
    ] = "data/raw_data"
    n_frames: Annotated[
        int | None,
        tyro.conf.arg(
            help="Maximum number of frames to process per file (all frames when omitted).",
        ),
    ] = None
    save_as: Annotated[
        Literal[tuple(SUPPORTED_FORMATS)],  # ty: ignore[invalid-type-form]
        tyro.conf.arg(
            help=f"Output format. One of: {', '.join(SUPPORTED_FORMATS)}.",
        ),
    ] = "gif"
    keep_keys: Annotated[
        list[str],
        tyro.conf.arg(
            help="List of pipeline output keys to forward to the next frame iteration.",
        ),
    ] = field(default_factory=lambda: ["maxval"])
    timings: Annotated[
        bool,
        tyro.conf.arg(
            help="Record dataloader and pipeline timings and save to YAML files in save_dir.",
        ),
    ] = False
    num_threads: Annotated[
        int,
        tyro.conf.arg(
            help="Number of threads for the dataloader. Default: 16.",
        ),
    ] = 16
    revision: Annotated[
        str | None,
        tyro.conf.arg(
            help="HuggingFace revision for the dataset (branch, tag, or commit hash). "
            "Only used for hf:// paths."
        ),
    ] = None
    config_revision: Annotated[
        str | None,
        tyro.conf.arg(
            help="HuggingFace revision for the config (branch, tag, or commit hash). "
            "Defaults to --revision if omitted."
        ),
    ] = None
    overwrite: Annotated[
        bool,
        tyro.conf.arg(
            help="Overwrite existing output files. Default: False.",
        ),
    ] = False
    keep_dynamic_range: Annotated[
        bool,
        tyro.conf.arg(
            help="Store pipeline output as-is (float32 dB) instead of converting to uint8. "
            "Only valid when --save-as hdf5."
        ),
    ] = False


# ── Data file manipulation subcommands (``zea data …``) ───────────────────────
#
# Each dataclass's fields become CLI arguments; its ``run`` method dispatches to
# the matching operation in :mod:`zea.data.file_operations`. That module is
# imported lazily inside ``run`` so that parsing ``zea --help`` / ``zea data
# --help`` stays free of heavy imports (keras, ``zea.data`` …).


@dataclass
class _Sum:
    """Sum the raw data of multiple files or folders."""

    input_paths: tyro.conf.Positional[list[Path]]
    """Paths to the input files or folders."""
    output_path: Path
    """Output HDF5 file. Passed as ``--output-path`` because the inputs are variadic."""
    overwrite: bool = False
    """Overwrite existing output file."""

    def run(self):
        from zea.data.file_operations import sum_data

        sum_data(
            input_paths=self.input_paths, output_path=self.output_path, overwrite=self.overwrite
        )


@dataclass
class _CompoundFrames:
    """Compound frames to increase SNR."""

    input_path: tyro.conf.Positional[Path]
    """Input HDF5 file or folder."""
    output_path: tyro.conf.Positional[Path]
    """Output HDF5 file or folder."""
    overwrite: bool = False
    """Overwrite existing output file."""

    def run(self):
        from zea.data.file_operations import compound_frames

        compound_frames(
            input_path=self.input_path, output_path=self.output_path, overwrite=self.overwrite
        )


@dataclass
class _CompoundTransmits:
    """Compound transmits to increase SNR."""

    input_path: tyro.conf.Positional[Path]
    """Input HDF5 file or folder."""
    output_path: tyro.conf.Positional[Path]
    """Output HDF5 file or folder."""
    overwrite: bool = False
    """Overwrite existing output file."""

    def run(self):
        from zea.data.file_operations import compound_transmits

        compound_transmits(
            input_path=self.input_path, output_path=self.output_path, overwrite=self.overwrite
        )


@dataclass
class _Resave:
    """Resave a file to change format version."""

    input_path: tyro.conf.Positional[Path]
    """Input HDF5 file or folder."""
    output_path: tyro.conf.Positional[Path]
    """Output HDF5 file or folder."""
    overwrite: bool = False
    """Overwrite existing output file."""
    chunk_axes: tuple[str, ...] = ("n_frames",)
    """Dimension names to chunk with HDF5 chunk size 1 (others stored at full extent),
    so partial/streamed reads fetch only the requested frames. Defaults to one chunk per
    frame, mirroring zea.data.spec.DEFAULT_CHUNK_AXES
    """

    def run(self):
        from zea.data.file_operations import resave

        resave(
            input_path=self.input_path,
            output_path=self.output_path,
            overwrite=self.overwrite,
            chunk_axes=self.chunk_axes,
        )


@dataclass
class _Extract:
    """Extract subset of frames or transmits."""

    input_path: tyro.conf.Positional[Path]
    """Input HDF5 file or folder."""
    output_path: tyro.conf.Positional[Path]
    """Output HDF5 file or folder."""
    transmits: list[str] = field(default_factory=lambda: ["all"])
    """Target transmits. Can be a list of integers or ranges (e.g. 0-3 7)."""
    frames: list[str] = field(default_factory=lambda: ["all"])
    """Target frames. Can be a list of integers or ranges (e.g. 0-3 7)."""
    overwrite: bool = False
    """Overwrite existing output file."""

    def run(self):
        from zea.data.file_operations import _interpret_indices, extract_frames_transmits

        extract_frames_transmits(
            input_path=self.input_path,
            output_path=self.output_path,
            frame_indices=_interpret_indices(self.frames),
            transmit_indices=_interpret_indices(self.transmits),
            overwrite=self.overwrite,
        )


@dataclass
class _Summary:
    """Print a summary of a zea data file to the console."""

    input_path: tyro.conf.Positional[Path]
    """Input HDF5 file."""

    def run(self):
        from zea.data.file_operations import summary

        summary(input_path=self.input_path)


@dataclass
class _Copy:
    """Copy zea files or folders to a new location.

    You can specify a data key to copy only a subset of the data.
    If the destination file already exists, you can specify a mode to control
    how the data is written (append, overwrite, etc.).
    """

    src: tyro.conf.Positional[Path]
    """Source file or folder path."""
    dst: tyro.conf.Positional[Path]
    """Destination folder path."""
    key: str
    """Key to access in the HDF5 files."""
    mode: Literal["a", "w", "r+", "x"] | None = None
    """HDF5 file mode for the destination files. Defaults to auto-selection."""

    def run(self):
        from zea.data.file_operations import copy

        copy(src=self.src, dst=self.dst, key=self.key, mode=self.mode)


@dataclass
class _Virtualize:
    """Build a virtual (Zarr) reference for cloud-optimized reads.

    Reads only each file's HDF5 metadata — over HTTP for ``hf://`` inputs, so nothing is
    downloaded — and writes one JSON reference combining all files. Publish it in the
    dataset at ``virtual/index.json`` to let readers use ``Dataset(..., lazy='virtual')``,
    which fetches array data straight from chunk byte ranges.

    Files must be written with a Zarr-decodable codec (the Blosc default since zea
    0.1.3); resave older lzf files first.
    """

    input_path: tyro.conf.Positional[str]
    """Input HDF5 file, folder, or hf:// path."""
    output_path: tyro.conf.Positional[Path]
    """Output JSON reference file."""
    revision: str | None = None
    """HuggingFace revision (branch, tag, or commit hash) to pin the chunk URLs to.
    Only used for hf:// inputs. Pin to a commit hash so the reference cannot go stale."""
    overwrite: bool = False
    """Overwrite an existing reference file."""

    def run(self):
        from zea.data.file_operations import virtualize

        virtualize(
            input_path=self.input_path,
            output_path=self.output_path,
            revision=self.revision,
        )


@dataclass
class _Publish:
    """Publish a dataset to the Hugging Face Hub in the cloud-optimized layout.

    Resaves every file with the current defaults (Blosc + one chunk per frame), uploads
    them, and publishes a virtual reference pinned to that commit — so readers can use
    ``Dataset(..., lazy='virtual')``. Also the migration path for datasets published
    before zea 0.1.3 (lzf): pass their ``hf://`` path as the input.

    Writes to a remote repository (creates it if needed, then commits twice). Needs
    write access: set HF_TOKEN, or run `hf auth login`.
    """

    input_path: tyro.conf.Positional[str]
    """Dataset to publish: a local file/folder, or an hf:// path to migrate."""
    repo_id: tyro.conf.Positional[str]
    """Target HuggingFace dataset repo, e.g. zeahub/my-dataset."""
    branch: str | None = None
    """Branch to commit to. Defaults to the repo's default branch."""
    private: bool = False
    """Create the repo private, if it does not exist yet."""
    no_resave: bool = False
    """Upload the files as they are. Only for files already written with the current
    codec and chunking; the virtual reference cannot be built otherwise."""
    workdir: Path | None = None
    """Where to write the resaved files. Defaults to a temporary directory."""

    def run(self):
        from zea.data.publish import publish_dataset

        publish_dataset(
            input_path=self.input_path,
            repo_id=self.repo_id,
            resave=not self.no_resave,
            branch=self.branch,
            private=self.private,
            workdir=self.workdir,
        )


DataCommand = Union[
    Annotated[_Sum, tyro.conf.subcommand("sum")],
    Annotated[_CompoundFrames, tyro.conf.subcommand("compound_frames")],
    Annotated[_CompoundTransmits, tyro.conf.subcommand("compound_transmits")],
    Annotated[_Resave, tyro.conf.subcommand("resave")],
    Annotated[_Extract, tyro.conf.subcommand("extract")],
    Annotated[_Summary, tyro.conf.subcommand("summary")],
    Annotated[_Copy, tyro.conf.subcommand("copy")],
    Annotated[_Virtualize, tyro.conf.subcommand("virtualize")],
    Annotated[_Publish, tyro.conf.subcommand("publish")],
]


def _run_data_command(command) -> None:
    """Guard the output path (unless ``--overwrite``) and run a data subcommand.

    Read-only operations such as ``summary`` have no ``output_path`` and are never
    blocked. For folder operations the output is a directory; per-file outputs are
    still guarded inside the operation itself, so only an existing output *file* is
    blocked here.
    """
    from zea.log import logger

    output_path = getattr(command, "output_path", None)
    if (
        output_path is not None
        and Path(output_path).is_file()
        and not getattr(command, "overwrite", False)
    ):
        logger.error(f"Output file {output_path} already exists. Use --overwrite to overwrite it.")
        raise SystemExit(1)
    command.run()


@dataclass
class DataArgs:
    """Manipulate zea data files (sum, compound, resave, extract, summary, copy,
    virtualize, publish).

    All operations accept files; folder inputs are also supported. For file-to-file
    operations, each zea file in the input folder is processed and written to a
    mirrored path in the output folder.
    """

    subcommand: tyro.conf.OmitSubcommandPrefixes[DataCommand]

    def run(self) -> None:
        _run_data_command(self.subcommand)
