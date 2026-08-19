"""Lightweight CLI argument definitions for the ``zea`` command line tool.

Kept free of heavy imports (keras, ``zea.data``, …) so that ``zea --help`` and
``zea process --help`` can be rendered without loading an ML backend. This
module lives at the top level of the package (rather than under ``zea.data``)
because importing ``zea.data`` eagerly pulls in keras. The actual processing
code lives in :mod:`zea.data.process`.

Every subcommand dataclass exposes a ``run()`` method that imports its implementation
lazily and dispatches to it, so :mod:`zea.__main__` only has to parse and call ``run()``,
and each module's own ``python -m`` entry point can reuse the same dispatch.
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Union

import numpy as np
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

    def run(self) -> None:
        """Launch the Gradio dataset visualizer."""
        try:
            import gradio as gr
        except ImportError as exc:
            raise ImportError(
                "gradio is required for the zea app. Install with: pip install 'zea[app]'"
            ) from exc

        from zea.data.app import CSS, build_interface

        build_interface().launch(
            share=self.share,
            server_port=self.server_port,
            theme=gr.themes.Soft(primary_hue="violet", secondary_hue="yellow"),
            css=CSS,
        )


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

    def run(self) -> None:
        """Beamform the dataset with the configured pipeline."""
        from zea.data.process import run_processing

        run_processing(
            dataset_path=self.dataset,
            config_path=self.config,
            key=self.key,
            n_frames=self.n_frames,
            save_dir=self.save_dir,
            save_as=self.save_as,
            keep_keys=self.keep_keys,
            timings=self.timings,
            num_threads=self.num_threads,
            overwrite=self.overwrite,
            keep_dynamic_range=self.keep_dynamic_range,
            revision=self.revision,
            config_revision=self.config_revision,
        )


# ── Data file manipulation subcommands (``zea data …``) ───────────────────────
#
# Each dataclass's fields become CLI arguments; its ``run`` method dispatches to
# the matching operation in :mod:`zea.data.file_operations`. That module is
# imported lazily inside ``run`` so that parsing ``zea --help`` / ``zea data
# --help`` stays free of heavy imports (keras, ``zea.data`` …).


@dataclass
class _Sum:
    """Sum the raw data of multiple files or folders."""

    input_paths: tyro.conf.Positional[list[str]]
    """Paths to the input files or folders. Also accepts 'hf://' paths."""
    output_path: str
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

    input_path: tyro.conf.Positional[str]
    """Input HDF5 file or folder. Also accepts an 'hf://' path."""
    output_path: tyro.conf.Positional[str]
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

    input_path: tyro.conf.Positional[str]
    """Input HDF5 file or folder. Also accepts an 'hf://' path."""
    output_path: tyro.conf.Positional[str]
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

    input_path: tyro.conf.Positional[str]
    """Input HDF5 file or folder. Also accepts an 'hf://' path."""
    output_path: tyro.conf.Positional[str]
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

    input_path: tyro.conf.Positional[str]
    """Input HDF5 file or folder. Also accepts an 'hf://' path."""
    output_path: tyro.conf.Positional[str]
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

    input_path: tyro.conf.Positional[str]
    """Input HDF5 file. Also accepts an 'hf://' path."""

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

    src: tyro.conf.Positional[str]
    """Source file or folder path. Also accepts an 'hf://' path."""
    dst: tyro.conf.Positional[str]
    """Destination folder path."""
    key: str
    """Key to access in the HDF5 files."""
    mode: Literal["a", "w", "r+", "x"] | None = None
    """HDF5 file mode for the destination files. Defaults to auto-selection."""

    def run(self):
        from zea.data.file_operations import copy

        copy(src=self.src, dst=self.dst, key=self.key, mode=self.mode)


@dataclass
class _DecodeHadamard:
    """Decode Hadamard-encoded data in a zea file or folder."""

    input_path: tyro.conf.Positional[Path]
    """Input HDF5 file or folder."""
    output_path: tyro.conf.Positional[Path]
    """Output HDF5 file or folder."""
    overwrite: bool = False
    """Overwrite existing output file."""

    def run(self):
        from zea.data.file_operations import decode_hadamard_file_operation

        decode_hadamard_file_operation(
            input_path=self.input_path,
            output_path=self.output_path,
            overwrite=self.overwrite,
        )


@dataclass
class _SAToVirtualFocus:
    """Convert a synthetic aperture dataset to a virtual focus dataset."""

    input_path: tyro.conf.Positional[Path]
    """Input HDF5 file or folder."""
    output_path: tyro.conf.Positional[Path]
    """Output HDF5 file or folder."""
    polar_angle_deg: float = 0.0
    """Polar angle for the virtual focus."""
    azimuth_angle_deg: float = 0.0
    """Azimuth angle for the virtual focus."""
    focus_distance: float = float("inf")
    """Focus distance for the virtual focus."""
    transmit_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Origin of the transmit."""
    tx_apodization: str = "kaiser"
    """Type of transmit apodization to apply."""
    overwrite: bool = False
    """Overwrite existing output file."""

    def run(self):
        from zea.data.file_operations import sa_to_virtual_focus

        sa_to_virtual_focus(
            polar_angle=np.deg2rad(self.polar_angle_deg),
            azimuth_angle=np.deg2rad(self.azimuth_angle_deg),
            focus_distance=self.focus_distance,
            transmit_origin=self.transmit_origin,
            tx_apodization=self.tx_apodization,
            input_path=self.input_path,
            output_path=self.output_path,
            overwrite=self.overwrite,
        )


DataCommand = Union[
    Annotated[_Sum, tyro.conf.subcommand("sum")],
    Annotated[_CompoundFrames, tyro.conf.subcommand("compound_frames")],
    Annotated[_CompoundTransmits, tyro.conf.subcommand("compound_transmits")],
    Annotated[_Resave, tyro.conf.subcommand("resave")],
    Annotated[_Extract, tyro.conf.subcommand("extract")],
    Annotated[_Summary, tyro.conf.subcommand("summary")],
    Annotated[_Copy, tyro.conf.subcommand("copy")],
    Annotated[_DecodeHadamard, tyro.conf.subcommand("decode_hadamard")],
    Annotated[_SAToVirtualFocus, tyro.conf.subcommand("sa_to_virtual_focus")],
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
    dst = getattr(command, "dst", None)
    if (output_path is not None and str(output_path).startswith("hf://")) or (
        dst is not None and str(dst).startswith("hf://")
    ):
        logger.error("Output path cannot be an 'hf://' path; 'hf://' is only supported for inputs.")
        raise SystemExit(1)
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
    """Manipulate zea data files (sum, compound, resave, extract, summary, copy).

    All operations accept files; folder inputs are also supported. For file-to-file
    operations, each zea file in the input folder is processed and written to a
    mirrored path in the output folder. Inputs also accept ``hf://`` paths (a single
    file or a folder in a Hugging Face dataset repo); outputs must be local paths.
    """

    subcommand: tyro.conf.OmitSubcommandPrefixes[DataCommand]

    def run(self) -> None:
        _run_data_command(self.subcommand)


# ── Interactive tooling subcommands (``zea tools …``) ─────────────────────────
#
# Only the tools in :mod:`zea.tools` that are meant to be driven from a terminal are
# exposed here. The heavy implementation is imported lazily inside ``run`` so that
# ``zea --help`` / ``zea tools --help`` stay free of keras and matplotlib imports.


@dataclass
class _Select:
    """Interactively select regions of interest in images or a sequence.

    With one or more images, two regions are selected in the first image and compared
    across all of them with ``--metric``. With a single video, gif or zea file, a number
    of key frames are annotated, the masks are interpolated over all frames and written
    next to the input as a zea HDF5 file (image + segmentation), together with a preview
    animation.

    Any option that is omitted is asked for interactively, so ``zea tools select``
    without arguments walks you through the whole process.
    """

    files: tyro.conf.Positional[list[str]] = field(default_factory=list)
    """Input images (png/jpg), or a single video / gif or zea HDF5 file. zea files also
    accept an 'hf://' path (parsed as a str so the '//' is not collapsed). Omit to pick
    files through a file dialog."""
    selector: Literal["rectangle", "lasso"] | None = None
    """Shape of the selection tool. Asked for interactively when omitted."""
    title: str | None = None
    """Name of what is being selected. Used as the segmentation label and in the output
    filenames (sequence mode only). Asked for interactively when omitted."""
    num_selections: int | None = None
    """Number of key frames to annotate before interpolating (sequence mode only). Asked
    for interactively when omitted."""
    fps: int | None = None
    """Frame rate of the saved preview animation (sequence mode only). Asked for
    interactively when omitted."""
    metric: str | None = "gcnr"
    """Metric from ``zea.metrics`` computed between the two selections (image mode
    only). Pass ``None`` to skip the metric."""
    output_dir: Path | None = None
    """Directory the annotations and animation are written to. Defaults to the folder of
    the input file, or the working directory for 'hf://' inputs."""
    animation: bool = True
    """Save a preview gif of the interpolated masks (sequence mode only)."""
    confirm: bool = True
    """Ask to confirm each selection before moving on, with a keypress in the plot
    window."""
    overwrite: bool = False
    """Overwrite existing output files."""

    def run(self):
        from zea.tools.selection_tool import run_selection_tool

        run_selection_tool(
            files=self.files,
            selector=self.selector,
            title=self.title,
            num_selections=self.num_selections,
            fps=self.fps,
            metric=self.metric,
            output_dir=self.output_dir,
            save_animation=self.animation,
            confirm_selection=self.confirm,
            overwrite=self.overwrite,
        )


ToolsCommand = Union[
    Annotated[_Select, tyro.conf.subcommand("select")],
    # A one-element ``Union`` collapses to the element itself, which would inline the
    # tool's arguments into ``zea tools`` rather than put them behind ``select``. The
    # suppressed ``None`` keeps this a real subcommand union; drop it once a second
    # tool is added here.
    Annotated[None, tyro.conf.Suppress],
]


@dataclass
class ToolsArgs:
    """Interactive tools from :mod:`zea.tools`.

    These are GUI-driven helpers rather than batch operations; they open matplotlib
    windows and file dialogs, so they need a display::

        zea tools select                  # pick files through a dialog
        zea tools select a.png b.png      # compare two images with gCNR
        zea tools select clip.mp4         # annotate a video
        zea tools select scan.hdf5        # annotate a zea file

    Run ``zea tools <tool> --help`` for the per-tool options.
    """

    subcommand: tyro.conf.OmitSubcommandPrefixes[ToolsCommand]

    def run(self) -> None:
        # The `None` padding in ToolsCommand is suppressed, so tyro never selects it.
        assert self.subcommand is not None, "No tool selected."
        self.subcommand.run()


# ── Dataset conversion subcommands (``zea convert …``) ────────────────────────
#
# Thin CLI dataclasses for converting raw open-source ultrasound datasets to the
# zea format. Each dataset's fields become CLI arguments; its ``run`` method
# imports the heavy converter from :mod:`zea.data.convert` lazily and dispatches to
# it. They live here (rather than under ``zea.data.convert``) so that ``zea
# --help`` / ``zea convert --help`` render without importing an ML backend —
# importing ``zea.data`` eagerly pulls in keras.


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


ConvertDataset = Union[
    Annotated[_Echonet, tyro.conf.subcommand("echonet")],
    Annotated[_EchonetLVH, tyro.conf.subcommand("echonetlvh")],
    Annotated[_Camus, tyro.conf.subcommand("camus")],
    Annotated[_Cetus, tyro.conf.subcommand("cetus")],
    Annotated[_Picmus, tyro.conf.subcommand("picmus")],
    Annotated[_Verasonics, tyro.conf.subcommand("verasonics")],
    Annotated[_EchoXFlow, tyro.conf.subcommand("echoxflow")],
]


@dataclass
class ConvertArgs:
    """Convert raw open-source ultrasound datasets to the zea format.

    Pick a dataset subcommand and provide the source and destination folders::

        zea convert camus ./raw ./output --download
        zea convert echonet ./raw ./output
        zea convert echoxflow ./raw ./output

    Run ``zea convert <dataset> --help`` for the per-dataset options.
    """

    subcommand: tyro.conf.OmitSubcommandPrefixes[ConvertDataset]

    def run(self) -> None:
        self.subcommand.run()
