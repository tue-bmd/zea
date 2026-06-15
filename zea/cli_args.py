"""Lightweight CLI argument definitions for the ``zea`` command line tool.

Kept free of heavy imports (keras, ``zea.data``, …) so that ``zea --help`` and
``zea process --help`` can be rendered without loading an ML backend. This
module lives at the top level of the package (rather than under ``zea.data``)
because importing ``zea.data`` eagerly pulls in keras. The actual processing
code lives in :mod:`zea.data.process`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro


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
    save_dir: Path = Path("output")
    key: str = "data/raw_data"
    n_frames: int | None = None
    save_as: str = "gif"
    keep_keys: list[str] = field(default_factory=lambda: ["maxval"])
    timings: bool = False
    num_threads: int = 16
    revision: str | None = None
    config_revision: str | None = None
    overwrite: bool = False
    keep_dynamic_range: bool = False
    device: Annotated[
        str,
        tyro.conf.arg(
            help=(
                "Compute device ('cuda:0', 'cpu', 'auto:1', …). "
                "Only relevant when running the beamformer pipeline."
            ),
        ),
    ] = "auto:1"
