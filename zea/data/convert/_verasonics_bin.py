"""Read Verasonics RF buffers stored as external ``.bin`` files instead of in ``RcvData``.

Saving a large ``RcvData`` to a ``-v7.3`` ``.mat`` is slow. Some setups therefore
write each RF buffer to its own raw binary file and save a ``.mat`` without
``RcvData``. This module reads that layout. A normal workspace ``.mat`` never
reaches this code.

Layout of a measurement directory::

    measurement_dir/
    ├── <any name>.mat   # -v7.3 workspace without RcvData, plus:
    │     RF_rows   (1, n_buf)       fast-time samples per buffer
    │     RF_cols   (1, n_buf)       number of saved channels per buffer
    │     RF_frames (1, n_buf)       frames per buffer
    │     NonzeroRFcolumns {1, n_buf} (optional) logical mask per buffer. Its
    │                                length is the number of hardware channels
    │                                and it is true for each saved channel.
    ├── RF_data_1.bin    # RcvData{1}: saved channels only, int16 little-endian,
    ├── RF_data_2.bin    #   MATLAB column-major (RF_rows x RF_cols x RF_frames)
    └── ...

A buffer without a ``.bin`` file was not saved, so it is left out.
"""

from pathlib import Path

import numpy as np

from zea import log

_BINARY_TEMPLATE = "RF_data_{k}.bin"


def is_binary_workspace(file):
    """Whether an open ``VerasonicsFile`` stores its RF in external ``.bin`` files."""
    return "RcvData" not in file and "RF_frames" in file


def _binary_path(file, buffer_index):
    # The files use the 1-based MATLAB buffer number.
    directory = Path(file.filename).resolve().parent
    return directory / _BINARY_TEMPLATE.format(k=buffer_index + 1)


def _read_dimension(file, name, buffer_index):
    values = np.asarray(file[name][:]).reshape(-1)
    if buffer_index >= values.size:
        raise IndexError(f"buffer_index={buffer_index} is out of range for '{name}'.")
    return int(values[buffer_index])


def available_buffers(file):
    """0-based indices of the buffers that have a ``.bin`` file."""
    n_buffers = np.asarray(file["RF_frames"][:]).size
    return [i for i in range(n_buffers) if _binary_path(file, i).is_file()]


def read_buffer(file, buffer_index):
    """Read one buffer as ``(n_frames, n_hardware_channels, n_samples)`` int16, like ``RcvData``.

    Channels that were not saved are filled with zeros.
    """
    path = _binary_path(file, buffer_index)
    if not path.is_file():
        saved = [i + 1 for i in available_buffers(file)]
        raise FileNotFoundError(
            f"RF buffer {buffer_index + 1} was not saved: {path.name} does not exist. "
            f"Saved buffers (1-based): {saved}."
        )

    rows = _read_dimension(file, "RF_rows", buffer_index)
    cols = _read_dimension(file, "RF_cols", buffer_index)
    n_frames = _read_dimension(file, "RF_frames", buffer_index)

    raw = np.fromfile(path, dtype="<i2")
    if raw.size != rows * cols * n_frames:
        raise ValueError(
            f"{path.name} has {raw.size} int16 values but RF_rows*RF_cols*RF_frames = "
            f"{rows}*{cols}*{n_frames}. The binary and the stored dimensions are inconsistent."
        )
    # Column-major (rows, cols, frames) -> (frames, cols, rows).
    raw = raw.reshape((rows, cols, n_frames), order="F").transpose(2, 1, 0)

    if "NonzeroRFcolumns" not in file:
        log.warning(
            "'NonzeroRFcolumns' not found; assuming the binary holds all hardware "
            "channels in order."
        )
        return np.ascontiguousarray(raw)

    mask = np.asarray(file.dereference_index(file["NonzeroRFcolumns"], buffer_index))
    mask = mask.reshape(-1).astype(bool)
    if int(mask.sum()) != cols:
        raise ValueError(
            f"NonzeroRFcolumns for buffer {buffer_index + 1} marks {int(mask.sum())} "
            f"channels, but RF_cols={cols}."
        )
    full = np.zeros((n_frames, mask.size, rows), dtype=np.int16)
    full[:, mask, :] = raw
    return full
