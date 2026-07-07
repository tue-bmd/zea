"""Convert us4us (arrus + gui4us) pickle files to the zea format.

This converter supports pickle datasets acquired with ARRUS <= 0.14.x.

NOTE: this converter works ONLY with the single-axis array probes (linear/convex/phase/ring/etc.).

The arrus+gui4us software (https://us4us.eu) stores acquired ultrasound data as
Python pickle files containing:

- ``data["data"]``: a list of N frames, each frame is a tuple of M numpy arrays
  (one per pipeline output, e.g. image, beamformed IQ, raw channel data).
- ``data["metadata"]``: a tuple of M :class:`ConstMetadata` (ARRUS python package)
  objects, one per pipeline output, describing the acquisition context,
  probe model and sequence.

Example usage::

    # Single file → single file
    python -m zea.data.convert us4us input.pkl output.hdf5 \
        --mapping '{"0": "image", "1": "beamformed_data", "2": "raw_data"}'

    # Directory of .pkl files → directory of .hdf5 files
    # Each <name>.pkl in <src_dir> is written as <dst_dir>/<name>.hdf5.
    python -m zea.data.convert us4us src_dir/ dst_dir/ \
        --mapping '{"0": "image"}'

The ``mapping`` argument maps each us4us output index (position in the per-frame
tuple in the pickle dataset) to a zea data type string.

Default: ``{0: "image"}``.

The following mapping from arrus (<=0.14.x) metadata to zea scan/probe fields:

- ``metadata.context.raw_sequence.ops[j].tx.delays``  →  ``t0_delays``
- ``metadata.context.raw_sequence.ops[j].tx.aperture``  →  ``tx_apodizations``
- ``metadata.context.raw_sequence.ops[j].rx.sample_range``  →  ``initial_times``
- ``metadata.context.raw_sequence.ops[j].pri``  →  ``time_to_next_transmit``
- ``focus_distances`` is taken from ``metadata.context.sequence.tx_focus`` when
  the user-facing sequence is a ``SimpleTxRxSequence``
  (``LinSequence`` / ``PwiSequence`` / ``StaSequence``) and from
  ``metadata.context.raw_sequence.ops[j].tx.focus`` when it is a bare
  ``TxRxSequence``.
- ``polar_angles`` is taken from ``metadata.context.sequence.angles`` for a
  ``SimpleTxRxSequence`` and from ``metadata.context.raw_sequence.ops[j].tx.angle``
  for a ``TxRxSequence``.
- ``metadata.data_description.sampling_frequency``  →  ``sampling_frequency``
- ``metadata.context.device.probe[0].model``  →  probe geometry / element_width
- ``metadata.context.medium.speed_of_sound``  →  ``sound_speed``
"""

import pickle
from pathlib import Path

import numpy as np

from zea import log
from zea.data.file import File
from zea.data.spec import DEFAULT_COMPRESSION


# Converter-local allowlist of zea data types this module actually knows how
# to build in :func:`_prepare_output` and hand to :meth:`zea.File.create`.
# Kept intentionally narrower than ``DataSpec.SCHEMA``: the broader schema
# contains data types that no branch of ``_prepare_output`` produces, so
# accepting them here would silently fall through to the generic fallback and
# emit arrays that neither ``File.create`` nor downstream readers understand
# for that slot.  Any new supported type must both get an explicit branch in
# ``_prepare_output`` *and* be added here.
_SUPPORTED_MAPPING_DATA_TYPES: frozenset[str] = frozenset({
    "raw_data",
    "image",
    "beamformed_data",
    "envelope_data",
    "aligned_data",
})


# ---------------------------------------------------------------------------
# Stub / allowlisted unpickler
#
# TRUST BOUNDARY:
# Pickle deserialization is not a safe format for arbitrary input -- every
# ``GLOBAL`` / ``STACK_GLOBAL`` opcode resolves a class that can run code
# through ``__reduce__`` / ``__setstate__`` / ``__new__`` hooks.  To keep this
# converter usable without the third-party ``arrus`` Python package installed
# (and to prevent a hostile ``.pkl`` from popping a shell via
# ``os.system``/``subprocess.Popen``/etc.), we restrict ``find_class`` to an
# explicit allowlist:
#
#   * any ``arrus.*`` global is replaced by :class:`_ArrusStub`;
#   * any ``numpy.*`` global is resolved normally (arrays / dtypes / scalars);
#   * everything else raises :class:`pickle.UnpicklingError` before the
#     class is looked up.
#
# Even with this allowlist, only run this loader against ``.pkl`` files
# produced by us4us / arrus acquisition pipelines that you control.  Do not
# point it at ``.pkl`` files received from untrusted third parties.
# ---------------------------------------------------------------------------
_ALLOWED_PICKLE_MODULE_PREFIXES = ("numpy",)


class _ArrusStub:
    """Namespace placeholder for any arrus class encountered during unpickling."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


class _ArrusUnpickler(pickle.Unpickler):
    """Allowlisted unpickler for us4us pickle files.

    Swaps ``arrus.*`` classes for :class:`_ArrusStub`, resolves ``numpy.*``
    globals normally, and refuses everything else so that malicious payloads
    referencing e.g. ``os.system`` or ``subprocess.Popen`` cannot execute.
    """

    def find_class(self, module, name):
        if "arrus" in module:
            return _ArrusStub
        if module == "numpy" or any(
            module.startswith(p + ".") for p in _ALLOWED_PICKLE_MODULE_PREFIXES
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to load class {module}.{name} from us4us pickle: "
            "module is not in the us4us converter allowlist. Only trusted "
            "us4us .pkl files should be processed by this loader."
        )


# ---------------------------------------------------------------------------
# Pickle loading
# ---------------------------------------------------------------------------
def _load_us4us_pickle(path: Path) -> dict:
    """Load an us4us pickle with an allowlisted unpickler.

    .. warning::
        Pickle is not a safe format for arbitrary input.  Even with the
        allowlist enforced by :class:`_ArrusUnpickler`, only invoke this on
        ``.pkl`` files produced by us4us / arrus acquisition pipelines that
        you control — never on files received from untrusted third parties.
    """
    with open(path, "rb") as f:
        return _ArrusUnpickler(f).load()


# ---------------------------------------------------------------------------
# Probe extraction
# ---------------------------------------------------------------------------
def _extract_probe_dict(probe_dto, ops) -> dict:
    """
    Builds a :class:`~zea.data.spec.ProbeSpec`-compatible dict
    from an arrus.devices.probe.ProbeDTO stub.
    """
    model = probe_dto.model
    n_el = int(model.n_elements)

    x = np.asarray(model.element_pos_x, dtype=np.float32).ravel()
    z = np.asarray(model.element_pos_z, dtype=np.float32).ravel()
    y = np.zeros(n_el, dtype=np.float32)
    probe_geometry = np.stack([x, y, z], axis=1)  # (n_el, 3)

    probe_dict: dict = {"probe_geometry": probe_geometry}

    if hasattr(model, "pitch") and model.pitch:
        probe_dict["element_width"] = np.float32(model.pitch)

    # Probe type
    cr = getattr(model, "curvature_radius", None)
    if cr is not None:
        probe_dict["type"] = "linear" if float(cr) == 0.0 else "curved"

    # Name from model_id if available
    model_id = getattr(model, "model_id", None)
    probe_dict["name"] = getattr(model_id, "name", "generic") or "generic"

    # Heuristic: center frequency from the TX excitation
    # TODO: the below heuristic is not the best way to handle that,
    # and in the future consider using tx_frequency_range (currently not
    # available in the Python api).
    if ops:
        exc = ops[0].tx.excitation
        cf = getattr(exc, "center_frequency", None)
        if cf is not None:
            probe_dict["probe_center_frequency"] = np.float32(cf)

    # Lens parameters
    lens = getattr(model, "lens", None)
    if lens is not None:
        t = getattr(lens, "thickness", None)
        c = getattr(lens, "speed_of_sound", None)
        if t and float(t) > 0:
            probe_dict["lens_thickness"] = np.float32(t)
        if c and float(c) > 0:
            probe_dict["lens_sound_speed"] = np.float32(c)

    return probe_dict


# ---------------------------------------------------------------------------
# TX/RX sequence -> scan
# ---------------------------------------------------------------------------
def _extract_scan_dict(context, data_char, n_frames: int) -> dict:
    """
    Build a :class:`~zea.data.spec.ScanSpec`-compatible dict from arrus context.
    """
    raw_seq = context.raw_sequence
    sequence = context.sequence
    ops = raw_seq.ops
    n_tx = len(ops)

    device = context.device
    model = device.probe[0].model
    n_el = int(model.n_elements)

    x_pos = np.asarray(model.element_pos_x, dtype=np.float64).ravel()
    z_pos = np.asarray(model.element_pos_z, dtype=np.float64).ravel()

    sampling_frequency = np.float32(data_char.sampling_frequency)

    # Center / demodulation frequency from TX excitation
    exc = ops[0].tx.excitation
    center_frequency = np.float32(getattr(exc, "center_frequency", 0.0))
    demodulation_frequency = center_frequency

    # -----------------------------------------------------------------------
    # t0_delays and tx_apodizations — (n_tx, n_el)
    # -----------------------------------------------------------------------
    t0_delays = np.zeros((n_tx, n_el), dtype=np.float64)
    tx_apodizations = np.zeros((n_tx, n_el), dtype=np.float32)

    for i, op in enumerate(ops):
        aperture = np.asarray(op.tx.aperture, dtype=bool)
        delays = np.asarray(op.tx.delays, dtype=np.float64)
        t0_delays[i, aperture] = delays
        tx_apodizations[i, aperture] = 1.0

    # Shift each transmit so the first active element fires at t = 0, then
    # clip to suppress floating-point underflow below zero.
    for i in range(n_tx):
        active = tx_apodizations[i] > 0
        if np.any(active):
            t0_delays[i] -= t0_delays[i, active].min()

    t0_delays = np.clip(t0_delays, 0.0, None).astype(np.float32)

    # -----------------------------------------------------------------------
    # initial_times — (n_tx,)
    # -----------------------------------------------------------------------
    initial_times = np.zeros(n_tx, dtype=np.float32)
    for i, op in enumerate(ops):
        sr = getattr(op.rx, "sample_range", None)
        if sr is not None:
            initial_times[i] = np.float32(sr[0] / sampling_frequency)

    # -----------------------------------------------------------------------
    # focus_distances — (n_tx,)
    # -----------------------------------------------------------------------
    # SimpleTxRxSequence (LinSequence/PwiSequence/StaSequence) carries a
    # scalar `tx_focus` on the sequence object. A bare TxRxSequence stores
    # `focus` per-op on each Tx (or None when the user supplied raw `delays`).
    tx_focus = getattr(sequence, "tx_focus", None)
    if tx_focus is not None:
        focus_distances = np.full(n_tx, float(tx_focus), dtype=np.float32)
    else:
        focus_distances = np.array(
            [
                float(op.tx.focus) if getattr(op.tx, "focus", None) is not None else np.inf
                for op in ops
            ],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------------
    # transmit_origins — (n_tx, 3): centre of active TX aperture
    # -----------------------------------------------------------------------
    transmit_origins = np.zeros((n_tx, 3), dtype=np.float32)
    for i, op in enumerate(ops):
        aperture = np.asarray(op.tx.aperture, dtype=bool)
        if np.any(aperture):
            transmit_origins[i, 0] = float(x_pos[aperture].mean())
            transmit_origins[i, 2] = float(z_pos[aperture].mean())

    # -----------------------------------------------------------------------
    # polar_angles — (n_tx,)
    # -----------------------------------------------------------------------
    # SimpleTxRxSequence exposes `angles` (scalar or per-TX list) on the
    # sequence. A bare TxRxSequence stores `angle` per-op on each Tx.
    seq_angles = getattr(sequence, "angles", None)
    if seq_angles is not None:
        angles_arr = np.atleast_1d(np.asarray(seq_angles, dtype=np.float32)).ravel()
        if angles_arr.size == n_tx:
            polar_angles = angles_arr
        else:
            polar_angles = np.full(n_tx, float(angles_arr[0]), dtype=np.float32)
    else:
        polar_angles = np.array(
            [
                float(op.tx.angle) if getattr(op.tx, "angle", None) is not None else 0.0
                for op in ops
            ],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------------
    # time_to_next_transmit — (n_frames, n_tx)
    # -----------------------------------------------------------------------
    pris = np.array([op.pri for op in ops], dtype=np.float32)
    time_to_next_transmit = np.tile(pris[np.newaxis, :], (n_frames, 1))

    scan_dict: dict = {
        "sampling_frequency": sampling_frequency,
        "center_frequency": center_frequency,
        "demodulation_frequency": demodulation_frequency,
        "initial_times": initial_times,
        "t0_delays": t0_delays,
        "tx_apodizations": tx_apodizations,
        "focus_distances": focus_distances,
        "transmit_origins": transmit_origins,
        "polar_angles": polar_angles,
        "time_to_next_transmit": time_to_next_transmit,
    }

    medium = getattr(context, "medium", None)
    if medium is not None:
        sos = getattr(medium, "speed_of_sound", None)
        if sos is not None:
            scan_dict["sound_speed"] = np.float32(sos)

    return scan_dict


# ---------------------------------------------------------------------------
# Data (coordinates, etc.)
# ---------------------------------------------------------------------------
def _get_image_coordinates(data_char) -> "np.ndarray | None":
    """
    Try to extract a (n_z, n_x, 3) coordinate grid from arrus spacing metadata.
    """
    spacing = getattr(data_char, "spacing", None)
    if spacing is None:
        print("NO SPACING")
        return None
    coords = getattr(spacing, "coordinates", None)
    if coords is None or len(coords) < 2:
        print("NO COORDS")
        return None
    try:
        z_vals = np.asarray(coords[0], dtype=np.float32).ravel()
        x_vals = np.asarray(coords[1], dtype=np.float32).ravel()
        Z, X = np.meshgrid(z_vals, x_vals, indexing="ij")
        Y = np.zeros_like(Z)
        return np.stack([X, Y, Z], axis=-1)  # (n_z, n_x, 3)
    except Exception as exc:
        print("EXCEPTION WHILE CALCULATING coords")
        log.warning(f"Could not build image coordinates: {exc}")
        return None


def _prepare_output(
    arrays, output_idx: int, data_type: str, meta_entry, raw_ops=None
) -> "dict | np.ndarray":
    """Stack per-frame arrays and reshape into the expected zea format.

    Args:
        arrays: List of per-frame tuples.
        output_idx: Index into each tuple.
        data_type: Target zea data type string.
        meta_entry: Corresponding :class:`ConstMetadata` stub for this output.
        raw_ops: Optional list of arrus ``TxRx`` stubs from the raw sequence.
            When provided, ``raw_data`` arrays are reconstructed from
            the hardware sub-aperture back to the full-probe element count
            using ``rx.aperture`` and ``rx.padding``.

    Returns:
        A numpy array (for ``raw_data``) or a dict with at least a ``"values"``
        key (for all map-based types).
    """
    raw = [frame[output_idx] for frame in arrays]
    sample = raw[0]

    if data_type == "raw_data":
        # arrus: (1, n_tx, n_ax, rx_aperture_size) → zea: (n_frames, n_tx, n_ax, n_el, 1)
        stacked = np.stack([a.squeeze(0) for a in raw], axis=0)
        # stacked: (n_frames, n_tx, n_ax, rx_aperture_size)

        if raw_ops is not None:
            # Distribute receive aperture elements back to full probe aperture.
            # Each firing i has:
            #   rx.padding = (left_pad, right_pad) — zeros prepended/appended
            #   rx.aperture — bool mask over all probe elements; True = active
            # Channel layout in stacked[..., i, :, :, :]:
            #   [left_pad zeros] [n_active real samples] [right_pad zeros]
            n_el_full = len(raw_ops[0].rx.aperture)
            full = np.zeros((*stacked.shape[:3], n_el_full), dtype=stacked.dtype)
            for i, op in enumerate(raw_ops):
                left_pad = int(op.rx.padding[0])
                rx_aper = np.asarray(op.rx.aperture, dtype=bool)
                active_idx = np.where(rx_aper)[0]
                # Assign one element at a time to avoid NumPy's fancy-index
                # dimension transposition when mixing integer and array indices.
                for k, el in enumerate(active_idx):
                    full[:, i, :, int(el)] = stacked[:, i, :, left_pad + k]
            stacked = full

        return stacked[..., np.newaxis]

    elif data_type == "image":
        # arrus: a sequence of (n_z, n_x) → zea Image: (n_frames, n_z, n_x) float32 <= 0
        stacked = np.stack(raw, axis=0).astype(np.float32)
        # Shift so the maximum value equals 0 (dB scale requirement)
        max_val = float(stacked.max())
        if max_val > 0:
            stacked = stacked - max_val
        result: dict = {"values": stacked}
        coords = _get_image_coordinates(meta_entry._data_char)
        if coords is not None:
            result["coordinates"] = coords
        return result

    elif data_type == "beamformed_data":
        # arrus IQ: a sequence of (1, n_tx, n_ax) complex64
        # → zea BeamformedData: (n_frames, n_ax, n_tx, 2) = (n_frames, z, x, n_ch)
        stacked = np.stack([a.squeeze(0) for a in raw], axis=0)  # (n_frames, n_tx, n_ax)
        stacked = stacked.transpose(0, 2, 1)  # (n_frames, n_ax=z, n_tx=x)
        iq = np.stack([stacked.real, stacked.imag], axis=-1).astype(np.float32)
        return {"values": iq, "labels": np.array(["I", "Q"], dtype=np.str_)}

    elif data_type == "envelope_data":
        # arrus: a sequence of (1, n_tx, n_ax) real
        # → zea EnvelopeData: (n_frames, n_ax, n_tx)
        stacked = np.stack([a.squeeze(0) for a in raw], axis=0)
        if np.iscomplexobj(stacked):
            stacked = np.abs(stacked)
        stacked = stacked.transpose(0, 2, 1).astype(np.float32)
        return {"values": stacked}
    else:
        # Generic fallback: squeeze a leading size-1 dim if present, stack frames
        if sample.ndim > 1 and sample.shape[0] == 1:
            stacked = np.stack([a.squeeze(0) for a in raw], axis=0)
        else:
            stacked = np.stack(raw, axis=0)
        return {"values": stacked}


# ---------------------------------------------------------------------------
# Metadata selection helper
# ---------------------------------------------------------------------------
def _pick_scan_metadata(metadata_tuple, mapping: dict):
    """Return the best metadata entry for scan parameter extraction.

    Prefers the entry mapped to ``"raw_data"`` (highest ADC fidelity); falls
    back to the entry with the highest reported sampling frequency, then to the
    first available entry.
    """
    # Prefer raw_data mapping
    for idx, dtype in mapping.items():
        if dtype == "raw_data":
            return metadata_tuple[idx]

    # Fallback: highest sampling frequency → most likely raw ADC data
    best = metadata_tuple[0]
    best_fs = getattr(getattr(best, "_data_char", None), "sampling_frequency", 0) or 0
    for meta in metadata_tuple[1:]:
        fs = getattr(getattr(meta, "_data_char", None), "sampling_frequency", 0) or 0
        if fs > best_fs:
            best_fs = fs
            best = meta

    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _convert_single_us4us_pickle(
    src: Path, dst: Path, mapping: dict, *, overwrite: bool = False
) -> None:
    """Convert one us4us ``.pkl`` file to a single zea ``.hdf5`` file.

    Args:
        src: Source ``.pkl`` file.
        dst: Destination ``.hdf5`` file.
        mapping: Maps each pipeline output index to a zea data type string.
        overwrite: When ``True``, an existing ``dst`` file is replaced.
            When ``False`` (the default), :meth:`zea.File.create` raises
            rather than silently overwriting an existing converted output.
    """
    log.info(f"Loading us4us pickle: {log.yellow(src)}")
    data = _load_us4us_pickle(src)

    frames_data = data["data"]        # list[tuple[np.ndarray, ...]]
    metadata_tuple = data["metadata"]  # tuple[ConstMetadata, ...]
    n_frames = len(frames_data)

    log.info(f"Frames: {n_frames}, "
             f"pipeline outputs: {len(metadata_tuple)}, "
             f"mapping: {mapping}")

    # Scan and probe parameters come from the most informative metadata entry
    scan_meta = _pick_scan_metadata(metadata_tuple, mapping)
    context = scan_meta._context
    data_char = scan_meta._data_char

    probe_dto = context.device.probe[0]
    probe_dict = _extract_probe_dict(probe_dto, context.raw_sequence.ops)
    scan_dict = _extract_scan_dict(context, data_char, n_frames)

    # Reject duplicate target data types: two pipeline outputs mapped to the
    # same zea data type (e.g. ``{"0": "image", "1": "image"}``) would silently
    # overwrite each other in ``outputs`` below, since ``outputs`` is keyed by
    # ``data_type``.  The current data model has one slot per data type, so
    # this must be rejected up-front rather than emitting a lossy file.
    seen: dict[str, int] = {}
    for output_idx, data_type in mapping.items():
        if data_type in seen:
            raise ValueError(
                f"mapping assigns data type {data_type!r} to multiple pipeline "
                f"outputs (indices {seen[data_type]} and {output_idx}). Each "
                "zea data type can only be written once per file; pick a "
                "different data type for one of them (e.g. 'image' vs "
                "'beamformed_data')."
            )
        seen[data_type] = output_idx

    # Build per-output data dicts
    raw_ops = context.raw_sequence.ops
    outputs: dict[str, dict | np.ndarray] = {}

    for output_idx, data_type in mapping.items():
        log.info(f"  output[{output_idx}] → {data_type}")
        meta_entry = metadata_tuple[output_idx]
        outputs[data_type] = _prepare_output(
            frames_data,
            output_idx,
            data_type,
            meta_entry,
            raw_ops=raw_ops if data_type == "raw_data" else None,
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing zea file: {log.yellow(dst)}")

    if len(outputs) == 1:
        # Single output: flat single-track layout
        data_type, arr = next(iter(outputs.items()))
        File.create(
            path=dst,
            data={data_type: arr},
            scan=scan_dict,
            probe=probe_dict,
            description="us4us (gui4us+arrus) data converted to zea format",
            us_machine="us4R",
            compression=DEFAULT_COMPRESSION,
            overwrite=overwrite,
        )
    else:
        # NOTE: us4us "raw_data" can be either RF or IQ data. The following
        # combination (RF raw_data, IQ beamformed_data) will cause
        # "incosistent sizes" error (zea 0.1.1).
        tracks = [
            {"data": {data_type: arr}, "scan": scan_dict, "label": data_type}
            for data_type, arr in outputs.items()
        ]
        File.create(
            path=dst,
            tracks=tracks,
            probe=probe_dict,
            description="us4us (gui4us+arrus) data converted to zea format",
            us_machine="us4R",
            compression=DEFAULT_COMPRESSION,
            overwrite=overwrite,
        )

    log.success(f"Converted {log.yellow(src)} → {log.yellow(dst)}")


def convert_us4us(args):
    """Convert one or more us4us (arrus) pickle files to the zea HDF5 format.

    Args:
        args (argparse.Namespace): An object with attributes:

            - src (str | Path): Source ``.pkl`` file, or a directory containing
              one or more ``.pkl`` files (matched by ``*.pkl`` at the top level).
            - dst (str | Path): Destination ``.hdf5`` file when ``src`` is a
              file, or destination directory when ``src`` is a directory. In
              the directory case each input ``<name>.pkl`` is written as
              ``<dst>/<name>.hdf5``.
            - mapping (dict, optional): Maps each output index (integer) in
              the per-frame tuple to a zea data type string, e.g.
              ``{0: "image", 1: "beamformed_data", 2: "raw_data"}``. Supported
              types: ``raw_data``, ``image``, ``beamformed_data``,
              ``envelope_data``, ``aligned_data``. Defaults to ``{0: "image"}``.
            - overwrite (bool, optional): When ``True``, existing destination
              ``.hdf5`` files are replaced. Defaults to ``False`` so previously
              converted outputs are never silently overwritten; the caller must
              opt in explicitly (via ``--overwrite`` on the CLI or
              ``args.overwrite = True`` in code).

    The function is intentionally lenient: ``arrus`` does not need to be
    installed. All arrus classes in the pickle are replaced with lightweight
    namespace stubs that preserve the original attribute values.

    Raises:
        FileNotFoundError: If ``src`` does not exist, or if ``src`` is a
            directory that contains no ``.pkl`` files.
        ValueError: If a requested ``mapping`` value is not a supported us4us
            converter data type.
    """
    src = Path(args.src)
    dst = Path(args.dst)
    mapping: dict = getattr(args, "mapping", {0: "image"})
    overwrite: bool = bool(getattr(args, "overwrite", False))

    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")

    unknown = set(mapping.values()) - _SUPPORTED_MAPPING_DATA_TYPES
    if unknown:
        raise ValueError(
            f"Unknown zea data type(s) in mapping: {sorted(unknown)}. "
            "The us4us converter supports only the types it can build in "
            f"_prepare_output: {sorted(_SUPPORTED_MAPPING_DATA_TYPES)}."
        )

    if src.is_dir():
        pkl_files = sorted(src.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in directory: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        file_pairs = [(p, dst / f"{p.stem}.hdf5") for p in pkl_files]
    else:
        if dst.exists() and dst.is_dir():
            dst_file = dst / f"{src.stem}.hdf5"
        else:
            dst_file = dst
        file_pairs = [(src, dst_file)]

    for src_pkl, dst_h5 in file_pairs:
        _convert_single_us4us_pickle(src_pkl, dst_h5, mapping, overwrite=overwrite)

