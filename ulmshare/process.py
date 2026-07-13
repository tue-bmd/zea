"""Process an ULMShare acquisition into a Power-Doppler image.

Python port of ``process.m``. It replaces the MATLAB/MUST beamforming with
`zea <https://github.com/tue-bmd/zea>`_ (delay-and-sum, JAX backend) and does
the remaining steps -- IQ compounding, SVD clutter filtering and Power-Doppler
-- in JAX.

Compared to ``process.m`` this only goes up to the Power-Doppler image: the TAL
ULM localization/tracking stage is intentionally not ported (there is no Python
equivalent of the TAL toolbox to port from).

Data format
-----------
Each ``dataXXXX.bin`` file (the "reshaped" ULMShare variant that MATLAB reads
with ``load_reshaped_bin``) is a little-endian ``int16`` array prefixed with a
12-byte ``int32`` header holding the shape ``(2 * n_ax, n_el, n_angles * n_frames)``.
The first axis interleaves the I and Q channels (``iq = raw[0::2] - 1j*raw[1::2]``,
matching the MATLAB ``complex(raw(1:2:end), -raw(2:2:end))``).

For ``data1.bin`` this header reads ``[640, 128, 4400]``:
``n_ax = 320``, ``n_el = 128``, ``4400 = 11 angles x 400 frames``.

Acquisition parameters
-----------------------
Read from the acquisition's ``sequence.json`` (the same file MATLAB's
``process.m`` reads) -- transducer, plane-wave angles (in their recorded
ping-pong order), centre frequency, sampling mode, sound speed and the
start/end depth of the axial window. The probe geometry (element count + pitch)
is looked up from the transducer name, mirroring ``matlab/getProbe.m``.

Usage
-----
    python process.py --data .../mouse_1/acquisition_1/data1.bin
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from zea import Parameters, init_device  # noqa: E402  (import after KERAS_BACKEND is set)
from zea.beamform.delays import compute_t0_delays_planewave
from zea.ops import Beamform
from zea.probes import create_probe_geometry

# --------------------------------------------------------------------------- #
# Acquisition parameters.
#
# These come from the acquisition's ``sequence.json`` (same file the MATLAB
# ``process.m`` reads). The probe geometry (element count + pitch) is looked up
# from the transducer name, mirroring ``matlab/getProbe.m``.
# --------------------------------------------------------------------------- #

# Probe geometry table (from matlab/getProbe.m): name -> (n_el, pitch [m]).
PROBE_TABLE = {
    "L22-14v": (128, 0.08e-3),
    "L22-14vX": (128, 0.08e-3),
    "GEL818iD": (168, 0.15e-3),
}

# Beamforming / processing parameters (from process.m).
FNUMBER = 1.4
START_X, END_X = -0.5e-2, 0.5e-2  # BF grid x extent [m]
START_Z, END_Z = 0.05e-2, 0.85e-2  # BF grid z extent [m]
DELTA_GRID = 2.5e-5  # BF grid resolution [m]
CLUTTER_FILTER_CUT = 5 / 100  # SVD clutter-filter cutoff (fraction of frames)


def load_sequence_json(path):
    """Parse an ULMShare ``sequence.json`` into a typed parameter dict.

    Mirrors how ``process.m`` reads ``param_acq`` (all values are strings in the
    file). The important, easy-to-get-wrong fields:

    * ``angles_deg`` is a Verasonics *ping-pong* ordered list
      (e.g. ``-5, 5, -4, 4, ..., 0``), **not** monotonically increasing. Each
      transmit must be paired with its own angle in this order or the compounded
      image smears.
    * ``startDepth_wavelengths`` / ``endDepth_wavelengths`` set the axial window:
      the first recorded IQ sample is at ``startDepth`` (round-trip), and the
      effective IQ sampling frequency follows from the window spanning ``n_ax``
      samples (see :func:`build_parameters`).

    Args:
        path (str | Path): Path to ``sequence.json``.

    Returns:
        dict: Parsed parameters (native Python types).
    """
    with open(path) as f:
        raw = json.load(f)

    angles_deg = np.array([float(a) for a in raw["angles_deg"].split(",")], dtype=np.float32)
    n_angles = int(raw["nbAngles"])
    assert len(angles_deg) == n_angles, (
        f"nbAngles={n_angles} but angles_deg has {len(angles_deg)} entries."
    )

    return {
        "transducer": raw["transducer"],
        "n_angles": n_angles,
        "angles_deg": angles_deg,  # ping-pong order, as recorded
        "center_frequency": float(raw["frequency_MHz"]) * 1e6,
        "sampling": raw["sampling"],  # BS100BW / BS50BW
        "sound_speed": float(raw.get("speedOfSound", 1540.0)),
        "frame_rate_hz": float(raw["frameRate_Hz"]),
        "start_depth_wl": float(raw["startDepth_wavelengths"]),
        "end_depth_wl": float(raw["endDepth_wavelengths"]),
    }


def load_reshaped_bin(path):
    """Load a reshaped ULMShare ``.bin`` file.

    Mirrors the MATLAB ``load_reshaped_bin``: a 12-byte ``int32`` shape header
    followed by an ``int16`` payload stored in column-major (Fortran) order.

    Args:
        path (str | Path): Path to the ``dataXXXX.bin`` file.

    Returns:
        np.ndarray: ``int16`` array of shape ``(2 * n_ax, n_el, n_angles * n_frames)``.
    """
    path = Path(path)
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, dtype="<i4", count=3)
        payload = np.fromfile(fid, dtype="<i2")
    expected = int(np.prod(shape))
    if payload.size != expected:
        raise ValueError(
            f"{path.name}: payload has {payload.size} int16 values, "
            f"header shape {tuple(shape)} expects {expected}."
        )
    # MATLAB writes in column-major order, so read back with order="F".
    return payload.reshape(tuple(shape), order="F")


def raw_to_iq(raw, n_angles):
    """Convert interleaved reshaped raw data into zea-shaped IQ data.

    Args:
        raw (np.ndarray): Array of shape ``(2 * n_ax, n_el, n_angles * n_frames)``.
        n_angles (int): Number of plane-wave angles per compounded frame.

    Returns:
        np.ndarray: Real IQ data of shape
        ``(n_frames, n_angles, n_ax, n_el, 2)``, ready for zea (channel axis
        last, ``[I, Q]``). ``I`` and ``Q`` are stacked rather than complex so it
        can flow through the (real-valued) zea beamforming pipeline.
    """
    raw = raw.astype(np.float64)
    # De-interleave I/Q along the fast (axial) axis -> (n_ax, n_el, n_ang*n_fr).
    i = raw[0::2]
    q = -raw[1::2]  # note the sign, matching MATLAB complex(.., -..)

    n_ax, n_el, n_events = i.shape
    n_frames = n_events // n_angles

    def _reshape(x):
        # (n_ax, n_el, n_angles * n_frames) -> (n_frames, n_angles, n_ax, n_el).
        # The event axis is ordered [angle varies fastest, then frame], matching
        # the MATLAB reshape(iq, .., .., nbAngles, []).
        x = x.reshape(n_ax, n_el, n_angles, n_frames, order="F")
        return np.transpose(x, (3, 2, 0, 1))

    iq = np.stack([_reshape(i), _reshape(q)], axis=-1)  # (n_fr, n_ang, n_ax, n_el, 2)
    return iq.astype(np.float32)


def build_parameters(seq, n_ax):
    """Build a :class:`zea.Parameters` for the ULMShare plane-wave sequence.

    Args:
        seq (dict): Parsed ``sequence.json`` (see :func:`load_sequence_json`).
        n_ax (int): Number of axial IQ samples.

    Returns:
        zea.Parameters: Fully specified acquisition parameters (one compounded
        frame worth of transmits).
    """
    if seq["transducer"] not in PROBE_TABLE:
        raise NotImplementedError(f"Unknown transducer: {seq['transducer']}")
    n_el, pitch = PROBE_TABLE[seq["transducer"]]
    fc = seq["center_frequency"]
    c = seq["sound_speed"]

    probe_geometry = create_probe_geometry(n_el, pitch)
    polar_angles = np.deg2rad(seq["angles_deg"]).astype(np.float32)

    # Plane-wave transmit delays from the steering angles (seconds, min == 0).
    t0_delays = compute_t0_delays_planewave(probe_geometry, polar_angles, sound_speed=c).astype(
        np.float32
    )

    # --- Sampling / demodulation frequency ---------------------------------
    # Match MUST's ``das``: for baseband IQ the sampling frequency IS the
    # demodulation (carrier) frequency. BS100BW -> fs = fc; BS50BW -> fs = fc/2
    # (see the sampling branch in process.m). The startDepth/endDepth fields in
    # sequence.json describe the transmit depth-of-interest, NOT the IQ sample
    # rate -- do not derive fs from them (doing so over-compresses the axial
    # scale and warps the speckle).
    if seq["sampling"] == "BS50BW":
        sampling_frequency = fc / 2
    elif seq["sampling"] == "BS100BW":
        sampling_frequency = fc
    else:
        raise NotImplementedError(f"Unsupported sampling mode: {seq['sampling']}")
    demodulation_frequency = sampling_frequency

    # --- Recording start time (t0) -----------------------------------------
    # MUST's ``das`` uses tau = (dTX + dRX)/c with idxt = (tau - t0)*fs and
    # t0 = 0 by default, where dTX = min_el(txdelay*c + dist) already carries the
    # zero-aligned transmit delays. There is no additional per-transmit offset,
    # so initial_times = 0 (matches MUST exactly).
    initial_times = np.zeros(seq["n_angles"], dtype=np.float32)

    n_tx = seq["n_angles"]
    parameters = Parameters(
        probe_geometry=probe_geometry,
        n_el=n_el,
        n_ax=n_ax,
        n_ch=2,  # IQ
        n_tx=n_tx,
        center_frequency=fc,
        demodulation_frequency=demodulation_frequency,
        sampling_frequency=sampling_frequency,
        sound_speed=c,
        # Receive f-number aperture: MUST's ``das`` uses a *rectangular* aperture
        # (|x_el - x_pix| <= z/(2*fnumber)), which is exactly zea's built-in
        # f-number mask (default fnum_window_fn_rect). Use it directly to match.
        f_number=FNUMBER,
        polar_angles=polar_angles,
        focus_distances=np.full(n_tx, np.inf, dtype=np.float32),  # plane waves
        t0_delays=t0_delays,
        initial_times=initial_times,
        # MUST references the IQ samples to the emission time and adds no
        # pulse-peak offset, so t_peak=0 (zea's default is 1/fc, which would
        # shift the image ~half a wavelength deeper).
        t_peak=np.zeros(n_tx, dtype=np.float32),
        xlims=(START_X, END_X),
        zlims=(START_Z, END_Z),
        grid_size_x=int(round((END_X - START_X) / DELTA_GRID)),
        grid_size_z=int(round((END_Z - START_Z) / DELTA_GRID)),
    )
    parameters.set_transmits("all")
    return parameters


def beamform_frames(iq, parameters):
    """Delay-and-sum beamform + angle-compound every frame with zea.

    Args:
        iq (np.ndarray): IQ data of shape ``(n_frames, n_angles, n_ax, n_el, 2)``.
        parameters (zea.Parameters): Acquisition parameters.

    Returns:
        np.ndarray: Complex beamformed IQ movie of shape
        ``(grid_size_z, grid_size_x, n_frames)``.
    """
    pipeline = Beamform(
        beamformer="delay_and_sum",
        enable_pfield=False,
        num_patches=100,
        with_batch_dim=False,
        jit_options="pipeline",
    )
    inputs = pipeline.prepare_parameters(parameters)

    n_frames = iq.shape[0]
    frames = []
    for k in range(n_frames):
        # Beamform one compounded frame: (n_angles, n_ax, n_el, 2). The Beamform
        # pipeline sums over channels (DAS) and over transmits (compounding),
        # returning a (grid_size_z, grid_size_x, n_ch=2) IQ image.
        out = pipeline(**{**inputs, pipeline.key: iq[k]})
        bf = np.asarray(out[pipeline.output_key])  # (Nz, Nx, 2)
        frames.append(bf[..., 0] + 1j * bf[..., 1])
        if (k + 1) % 50 == 0 or k == n_frames - 1:
            print(f"  beamformed frame {k + 1}/{n_frames}")

    return np.stack(frames, axis=-1)  # (Nz, Nx, n_frames)


def svd_clutter_filter(iq_bf, cutoff):
    """Remove tissue clutter via an SVD (spatio-temporal) filter.

    Same algorithm as :func:`zea.func.ultrasound.suppress_tissue` (eigendecompose
    the temporal Gram matrix, drop the strongest ``round(n_frames*cutoff)``
    tissue components), but with a **conjugate** transpose so it is correct for
    the *complex* IQ movie -- exactly the ``iq' * iq`` in ``process.m``.

    .. note::
        ``suppress_tissue`` uses a plain (non-conjugate) transpose, which is
        designed for real-valued input; on complex IQ it does not suppress the
        tissue (the Gram matrix is wrong). Hence the local implementation here.

    Args:
        iq_bf (np.ndarray): Complex beamformed movie ``(Nz, Nx, n_frames)``.
        cutoff (float): Fraction of leading (tissue) components to remove.

    Returns:
        np.ndarray: Clutter-filtered movie ``(Nz, Nx, n_frames)``.
    """
    import jax.numpy as jnp

    Nz, Nx, n_frames = iq_bf.shape
    n_cut = int(round(n_frames * cutoff))  # process.m: Ncut = round(nFrames * 5%)
    casorati = jnp.asarray(iq_bf.reshape(-1, n_frames))  # (n_pix, n_frames)

    # Temporal covariance (Hermitian) and its eigenvectors, energy-sorted.
    gram = jnp.conj(casorati.T) @ casorati  # (n_frames, n_frames)
    eig_vect, _, _ = jnp.linalg.svd(gram)

    # Keep everything except the strongest tissue components. process.m keeps
    # eig_vect(:, Ncut:end) (1-based), i.e. drops the first Ncut-1 columns.
    keep = eig_vect[:, max(n_cut - 1, 0) :]
    filtered = (casorati @ keep) @ jnp.conj(keep.T)

    return np.asarray(filtered).reshape(Nz, Nx, n_frames)


def power_doppler(iq_cf, frame=None):
    """Power-Doppler image in dB (normalised to 0 dB max).

    Args:
        iq_cf (np.ndarray): Clutter-filtered movie ``(Nz, Nx, n_frames)``.
        frame (int, optional): Show a single frame's clutter-filtered magnitude.
            When ``None`` (default) the magnitude is integrated (summed) over all
            frames (the standard Power-Doppler image).

    Returns:
        np.ndarray: Power-Doppler image ``(Nz, Nx)`` in dB.
    """
    if frame is None:
        pd = 20 * np.log10(np.sum(np.abs(iq_cf), axis=-1) + 1e-12)
    else:
        pd = 20 * np.log10(np.abs(iq_cf[..., frame]) + 1e-12)
    return pd - pd.max()


def bmode(iq_bf, frame=None):
    """B-mode image in dB (normalised to 0 dB max).

    The beamformed IQ is already the analytic (complex) signal, so its magnitude
    is the envelope directly -- no Hilbert transform needed.

    Args:
        iq_bf (np.ndarray): Beamformed (pre-clutter-filter) movie
            ``(Nz, Nx, n_frames)``.
        frame (int, optional): Show a single frame. When ``None`` (default) the
            envelope is averaged over all frames (a cleaner static B-mode).

    Returns:
        np.ndarray: B-mode image ``(Nz, Nx)`` in dB.
    """
    if frame is None:
        env = np.mean(np.abs(iq_bf), axis=-1)
    else:
        env = np.abs(iq_bf[..., frame])
    bm = 20 * np.log10(env + 1e-12)
    return bm - bm.max()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="/mnt/z/usbmd/oisin/ULMShare/mouse_1/acquisition_1/data1.bin",
        help="Path to a reshaped ULMShare dataXXXX.bin file.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Path to sequence.json (default: next to --data).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("power_doppler.png")),
        help="Where to save the Power-Doppler image.",
    )
    parser.add_argument(
        "--bmode-out",
        default=str(Path(__file__).with_name("bmode.png")),
        help="Where to save the B-mode image.",
    )
    parser.add_argument(
        "--pd-frame-out",
        default=str(Path(__file__).with_name("power_doppler_frame.png")),
        help=(
            "Where to save the single-frame Power-Doppler snapshot "
            "(only saved when --frame is given)."
        ),
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Optionally limit the number of frames (for a quick run).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help=(
            "Select a single frame (0-based) for the B-mode and for an extra "
            "single-frame Power-Doppler snapshot (--pd-frame-out). The integrated "
            "Power-Doppler is always saved. Default: frame-averaged B-mode, no snapshot."
        ),
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help=(
            "Matplotlib colormap for the images. Use 'parula' (or 'viridis') for "
            "the MATLAB-style blue-yellow map. Default: gray."
        ),
    )
    args = parser.parse_args()

    init_device(verbose=False)

    seq_path = args.sequence or str(Path(args.data).with_name("sequence.json"))
    print(f"Loading sequence parameters from {seq_path} ...")
    seq = load_sequence_json(seq_path)
    print(
        f"  {seq['transducer']}, {seq['n_angles']} angles "
        f"[{', '.join(f'{a:g}' for a in seq['angles_deg'])}] deg, "
        f"fc={seq['center_frequency'] / 1e6:g} MHz, {seq['sampling']}"
    )

    print(f"Loading {args.data} ...")
    raw = load_reshaped_bin(args.data)
    iq = raw_to_iq(raw, seq["n_angles"])  # (n_fr, n_ang, n_ax, n_el, 2)
    if args.n_frames is not None:
        iq = iq[: args.n_frames]
    n_frames, n_angles, n_ax, n_el, _ = iq.shape
    print(f"  IQ data: {n_frames} frames x {n_angles} angles x {n_ax} ax x {n_el} el")

    parameters = build_parameters(seq, n_ax)

    print("Beamforming (zea delay-and-sum, JAX)...")
    iq_bf = beamform_frames(iq, parameters)  # (Nz, Nx, n_frames)

    if args.frame is not None:
        print(f"Computing B-mode (frame {args.frame})...")
    else:
        print("Computing B-mode (frame-averaged)...")
    bm = bmode(iq_bf, frame=args.frame)

    print("SVD clutter filtering (JAX)...")
    iq_cf = svd_clutter_filter(iq_bf, CLUTTER_FILTER_CUT)

    print("Computing Power Doppler (frame-integrated)...")
    pd = power_doppler(iq_cf)

    # Like process.m (which shows both the per-frame powerDopplerMovie and the
    # integrated powerDoppler), also compute a single-frame Power-Doppler
    # snapshot when a frame is selected.
    pd_frame = None
    if args.frame is not None:
        print(f"Computing Power Doppler (frame {args.frame})...")
        pd_frame = power_doppler(iq_cf, frame=args.frame)

    # ---- Display / save ---------------------------------------------------- #
    import matplotlib.pyplot as plt

    from zea.visualize import set_mpl_style

    set_mpl_style()
    extent = [START_X * 1e3, END_X * 1e3, END_Z * 1e3, START_Z * 1e3]  # mm

    def _show(image, title, out_path, vmin, vmax, cmap):
        plt.figure()
        plt.imshow(image, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, aspect="equal")
        plt.title(title)
        plt.xlabel("x axis (mm)")
        plt.ylabel("z axis (mm)")
        plt.colorbar(ticks=[vmin, vmax])
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved {title} to {out_path}")

    bmode_title = "B-mode (dB)" if args.frame is None else f"B-mode (dB), frame {args.frame}"
    _show(bm, bmode_title, args.bmode_out, vmin=-50, vmax=0, cmap="gray")
    _show(pd, "Power Doppler (dB)", args.out, vmin=-40, vmax=0, cmap="viridis")
    if pd_frame is not None:
        _show(
            pd_frame,
            f"Power Doppler (dB), frame {args.frame}",
            args.pd_frame_out,
            vmin=-40,
            vmax=0,
            cmap="viridis",
        )


if __name__ == "__main__":
    main()
