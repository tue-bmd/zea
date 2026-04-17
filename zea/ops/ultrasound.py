import uuid
from typing import Tuple

import keras
import numpy as np
from keras import ops

from zea import log
from zea.backend import jit as backend_jit
from zea.beamform.beamformer import calculate_delays, tof_correction
from zea.display import scan_convert
from zea.func.tensor import (
    apply_along_axis,
    correlate,
    extend_n_dims,
    gaussian_filter,
    reshape_axis,
)
from zea.func.ultrasound import (
    channels_to_complex,
    complex_to_channels,
    demodulate,
    envelope_detect,
    get_band_pass_filter,
    get_low_pass_iq_filter,
    log_compress,
    upmix,
)
from zea.internal.core import (
    DEFAULT_DYNAMIC_RANGE,
    DataTypes,
)
from zea.internal.registry import ops_registry
from zea.ops.base import Filter, Operation
from zea.simulator import simulate_rf
from zea.utils import canonicalize_axis

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import jax
except ImportError:
    jax = None


@ops_registry("simulate_rf")
class Simulate(Operation):
    """Simulate RF data."""

    # Define operation-specific static parameters
    STATIC_PARAMS = ["n_ax", "apply_lens_correction"]

    def __init__(self, **kwargs):
        super().__init__(
            output_data_type=DataTypes.RAW_DATA,
            additional_output_keys=["n_ch"],
            **kwargs,
        )

    def call(
        self,
        scatterer_positions,
        scatterer_magnitudes,
        probe_geometry,
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        n_ax,
        center_frequency,
        sampling_frequency,
        t0_delays,
        initial_times,
        element_width,
        attenuation_coef,
        tx_apodizations,
        **kwargs,
    ):
        simulate_kwargs = {
            "probe_geometry": probe_geometry,
            "apply_lens_correction": apply_lens_correction,
            "lens_thickness": lens_thickness,
            "lens_sound_speed": lens_sound_speed,
            "sound_speed": sound_speed,
            "n_ax": n_ax,
            "center_frequency": center_frequency,
            "sampling_frequency": sampling_frequency,
            "t0_delays": t0_delays,
            "initial_times": initial_times,
            "element_width": element_width,
            "attenuation_coef": attenuation_coef,
            "tx_apodizations": tx_apodizations,
        }
        if not self.with_batch_dim:
            simulated_rf = simulate_rf(
                scatterer_positions=scatterer_positions,
                scatterer_magnitudes=scatterer_magnitudes,
                **simulate_kwargs,
            )
        else:
            simulated_rf = ops.map(
                lambda inputs: simulate_rf(
                    scatterer_positions=inputs["positions"],
                    scatterer_magnitudes=inputs["magnitudes"],
                    **simulate_kwargs,
                ),
                {
                    "positions": scatterer_positions,
                    "magnitudes": scatterer_magnitudes,
                },
            )

        return {
            self.output_key: simulated_rf,
            "n_ch": 1,  # Simulate always returns RF data (so single channel)
        }


@ops_registry("tof_correction")
class TOFCorrection(Operation):
    """Time-of-flight correction operation for ultrasound data."""

    # Define operation-specific static parameters
    STATIC_PARAMS = ["f_number", "apply_lens_correction"]

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.ALIGNED_DATA,
            **kwargs,
        )

    def call(
        self,
        flatgrid,
        sound_speed,
        polar_angles,
        focus_distances,
        sampling_frequency,
        f_number,
        demodulation_frequency,
        t0_delays,
        tx_apodizations,
        initial_times,
        probe_geometry,
        t_peak,
        tx_waveform_indices,
        transmit_origins,
        apply_lens_correction=None,
        lens_thickness=None,
        lens_sound_speed=None,
        **kwargs,
    ):
        """Perform time-of-flight correction on raw RF data.

        Args:
            raw_data (ops.Tensor): Raw RF data to correct
            flatgrid (ops.Tensor): Grid points at which to evaluate the time-of-flight
            sound_speed (float): Sound speed in the medium
            polar_angles (ops.Tensor): Polar angles for scan lines
            focus_distances (ops.Tensor): Focus distances for scan lines
            sampling_frequency (float): Sampling frequency
            f_number (float): F-number for apodization
            demodulation_frequency (float): Demodulation frequency
            t0_delays (ops.Tensor): T0 delays
            tx_apodizations (ops.Tensor): Transmit apodizations
            initial_times (ops.Tensor): Initial times
            probe_geometry (ops.Tensor): Probe element positions
            t_peak (float): Time to peak of the transmit pulse
            tx_waveform_indices (ops.Tensor): Index of the transmit waveform for each
                transmit. (All zero if there is only one waveform)
            transmit_origins (ops.Tensor): Transmit origins of shape (n_tx, 3)
            apply_lens_correction (bool): Whether to apply lens correction
            lens_thickness (float): Lens thickness
            lens_sound_speed (float): Sound speed in the lens

        Returns:
            dict: Dictionary containing tof_corrected_data
        """

        raw_data = kwargs[self.key]

        tof_kwargs = {
            "flatgrid": flatgrid,
            "t0_delays": t0_delays,
            "tx_apodizations": tx_apodizations,
            "sound_speed": sound_speed,
            "probe_geometry": probe_geometry,
            "initial_times": initial_times,
            "sampling_frequency": sampling_frequency,
            "demodulation_frequency": demodulation_frequency,
            "f_number": f_number,
            "polar_angles": polar_angles,
            "focus_distances": focus_distances,
            "t_peak": t_peak,
            "tx_waveform_indices": tx_waveform_indices,
            "transmit_origins": transmit_origins,
            "apply_lens_correction": apply_lens_correction,
            "lens_thickness": lens_thickness,
            "lens_sound_speed": lens_sound_speed,
        }

        if not self.with_batch_dim:
            tof_corrected = tof_correction(raw_data, **tof_kwargs)
        else:
            tof_corrected = ops.map(
                lambda data: tof_correction(data, **tof_kwargs),
                raw_data,
            )

        return {self.output_key: tof_corrected}


@ops_registry("mach_beamform")
class MachBeamform(Operation):
    """Mach-based delay-and-sum beamforming for RF or IQ data.

    Beamforming is split into three stages:

    1. :meth:`_prepare_inputs` — translate Zea scan parameters and data to
       the layout expected by ``mach.kernel.beamform``.  Uses only Keras/Zea
       ops so it can be JIT-compiled.
    2. :meth:`_run_cuda_kernel` — loop over transmits and accumulate the
       CUDA beamform result.  Not JIT-compilable (GPU kernel invocation).
    3. :meth:`_prepare_outputs` — convert the accumulated ``(n_pix, n_frames)``
       result back to the Zea data layout.  Uses only Keras/Zea ops so it
       can be JIT-compiled.
    """

    STATIC_PARAMS = ["interp_type", "tukey_alpha"]

    def __init__(self, interp_type: str = "linear", tukey_alpha: float = 0, **kwargs):
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            jittable=False,
            **kwargs,
        )

        try:
            import mach  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mach is not installed. Please run "
                "`pip install mach-beamform` to use this operation."
            ) from exc

        if cp is None:
            raise ImportError(
                "cupy is not installed. Please run "
                "`pip install cupy` to use this operation."
            )

        self.interp_type = interp_type
        self.tukey_alpha = float(tukey_alpha)

        # Resolve the interp_type enum once at construction time so it can be
        # passed straight to the kernel without a string lookup each call.
        from mach.kernel import InterpolationType

        _interp_map = {
            "nearest": InterpolationType.NearestNeighbor,
            "linear": InterpolationType.Linear,
            "quadratic": InterpolationType.Quadratic,
        }
        if isinstance(interp_type, str):
            key = interp_type.lower()
            if key not in _interp_map:
                raise ValueError(
                    f"Unsupported interp_type '{interp_type}'. "
                    "Use 'nearest', 'linear', or 'quadratic'."
                )
            self._interp_type_enum = _interp_map[key]
        else:
            self._interp_type_enum = interp_type

        # if keras.backend.backend() == "jax":
        #     #self._calculate_delays = backend_jit(calculate_delays, static_argnums=(7, 8, 14))
        #     #self._channels_to_complex = backend_jit(channels_to_complex)
        #     # apply_lens_correction (arg 13) is a Python bool that must be concrete
        #     # for the inner _calculate_delays static arg to work under JAX JIT.
        #     # is_iq (arg 1 of _prepare_outputs) drives Python control flow.
        self._prepare_inputs = backend_jit(
            self._prepare_inputs, static_argnums=(13,)
        )
        self._prepare_outputs = backend_jit(
            self._prepare_outputs, static_argnums=(1,)
        )
        # else:
        #     # TODO: check if this is needed, passing the static_argnums is probably
        #     # always ok (even if it doesn't do anything outside JAX)
        #     #self._calculate_delays = backend_jit(calculate_delays)
        #     #self._channels_to_complex = channels_to_complex
        #     self._prepare_inputs = backend_jit(self._prepare_inputs)
        #     self._prepare_outputs = backend_jit(self._prepare_outputs)

    # ------------------------------------------------------------------
    # Stage 1: Zea → mach  (pure tensor ops, JIT-able)
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        data,
        flatgrid,
        probe_geometry,
        sampling_frequency,
        sound_speed,
        initial_times_for_tx,
        t0_delays,
        tx_apodizations,
        focus_distances,
        polar_angles,
        t_peak,
        tx_waveform_indices,
        transmit_origins,
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
    ):
        """Translate pre-validated Zea parameters and data into mach kernel inputs.

        This method contains only pure tensor operations and is JIT-compiled
        in ``__init__`` via ``backend_jit``.  All Python-level validation,
        None-resolution, and logging must be performed by the caller before
        invoking this method.

        Args:
            data: Input tensor with the channel dimension already removed and
                IQ data already converted to complex dtype.  Shape is
                ``([n_tx,] n_ax, n_el)`` (no batch) or
                ``(n_frames, [n_tx,] n_ax, n_el)`` (batch).
            flatgrid: ``(n_pix, 3)`` grid coordinates in metres.
            probe_geometry: ``(n_el, 3)`` element positions in metres.
            sampling_frequency: Sampling frequency in Hz.
            sound_speed: Speed of sound in m/s.
            initial_times_for_tx: Per-transmit time offsets ``(n_tx,)``
                (already zeroed-out when ``rx_start_s`` was externally set).
            t0_delays: Transmit delays ``(n_tx, n_el)`` in seconds.
            tx_apodizations: Transmit apodizations ``(n_tx, n_el)``.
            focus_distances: Focus distances ``(n_tx,)``.
            polar_angles: Polar angles ``(n_tx,)``.
            t_peak: Waveform peak times ``(n_waveforms,)``.
            tx_waveform_indices: Waveform index per transmit ``(n_tx,)``.
            transmit_origins: Transmit origins ``(n_tx, 3)``.
            apply_lens_correction: Python bool — must be a static JAX arg.
            lens_thickness: Lens thickness in metres (or ``None``).
            lens_sound_speed: Lens sound speed in m/s (or ``None``).

        Returns:
            Tuple of ``(channel_data_list, tx_wave_arrivals_list, n_frames)``
            where each element of the lists corresponds to one transmit.
        """
        # ---- Compute transmit wave-arrival times ---------------------
        # tx_delays: (n_pix, n_tx) in *samples*
        n_tx = int(t0_delays.shape[0])
        n_el = int(probe_geometry.shape[0])
        tx_delays, _ = calculate_delays(
            flatgrid,
            t0_delays,
            tx_apodizations,
            probe_geometry,
            initial_times_for_tx,
            sampling_frequency,
            sound_speed,
            n_tx,
            n_el,
            focus_distances,
            polar_angles,
            t_peak,
            tx_waveform_indices,
            transmit_origins,
            apply_lens_correction,
            lens_thickness,
            lens_sound_speed,
        )
        # Convert samples → seconds: (n_pix, n_tx)
        tx_wave_arrivals_s = tx_delays / sampling_frequency

        # ---- Reshape data to (n_tx, n_el, n_ax, n_frames) -----------
        # Zea layout (no batch):  ([n_tx,] n_ax, n_el)
        # Zea layout (batch):     (n_frames, [n_tx,] n_ax, n_el)
        # self.with_batch_dim is a closed-over Python bool; JAX evaluates
        # the if/else at trace time (shapes are always concrete under jit).
        if self.with_batch_dim:
            if data.ndim == 4:
                # (n_frames, n_tx, n_ax, n_el) → (n_tx, n_el, n_ax, n_frames)
                data = ops.transpose(data, (1, 3, 2, 0))
            else:
                # (n_frames, n_ax, n_el) → (1, n_el, n_ax, n_frames)
                data = ops.transpose(data, (2, 1, 0))
                data = ops.expand_dims(data, axis=0)
        else:
            if data.ndim == 3:
                # (n_tx, n_ax, n_el) → (n_tx, n_el, n_ax, 1)
                data = ops.transpose(data, (0, 2, 1))
                data = ops.expand_dims(data, axis=-1)
            else:
                # (n_ax, n_el) → (1, n_el, n_ax, 1)
                data = ops.transpose(data, (1, 0))
                data = ops.expand_dims(data, axis=0)
                data = ops.expand_dims(data, axis=-1)

        # data is now (n_tx, n_el, n_ax, n_frames)
        n_tx_data = int(data.shape[0])
        n_frames = int(data.shape[3])

        # Split along transmit axis → list of (n_el, n_ax, n_frames) tensors
        channel_data_list = [data[i] for i in range(n_tx_data)]
        # Split wave arrivals → list of (n_pix,) tensors
        tx_wave_arrivals_list = [tx_wave_arrivals_s[:, i] for i in range(n_tx)]

        return channel_data_list, tx_wave_arrivals_list, n_frames

    # ------------------------------------------------------------------
    # Stage 2: run CUDA kernel  (not JIT-able)
    # ------------------------------------------------------------------

    def _run_cuda_kernel(
        self,
        channel_data_list,
        tx_wave_arrivals_list,
        flatgrid,
        probe_geometry,
        rx_start_s,
        sampling_frequency,
        sound_speed,
        f_number,
        modulation_freq_hz,
        n_frames,
        is_iq,
    ):
        """Loop over transmits and accumulate beamformed output on GPU.

        All CuPy / DLPack conversions are confined to this method so that
        stages 1 and 3 remain backend-agnostic.

        Returns:
            output: CuPy array of shape ``(n_pix, n_frames)``, dtype
                ``complex64`` (IQ) or ``float32`` (RF).
        """
        from mach import kernel as mach_kernel

        def _to_cupy(arr):
            """Zero-copy where possible: CuPy stays as-is, JAX uses DLPack."""
            if isinstance(arr, cp.ndarray):
                return arr
            if jax is not None and isinstance(arr, jax.Array):
                return cp.from_dlpack(arr)
            # NumPy / TF / Torch tensors
            try:
                return cp.from_dlpack(arr)
            except Exception:
                return cp.asarray(ops.convert_to_numpy(arr))

        scan_coords_m = cp.ascontiguousarray(_to_cupy(flatgrid).astype(cp.float32, copy=False))
        rx_coords_m = cp.ascontiguousarray(_to_cupy(probe_geometry).astype(cp.float32, copy=False))

        n_pix = scan_coords_m.shape[0]
        output_dtype = cp.complex64 if is_iq else cp.float32
        out = cp.zeros((n_pix, n_frames), dtype=output_dtype)

        for i, (single_data, arrivals) in enumerate(
            zip(channel_data_list, tx_wave_arrivals_list)
        ):
            single_data_cp = cp.ascontiguousarray(
                _to_cupy(single_data).astype(output_dtype, copy=False)
            )
            arrivals_cp = cp.ascontiguousarray(
                _to_cupy(arrivals).astype(cp.float32, copy=False)
            )
            mach_kernel.beamform(
                channel_data=single_data_cp,
                rx_coords_m=rx_coords_m,
                scan_coords_m=scan_coords_m,
                tx_wave_arrivals_s=arrivals_cp,
                out=out,
                rx_start_s=rx_start_s,
                sampling_freq_hz=float(sampling_frequency),
                f_number=float(f_number),
                sound_speed_m_s=float(sound_speed),
                modulation_freq_hz=modulation_freq_hz,
                tukey_alpha=self.tukey_alpha,
                interp_type=self._interp_type_enum,
            )

        cp.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return out

    # ------------------------------------------------------------------
    # Stage 3: mach → Zea  (pure tensor ops, JIT-able)
    # ------------------------------------------------------------------

    def _prepare_outputs(self, output, is_iq):
        """Convert a ``(n_pix, n_frames)`` framework tensor to Zea layout.

        This method contains only pure tensor operations and is JIT-compiled
        in ``__init__`` via ``backend_jit``.  ``output`` must already be a
        framework tensor (i.e. the caller must call ``ops.convert_to_tensor``
        before invoking this method). Under JAX, ``is_iq`` is a static arg
        because it drives Python control flow.

        Returns:
            dict with output tensor in Zea layout:
            - RF, no batch: ``(n_pix, 1)``
            - RF, batch:    ``(n_frames, n_pix, 1)``
            - IQ, no batch: ``(n_pix, 2)``
            - IQ, batch:    ``(n_frames, n_pix, 2)``
        """
        if is_iq:
            channel = ops.stack([ops.real(output), ops.imag(output)], axis=-1)
        else:
            channel = ops.expand_dims(output, axis=-1)
        # channel: (n_pix, n_frames, n_ch)
        if self.with_batch_dim:
            channel = ops.transpose(channel, (1, 0, 2))
        else:
            channel = channel[:, 0, :]
        return {self.output_key: channel}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def call(
        self,
        flatgrid,
        probe_geometry,
        sampling_frequency,
        sound_speed,
        f_number,
        demodulation_frequency=None,
        initial_times=None,
        t0_delays=None,
        tx_apodizations=None,
        focus_distances=None,
        polar_angles=None,
        t_peak=None,
        tx_waveform_indices=None,
        transmit_origins=None,
        apply_lens_correction=False,
        lens_thickness=None,
        lens_sound_speed=None,
        rx_start_s=None,
        **kwargs,
    ):
        """Beamform input data with mach.kernel.beamform.

        Args:
            flatgrid: Flattened grid points of shape (n_pix, 3).
            probe_geometry: Receive element positions of shape (n_el, 3).
            sampling_frequency: Sampling frequency in Hz.
            sound_speed: Speed of sound in m/s.
            f_number: F-number for dynamic aperture.
            demodulation_frequency: Center frequency in Hz for IQ data.
            initial_times: Optional per-transmit receive start times (n_tx,).
            t0_delays: Transmit delays in seconds of shape (n_tx, n_el).
            tx_apodizations: Transmit apodizations of shape (n_tx, n_el).
            focus_distances: Focus distances of shape (n_tx,).
            polar_angles: Polar angles of shape (n_tx,).
            t_peak: Time of the transmit peak of shape (n_waveforms,).
            tx_waveform_indices: Waveform indices of shape (n_tx,).
            transmit_origins: Transmit origins of shape (n_tx, 3).
            apply_lens_correction: Whether to apply lens correction to delays.
            lens_thickness: Lens thickness in meters.
            lens_sound_speed: Lens sound speed in m/s.
            rx_start_s: Optional scalar receive start time in seconds.
        """
        data = kwargs[self.key]

        # ---- Validation (not JIT-able: None checks, logging, raises) ----
        required = {
            "t0_delays": t0_delays,
            "tx_apodizations": tx_apodizations,
            "focus_distances": focus_distances,
            "polar_angles": polar_angles,
            "initial_times": initial_times,
            "t_peak": t_peak,
            "tx_waveform_indices": tx_waveform_indices,
            "transmit_origins": transmit_origins,
        }
        missing = [name for name, val in required.items() if val is None]
        if missing:
            raise ValueError(
                "Missing Zea scan parameters required to compute tx_wave_arrivals_s: "
                + ", ".join(missing)
                + "."
            )

        if not isinstance(apply_lens_correction, (bool, np.bool_)):
            log.warning(
                "apply_lens_correction must be a Python bool for MachBeamform; "
                "defaulting to False for delay computation."
            )
            apply_lens_correction = False

        # ---- IQ / RF detection (shape is concrete; raises must stay here) ----
        n_ch = data.shape[-1]
        if n_ch == 1:
            data = ops.squeeze(data, axis=-1)
            is_iq = False
        elif n_ch == 2:
            data = channels_to_complex(data)
            is_iq = True
        else:
            is_iq = np.issubdtype(
                ops.convert_to_numpy(ops.zeros(1, dtype=data.dtype)).dtype,
                np.complexfloating,
            )
            if not is_iq:
                raise ValueError(
                    "MachBeamform expects RF (n_ch=1) or IQ (n_ch=2) data. "
                    f"Got last dimension {n_ch}."
                )

        if is_iq and demodulation_frequency is None:
            raise ValueError(
                "demodulation_frequency is required for IQ data. "
                "Set it to 0 if data is baseband."
            )
        modulation_freq_hz = (
            0.0 if demodulation_frequency is None else float(demodulation_frequency)
        )

        # ---- Resolve rx_start_s / initial_times interaction ----------
        # calculate_delays subtracts initial_times internally, so if the
        # caller supplies a manual rx_start_s we zero-out initial_times to
        # avoid double-counting.
        if rx_start_s is None:
            rx_start_s = 0.0
            initial_times_for_tx = initial_times
        else:
            rx_start_s = float(rx_start_s)
            if initial_times is not None:
                log.warning(
                    "rx_start_s provided; ignoring initial_times in tx delay "
                    "computation to avoid double offsets."
                )
            initial_times_for_tx = ops.zeros_like(initial_times)

        # ---- ndim validation (raise before JIT boundary) ----
        expected_ndim = {True: (3, 4), False: (2, 3)}
        if data.ndim not in expected_ndim[self.with_batch_dim]:
            label = "with_batch_dim=True" if self.with_batch_dim else "with_batch_dim=False"
            raise ValueError(
                f"MachBeamform {label} expects data with "
                f"{expected_ndim[self.with_batch_dim]} dims after channel squeeze, "
                f"got shape {data.shape}."
            )

        # Stage 1 — translate Zea → mach (JIT-compiled)
        channel_data_list, tx_wave_arrivals_list, n_frames = self._prepare_inputs(
            data,
            flatgrid,
            probe_geometry,
            sampling_frequency,
            sound_speed,
            initial_times_for_tx,
            t0_delays,
            tx_apodizations,
            focus_distances,
            polar_angles,
            t_peak,
            tx_waveform_indices,
            transmit_origins,
            apply_lens_correction,
            lens_thickness,
            lens_sound_speed,
        )

        # Stage 2 — CUDA kernel (not JIT-compilable)
        output = self._run_cuda_kernel(
            channel_data_list,
            tx_wave_arrivals_list,
            flatgrid,
            probe_geometry,
            rx_start_s,
            sampling_frequency,
            sound_speed,
            f_number,
            modulation_freq_hz,
            n_frames,
            is_iq,
        )

        # Stage 3 — translate mach → Zea (JIT-compiled)
        # CuPy → framework tensor conversion must happen outside JIT.
        output = ops.convert_to_tensor(output)
        return self._prepare_outputs(output, is_iq)


@ops_registry("pfield_weighting")
class PfieldWeighting(Operation):
    """Weighting aligned data with the pressure field."""

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.ALIGNED_DATA,
            **kwargs,
        )

    def call(self, flat_pfield=None, **kwargs):
        """Weight data with pressure field.

        Args:
            flat_pfield (ops.Tensor): Pressure field weight mask of shape (n_pix, n_tx)

        Returns:
            dict: Dictionary containing weighted data
        """
        data = kwargs[self.key]  # must start with ((batch_size,) n_tx, n_pix, ...)

        if flat_pfield is None:
            return {self.output_key: data}

        # Swap (n_pix, n_tx) to (n_tx, n_pix)
        flat_pfield = ops.swapaxes(flat_pfield, 0, 1)

        # Add batch dimension if needed
        if self.with_batch_dim:
            pfield_expanded = ops.expand_dims(flat_pfield, axis=0)
        else:
            pfield_expanded = flat_pfield

        append_n_dims = ops.ndim(data) - ops.ndim(pfield_expanded)
        pfield_expanded = extend_n_dims(pfield_expanded, axis=-1, n_dims=append_n_dims)

        # Perform element-wise multiplication with the pressure weight mask
        weighted_data = data * pfield_expanded

        return {self.output_key: weighted_data}


@ops_registry("scan_convert")
class ScanConvert(Operation):
    """Scan convert images to cartesian coordinates."""

    STATIC_PARAMS = ["fill_value"]

    def __init__(self, order=1, **kwargs):
        """Initialize the ScanConvert operation.

        Args:
            order (int, optional): Interpolation order. Defaults to 1. Currently only
                GPU support for order=1.
        """
        if order > 1:
            jittable = False
            log.warning(
                "GPU support for order > 1 is not available. " + "Disabling jit for ScanConvert."
            )
        else:
            jittable = True

        super().__init__(
            input_data_type=DataTypes.IMAGE,
            output_data_type=DataTypes.IMAGE_SC,
            jittable=jittable,
            additional_output_keys=[
                "resolution",
                "x_lim",
                "y_lim",
                "z_lim",
                "rho_range",
                "theta_range",
                "phi_range",
                "d_rho",
                "d_theta",
                "d_phi",
            ],
            **kwargs,
        )
        self.order = order

    def call(
        self,
        rho_range=None,
        theta_range=None,
        phi_range=None,
        resolution=None,
        coordinates=None,
        fill_value=None,
        **kwargs,
    ):
        """Scan convert images to cartesian coordinates.

        Args:
            rho_range (Tuple): Range of the rho axis in the polar coordinate system.
                Defined in meters.
            theta_range (Tuple): Range of the theta axis in the polar coordinate system.
                Defined in radians.
            phi_range (Tuple): Range of the phi axis in the polar coordinate system.
                Defined in radians.
            resolution (float): Resolution of the output image in meters per pixel.
                if None, the resolution is computed based on the input data.
            coordinates (Tensor): Coordinates for scan convertion. If None, will be computed
                based on rho_range, theta_range, phi_range and resolution. If provided, this
                operation can be jitted.
            fill_value (float): Value to fill the image with outside the defined region.

        """
        if fill_value is None:
            fill_value = np.nan

        data = kwargs[self.key]

        if self._jit_compile and self.jittable:
            assert coordinates is not None, (
                "coordinates must be provided to jit scan conversion."
                "You can set ScanConvert(jit_compile=False) to disable jitting."
            )

        data_out, parameters = scan_convert(
            data,
            rho_range,
            theta_range,
            phi_range,
            resolution,
            coordinates,
            fill_value,
            self.order,
            with_batch_dim=self.with_batch_dim,
        )

        return {self.output_key: data_out, **parameters}


@ops_registry("demodulate")
class Demodulate(Operation):
    """Demodulates the input data to baseband. After this operation, the carrier frequency
    is removed (0 Hz) and the data is in IQ format stored in two real valued channels."""

    def __init__(self, axis=-3, **kwargs):
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.RAW_DATA,
            jittable=True,
            additional_output_keys=["center_frequency", "n_ch"],
            **kwargs,
        )
        self.axis = axis

    def call(self, demodulation_frequency=None, sampling_frequency=None, **kwargs):
        data = kwargs[self.key]

        # Split the complex signal into two channels
        iq_data_two_channel = demodulate(
            data=data,
            demodulation_frequency=demodulation_frequency,
            sampling_frequency=sampling_frequency,
            axis=self.axis,
        )

        return {
            self.output_key: iq_data_two_channel,
            "center_frequency": 0.0,
            "n_ch": 2,
        }


@ops_registry("fir_filter")
class FirFilter(Operation):
    """Apply a FIR filter to the input signal using convolution.

    Looks for the filter taps in the input dictionary using the specified ``filter_key``.
    """

    def __init__(
        self,
        axis: int,
        complex_channels: bool = False,
        filter_key: str = "fir_filter_taps",
        **kwargs,
    ):
        """
        Args:
            axis (int): Axis along which to apply the filter. Cannot be the batch dimension and
                not the complex channel axis when ``complex_channels=True``.
            complex_channels (bool): Whether the last dimension of the input signal represents
                complex channels (real and imaginary parts). When True, it will convert the signal
                to ``complex`` dtype before filtering and convert it back to two channels
                after filtering.
            filter_key (str): Key in the input dictionary where the FIR filter taps are stored.
                Default is "fir_filter_taps".
        """
        super().__init__(**kwargs)
        self._check_axis(axis)

        self.axis = axis
        self.complex_channels = complex_channels
        self.filter_key = filter_key

    def _check_axis(self, axis, ndim=None):
        """Check if axis is not the batch dimension."""
        if self.with_batch_dim and (axis == 0 or (ndim is not None and axis == -ndim)):
            raise ValueError("Cannot apply FIR filter along batch dimension.")

    @property
    def valid_keys(self):
        """Get the valid keys for the `call` method."""
        return self._valid_keys.union({self.filter_key})

    def call(self, **kwargs):
        signal = kwargs[self.key]
        fir_filter_taps = kwargs[self.filter_key]

        ndim = ops.ndim(signal)
        self._check_axis(self.axis, ndim)
        axis = canonicalize_axis(self.axis, ndim)

        if self.complex_channels:
            assert axis < ndim - 1, (
                "When using complex_channels=True, the complex channels are removed to convert"
                " to complex numbers before filtering, so axis cannot be the last axis."
            )
            signal = channels_to_complex(signal)

        def _convolve(signal):
            """Apply the filter to the signal using correlation."""
            return correlate(signal, fir_filter_taps[::-1], mode="same")

        filtered_signal = apply_along_axis(_convolve, axis, signal)

        if self.complex_channels:
            filtered_signal = complex_to_channels(filtered_signal)

        return {self.output_key: filtered_signal}


@ops_registry("low_pass_filter")
class LowPassFilterIQ(FirFilter):
    """Apply a low-pass FIR filter to the demodulated IQ (n_ch=2) input signal using convolution.

    It is recommended to use :class:`FirFilter` with pre-computed filter taps for jittable
    operations. The :class:`LowPassFilterIQ` operation itself is not jittable and is provided
    for convenience only.

    Uses :func:`get_low_pass_iq_filter` to compute the filter taps.
    """

    def __init__(self, axis: int = -3, num_taps: int = 127, **kwargs):
        """Initialize the LowPassFilterIQ operation.

        Args:
            axis (int): Axis along which to apply the filter. Cannot be the batch dimension and
                cannot be the complex channel axis (the last axis). Default is -3, which is the
                ``n_ax`` axis for standard ultrasound data layout.
            num_taps (int): Number of taps in the FIR filter. Default is 127.
                Odd will result in a type I filter, even in a type II filter.
        """
        self._random_suffix = str(uuid.uuid4())
        kwargs.pop("filter_key", None)
        kwargs.pop("jittable", None)
        kwargs.pop("complex_channels", None)
        super().__init__(
            axis=axis,
            complex_channels=True,
            filter_key=f"low_pass_{self._random_suffix}",
            jittable=False,
            **kwargs,
        )
        self.num_taps = num_taps

    def call(self, bandwidth, sampling_frequency, center_frequency, **kwargs):
        lpf = get_low_pass_iq_filter(
            self.num_taps,
            ops.convert_to_numpy(sampling_frequency).item(),
            ops.convert_to_numpy(center_frequency).item(),
            ops.convert_to_numpy(bandwidth).item(),
        )
        kwargs[self.filter_key] = lpf
        return super().call(**kwargs)


@ops_registry("band_pass_filter")
class BandPassFilter(FirFilter):
    """Apply a band-pass FIR filter to the real input signal using convolution.

    The bandwidth parameter in the call method defines the passband centered around
    ``demodulation_frequency``, with edges at ``demodulation_frequency - bandwidth/2``
    and ``demodulation_frequency + bandwidth/2``. So, make sure this is used before demodulation
    to baseband.

    This operation is provided for convenience and will recompute the filter weights every
    time it is called. Alternatively, you can use :class:`FirFilter` with pre-computed
    filter taps.
    """

    def __init__(self, axis: int = -3, num_taps: int = 127, **kwargs):
        """Initialize the BandPassFilter operation.

        Args:
            axis (int): Axis along which to apply the filter. Cannot be the batch dimension.
                Default is -3, which is the ``n_ax`` axis for standard ultrasound data layout.
            num_taps (int): Number of taps in the FIR filter. Default is 127.
                Odd will result in a type I filter, even in a type II filter.
        """
        self._random_suffix = str(uuid.uuid4())
        kwargs.pop("filter_key", None)
        kwargs.pop("complex_channels", None)
        super().__init__(
            axis=axis,
            complex_channels=False,
            filter_key=f"band_pass_{self._random_suffix}",
            **kwargs,
        )
        self.num_taps = num_taps

    def call(self, sampling_frequency, demodulation_frequency, bandwidth, **kwargs):
        """Apply band-pass filter with specified bandwidth.

        Args:
            sampling_frequency (float): Sampling frequency in Hz.
            demodulation_frequency (float): Center frequency in Hz.
            bandwidth (float): Bandwidth in Hz. The filter will pass frequencies from
                ``demodulation_frequency - bandwidth/2`` to
                ``demodulation_frequency + bandwidth/2``.

        Returns:
            dict: Dictionary containing filtered signal.
        """
        f1 = demodulation_frequency - bandwidth / 2
        f2 = demodulation_frequency + bandwidth / 2

        bpf = get_band_pass_filter(
            self.num_taps, sampling_frequency, f1, f2, validate=not self._jit_compile
        )
        kwargs[self.filter_key] = bpf
        return super().call(**kwargs)


@ops_registry("channels_to_complex")
class ChannelsToComplex(Operation):
    def call(self, **kwargs):
        data = kwargs[self.key]
        output = channels_to_complex(data)
        return {self.output_key: output}


@ops_registry("complex_to_channels")
class ComplexToChannels(Operation):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, **kwargs):
        data = kwargs[self.key]
        output = complex_to_channels(data, axis=self.axis)
        return {self.output_key: output}


@ops_registry("lee_filter")
class LeeFilter(Filter):
    """
    The Lee filter is a speckle reduction filter commonly used in synthetic aperture radar (SAR)
    and ultrasound image processing. It smooths the image while preserving edges and details.
    This implementation uses Gaussian filter for local statistics and treats channels independently.

    Lee, J.S. (1980). Digital image enhancement and noise filtering by use of local statistics.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, (2), 165-168.
    """

    def __init__(
        self,
        sigma: float,
        mode: str = "symmetric",
        cval: float | None = None,
        truncate: float = 4.0,
        axes: Tuple[int] = (-3, -2),
        **kwargs,
    ):
        """
        Args:
            sigma (float or tuple): Standard deviation for Gaussian kernel. The standard deviations
                of the Gaussian filter are given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes.
            mode (str, optional): Padding mode for the input image. Default is 'symmetric'.
                See [keras docs](https://www.tensorflow.org/api_docs/python/tf/keras/ops/pad) for
                all options and [tensorflow docs](https://www.tensorflow.org/api_docs/python/tf/pad)
                for some examples. Note that the naming differs from scipy.ndimage.gaussian_filter!
            cval (float, optional): Value to fill past edges of input if mode is 'constant'.
                Default is None.
            truncate (float, optional): Truncate the filter at this many standard deviations.
                Default is 4.0.
            axes (Tuple[int], optional): If None, input is filtered along all axes. Otherwise, input
                is filtered along the specified axes. When axes is specified, any tuples used for
                sigma, order, mode and/or radius must match the length of axes. The ith entry in
                any of these tuples corresponds to the ith entry in axes. Default is (-3, -2),
                which corresponds to the height and width dimensions of a
                (..., height, width, channels) tensor.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.mode = mode
        self.cval = cval
        self.truncate = truncate
        self.axes = axes

    def call(self, **kwargs):
        """Apply the Lee filter to the input data.

        Args:
            data (ops.Tensor): Input image data of shape (height, width, channels) with
                optional batch dimension if ``self.with_batch_dim``.
        """
        data = kwargs.pop(self.key)
        axes = self._resolve_filter_axes(data, self.axes)

        # Apply Gaussian blur to get local mean
        img_mean = gaussian_filter(
            data, self.sigma, mode=self.mode, cval=self.cval, truncate=self.truncate, axes=axes
        )

        # Apply Gaussian blur to squared data to get local squared mean
        img_sqr_mean = gaussian_filter(
            data**2, self.sigma, mode=self.mode, cval=self.cval, truncate=self.truncate, axes=axes
        )

        # Calculate local variance
        img_variance = img_sqr_mean - img_mean**2

        # Calculate global variance (per channel)
        overall_variance = ops.var(data, axis=axes, keepdims=True)

        # Calculate adaptive weights
        eps = keras.config.epsilon()
        img_weights = img_variance / (img_variance + overall_variance + eps)

        # Apply Lee filter formula
        img_output = img_mean + img_weights * (data - img_mean)

        return {self.output_key: img_output}


@ops_registry("companding")
class Companding(Operation):
    """Companding according to the A- or μ-law algorithm.

    Invertible compressing operation. Used to compress
    dynamic range of input data (and subsequently expand).

    μ-law companding:
    https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    A-law companding:
    https://en.wikipedia.org/wiki/A-law_algorithm

    Args:
        expand (bool, optional): If set to False (default),
            data is compressed, else expanded.
        comp_type (str): either `a` or `mu`.
        mu (float, optional): compression parameter. Defaults to 255.
        A (float, optional): compression parameter. Defaults to 87.6.
    """

    def __init__(self, expand=False, comp_type="mu", **kwargs):
        super().__init__(**kwargs)
        self.expand = expand
        self.comp_type = comp_type.lower()
        if self.comp_type not in ["mu", "a"]:
            raise ValueError("comp_type must be 'mu' or 'a'.")

        if self.comp_type == "mu":
            self._compand_func = self._mu_law_expand if self.expand else self._mu_law_compress
        else:
            self._compand_func = self._a_law_expand if self.expand else self._a_law_compress

    @staticmethod
    def _mu_law_compress(x, mu=255, **kwargs):
        x = ops.clip(x, -1, 1)
        return ops.sign(x) * ops.log(1.0 + mu * ops.abs(x)) / ops.log(1.0 + mu)

    @staticmethod
    def _mu_law_expand(y, mu=255, **kwargs):
        y = ops.clip(y, -1, 1)
        return ops.sign(y) * ((1.0 + mu) ** ops.abs(y) - 1.0) / mu

    @staticmethod
    def _a_law_compress(x, A=87.6, **kwargs):
        x = ops.clip(x, -1, 1)
        x_sign = ops.sign(x)
        x_abs = ops.abs(x)
        A_log = ops.log(A)
        val1 = x_sign * A * x_abs / (1.0 + A_log)
        val2 = x_sign * (1.0 + ops.log(A * x_abs)) / (1.0 + A_log)
        y = ops.where((x_abs >= 0) & (x_abs < (1.0 / A)), val1, val2)
        return y

    @staticmethod
    def _a_law_expand(y, A=87.6, **kwargs):
        y = ops.clip(y, -1, 1)
        y_sign = ops.sign(y)
        y_abs = ops.abs(y)
        A_log = ops.log(A)
        val1 = y_sign * y_abs * (1.0 + A_log) / A
        val2 = y_sign * ops.exp(y_abs * (1.0 + A_log) - 1.0) / A
        x = ops.where((y_abs >= 0) & (y_abs < (1.0 / (1.0 + A_log))), val1, val2)
        return x

    def call(self, mu=255, A=87.6, **kwargs):
        data = kwargs[self.key]

        mu = ops.cast(mu, data.dtype)
        A = ops.cast(A, data.dtype)

        data_out = self._compand_func(data, mu=mu, A=A)
        return {self.output_key: data_out}


@ops_registry("downsample")
class Downsample(Operation):
    """Downsample data along a specific axis."""

    def __init__(self, factor: int = 1, phase: int = 0, axis: int = -3, **kwargs):
        super().__init__(
            additional_output_keys=["sampling_frequency", "n_ax"],
            **kwargs,
        )
        if factor < 1:
            raise ValueError("Downsample factor must be >= 1.")
        if phase < 0 or phase >= factor:
            raise ValueError("phase must satisfy 0 <= phase < factor.")
        self.factor = factor
        self.phase = phase
        self.axis = axis

    def call(self, sampling_frequency=None, n_ax=None, **kwargs):
        data = kwargs[self.key]
        length = ops.shape(data)[self.axis]
        sample_idx = ops.arange(self.phase, length, self.factor)
        data_downsampled = ops.take(data, sample_idx, axis=self.axis)

        output = {self.output_key: data_downsampled}
        # downsampling also affects the sampling frequency
        if sampling_frequency is not None:
            sampling_frequency = sampling_frequency / self.factor
            output["sampling_frequency"] = sampling_frequency
        if n_ax is not None:
            n_ax = n_ax // self.factor
            output["n_ax"] = n_ax
        return output


@ops_registry("anisotropic_diffusion")
class AnisotropicDiffusion(Operation):
    """Speckle Reducing Anisotropic Diffusion (SRAD) filter.

    Reference:
    - https://www.researchgate.net/publication/5602035_Speckle_reducing_anisotropic_diffusion
    - https://nl.mathworks.com/matlabcentral/fileexchange/54044-image-despeckle-filtering-toolbox
    """

    def call(self, niter=100, lmbda=0.1, rect=None, eps=1e-6, **kwargs):
        """Anisotropic diffusion filter.

        Assumes input data is non-negative.

        Args:
            niter: Number of iterations.
            lmbda: Lambda parameter.
            rect: Rectangle [x1, y1, x2, y2] for homogeneous noise (optional).
            eps: Small epsilon for stability.
        Returns:
            Filtered image (2D tensor or batch of images).
        """
        data = kwargs[self.key]

        if not self.with_batch_dim:
            data = ops.expand_dims(data, axis=0)

        batch_size = ops.shape(data)[0]

        results = []
        for i in range(batch_size):
            image = data[i]
            image_out = self._anisotropic_diffusion_single(image, niter, lmbda, rect, eps)
            results.append(image_out)

        result = ops.stack(results, axis=0)

        if not self.with_batch_dim:
            result = ops.squeeze(result, axis=0)

        return {self.output_key: result}

    def _anisotropic_diffusion_single(self, image, niter, lmbda, rect, eps):
        """Apply anisotropic diffusion to a single image (2D)."""
        image = ops.exp(image)
        M, N = image.shape

        for _ in range(niter):
            iN = ops.concatenate([image[1:], ops.zeros((1, N), dtype=image.dtype)], axis=0)
            iS = ops.concatenate([ops.zeros((1, N), dtype=image.dtype), image[:-1]], axis=0)
            jW = ops.concatenate([image[:, 1:], ops.zeros((M, 1), dtype=image.dtype)], axis=1)
            jE = ops.concatenate([ops.zeros((M, 1), dtype=image.dtype), image[:, :-1]], axis=1)

            if rect is not None:
                x1, y1, x2, y2 = rect
                imageuniform = image[x1:x2, y1:y2]
                q0_squared = (ops.std(imageuniform) / (ops.mean(imageuniform) + eps)) ** 2

            dN = iN - image
            dS = iS - image
            dW = jW - image
            dE = jE - image

            G2 = (dN**2 + dS**2 + dW**2 + dE**2) / (image**2 + eps)
            L = (dN + dS + dW + dE) / (image + eps)
            num = (0.5 * G2) - ((1 / 16) * (L**2))
            den = (1 + ((1 / 4) * L)) ** 2
            q_squared = num / (den + eps)

            if rect is not None:
                den = (q_squared - q0_squared) / (q0_squared * (1 + q0_squared) + eps)
            c = 1.0 / (1 + den)
            cS = ops.concatenate([ops.zeros((1, N), dtype=image.dtype), c[:-1]], axis=0)
            cE = ops.concatenate([ops.zeros((M, 1), dtype=image.dtype), c[:, :-1]], axis=1)

            D = (cS * dS) + (c * dN) + (cE * dE) + (c * dW)
            image = image + (lmbda / 4) * D

        result = ops.log(image)
        return result


@ops_registry("envelope_detect")
class EnvelopeDetect(Operation):
    """Envelope detection of RF signals."""

    def __init__(
        self,
        axis=-3,
        **kwargs,
    ):
        super().__init__(
            input_data_type=DataTypes.BEAMFORMED_DATA,
            output_data_type=DataTypes.ENVELOPE_DATA,
            **kwargs,
        )
        self.axis = axis

    def call(self, **kwargs):
        """
        Args:
            - data (Tensor): The beamformed data of shape (..., grid_size_z, grid_size_x, n_ch).
        Returns:
            - envelope_data (Tensor): The envelope detected data
                of shape (..., grid_size_z, grid_size_x).
        """
        data = kwargs[self.key]

        data = envelope_detect(data, axis=self.axis)

        return {self.output_key: data}


@ops_registry("upmix")
class UpMix(Operation):
    """Upmix IQ data to RF data."""

    def __init__(
        self,
        upsampling_rate=1,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.upsampling_rate = upsampling_rate

    def call(self, sampling_frequency=None, demodulation_frequency=None, **kwargs):
        data = kwargs[self.key]

        if data.shape[-1] == 1:
            log.warning("Upmixing is not applicable to RF data.")
            return {self.output_key: data}
        elif data.shape[-1] == 2:
            data = channels_to_complex(data)

        data = upmix(data, sampling_frequency, demodulation_frequency, self.upsampling_rate)
        data = ops.expand_dims(data, axis=-1)
        return {self.output_key: data}


@ops_registry("log_compress")
class LogCompress(Operation):
    """Logarithmic compression of data."""

    def __init__(self, clip: bool = True, **kwargs):
        """Initialize the LogCompress operation.

        Args:
            clip (bool): Whether to clip the output to a dynamic range. Defaults to True.
        """
        super().__init__(
            input_data_type=DataTypes.ENVELOPE_DATA,
            output_data_type=DataTypes.IMAGE,
            **kwargs,
        )
        self.clip = clip

    def call(self, dynamic_range=None, **kwargs):
        """Apply logarithmic compression to data.

        Args:
            dynamic_range (tuple, optional): Dynamic range in dB. Defaults to (-60, 0).

        Returns:
            dict: Dictionary containing log-compressed data
        """
        data = kwargs[self.key]

        if dynamic_range is None:
            dynamic_range = ops.array(DEFAULT_DYNAMIC_RANGE)
        dynamic_range = ops.cast(dynamic_range, data.dtype)

        compressed_data = log_compress(data)
        if self.clip:
            compressed_data = ops.clip(compressed_data, dynamic_range[0], dynamic_range[1])

        return {self.output_key: compressed_data}


@ops_registry("reshape_grid")
class ReshapeGrid(Operation):
    """Reshape flat grid data to grid shape."""

    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, grid, **kwargs):
        """
        Args:
            - data (Tensor): The flat grid data of shape (..., n_pix, ...).
        Returns:
            - reshaped_data (Tensor): The reshaped data of shape (..., grid.shape, ...).
        """
        data = kwargs[self.key]
        reshaped_data = reshape_axis(data, grid.shape[:-1], self.axis + int(self.with_batch_dim))
        return {self.output_key: reshaped_data}


@ops_registry("apply_window")
class ApplyWindow(Operation):
    """Apply a window function to the input data along a specific axis.

    This operation can be used to zero out the end and/or beginning of the signal and apply a window
    of some size to transition from the zeroed region to the unmodified region.

    The axis is divided into five regions:
    [start (zero)] - [size (window)] - [middle (unmodified)] - [size (window)] - [end (zero)]
    """

    STATIC_PARAMS = ["axis", "size", "window_type", "start", "end"]

    def __init__(self, axis=-3, size=32, start=16, end=0, window_type="hanning", **kwargs):
        """
        Args:
            axis (int): Axis along which to apply the window.
            size (int): Size of the window to apply at the start and end regions.
            start (int): Number of elements to zero at the end.
            end (int): Number of elements to zero at the end.
            window_type (str): Type of window to apply. Supported types are "hanning" and "linear".
        """
        super().__init__(**kwargs)
        self.axis = axis
        self.size = int(size)
        self.start = int(start)
        self.end = int(end)
        self._check_inputs()
        self.window_type = window_type
        self.window = self._get_window(self.window_type, size, "float32")

    def _check_inputs(self):
        if self.start < 0:
            raise ValueError("start must be >= 0.")
        if self.end < 0:
            raise ValueError("end must be >= 0.")
        if self.size < 0:
            raise ValueError("size must be >= 0.")

    @staticmethod
    def _get_window(window_type, size, dtype):
        if window_type == "hanning":
            window = ops.hanning(size * 2)
        elif window_type == "linear":
            window = ops.concatenate(
                [ops.linspace(0.0, 1.0, size), ops.linspace(1.0, 0.0, size)], axis=0
            )
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
        return ops.cast(window, dtype)

    def call(self, **kwargs):
        data = kwargs[self.key]
        dtype = data.dtype
        axis = canonicalize_axis(self.axis, ops.ndim(data))

        length = ops.shape(data)[axis]

        if self.start + self.size * 2 + self.end > length:
            raise ValueError("start, size, and end are larger than the axis length.")

        window = ops.cast(self.window, dtype)

        ones = ops.ones((length,), dtype=dtype)
        mask = ops.concatenate(
            [
                ops.zeros((self.start,), dtype=dtype),
                window[: self.size],
                ones[self.size + self.start : -(self.end + self.size)],
                window[self.size :],
                ops.zeros((self.end,), dtype=dtype),
            ],
            axis=0,
        )

        shape = [1] * ops.ndim(data)
        shape[axis] = length
        mask = ops.reshape(mask, shape)

        return {self.output_key: data * mask}
