import uuid
from typing import Tuple

import keras
import numpy as np
from keras import ops

from zea import log
from zea.beamform.beamformer import tof_correction
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
