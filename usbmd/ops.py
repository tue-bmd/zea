from abc import ABC, abstractmethod

import numpy as np
import scipy
import tensorflow as tf
import torch
from scipy import ndimage, signal

from usbmd.display import scan_convert
from usbmd.probes import Probe
from usbmd.pytorch_ultrasound.processing import on_device_torch
from usbmd.registry import tf_beamformer_registry, torch_beamformer_registry
from usbmd.scan import Scan
from usbmd.tensorflow_ultrasound.processing import on_device_tf
from usbmd.utils import log
from usbmd.utils.checks import get_check
from usbmd.utils.config import Config
from usbmd.utils.utils import translate


class Operation(ABC):
    def __init__(
        self,
        name=None,
        input_data_type=None,
        output_data_type=None,
        batch_dim=True,
    ):
        self.name = name
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.batch_dim = batch_dim

        self._ops = None

    @property
    def ops(self):
        assert self._ops is not None, "ops package is not set"
        return self._ops

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @abstractmethod
    def process(self, data):
        return data

    def set_ops_pkg(self, ops):
        self.ops = ops

    def __call__(self, data, *args, **kwargs):
        if self.input_data_type:
            check = get_check(self.input_data_type)
            check(data, self.batch_dim)
        return self.process(data, *args, **kwargs)

    @property
    def _ready(self):
        # Check if the operation is ready to be used
        # and see if necessary parameters are set
        return True

    def initialize(self):
        # Initialize the operation
        pass

    def set_params(self, config: Config, scan: Scan, probe: Probe):
        self._assign_probe_params(probe)
        self._assign_scan_params(scan)
        self._assign_config_params(config)

    def _assign_config_params(self, config: Config):
        # Assign the config parameters to the operation
        pass

    def _assign_scan_params(self, scan: Scan):
        # Assign the scan parameters to the operation
        pass

    def _assign_probe_params(self, probe: Probe):
        # Assign the probe parameters to the operation
        pass


class Pipeline:
    def __init__(self, operations, ops=np, batch_dim=True, device=None):
        self.operations = operations
        self.device = device

        for operation in self.operations:
            operation.ops = ops
            operation.batch_dim = batch_dim

        # check if the operations are compatible
        for i in range(len(self.operations) - 1):
            if self.operations[i].output_data_type is None:
                continue
            if self.operations[i + 1].input_data_type is None:
                continue
            if (
                self.operations[i].output_data_type
                != self.operations[i + 1].input_data_type
            ):
                raise ValueError(
                    f"Operation {self.operations[i].name} output data type is not compatible "
                    f"with the input data type of operation {self.operations[i + 1].name}"
                )

    @property
    def ops(self):
        return self.operations[0].ops

    def on_device(self, func, data, device=None):
        if self.ops == np:
            return func(data)
        elif self.ops == tf:
            return on_device_tf(func, data, device=device)
        elif self.ops == torch:
            return on_device_torch(func, data, device=device)
        else:
            raise ValueError("Unsupported operations package.")

    def set_params(self, config: Config, scan: Scan, probe: Probe):
        for operation in self.operations:
            operation.set_params(config, scan, probe)

    def process(self, data):
        data = self.prepare_tensor(data)
        if not all(operation._ready for operation in self.operations):
            operations_not_ready = [
                operation.name for operation in self.operations if not operation._ready
            ]
            raise ValueError(
                log.error(
                    f"Operations {operations_not_ready} are not ready to be used, "
                    "please set parameters using `op.set_params(config, scan, probe)` "
                    "and initialize them using `op.initialize()`."
                )
            )

        if self.device:
            return self.on_device(self._process, data, device=self.device)
        return self._process(data)

    def _process(self, data):
        for operation in self.operations:
            data = operation(data)
        return data

    def initialize(self):
        for operation in self.operations:
            operation.initialize()

    # def compile(self, jit=False):
    #     if self.ops == np:
    #         return
    #     elif self.ops == tf:
    #         # jit compile the pipeline
    #         self.process = tf.function(self.process, jit_compile=jit)
    #     elif self.ops == jax:
    #         if not jit:
    #             return
    #         self.process = jax.jit(self.process)

    def prepare_tensor(self, x):
        if self.ops == np:
            return np.array(x)
        elif self.ops == tf:
            return tf.convert_to_tensor(x)
        elif self.ops == torch:
            return torch.tensor(x)
        else:
            raise ValueError("Unsupported operations package.")


class Beamform(Operation):
    def __init__(self, beamformer=None):
        super().__init__(
            name="Beamform",
            input_data_type="raw_data",
            output_data_type="beamformed_data",
        )

        self.beamformer = beamformer

    def initialize(self):
        if self.beamformer is not None:
            return

        beamformer_type = self.config.model.beamformer.type
        # pylint: disable=import-outside-toplevel
        if self.config.get("ml_library") == "torch":
            from usbmd.pytorch_ultrasound.layers.beamformers import get_beamformer

            _BEAMFORMER_TYPES = torch_beamformer_registry.registered_names()
        elif self.config.get("ml_library") == "tensorflow":
            from usbmd.tensorflow_ultrasound.layers.beamformers import get_beamformer

            _BEAMFORMER_TYPES = tf_beamformer_registry.registered_names()
        elif self.config.get("ml_library") == "disable":
            get_beamformer = None
            _BEAMFORMER_TYPES = []
        else:
            log.warning(
                "Cannot find ML library, proceeding without. Be aware of limitations."
            )
            get_beamformer = None
            _BEAMFORMER_TYPES = []

        assert beamformer_type in _BEAMFORMER_TYPES

        self.beamformer = get_beamformer(self.probe, self.scan, self.config)

    @property
    def _ready(self):
        return self.beamformer is not None

    def _assign_config_params(self, config: Config):
        self.config = config

    def _assign_scan_params(self, scan: Scan):
        self.scan = scan

    def _assign_probe_params(self, probe: Probe):
        self.probe = probe

    def process(self, data):
        if self.batch_dim is False:
            data = self.ops.expand_dims(data, axis=0)
        data = self.beamformer(data, probe=self.probe, scan=self.scan)
        if self.batch_dim is False:
            data = self.ops.squeeze(data, axis=0)
        return data


class Mean(Operation):
    def __init__(self, axis):
        super().__init__(
            name="Mean",
            input_data_type=None,
            output_data_type=None,
        )
        self.axis = axis

    def process(self, data):
        return self.ops.mean(data, axis=self.axis)


class Normalize(Operation):
    def __init__(self, output_range=None, input_range=None):
        """Normalize data to a given range.

        Args:
            output_range (Tuple, optional): Range to which data should be mapped.
                Defaults to (0, 1).
            input_range (Tuple, optional): Range of input data. If None, the range
                of the input data will be computed. Defaults to None.
        """
        super().__init__(
            name="Normalize",
            input_data_type=None,
            output_data_type=None,
        )
        self.output_range = output_range
        self.input_range = input_range

    def _assign_config_params(self, config):
        self.input_range = config.data.input_range
        self.output_range = None

    def process(self, data):
        if self.output_range is None:
            self.output_range = (0, 1)

        if self.input_range is None:
            minimum = self.ops.min(data)
            maximum = self.ops.max(data)
            self.input_range = (minimum, maximum)
        else:
            a_min, a_max = self.input_range
            data = self.ops.clip(data, a_min, a_max)
        return translate(data, self.input_range, self.output_range)


class LogCompress(Operation):
    def __init__(self, dynamic_range=None):
        super().__init__(
            name="LogCompress",
            input_data_type=None,
            output_data_type=None,
        )
        self.dynamic_range = dynamic_range

    def _assign_config_params(self, config):
        self.dynamic_range = config.data.dynamic_range

    def process(self, data):
        if self.dynamic_range is None:
            self.dynamic_range = (-60, 0)

        data[data == 0] = np.finfo(float).eps
        compressed_data = 20 * self.ops.log10(data)
        compressed_data = self.ops.clip(compressed_data, *self.dynamic_range)
        return compressed_data


class Downsample(Operation):
    def __init__(self, factor: int = None, phase: int = None, axis: int = -1):
        super().__init__(
            name="Downsample",
            input_data_type=None,
            output_data_type=None,
        )
        self.factor = factor
        self.phase = phase
        self.axis = axis

    def _assign_config_params(self, config):
        self.factor = config.scan.downsample

    def process(self, data):
        length = self.ops.shape(data)[self.axis]
        if self.phase is None:
            self.phase = 0
        sample_idx = self.ops.arange(self.phase, length, self.factor)
        return self.ops.take(data, sample_idx, axis=self.axis)


class Companding(Operation):
    def __init__(self, expand=False, comp_type=None, mu=255, A=87.6):
        super().__init__(
            name="Companding",
            input_data_type=None,
            output_data_type=None,
        )
        self.expand = expand
        self.comp_type = comp_type
        self.mu = mu
        self.A = A

    def _assign_config_params(self, config):
        self.expand = config.expand
        self.comp_type = config.comp_type
        self.mu = config.mu
        self.A = config.A

    def process(self, data):
        data = self.ops.clip(data, -1, 1)

        def mu_law_compress(x):
            y = (
                self.ops.sign(x)
                * self.ops.log(1 + self.mu * self.ops.abs(x))
                / self.ops.log(1 + self.mu)
            )
            return y

        def mu_law_expand(y):
            x = self.ops.sign(y) * ((1 + self.mu) ** (self.ops.abs(y)) - 1) / self.mu
            return x

        def a_law_compress(x):
            x_sign = self.ops.sign(x)
            x_abs = self.ops.abs(x)
            A_log = self.ops.log(self.A)

            idx_1 = self.ops.where((x_abs >= 0) & (x_abs < (1 / self.A)))
            idx_2 = self.ops.where((x_abs >= (1 / self.A)) & (x_abs <= 1))

            y = x_sign
            y[idx_1] *= self.A * x_abs[idx_1] / (1 + A_log)
            y[idx_2] *= (1 + self.ops.log(self.A * x_abs[idx_2])) / (1 + A_log)
            return y

        def a_law_expand(y):
            y_sign = self.ops.sign(y)
            y_abs = self.ops.abs(y)
            A_log = self.ops.log(self.A)

            idx_1 = self.ops.where((y_abs >= 0) & (y_abs < (1 / (1 + A_log))))
            idx_2 = self.ops.where((y_abs >= (1 / (1 + A_log))) & (y_abs <= 1))

            x = y_sign
            x[idx_1] *= y_abs[idx_1] * (1 + A_log) / self.A
            x[idx_2] *= self.ops.exp(y_abs[idx_2] * (1 + A_log) - 1) / self.A
            return x

        if self.comp_type.lower() == "mu":
            if self.expand:
                array_out = mu_law_expand(data)
            else:
                array_out = mu_law_compress(data)
        elif self.comp_type.lower() == "a":
            if self.expand:
                array_out = a_law_expand(data)
            else:
                array_out = a_law_compress(data)

        return array_out


class EnvelopeDetect(Operation):
    def __init__(self, axis=-3):
        super().__init__(
            name="EnvelopeDetection",
            input_data_type="beamformed_data",
            output_data_type="envelope_data",
        )
        self.axis = axis

    def process(self, data):
        if data.shape[-1] == 2:
            data = channels_to_complex(data, axis=-1)
        else:
            n_ax = data.shape[self.axis]
            M = 2 ** int(np.ceil(np.log2(n_ax)))
            data = scipy.signal.hilbert(data, N=M, axis=self.axis)
            data = self.ops.take(data, np.arange(n_ax), axis=self.axis)
            data = self.ops.squeeze(data, axis=-1)

        return self.ops.abs(data)


class Demodulate(Operation):
    def __init__(self, fs=None, fc=None, bandwidth=None, filter_coeff=None):
        super().__init__(
            name="Demodulate",
            input_data_type=None,
            output_data_type=None,
        )
        self.fs = fs
        self.fc = fc
        self.bandwidth = bandwidth
        self.filter_coeff = filter_coeff

    def process(self, data):
        if data.shape[-1] == 2:
            log.warning("Demodulation is not applicable to IQ data.")
            return data
        data = self.ops.squeeze(data, axis=-1)
        data = demodulate(data, self.fs, self.fc, self.bandwidth, self.filter_coeff)
        return complex_to_channels(data, axis=-1)

    def _assign_scan_params(self, scan):
        self.fs = scan.fs
        self.fc = scan.fc
        self.bandwidth = scan.bandwidth_percent

    def _assign_config_params(self, config):
        if config.scan.sampling_frequency is not None:
            self.fs = config.scan.sampling_frequency
        if config.scan.center_frequency is not None:
            self.fc = config.scan.center_frequency


class BandPassFilter(Operation):
    def __init__(self, axis=None, num_taps=None, fs=None, fc=None, f1=None, f2=None):
        super().__init__(
            name="BandPassFilter",
            input_data_type=None,
            output_data_type=None,
        )
        self.axis = axis
        self.num_taps = num_taps
        self.fs = fs
        self.fc = fc
        self.f1 = f1
        self.f2 = f2

        if self._ready:
            self.initialize()

    def initialize(self):
        self.filter = get_band_pass_filter(self.num_taps, self.fs, self.f1, self.f2)

    @property
    def _ready(self):
        return (
            self.axis is not None
            and self.num_taps is not None
            and self.fs is not None
            and self.f1 is not None
            and self.f2 is not None
        )

    def process(self, data):
        return ndimage.convolve1d(data, self.filter, mode="wrap", axis=self.axis)


class ScanConvert(Operation):
    def __init__(
        self,
        x_axis=None,
        z_axis=None,
        spline_order=1,
        fill_value=None,
        n_pixels=None,
    ):
        """Scan convert image.

        Args:
            image (ndarray): Input image (in polar coordinates).
            spline_order (int, optional): Order of spline interpolation.
                Defaults to 1.
            fill_value (float, optional): Value of the points that cannot be
                mapped from sample_points to grid. Defaults to None, which sets
                value to minimun of dynamic range.
            n_pixels (int, optional): Number of pixels (widht) in output image.
                height is derived by division of sqrt(2) of width. Defaults to None.
                in this case n_pixels is set using default arg of scan_convert func.

        Returns:
            image_sc (ndarray): Output image (converted to cartesian coordinates).

        """
        super().__init__(
            name="ScanConvert",
            input_data_type=None,
            output_data_type=None,
        )
        self.x_axis = x_axis
        self.z_axis = z_axis
        self.spline_order = spline_order
        self.fill_value = fill_value
        self.n_pixels = n_pixels
        self.probe_type = None

    @property
    def _ready(self):
        if self.probe_type != "phased":
            return True
        return (
            self.x_axis is not None
            and self.z_axis is not None
            and self.spline_order is not None
            and self.fill_value is not None
            and self.n_pixels is not None
        )

    def process(self, data):
        if self.probe_type != "phased":
            return data

        return scan_convert(
            data,
            self.x_axis,
            self.z_axis,
            n_pixels=self.n_pixels,
            spline_order=self.spline_order,
            fill_value=self.fill_value,
        )

    def _assign_config_params(self, config):
        self.n_pixels = config.data.output_size
        self.fill_value = config.data.dynamic_range[0]

    def _assign_probe_params(self, probe):
        self.probe_type = probe.probe_type
        if self.probe_type == "phased":
            self.x_axis = probe.angle_deg_axis

    def _assign_scan_params(self, scan):
        self.z_axis = scan.z_axis


def demodulate(rf_data, fs=None, fc=None, bandwidth=None, filter_coeff=None):
    """Demodulates an RF signal to complex base-band (IQ).

    Demodulates the radiofrequency (RF) bandpass signals and returns the
    Inphase/Quadrature (I/Q) components. IQ is a complex whose real (imaginary)
    part contains the in-phase (quadrature) component.

    This function operates (i.e. demodulates) on the RF signal over the
    (fast-) time axis which is assumed to be the last axis.

    Args:
        rf_data (ndarray): real valued input array of size [..., n_ax, n_el].
            second to last axis is fast-time axis.
        fs (float): the sampling frequency of the RF signals (in Hz).
            Only not necessary when filter_coeff is provided.
        fc (float, optional): represents the center frequency (in Hz).
            Defaults to None.
        bandwidth (float, optional): Bandwidth of RF signal in % of center
            frequency. Defaults to None.
            The bandwidth in % is defined by:
            B = Bandwidth_in_% = Bandwidth_in_Hz*(100/fc).
            The cutoff frequency:
            Wn = Bandwidth_in_Hz/Fs, i.e:
            Wn = B*(Fc/100)/Fs.
        filter_coeff (list, optional): (b, a), numerator and denominator coefficients
            of FIR filter for quadratic band pass filter. All other parameters are ignored
            if filter_coeff are provided. Instead the given filter_coeff is directly used.
            If not provided, a filter is derived from the other params (fs, fc, bandwidth).
            see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html

    Returns:
        iq_data (ndarray): complex valued base-band signal.

    """
    assert np.isreal(rf_data).all(), "RF must contain real RF signals."

    input_shape = rf_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, n_el = input_shape
    else:
        n_ax, n_el = input_shape

    if filter_coeff is None:
        assert fs is not None, "provide fs when no filter is given."
        # Time vector
        t = np.arange(n_ax) / fs
        t0 = 0
        t = t + t0

        # Estimate center frequency
        if fc is None:
            # Keep a maximum of 100 randomly selected scanlines
            idx = np.arange(n_el)
            if n_el > 100:
                idx = np.random.permutation(idx)[:100]
            # Power Spectrum
            P = np.sum(
                np.abs(np.fft.fft(np.take(rf_data, idx, axis=-1), axis=-2)) ** 2,
                axis=-1,
            )
            P = P[: n_ax // 2]
            # Carrier frequency
            idx = np.sum(np.arange(n_ax // 2) * P) / np.sum(P)
            fc = idx * fs / n_ax

        # Normalized cut-off frequency
        if bandwidth is None:
            Wn = min(2 * fc / fs, 0.5)
            bandwidth = fc * Wn
        else:
            assert np.isscalar(
                bandwidth
            ), "The signal bandwidth (in %) must be a scalar."
            assert (bandwidth > 0) & (
                bandwidth <= 200
            ), "The signal bandwidth (in %) must be within the interval of ]0,200]."
            # bandwidth in Hz
            bandwidth = fc * bandwidth / 100
            Wn = bandwidth / fs
        assert (Wn > 0) & (Wn <= 1), (
            "The normalized cutoff frequency is not within the interval of (0,1). "
            "Check the input parameters!"
        )

        # Down-mixing of the RF signals
        carrier = np.exp(-1j * 2 * np.pi * fc * t)
        # add the singleton dimensions
        carrier = np.reshape(carrier, (*[1] * (n_dim - 2), n_ax, 1))
        iq_data = rf_data * carrier

        # Low-pass filter
        N = 5
        b, a = scipy.signal.butter(N, Wn, "low")

        # factor 2: to preserve the envelope amplitude
        iq_data = scipy.signal.filtfilt(b, a, iq_data, axis=-2) * 2

        # Display a warning message if harmful aliasing is suspected
        # the RF signal is undersampled
        if fs < (2 * fc + bandwidth):
            # lower and higher frequencies of the bandpass signal
            fL = fc - bandwidth / 2
            fH = fc + bandwidth / 2
            n = fH // (fH - fL)
            harmless_aliasing = any(
                (2 * fH / np.arange(1, n) <= fs) & (fs <= 2 * fL / np.arange(1, n))
            )
            if not harmless_aliasing:
                log.warning(
                    "rf2iq:harmful_aliasing Harmful aliasing is present: the aliases"
                    " are not mutually exclusive!"
                )
    else:
        b, a = filter_coeff
        iq_data = scipy.signal.lfilter(b, a, rf_data, axis=-2) * 2

    return iq_data


def upmix(iq_data, fs, fc, upsampling_rate=6):
    """Upsamples and upmixes complex base-band signals (IQ) to RF.

    Args:
        iq_data (ndarray): complex valued input array of size [..., n_ax, n_el]. second
            to last axis is fast-time axis.
        fs (float): the sampling frequency of the input IQ signal (in Hz).
            resulting fs of RF data is upsampling_rate times higher.
        fc (float, optional): represents the center frequency (in Hz).

    Returns:
        rf_data (ndarray): output real valued rf data.

    """
    assert iq_data.dtype in [
        "complex64",
        "complex128",
    ], "IQ must contain all complex signals."

    input_shape = iq_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, _ = input_shape
    else:
        n_ax, _ = input_shape

    # Time vector
    n_ax *= upsampling_rate
    fs *= upsampling_rate

    t = np.arange(n_ax) / fs
    t0 = 0
    t = t + t0

    # interpolation
    iq_data_upsampled = signal.resample(iq_data, num=n_ax, axis=-2)

    # Up-mixing of the IQ signals
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    # add the singleton dimensions
    carrier = np.reshape(carrier, (*[1] * (n_dim - 2), n_ax, 1))

    rf_data = iq_data_upsampled * carrier
    rf_data = np.real(rf_data) * np.sqrt(2)

    return rf_data.astype(np.float32)


def apply_multi_band_pass_filter(
    beamformed_data,
    params,
    process=None,
    to_image=True,
    with_frame_dim=False,
):
    """Applies multiply band pass filters on beamformed data.

    Takes average in image domain of differend band passed filtered data if `to_image` set to true.
    Data is filtered in the RF / IQ domain. This function also can convert to image domain, since
    the compounding of filtered beamformed data takes place there (incoherent compounding).

    Args:
        beamformed_data (ndarray): input data, RF / IQ with shape [..., n_ax, n_el, n_ch].
            filtering is always applied over the n_ax axis.
        params (dict): dict with parameters for filter.
            Should include `num_taps`, `fs`, `fc` and two lists: `freqs` and `bandwidths`
            which define the filter characteristics. Lengths of those lists should
            be the same and is equal to the number of filters applied. Optionally the
            `units` can be specified, which is for instance `Hz` or `MHz`. Defaults to `Hz`.
        process (processing.Process, optional): process class for converting
            beamformed data to image domain. Defaults to None. Should be provided if `to_image`
            is set to True.
        to_image (bool, optional): Whether to convert to image domain or not.
        with_frame_dim (bool, optional): Whether to process data with frame (batch of images).
            Defaults to False. In that case data is processed for a single image.

    Returns:
        image: resulting image in image domain if `to_image` is set to True. Otherwise, a list
            is returned with the filtered beamformed_data with size of number of filters applied.
            each filtered beamformed_data has same shape as input [..., n_ax, n_el, n_ch].

    Example:
        >>> params = {
        >>>     'num_taps': 128,
        >>>     'fs': 50e6,
        >>>     'fc': 5e6,
        >>>     'freqs': [-2.5, 0, 2.5],
        >>>     'bandwidths': [1, 1, 1],
        >>>     'units': 'MHz'
        >>> }
        >>> process = usbmd.processing.Process()
        >>> image = apply_multi_band_pass_filter(beamformed_data, params, process)
    """
    # removing channel axis here (going to complex if IQ) for filtering
    # adding it back later
    if (beamformed_data).shape[-1] == 1:
        modtype = "rf"
        beamformed_data = np.squeeze(beamformed_data, axis=-1)
    elif (beamformed_data).shape[-1] == 2:
        modtype = "iq"
        beamformed_data = channels_to_complex(beamformed_data, axis=-1)
    else:
        raise ValueError(
            f"Unknown number of channels: {beamformed_data.shape[-1]}, "
            "should be 1 or 2 for RF / IQ respectively."
        )

    if to_image:
        assert (
            process is not None
        ), "Please provide process class to convert beamformed data to image domain."

    if "units" in params:
        units = ["Hz", "kHz", "MHz", "GHz"]
        factors = [1, 1e3, 1e6, 1e9]
        unit_factor = factors[units.index(params["units"])]
    else:
        unit_factor = 1

    offsets = params["freqs"] * unit_factor
    bandwidths = params["bandwidths"] * unit_factor
    num_taps = params["num_taps"]
    # make sure fs is correct for IQ (downsampled)
    fs = params["fs"] * unit_factor
    fc = params["fc"] * unit_factor  # fc is only used when RF

    if modtype == "iq":
        fc = 0  # fc is automatically set to zero if IQ
        params = [
            {"num_taps": num_taps, "fs": fs, "f": fc - offset, "bw": bw}
            for offset, bw in zip(offsets, bandwidths)
        ]
    elif modtype == "rf":
        params = [
            {
                "num_taps": num_taps,
                "fs": fs,
                "f1": fc - offset - bw / 2,
                "f2": fc - offset + bw / 2,
            }
            for offset, bw in zip(offsets, bandwidths)
        ]

    images = []
    for param in params:
        if modtype == "iq":
            filter_weights = get_low_pass_iq_filter(**param)
        elif modtype == "rf":
            filter_weights = get_band_pass_filter(**param)

        axial_axis = -2
        data_filtered = ndimage.convolve1d(
            beamformed_data, filter_weights, mode="wrap", axis=axial_axis
        )

        # adding back the channel dimension for process pipeline
        if modtype == "iq":
            data_filtered = complex_to_channels(data_filtered, axis=-1)
        else:
            data_filtered = np.expand_dims(data_filtered, axis=-1)
        if to_image:
            env_data = process.envelope_detect(
                data_filtered, with_frame_dim=with_frame_dim
            )
            images_filtered = process.run(
                env_data,
                dtype="envelope_data",
                to_dtype="image",
                with_frame_dim=with_frame_dim,
            )
        else:
            images_filtered = data_filtered

        images.append(images_filtered)

    # only compound the result in image domain
    if to_image:
        images = np.mean(np.stack(images), axis=0)
    return images


def get_band_pass_filter(num_taps, fs, f1, f2):
    """Band pass filter

    Args:
        num_taps (int): number of taps in filter.
        fs (float): sample frequency in Hz.
        f1 (float): cutoff frequency in Hz of left band edge.
        f2 (float): cutoff frequency in Hz of right band edge.

    Returns:
        ndarray: band pass filter
    """
    bpf = signal.firwin(num_taps, [f1, f2], pass_zero=False, fs=fs)
    return bpf


def get_low_pass_iq_filter(num_taps, fs, f, bw):
    """Design low pass filter.

    LPF with num_taps points and cutoff at bw / 2

    Args:
        num_taps (int): number of taps in filter.
        fs (float): sample frequency.
        f (float): center frequency.
        bw (float): bandwidth in Hz.
    Raises:
        AssertionError: if cutoff frequency (bw / 2) is not within (0, fs / 2)

    Returns:
        ndarray: complex LP filter
    """
    assert (bw / 2 > 0) & (bw / 2 < fs / 2), log.error(
        "Cutoff frequency must be within (0, fs / 2), "
        f"got {bw / 2} Hz, must be within (0, {fs / 2}) Hz"
    )
    t_qbp = np.arange(num_taps) / fs
    lpf = signal.firwin(num_taps, bw / 2, pass_zero=True, fs=fs) * np.exp(
        1j * 2 * np.pi * f * t_qbp
    )
    return lpf


def complex_to_channels(complex_data, axis=-1, ops=np):
    """Unroll complex data to separate channels.

    Args:
        complex_data (complex ndarray): complex input data.
        axis (int, optional): on which axis to extend. Defaults to -1.

    Returns:
        ndarray: real array with real and imaginary components
            unrolled over two channels at axis.
    """
    assert ops.iscomplex(complex_data).any()
    q_data = ops.imag(complex_data)
    i_data = ops.real(complex_data)

    i_data = ops.expand_dims(i_data, axis=axis)
    q_data = ops.expand_dims(q_data, axis=axis)

    iq_data = ops.concatenate((i_data, q_data), axis=axis)
    return iq_data


def channels_to_complex(data, axis=-1, ops=np):
    """Convert array with real and imaginary components at
    different channels to complex data array.

    Args:
        data (ndarray): input data, with at 0 index of axis
            real component and 1 index of axis the imaginary.
        axis (int, optional): axis with real and imag component
            as separate channels. Defaults to -1.

    Returns:
        ndarray: complex array with real and imaginary components.
    """
    assert data.shape[axis] == 2
    return ops.take(data, 0, axis=axis) + 1j * ops.take(data, 1, axis=axis)


# def to_image(envelope_data, dynamic_range=None, input_range=None):
#     """Convert envelope data to image.

#     envelope_data -> normalize [0, 1] -> log compress -> [*dynamic_range]

#     Args:
#         envelope_data (ndarray): Input array of envelope data.
#         dynamic_range (tuple): Dynamic range of output image values.
#             Defaults to [-60, 0] dB.
#         input_range (tuple): Range of input values (if none, min max is used).
#     Returns:
#         image (ndarray): Output image.

#     """
#     if dynamic_range is None:
#         dynamic_range = (-60, 0)

#     assert (
#         dynamic_range[0] < dynamic_range[1]
#     ), "dynamic range must be increasing tuple of size 2"
#     assert dynamic_range[0] < 0, "dynamic range must be negative (dB)"
#     assert (np.array(dynamic_range) <= 0).all(), "dynamic range must be negative (dB)"
#     output_range = (0, 1)
#     image = normalize(envelope_data, output_range, input_range)
#     image = log_compress(image, dynamic_range)
#     return image


# def rf2iq(
#     rf_data,
#     demod_type=None,
#     fs=None,
#     fc=None,
#     bandwidth=None,
#     filter_coeff=None,
#     separate_channels=False,
# ):
#     """Convert rf to iq.

#     TODO:
#         - add different type of demodulation methods
#             gabor
#             manual demodulation
#     Args:
#         rf_data (ndarray): real valued input array of size [..., n_ax, n_el].
#             second to last axis is fast-time axis.
#         demod_type (str, optional): demodulation type. Defaults to 'manual'.
#         fs (float): the sampling frequency of the RF signals (in Hz).
#             Only not necessary when filter_coeff is provided.
#         fc (float, optional): represents the center frequency (in Hz).
#             Defaults to None.
#         bandwidth (float, optional): Bandwidth of RF signal in % of center
#             frequency. Defaults to None.
#         filter_coeff (list, optional): (b, a), numerator and denominator coefficients
#             of FIR filter. see demodulation for description.
#         separate_channels (bool, optional): Whether to output the iq, imag and
#             real componants as separates channels. Defaults to False. If False,
#             iq is not split in channels and data is complex type.

#     Returns:
#         iq_data (ndarray): output iq data of size [n_ax, n_el] or [n_ax, n_el, 2].

#     """
#     if demod_type in [None, "manual"]:
#         if fs is None:
#             raise ValueError("fs must be provided if demod_type is manual")
#         iq_data = demodulate(rf_data, fs, fc, bandwidth, filter_coeff)
#     elif demod_type == "hilbert":
#         n_ax = rf_data.shape[-2]
#         M = 2 ** int(np.ceil(np.log2(n_ax)))
#         iq_data = signal.hilbert(rf_data, N=M, axis=-2)
#         iq_data = np.take(iq_data, np.arange(n_ax), axis=-2)
#     else:
#         raise ValueError(f"Unknown demodulation type: {demod_type}")

#     if separate_channels:
#         iq_data = complex_to_channels(iq_data, axis=-1)

#     return iq_data
