"""Automatic pressure field computation used for compounding multiple Tx events

- **Author(s)**     : Ruud van Sloun (initial code), Tristan Stevens (transferred to keras)
- **Date**          : 2024-07-24, 20240-10-09
"""

import keras
import numpy as np
from keras import ops

from usbmd.utils import log


def compute_pfield(
    scan,
    FrequencyStep=4,
    dBThresh=-1,
    downsample=10,
    downmix=4,
    alpha=1,
    perc=10,
):
    """
    Compute the pressure field for ultrasound imaging.

    Args:
        scan (Scan): The ultrasound scan object.
        FrequencyStep (int, optional): Frequency step. Default is 4.
            Higher is faster but less accurate.
        dBThresh (int, optional): dB threshold. Default is -1.
            Higher is faster but less accurate.
        downsample (int, optional): Downsample the grid for faster computation.
            Default is 10. Higher is faster but less accurate.
        downmix (int, optional): Downmixing the frequency to facilitate a smaller grid.
            Default is 4. Higher requires lower number of grid points but is less accurate.
        alpha (float, optional): Exponent to 'sharpen or smooth' the weighting. Default is 1.
        perc (int, optional): minium percentile threshold to keep in the weighting
            Higher is more aggressive) Default is 10.

    Returns:
        ops.array: The normalized pressure field (across tx events).
    """
    # medium params
    alpha_dB = 0  # currently we ignore attenuation in the compounding

    c = scan.sound_speed

    # probe params
    fc = scan.fc  # % central frequency (Hz)
    fc = fc / downmix  # % downmixing the frequency to facilitate a smaller grid

    BW = (
        scan.bandwidth_percent
    )  # % pulse-echo 6dB fractional bandwidth of the probe (%)

    # pulse params
    NoW = 1  # number of waveforms in the pulse - we don't have this in the scan object

    # array params
    probe_geometry = scan.probe_geometry

    NumberOfElements = scan.n_el  # % number of elements
    pitch = probe_geometry[1, 0] - probe_geometry[0, 0]  # % element pitch

    kerf = (
        0.1 * pitch
    )  # for now this is hardcoded - we don't have it in the probe object!
    ElementWidth = pitch - kerf

    n_transmits = len(scan.tx_apodizations)

    # %------------------------------------%
    # % POINT LOCATIONS, DISTANCES & GRIDS %
    # %------------------------------------%

    # subdivide elements into sub elements or not? (to satisfy Fraunhofer approximation)
    LambdaMin = c / (fc * (1 + BW / 200))
    M = ops.cast(ops.ceil(ElementWidth / LambdaMin), "int32")

    x_orig = scan.grid[:, :, 0]
    z_orig = scan.grid[:, :, 2]

    siz_orig = ops.shape(x_orig)

    x = x_orig[::downsample, ::downsample]
    z = z_orig[::downsample, ::downsample]
    siz0 = ops.shape(x)

    # %-- Coordinates of the points where pressure is needed
    x = ops.reshape(x, (-1,))
    z = ops.reshape(z, (-1,))

    # %-- Centers of the tranducer elements (x- and z-coordinates)
    xe = (ops.arange(0.0, NumberOfElements) - (NumberOfElements - 1) / 2) * pitch
    ze = ops.zeros(NumberOfElements)
    THe = ops.zeros(NumberOfElements)

    # %-- Centroids of the sub-elements
    # %-- note: Each elements is split into M sub-elements.
    # % X-position (xi) and Z-position (zi) of the centroids of the sub-elements
    # % (relative to the centers of the transducer elements).
    # % The values in xi,zi are in the range ]-ElementWidth/2 ElementWidth/2[
    # % (if M=1, then xi = zi = 0 for a rectilinear array).

    SegLength = ElementWidth / M
    tmp = -ElementWidth / 2 + SegLength / 2 + ops.arange(0, M) * SegLength
    xi = tmp
    zi = ops.zeros((M,))

    # %-- Distances between the points and the transducer elements
    # Expand dimensions to allow broadcasting
    x_expanded = x[:, None, None]  # Shape: (4000, 1, 1)
    xi_expanded = xi[None, :, None]  # Shape: (1, 7, 1)
    xe_expanded = xe[None, None, :]  # Shape: (1, 1, 128)

    # Perform the operation
    dxi = x_expanded - xi_expanded - xe_expanded

    z_expanded = z[:, None, None]  # Shape: (4000, 1, 1)
    zi_expanded = zi[None, :, None]  # Shape: (1, 7, 1)
    ze_expanded = ze[None, None, :]  # Shape: (1, 1, 128)

    d2 = dxi**2 + (z_expanded - zi_expanded - ze_expanded) ** 2
    r = ops.sqrt(d2)

    # Angle between the normal to the transducer and the line joining
    # the point and the transducer
    epss = keras.config.epsilon()
    Th = ops.arcsin((dxi + epss) / (ops.sqrt(d2) + epss)) - THe
    sinT = ops.sin(Th)

    mysinc = lambda x: ops.sin(ops.abs(x) + epss) / (ops.abs(x) + epss)

    T = NoW / fc  # % temporal pulse width
    wc = 2 * np.pi * fc

    pulseSpectrum = lambda w: 1j * (mysinc(T * (w - wc) / 2) - mysinc(T * (w + wc) / 2))

    # -- FREQUENCY RESPONSE of the ensemble PZT + probe
    wB = BW * wc / 100  # angular frequency bandwidth
    p = ops.log(126) / ops.log(epss + 2 * wc / wB)  # p adjusts the shape
    probeSpectrum = lambda w: ops.exp(
        -((ops.abs(w - wc) / (wB / 2 / ops.log(2) ** (1 / p))) ** p)
    )

    P_list = []
    for j in range(0, n_transmits):
        # print some progress
        if j % 10 == 0:
            log.info(f"Precomputing pressure fields, transmit {j}/{n_transmits}")

        # delays and apodization of transmit event
        delaysTX = ops.convert_to_tensor(scan.t0_delays[j])
        idx = ops.isnan(delaysTX)
        delaysTX[idx] = 0

        TXapodization = ops.convert_to_tensor(scan.tx_apodizations[j])
        TXapodization[ops.any(idx)] = 0
        TXapodization = ops.squeeze(TXapodization)

        # The frequency response is a pulse-echo (transmit + receive) response. A
        # square root is thus required when calculating the pressure field:
        # Note: The spectrum of the pulse (pulseSpectrum) will be then multiplied
        # by the frequency-domain tapering window of the transducer (probeSpectrum)
        # The frequency step df is chosen to avoid interferences due to
        # inadequate discretization.
        # -- df = frequency step (must be sufficiently small):
        # One has exp[-i(k r + w delay)] = exp[-2i pi(f r/c + f delay)] in the Eq.
        # One wants: the phase increment 2pi(df r/c + df delay) be < 2pi.
        # Therefore: df < 1/(r/c + delay).

        df = 1 / (ops.max(r.flatten() / c) + ops.max(delaysTX.flatten()))
        df = (
            FrequencyStep * df
        )  # df is here an upper bound; it will be recalculated below

        # -- FREQUENCY SAMPLES
        Nf = (
            2 * ops.cast(ops.ceil(fc / df), "int32") + 1
        )  # % number of frequency samples
        f = ops.linspace(0, 2 * fc, Nf)  # % frequency samples
        df = f[1]  # % update the frequency step

        # -- we keep the significant components only by using options.dBThresh
        S = ops.abs(pulseSpectrum(2 * np.pi * f) * probeSpectrum(2 * np.pi * f))
        GdB = 20 * ops.log10(epss + S / (ops.max(S)))  # % gain in dB
        IDX = GdB > dBThresh

        f = f[IDX]
        nSampling = len(f)

        pulseSPECT = pulseSpectrum(2 * np.pi * f)  # % pulse spectrum
        probeSPECT = probeSpectrum(2 * np.pi * f)  # % probe response

        # %-- EXPONENTIAL arrays of size [numel(x) NumberOfElements M]
        kw = 2 * np.pi * f[0] / c  # % wavenumber
        kwa = alpha_dB / 8.69 * f[0] / 1e6 * 1e2  # % attenuation-based wavenumber
        EXP = ops.exp(-kwa * r + 1j * ops.mod(kw * r, 2 * np.pi))
        # % faster than exp(-kwa*r+1j*kw*r)

        # %-- Exponential array for the increment wavenumber dk
        dkw = 2 * np.pi * df / c
        dkwa = alpha_dB / 8.69 * df / 1e6 * 1e2
        EXPdf = ops.exp((-dkwa + 1j * dkw) * r)

        EXP = EXP / ops.sqrt(r)
        EXP = EXP * ops.min(ops.sqrt(r))  # normalize the field

        kc = 2 * np.pi * fc / c  # % center wavenumber
        DIR = mysinc(kc * SegLength / 2 * sinT)  # directivity of each segment
        EXP = EXP * DIR

        # Render pressure field for all relevant frequencies and sum them up
        RP = 0
        RP = pfield_freqloop_torch(
            f,
            c,
            delaysTX,
            TXapodization,
            M,
            EXP,
            EXPdf,
            pulseSPECT,
            probeSPECT,
            z,
            nSampling,
        )

        RP = (
            RP.cpu().numpy()
        )  # Convert back to numpy... not ideal but needs to work with sc.ndimage.zoom

        # % RMS acoustic pressure
        P = ops.reshape(ops.sqrt(RP), siz0)

        # resize P to exactly the original grid size
        # P = sc.ndimage.zoom(P, (siz_orig[0] / siz0[0], siz_orig[1] / siz0[1]), order=1)
        P = ops.squeeze(
            ops.image.resize(P[..., None], siz_orig, interpolation="nearest"), axis=-1
        )

        P_list.append(P)

    P_arr = ops.convert_to_tensor(P_list)

    P_norm = normalize(P_arr, alpha=alpha, perc=perc)

    return P_norm


def normalize(P_arr, alpha=1, perc=10):
    """
    Normalize the input array of intensities.

    Args:
        P_arr (array): Sequence of intensity arrays.
        alpha (float, optional): Shape factor to tighten the beams. Default is 1.
        perc (int, optional): Percentile to keep. Default is 10.

    Returns:
        numpy.ndarray: Normalized intensity array.

    """
    # keep only the highest intensities
    # P_arr[P_arr < ops.percentile(P_arr, perc, axis=(1, 2))[:, None, None]] = 0

    # let's do manual percentile calculation
    # Flatten the last two dimensions, sort, and reshape back
    P_flat = ops.reshape(P_arr, (P_arr.shape[0], -1))
    P_sorted = ops.sort(P_flat, axis=1)
    perc_value = P_sorted[:, int(P_arr.shape[1] * P_arr.shape[2] * perc / 100)]
    P_arr = ops.where(P_arr < perc_value[:, None, None], 0, P_arr)

    P_arr = ops.convert_to_tensor(P_arr) ** alpha
    P_norm = P_arr / (keras.config.epsilon() + ops.sum(P_arr, axis=0))

    return P_norm


def pfield_freqloop_torch(
    f, c, delaysTX, TXapodization, M, EXP, EXPdf, pulseSPECT, probeSPECT, z, nSampling
):
    """
    Calculates the pressure field using frequency loop method in PyTorch.

    Args:
        f (list): List of frequencies.
        c (float): Speed of sound.
        delaysTX (list): List of transmit delays.
        TXapodization (list): List of transmit apodization values.
        M (int): Number of elements in the array.
        EXP (list): List of complex exponentials.
        EXPdf (list): List of complex exponential frequency shifts.
        pulseSPECT (list): List of pulse spectra.
        probeSPECT (list): List of probe spectra.
        z (list): List of z-coordinates.
        nSampling (int): Number of samples.

    Returns:
        RP (Tensor): Pressure field.

    """

    RP = 0
    kw = 2 * np.pi * f / c

    for k in range(nSampling):
        if k > 0:
            EXP *= EXPdf

        if M > 1:
            RPmono = ops.mean(EXP, axis=1)
        else:
            RPmono = ops.squeeze(EXP)

        DELAPOD = ops.exp(1j * kw[k] * c * delaysTX) * TXapodization
        RPk = ops.matmul(RPmono, DELAPOD)

        RPk *= pulseSPECT[k] * probeSPECT[k]

        isOUT = z < 0
        RPk[isOUT] = 0

        RP += ops.abs(RPk) ** 2

    return RP
