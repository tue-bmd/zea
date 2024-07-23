import numpy as np
import scipy as sc
import torch
import time


def pfield(probe, scan, options):
    """
    Compute the pressure field for ultrasound imaging.

    Args:
        probe (Probe): The ultrasound probe object.
        scan (Scan): The ultrasound scan object.
        options (dict): A dictionary containing various options for the computation.

    Returns:
        torch.Tensor: The normalized pressure field as a torch tensor.

    Raises:
        None

    """

    # options 
    FrequencyStep = options['FrequencyStep'] #% frequency step (scaling factor); default = 1. Higher is faster but less accurate.
    dBThresh = options['dBThresh'] # dB threshold for the frequency response; default = -60 dB. Higher is faster but less accurate.
    downsample = options['downsample'] # downsample the grid for faster computation; default = 1. Higher is faster but less accurate.
    downmix = options['downmix'] # downmixing the frequency to facilitate a smaller grid; default = 1. Higher requires lower number of grid points but is less accurate.
    
    alpha = options['alpha'] # default = 1; exponent to 'sharpen or smooth' the weighting (higher is sharper transitions) 
    perc = options['low_perc_th'] # default = 10; minium percentile threshold to keep in the weighting (higher is more aggressive)

    # medium params
    alpha_dB = 0 # currently we ignore attenuation in the compounding
    c = scan.sound_speed

    # probe params
    fc = scan.fc #% central frequency (Hz)
    fc = fc/downmix #% downmixing the frequency to facilitate a smaller grid

    BW = scan.bandwidth_percent #% pulse-echo 6dB fractional bandwidth of the probe (%)

    # pulse params
    NoW = 1 # number of waveforms in the pulse - we don't have this in the scan object

    # array params
    probe_geometry = probe.probe_geometry
    
    NumberOfElements = scan.n_el #% number of elements
    pitch = probe_geometry[1,0]-probe_geometry[0,0] #% element pitch

    kerf = 0.1*pitch # for now this is hardcoded - we don't have it in the probe object!
    ElementWidth = pitch - kerf

    n_transmits = len(scan.tx_apodizations)

    # %------------------------------------%
    # % POINT LOCATIONS, DISTANCES & GRIDS %
    # %------------------------------------%
    
    # subdivide elements into sub elements or not? (to satisfy Fraunhofer approximation)
    LambdaMin = c/(fc*(1+BW/200))
    M = np.int32(np.ceil(ElementWidth/LambdaMin))

    x_orig = scan.grid[:,:,0]
    z_orig = scan.grid[:,:,2]
    
    siz_orig = np.shape(x_orig)

    x = x_orig[::downsample,::downsample]
    z = z_orig[::downsample,::downsample]
    siz0 = np.shape(x)


    #%-- Coordinates of the points where pressure is needed
    x = x.flatten() 
    z = z.flatten()

    #%-- Centers of the tranducer elements (x- and z-coordinates)
    xe = (np.arange(0,NumberOfElements)-(NumberOfElements-1)/2)*pitch
    ze = np.zeros(NumberOfElements)
    THe = np.zeros(NumberOfElements)

    #%-- Centroids of the sub-elements
    #%-- note: Each elements is split into M sub-elements.
    #% X-position (xi) and Z-position (zi) of the centroids of the sub-elements
    #% (relative to the centers of the transducer elements).
    #% The values in xi,zi are in the range ]-ElementWidth/2 ElementWidth/2[
    #% (if M=1, then xi = zi = 0 for a rectilinear array).

    SegLength = ElementWidth/M
    tmp = -ElementWidth/2 + SegLength/2 + np.arange(0,M)*SegLength
    xi = tmp
    zi = np.zeros(M)

    # %-- Distances between the points and the transducer elements
    # Expand dimensions to allow broadcasting
    x_expanded = x[:, np.newaxis, np.newaxis]     # Shape: (4000, 1, 1)
    xi_expanded = xi[np.newaxis, :, np.newaxis]   # Shape: (1, 7, 1)
    xe_expanded = xe[np.newaxis, np.newaxis, :]   # Shape: (1, 1, 128)

    # Perform the operation
    dxi = x_expanded - xi_expanded - xe_expanded

    z_expanded = z[:, np.newaxis, np.newaxis]     # Shape: (4000, 1, 1)
    zi_expanded = zi[np.newaxis, :, np.newaxis]   # Shape: (1, 7, 1)
    ze_expanded = ze[np.newaxis, np.newaxis, :]   # Shape: (1, 1, 128)

    d2 = dxi**2+(z_expanded-zi_expanded-ze_expanded)**2
    r = np.sqrt(d2)

    #-- Angle between the normal to the transducer and the line joining the point and the transducer
    epss = np.finfo(np.float32).eps
    Th = np.arcsin((dxi+epss)/(np.sqrt(d2)+epss))-THe
    sinT = np.sin(Th)

    mysinc = lambda x: np.sin(np.abs(x)+epss)/(np.abs(x)+epss) 

    T = NoW/fc #% temporal pulse width
    wc = 2*np.pi*fc

    pulseSpectrum = lambda w: 1j*(mysinc(T*(w-wc)/2)-mysinc(T*(w+wc)/2))

    #-- FREQUENCY RESPONSE of the ensemble PZT + probe
    wB =BW*wc/100 # angular frequency bandwidth
    p = np.log(126)/np.log(epss+2*wc/wB) # p adjusts the shape
    probeSpectrum = lambda w: np.exp(-(np.abs(w-wc)/(wB/2/np.log(2)**(1/p)))**p)

    P_list = []
    for j in range(0,n_transmits):
        # print some progress
        if j%10==0:
            print(f'Precomputing pressure fields, transmit {j}/{n_transmits}')

        # delays and apodization of transmit event
        delaysTX = scan.t0_delays[j]
        idx = np.isnan(delaysTX)
        delaysTX[idx] = 0

        TXapodization = scan.tx_apodizations[j]
        TXapodization[np.any(idx)] = 0
        TXapodization = TXapodization.squeeze()

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

        df = 1/(np.max(r.flatten()/c) + np.max(delaysTX.flatten()))
        df = FrequencyStep*df # df is here an upper bound; it will be recalculated below

        #-- FREQUENCY SAMPLES
        Nf = 2*np.ceil(fc/df).astype(np.int32)+1 #% number of frequency samples
        f = np.linspace(0,2*fc,Nf) #% frequency samples
        df = f[1] #% update the frequency step

        #-- we keep the significant components only by using options.dBThresh
        S = np.abs(pulseSpectrum(2*np.pi*f)*probeSpectrum(2*np.pi*f))
        GdB = 20*np.log10(epss+S/(np.max(S))) #% gain in dB
        IDX = GdB>dBThresh

        f = f[IDX]
        nSampling = len(f)

        pulseSPECT = pulseSpectrum(2*np.pi*f) #% pulse spectrum
        probeSPECT = probeSpectrum(2*np.pi*f) #% probe response

        #%-- EXPONENTIAL arrays of size [numel(x) NumberOfElements M]
        kw = 2*np.pi*f[0]/c #% wavenumber
        kwa = alpha_dB/8.69*f[0]/1e6*1e2 #% attenuation-based wavenumber
        EXP = np.exp(-kwa*r + 1j*np.mod(kw*r,2*np.pi)); #% faster than exp(-kwa*r+1j*kw*r)
        
        #%-- Exponential array for the increment wavenumber dk
        dkw = 2*np.pi*df/c
        dkwa = alpha_dB/8.69*df/1e6*1e2
        EXPdf = np.exp((-dkwa + 1j*dkw)*r)

        EXP = EXP/np.sqrt(r)
        EXP = EXP*np.min(np.sqrt(r)) # normalize the field

        kc = 2*np.pi*fc/c #% center wavenumber
        DIR = mysinc(kc*SegLength/2*sinT) # directivity of each segment
        EXP = EXP*DIR

        # Render pressure field for all relevant frequencies and sum them up
        RP = 0
        RP = pfield_freqloop_torch(f, c, delaysTX, TXapodization, M, EXP, EXPdf, pulseSPECT, probeSPECT, z, nSampling)

        RP = RP.cpu().numpy()  # Convert back to numpy... not ideal but needs to work with sc.ndimage.zoom

        # % RMS acoustic pressure
        P = np.reshape(np.sqrt(RP),siz0)

        # resize P to exactly the original grid size
        P = sc.ndimage.zoom(P, (siz_orig[0]/siz0[0],siz_orig[1]/siz0[1]), order=1)

        P_list.append(P)
    
    P_norm = normalize(P_list, alpha = alpha, perc = perc)
    P_norm = torch.tensor(P_norm, dtype=torch.float32) # convert to torch tensor

    return P_norm

def normalize(P_list, alpha =1, perc = 10):
    # alpha: shape factor to tighten te beams (default = 1)
    # perc: percentile to keep (default = 10)

    # keep only the highest intensities
    P_arr  = np.array(P_list)
    P_arr[P_arr < np.percentile(P_arr, perc, axis=(1,2))[:,np.newaxis,np.newaxis]] = 0  

    P_arr=np.array(P_arr)**alpha
    P_norm = P_arr/(1e-10+np.sum(P_arr, axis=0))

    return P_norm


def pfield_freqloop_torch(f, c, delaysTX, TXapodization, M, EXP, EXPdf, pulseSPECT, probeSPECT, z, nSampling):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_tensor = torch.tensor(f, dtype=torch.float32, device=device)
    c_tensor = torch.tensor(c, dtype=torch.float32, device=device)
    delaysTX_tensor = torch.tensor(delaysTX, dtype=torch.float32, device=device)
    TXapodization_tensor = torch.tensor(TXapodization, dtype=torch.complex64, device=device)
    pulseSPECT_tensor = torch.tensor(pulseSPECT, dtype=torch.complex64, device=device)
    probeSPECT_tensor = torch.tensor(probeSPECT, dtype=torch.complex64, device=device)
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device)
    EXPdf_tensor = torch.tensor(EXPdf, dtype=torch.complex64, device=device)
    EXP_tensor = torch.tensor(EXP, dtype=torch.complex64, device=device)

    RP = 0
    kw = 2 * np.pi * f_tensor / c_tensor

    for k in range(nSampling):
        if k > 0:
            EXP_tensor *= EXPdf_tensor

        if M > 1:
            RPmono = torch.mean(EXP_tensor, dim=1)
        else:
            RPmono = EXP_tensor.squeeze()

        DELAPOD = torch.exp(1j * kw[k] * c_tensor * delaysTX_tensor) * TXapodization_tensor
        RPk = torch.matmul(RPmono, DELAPOD)

        RPk *= pulseSPECT_tensor[k] * probeSPECT_tensor[k]

        isOUT = z_tensor < 0
        RPk[isOUT] = 0

        RP += torch.abs(RPk) ** 2

    return RP


# Some plotting for debugging purposes
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import ipywidgets as widgets
from ipywidgets.embed import embed_minimal_html

def pfield_widget(pfields):
    def plot_image(index):
        plt.imshow(pfields[index].cpu().numpy(), cmap='hot')
        plt.title(f'Tx {index}')

    # Create an interactive slider
    num_images = len(pfields)
    slider = IntSlider(min=0, max=num_images-1, step=1, value=0, description='Transmit index')
    # Use the interact function to create the interactive plot
    interact(plot_image, index=slider)


# # Numpy implementation
# def pfield_freqloop(f, c, delaysTX, TXapodization, M, EXP, EXPdf, pulseSPECT, probeSPECT, z, nSampling):
#     RP =
#     for k in range(0,nSampling,1):
        
#         kw = 2*np.pi*f[k]/c #wavenumber
        
#         # %-- Exponential array of size [numel(x) NumberOfElements M]
#         # % For all k, we need: EXP = exp((-kwa+1i*kw)*r)
#         # %                         with kw = 2*pi*f(k)/c;
#         # %                     and with kwa = alpha_dB/8.7*f(k)/1e6*1e2;
#         # % Since f(k) = f(1)+(k-1)df, we use the following recursive product:
#         if k>0:
#             EXP = EXP*EXPdf

            
#         # %-- Radiation patterns of the single elements
#         # % They are the combination of the far-field patterns of the M small
#         # % segments that make upR the single elements
#         # %--

#         if M>1:
#             RPmono = np.mean(EXP,1) #% summation over the M small segments
#         else:
#             RPmono = EXP.squeeze()


#         #%-- Transmit delays + Transmit apodization
#         #% use of SUM: summation over the number of delay series (e.g. MLT)
#         DELAPOD = np.exp(1j*kw*c*delaysTX)*TXapodization
        
#         # %-- Summing the radiation patterns generating by all the elements
#         # % This is a matrix-vector product.
#         # %      RPmono is of size [number_of_points number_of_elements]
#         # %      DELAPOD is of size [number_of_elements 1]
#         RPk = RPmono.dot(DELAPOD)
#         #%- include spectrum responses:
#         RPk = pulseSPECT[k]*probeSPECT[k]*RPk

#         isOUT = z<0
#         RPk[isOUT] = 0

#         RP = RP + abs(RPk)**2 #% acoustic intensity
    
#     return RP