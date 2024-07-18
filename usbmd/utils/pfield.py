import numpy as np

def pfield(x,z,param,options):
    # medium params
    alpha_dB = param['attenuation']
    c = param['c']

    # waveform params
    fc = param['fc'] #% central frequency (Hz)
    BW = param['bandwidth'] #% pulse-echo 6dB fractional bandwidth (%)
    NoW = param['TXnow'] # number of waveforms in the pulse

    # array params
    NumberOfElements = param['Nelements'] #% number of elements
    pitch = param['pitch']
    ElementWidth = pitch - param['kerf']

    # delays and apodization of transmit event
    delaysTX = param['delaysTX']
    idx = np.isnan(delaysTX)
    delaysTX[idx] = 0
    TXapodization = param['TXapodization']
    TXapodization[np.any(idx)] = 0

    APOD = param['TXapodization'].squeeze()

    # options for acceleration
    FrequencyStep = options['FrequencyStep'] #% frequency step (scaling factor); default = 1. Higher is faster but less accurate.
    dBThresh = options['dBThresh'] # % dB threshold for the frequency response; default = -60 dB. Higher is faster but less accurate.
    
    # subdivide elements into sub elements or not? (to satisfy Fraunhofer approximation)
    LambdaMin = c/(fc*(1+BW/200))
    M = np.int32(np.ceil(ElementWidth/LambdaMin))

    # %------------------------------------%
    # % POINT LOCATIONS, DISTANCES & GRIDS %
    # %------------------------------------%

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

    # %-- Angle between the normal to the transducer and the line joining the point and the transducer
    epss = np.finfo(np.float32).eps
    Th = np.arcsin((dxi+epss)/(np.sqrt(d2)+epss))-THe
    sinT = np.sin(Th)

    mysinc = lambda x: np.sin(np.abs(x)+epss)/(np.abs(x)+epss) 

    T = NoW/fc #% temporal pulse width
    wc = 2*np.pi*fc

    pulseSpectrum = lambda w: 1j*(mysinc(T*(w-wc)/2)-mysinc(T*(w+wc)/2))


    #%-- FREQUENCY RESPONSE of the ensemble PZT + probe
    #% We want a generalized normal window (6dB-bandwidth = PARAM.bandwidth)
    #% (https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window)

    wB =BW*wc/100 # angular frequency bandwidth
    p = np.log(126)/np.log(2*wc/wB) # p adjusts the shape
    probeSpectrum = lambda w: np.exp(-(np.abs(w-wc)/(wB/2/np.log(2)**(1/p)))**p)

    #% The frequency response is a pulse-echo (transmit + receive) response. A
    #% square root is thus required when calculating the pressure field:

    #% Note: The spectrum of the pulse (pulseSpectrum) will be then multiplied
    #% by the frequency-domain tapering window of the transducer (probeSpectrum)

    # % The frequency step df is chosen to avoid interferences due to
    # % inadequate discretization.
    # % -- df = frequency step (must be sufficiently small):
    # % One has exp[-i(k r + w delay)] = exp[-2i pi(f r/c + f delay)] in the Eq.
    # % One wants: the phase increment 2pi(df r/c + df delay) be < 2pi.
    # % Therefore: df < 1/(r/c + delay).

    df = 1/(np.max(r.flatten()/c) + np.max(delaysTX.flatten()))
    df = FrequencyStep*df
    #% note: df is here an upper bound; it will be recalculated below

    #%-- FREQUENCY SAMPLES
    Nf = 2*np.ceil(fc/df).astype(np.int32)+1 #% number of frequency samples
    f = np.linspace(0,2*fc,Nf) #% frequency samples
    df = f[1] #% update the frequency step

    #%- we keep the significant components only by using options.dBThresh
    S = np.abs(pulseSpectrum(2*np.pi*f)*probeSpectrum(2*np.pi*f))
    GdB = 20*np.log10(epss+S/(np.max(S))) #% gain in dB
    IDX = GdB>dBThresh

    f = f[IDX]
    nSampling = len(f)

    #print(f'Number of frequency samples above dB threshold: {nSampling} out of {Nf}')

    pulseSPECT = pulseSpectrum(2*np.pi*f) #% pulse spectrum
    probeSPECT = probeSpectrum(2*np.pi*f) #% probe response

    #%-- Initialization
    RP = 0 #% RP = Radiation Pattern

    #%-- Obliquity factor (baffle property)
    ObliFac = 1

    #%-- EXPONENTIAL arrays of size [numel(x) NumberOfElements M]
    kw = 2*np.pi*f[0]/c #% wavenumber
    kwa = alpha_dB/8.69*f[0]/1e6*1e2 #% attenuation-based wavenumber
    EXP = np.exp(-kwa*r + 1j*np.mod(kw*r,2*np.pi)); #% faster than exp(-kwa*r+1j*kw*r)
    #%-- Exponential array for the increment wavenumber dk
    dkw = 2*np.pi*df/c
    dkwa = alpha_dB/8.69*df/1e6*1e2
    EXPdf = np.exp((-dkwa + 1j*dkw)*r)

    EXP = EXP*ObliFac/np.sqrt(r)

    kc = 2*np.pi*fc/c #% center wavenumber
    DIR = mysinc(kc*SegLength/2*sinT) # directivity of each segment
    EXP = EXP*DIR

    # %-----------------------------%
    # % SUMMATION OVER THE SPECTRUM %
    # %-----------------------------%

    for k in range(0,nSampling,1):
    #for k in range(70,150,1):

        kw = 2*np.pi*f[k]/c #wavenumber
        
        # %-- Exponential array of size [numel(x) NumberOfElements M]
        # % For all k, we need: EXP = exp((-kwa+1i*kw)*r)
        # %                         with kw = 2*pi*f(k)/c;
        # %                     and with kwa = alpha_dB/8.7*f(k)/1e6*1e2;
        # % Since f(k) = f(1)+(k-1)df, we use the following recursive product:
        if k>0:
            EXP = EXP*EXPdf

            
        # %-- Radiation patterns of the single elements
        # % They are the combination of the far-field patterns of the M small
        # % segments that make upR the single elements
        # %--

        if M>1:
            RPmono = np.mean(EXP,1) #% summation over the M small segments
            RPmono = EXP[:,0,:]
        else:
            RPmono = EXP.squeeze()


        #%-- Transmit delays + Transmit apodization
        #% use of SUM: summation over the number of delay series (e.g. MLT)
        DELAPOD = np.sum(np.exp(1j*kw*c*delaysTX),1)*APOD
        
        # %-- Summing the radiation patterns generating by all the elements
        # % This is a matrix-vector product.
        # %      RPmono is of size [number_of_points number_of_elements]
        # %      DELAPOD is of size [number_of_elements 1]
        RPk = RPmono.dot(DELAPOD)
        #%- include spectrum responses:
        RPk = pulseSPECT[k]*probeSPECT[k]*RPk

        isOUT = z<0
        RPk[isOUT] = 0

        RP = RP + abs(RPk)**2 #% acoustic intensity

        
    # % RMS acoustic pressure
    P = np.reshape(np.sqrt(RP),siz0)
    return P

#%%

def example():
    #% Grid
    Nx = 50
    Nz = 50
    x = np.linspace(-4e-2,4e-2,Nx)
    z = np.linspace(0,10e-2,Nz)
    [x,z] = np.meshgrid(x,z)


    options = {'FrequencyStep': 2, 'dBThresh': -10, 'FullFrequencyDirectivity': False}


    L = 10 # number of elements per transmit event = 2*L+1
    P_list = []
    for i in range(0,128,1):

        delaysTX = np.zeros((128,1)) # size number of elements x number of delay series (e.g. MLT)
        apodTX = np.zeros(128)
        apodTX[np.maximum(i-L,0):np.minimum(i+L,128-1)] = 1
        
        # apodTX = np.ones(128)
        # scale = 1-2*i/128
        # delaysTX = np.linspace(-1e-6*scale,1e-6*scale,128).reshape(128,1)

        param = {'Nelements': 128, 'pitch': 0.3e-3, 'fc': 2.7e6, 
                'kerf': 0.1e-3, 'radius': np.infty, 'bandwidth': 75, 
                'c': 1540, 'attenuation': 0, 
                'TXapodization': apodTX, 'delaysTX': delaysTX, 'TXnow': 1}


        #% RMS pressure field calculation
        start = time.time()
        P = pfield(x,z,param,options)
        end = time.time()

        #print(f'Progress: {100*i/128}%, Elapsed time: {end-start}')


        P_list.append(P)


        
    alpha = 1 # shape factor to tighten te beams

    # keep only the highest intensities
    P_arr[P_arr < np.percentile(P_arr, 10, axis=(1,2))[:,np.newaxis,np.newaxis]] = 0  

    P_arr=np.array(P_arr)**alpha
    P_norm = P_arr/(1+np.sum(P_arr, axis=0))


    return P_norm



