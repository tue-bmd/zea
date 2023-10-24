clearvars
usbmd_Globals

filename = fullfile(usbmd_g_MatFilesDir, mfilename);

% Specify Trans structure array.
Trans = usbmd_InitTrans(usbmd_g_TransName, usbmd_g_TransFreq);

%% Provide the input parameters for this script
pulseFreqMHz         = Trans.frequency;
nrPulseCycles        = 1;
startDepthWL         = Trans.startDepthWL;
endDepthWL           = 400 + startDepthWL;
depthRangeWL         = endDepthWL - startDepthWL;
pulseRate            = 1000;   % pulse repetition frequency [Hz]
soundSpeed           = 1.540;  % speed of sound [mm/us]
numAcqs              = 128;    % nr of acquisitions before DMA
enableAveraging      = 0;      % when 1 the RF lines of 1 DMA are averaged
upsampleFactor       = 1;      % upsamples the displayed RF lines by a factor (unsigned integer)
noopValue            = 20;
activeTX             = ceil( Trans.numelements / 2 );    % the index of the transducer to transmit
activeRXa            = ceil( Trans.numelements / 2 );    % the index of the 1st transducer to receive
activeRXb            = ceil( Trans.numelements / 2 );    % the index of the 2nd transducer to receive
plotXAxis            = 'distance';  % 'distance' or 'time', distance in mm, time in us

samplesPerWave       = 4;

%% Do not change the following initial values related to UI
enableFlash          = 0;     % all TDX simultaneously (or not)

disp(['Selected transducer: ', Trans.secondName]);

%% Specify resource parameters
Resource.Parameters.connector             = 1;
Resource.Parameters.numTransmit           = 128;
Resource.Parameters.numRcvChannels        = 128;
Resource.Parameters.speedOfSound          = 1540;
Resource.Parameters.speedCorrectionFactor = 1.0;
Resource.Parameters.startEvent            = 1;
Resource.Parameters.simulateMode          = 0; 
Resource.Parameters.fakeScanhead          = 1;
%Resource.Parameters.UpdateFunction        = 'update';
%Resource.Parameters.GUI                   = 'vsx_gui';
Resource.Parameters.verbose               = 2;

%% Define medium
% pt1;
Media.MP = [40*rand(100,1), zeros(100,1), 100*rand(100,1), rand(100,1)]; % [x,y,z,reflectivity]
% Media.MP(2,:) = [0,0,25,1.0]; % [x,y,z,reflectivity]
% Media.MP(3,:) = [-10,0,20,1.0]; % [x,y,z,reflectivity]
% Gamex404GS;
Media.function = 'movePoints';

%% Specify Resources
Resource.RcvBuffer(1).datatype     = 'int16';
Resource.RcvBuffer(1).rowsPerFrame = numAcqs * ceil( depthRangeWL * 2 * samplesPerWave / 128) * 128;
Resource.RcvBuffer(1).colsPerFrame = Resource.Parameters.numRcvChannels;
Resource.RcvBuffer(1).numFrames    = 1;       % minimum size is 1 frame.

%% Specify TW structure array
%TW = usbmd_ComputeTW(pulseFreqMHz,nrPulseCycles);
TW.type = 'parametric';
TW.Parameters = [Trans.frequency,.67,2,1];

%% Specify transmit operation with the TX structure array
ApodT              = zeros(1, Trans.numelements);
ApodT(activeTX)    = 1;

TX = repmat(struct('waveform', 1, ...
                   'Apod', ApodT, ...
                   'Origin', [0.0, 0.0, 0.0], ...
                   'focus', 0, ...
                   'Steer', [0.0, 0.0], ...
                   'Delay', zeros(1,Trans.numelements)), 1, 1 );

%% Specify receive structure array
ApodR                        = zeros(1, Trans.numelements);
ApodR([activeRXa activeRXb]) = 1;
lowPassCoef = [0 0 0 0 0 0 0 0 0 0 0 1];

inputFilter = firpm(40, [0 0.02 0.22 0.78 0.98 1], [0 0 1 1 0 0]);
inputFilter = inputFilter(1:21);

Receive = repmat(struct(...
                 'Apod'           , ApodR, ...
                 'startDepth'     , startDepthWL, ...
                 'endDepth'       , endDepthWL, ...
                 'TGC'            , 1, ...
                 'mode'           , 0, ...
                 'bufnum'         , 1, ...
                 'framenum'       , 1, ...       
                 'acqNum'         , 1, ...
                 'sampleMode'     , 'NS200BW', ...
                 'LowPassCoef'    , lowPassCoef, ...
                 'InputFilter'    , inputFilter), ...
                 1, numAcqs);
% Set event specific Receive attributes
for n = 1 : numAcqs
    Receive(n).acqNum = n;
end

%% Specify TGC Waveform structure for receive
TGC(1).CntrlPts = usbmd_g_TGCSetPoints;  % values equally spaced along depth range, value range is 0 to 1023.
TGC(1).rangeMax = endDepthWL;            % maximum depth in the range, expressed in wavelengths
TGC(1).Waveform = computeTGCWaveform(TGC);

%% Set TPCHighVoltage for profile one to desired amount of Volts
UI(1).Statement = '[result,hv] = setTpcProfileHighVoltage(usbmd_g_TransVoltage,1);';
UI(2).Statement = 'hv1Sldr = findobj(''Tag'',''hv1Sldr'');';
UI(3).Statement = 'set(hv1Sldr,''Value'',hv);';
UI(4).Statement = 'hv1Value = findobj(''Tag'',''hv1Value'');';
UI(5).Statement = 'set(hv1Value,''String'',num2str(hv,''%.1f''));';
% Run callback function to ensure new settings are transferred to hw.
%%%%UI(6).Statement = 'cb=get(hv1Value, ''Callback''); cb{1}(0,0,cb{2},cb{3});';

%% Specify external processes
Process(1).classname  = 'External'; % Identifies the processing as external.
Process(1).method     = 'RFLinePlot';
Process(1).Parameters = {'srcbuffer'  , 'receive',...
                         'srcbufnum'  , 1,...
                         'srcframenum', -1,...
                         'dstbuffer'  , 'none' };
EF(1).Function        = text2cell('%EF#1');

%% Some sequence control definitions
SeqControl(1).command  = 'timeToNextAcq'; 
SeqControl(1).argument = 1e6 * ( 1 / pulseRate);  % [us]
SeqControl(2).command  = 'sync';   % Synchronize hardware and software sequencers
SeqControl(2).argument = 1000000;  % 0.25 sec timeout for software sequencer (default is 0.5 seconds)
SeqControl(3).command  = 'jump';   % by default 'jump' includes 'returnToMatlab' sequence command
SeqControl(3).argument = 1;        % number of event to jump to
SeqControl(4).command  = 'triggerOut';
SeqControl(5).command  = 'noop';   % no action
SeqControl(5).argument = noopValue;% value*200ns; Note additional time constant: 3.5 us after the trigger follows TX (Verasonics intrinsic delay)
                                   % Trigger - 3.5 us Verasonics delay - value*200ns delay - TX
                                   % Set the RWN generator to: burst delay = 0ns, pulse width = 8us   
SeqControl(5).condition = 'Hw&Sw';
SeqControl(6).command   = 'noop';  % no action
SeqControl(6).argument  = round((1/pulseRate)/200e-9) - SeqControl(5).argument;   % value*200ns delay; frame rate
SeqControl(6).condition = 'Hw&Sw';

nsc = 7;    % start index for new SeqControl

%% Event definitions
n = 1;
for i = 1 : numAcqs
    % Acquires raw data (Hardware sequencer)
    Event(n).info         = 'Acquire RF Data.';
    Event(n).tx           = 1;
    Event(n).rcv          = i;
    Event(n).recon        = 0;
    Event(n).process      = 0;
    Event(n).seqControl   = [1,4];  % TTNA and trigger out
    n = n + 1;
end
Event(n-1).seqControl = [1,4,nsc];
    SeqControl(nsc).command  = 'transferToHost';
    nsc = nsc + 1;

% External processing
Event(n).info       = 'Call external displaying function and do sync';
Event(n).tx         = 0;
Event(n).rcv        = 0;
Event(n).recon      = 0;
Event(n).process    = 1;
Event(n).seqControl = 2;
n = n + 1;

Event(n).info         = 'Jump back to Event 1';
Event(n).tx           = 0;
Event(n).rcv          = 0;
Event(n).recon        = 0;
Event(n).process      = 0;
Event(n).seqControl   = 3;

%% Deal with UI changes
% - activeRXb
if Trans.numelements > 1
    UI(1).Control = {'UserB1', 'Style', 'VsSlider', ...
                     'Label', 'active RX B', ...
                     'SliderMinMaxVal', [1, Trans.numelements, activeRXb], ...
                     'SliderStep', [1/(Trans.numelements-1), 1/(Trans.numelements-1)], ...
                     'ValueFormat', '%3.0f'};
    UI(1).Callback = text2cell('%CB#1');
end

% - activeRXa
if Trans.numelements > 1
    UI(2).Control = {'UserB2', 'Style', 'VsSlider', ...
                     'Label', 'active RX A', ...
                     'SliderMinMaxVal', [1, Trans.numelements, activeRXa], ...
                     'SliderStep', [1/(Trans.numelements-1), 1/(Trans.numelements-1)], ...
                     'ValueFormat', '%3.0f'};
    UI(2).Callback = text2cell('%CB#2');
end

measureStartIndex = 1;%round(Resource.RcvBuffer(1).rowsPerFrame / numAcqs / 20); % initial value
measureStopIndex  = 2;%round(Resource.RcvBuffer(1).rowsPerFrame / numAcqs / 2 ); % initial value
maxIndex          = upsampleFactor * Resource.RcvBuffer(1).rowsPerFrame / numAcqs;
UI(3).Control = {'UserB3', 'Style', 'VsSlider', ...
                 'Label', 'stop index', ...
                 'SliderMinMaxVal', [1, maxIndex, measureStopIndex], ...
                 'SliderStep', [1 * numAcqs/Resource.RcvBuffer(1).rowsPerFrame, 100 * numAcqs/Resource.RcvBuffer(1).rowsPerFrame], ...
                 'ValueFormat', '%3.0f'};
UI(3).Callback = text2cell('%CB#3');
UI(4).Control = {'UserB4', 'Style', 'VsSlider', ...
                 'Label', 'start index', ...
                 'SliderMinMaxVal', [1, maxIndex, measureStartIndex], ...
                 'SliderStep', [1 * numAcqs/Resource.RcvBuffer(1).rowsPerFrame, 100 * numAcqs/Resource.RcvBuffer(1).rowsPerFrame], ...
                 'ValueFormat', '%3.0f'};
UI(4).Callback = text2cell('%CB#4');

% - active TX
if Trans.numelements > 1
    UI(11).Control = {'UserB5', 'Style', 'VsSlider', ...
                     'Label', 'active TX', ...
                     'SliderMinMaxVal', [1, Trans.numelements, activeTX], ...
                     'SliderStep', [1/(Trans.numelements-1), 1/(Trans.numelements-1)], ...
                     'ValueFormat', '%3.0f'};
    UI(11).Callback = text2cell('%CB#011');
end

% - Pulse Freq (MHz)
UI(5).Control = {'UserA2','Style','VsSlider','Label','Pulse freq MHz',...
                 'SliderMinMaxVal',[1,42,pulseFreqMHz],'SliderStep',[1/41,2/41],'ValueFormat','%d'};
UI(5).Callback = text2cell('%-CB#5Callback');

% - Number of pulses
UI(6).Control = {'UserA1','Style','VsSlider','Label','Pulse nr cycles',...
                 'SliderMinMaxVal',[1,16,nrPulseCycles], ...        % Set to 16 (from 10) to allow increased pulse lengths (MK, 09 Mar 16)
                 'SliderStep',[1/15,2/15],'ValueFormat','%d'};      % Changed from 1/9 and 2/9 to 1/15 and 2/15 to maintain integer values
UI(6).Callback = text2cell('%-CB#6Callback');

% - Toggle transmit all
UI(10).Control = {'UserC1','Style','VsPushButton',...
                'Label','transmit all'};
UI(10).Callback = text2cell('%-UI#10Callback');

%% Save all the structures to a .mat file and call VSX.
save(filename); VSX

return


%% Callback functions

%CB#1 - activeRXb change (note: CB#zero-eleventh is needed, otherwise text2cell.m function fails)
numAcqs = evalin('base', 'numAcqs'); 
Receive = evalin('base', 'Receive');
Trans   = evalin('base', 'Trans');
activeRXa = evalin('base', 'activeRXa');
for n = 1 : numAcqs
    ApodR = zeros(1, Trans.numelements);
    ApodR([round(UIValue) activeRXa]) = 1;
    Receive(n).Apod = ApodR;
end
assignin('base', 'Receive', Receive);
Control.Command = 'update&Run';                 
Control.Parameters = {'Receive'};
assignin('base', 'activeRXb', round(UIValue));
assignin('base', 'Control' , Control);
return
%CB#1

%CB#2 - activeRXa change
numAcqs = evalin('base', 'numAcqs'); 
Receive = evalin('base', 'Receive');
Trans   = evalin('base', 'Trans');
activeRXb = evalin('base', 'activeRXb');
for n = 1 : numAcqs
    ApodR = zeros(1, Trans.numelements);
    ApodR([round(UIValue) activeRXb]) = 1;
    Receive(n).Apod = ApodR;
end
assignin('base', 'Receive', Receive);
Control.Command = 'update&Run';                 
Control.Parameters = {'Receive'};
assignin('base', 'activeRXa', round(UIValue));
assignin('base', 'Control' , Control);     
return
%CB#2

%CB#3 - measurement start depth
assignin('base', 'measureStopIndex', round(UIValue));
return
%CB#3

%CB#4 - measurement stop depth
assignin('base', 'measureStartIndex', round(UIValue));
return
%CB#4

%-CB#5Callback (pulse freq MHz)
pulseFreqMHz = UIValue;
assignin('base', 'pulseFreqMHz', pulseFreqMHz);
nrPulseCycles = evalin('base', 'nrPulseCycles');
TW = usbmd_ComputeTW(pulseFreqMHz,nrPulseCycles);
assignin('base', 'TW', TW);
Control = evalin('base', 'Control');
Control.Command = 'update&Run';
Control.Parameters = {'TW'};
assignin('base', 'Control', Control);
return
%-CB#5Callback

%CB#011 - activeTX change
TX    = evalin('base', 'TX'); % load the TX structure from the Matlab workspace
TW    = evalin('base', 'TW'); % Need to be updates when TX or TW are changed during freeze
Trans = evalin('base', 'Trans');
ApodT = zeros(1, Trans.numelements);
ApodT(round(UIValue)) = 1;
TX.Apod = ApodT;
assignin('base', 'TX', TX);
Control.Command    = 'update&Run';
Control.Parameters = {'TX', 'TW'};
assignin('base', 'activeTX', round(UIValue));
assignin('base', 'Control', Control);
% when pressing this button the transmit flash mode needs to be turned off
% automatically
handleToGUIFigure = findobj('tag','UI');
set(handleToGUIFigure, 'Color', [0.8 0.8 0.8]); % default GUI grayend
enableFlash = 0;
assignin( 'base', 'enableFlash', enableFlash );
return
%CB#011

%-CB#6Callback - Pulse change
nrPulseCycles = UIValue;
assignin('base', 'nrPulseCycles', nrPulseCycles);
pulseFreqMHz = evalin('base','pulseFreqMHz');
TW = usbmd_ComputeTW(pulseFreqMHz,nrPulseCycles);
assignin('base','TW', TW);
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'TW'};
assignin('base','Control', Control);
return
%-CB#6Callback

%-UI#10Callback (toggle transmit flash mode)
handleToGUIFigure = findobj('tag','UI');
enableFlash = evalin( 'base', 'enableFlash' );
if enableFlash == 0
    enableFlash = 1;
    scope_flash_on();
    set(handleToGUIFigure, 'Color', [0.7 0.8 0.7]); % GUI green
else
    enableFlash = 0;
    scope_flash_off();
    set(handleToGUIFigure, 'Color', [0.8 0.8 0.8]); % default GUI grayend
end
%-update enableFlash value in base workspace
assignin( 'base', 'enableFlash', enableFlash );
return
%-UI#10Callback

%% External functions
%EF#1
RFLinePlot(RFData)

Trans        = evalin('base', 'Trans');
activeTX     = evalin('base', 'activeTX');
activeRXa    = evalin('base', 'activeRXa');
activeRXb    = evalin('base', 'activeRXb');
pinRxA       = Trans.Connector(activeRXa);
pinRxB       = Trans.Connector(activeRXb);
numAcqs      = evalin('base', 'numAcqs');
enableAveraging = evalin('base', 'enableAveraging');
rowsPerFrame = evalin('base', 'Resource.RcvBuffer(1).rowsPerFrame');
Fs           = evalin('base', 'samplesPerWave') * Trans.frequency;
measureStartIndex = evalin('base', 'measureStartIndex');
measureStopIndex  = evalin('base', 'measureStopIndex');
measureStopIndex  = max( measureStopIndex, measureStartIndex);
upsampleFactor    = evalin('base', 'upsampleFactor');
enableAveraging = evalin('base','enableAveraging');
plotXAxis = evalin('base', 'plotXAxis');

lineLength   = rowsPerFrame / numAcqs;
if strcmp(plotXAxis,'distance')
    plotXAxisData = linspace(0,1540*lineLength/(2*1e3*Fs),lineLength);  % [mm]
    plotXAxisLabel = 'Distance [mm]';
else
    plotXAxisData = linspace(0,lineLength/Fs,lineLength);  % [us]
    plotXAxisLabel = 'Acquisition time [\mus]';
end
RFLineA = double(RFData(1:lineLength, pinRxA));
RFLineB = double(RFData(1:lineLength, pinRxB));
if enableAveraging
    for i = 2 : numAcqs
        idxRange   = (i-1)*lineLength+1 : i*lineLength;
        RFLineA = RFLineA + double(RFData(idxRange, pinRxA));
        RFLineB = RFLineB + double(RFData(idxRange, pinRxB));
    end
    RFLineA = RFLineA ./ numAcqs;
    RFLineB = RFLineB ./ numAcqs;
else
    idxRange = lineLength+1 : 2*lineLength;
    RFLineA   = double(RFData(idxRange, pinRxA));
    RFLineB   = double(RFData(idxRange, pinRxB));
end

FsUp    = upsampleFactor*Fs;
RFLineA = resample(RFLineA,upsampleFactor,1);
RFLineB = resample(RFLineB,upsampleFactor,1);
plotXAxisData = linspace(min(plotXAxisData),max(plotXAxisData),upsampleFactor*numel(plotXAxisData));
Fs = FsUp;

hA = RFLineA(measureStartIndex:measureStopIndex);
hA_env = abs(hilbert(hA));
L = length(hA_env);
hA_env_interp = interp1(0:L-1, hA_env, 0:0.1:L-1,'spline');
[~,hA_env_max_idx] = max(hA_env_interp);
M  = 2^nextpow2(length(hA)); % will be half the fft size
Ha = fft(hA, 2*M);
Ha = 20*log10(abs(Ha(1:M+1)));
freqA     = linspace(0, Fs/2, M+1);
idxHMaxA  = find(Ha ==max(Ha),1,'first');
idxBWMinA = find(Ha > Ha(idxHMaxA)-6, 1, 'first');
idxBWMaxA = find(Ha > Ha(idxHMaxA)-6, 1, 'last');
idxHCenterA = round((idxBWMaxA - idxBWMinA)/2) + idxBWMinA;
if isempty(idxBWMinA), idxBWMinA = 1; end
if isempty(idxBWMaxA), idxBWMaxA = 1; end
if isempty(idxHCenterA), idxHCenterA = 1; end
bandWidthA    = freqA(idxBWMaxA)-freqA(idxBWMinA);
relBandWidthA = 100 * bandWidthA / (freqA(idxHCenterA) + eps);
p2phA         = abs(min(hA) - max(hA));

hB = RFLineB(measureStartIndex:measureStopIndex);
hB_env = abs(hilbert(hB));
L = length(hB_env);
hB_env_interp = interp1(0:L-1, hB_env, 0:0.1:L-1,'spline');
[~,hB_env_max_idx] = max(hB_env_interp);
M  = 2^nextpow2(length(hB)); % will be half the fft size
Hb = fft(hB, 2*M);
Hb = 20*log10(abs(Hb(1:M+1)));
freqB     = linspace(0, Fs/2, M+1);
idxHMaxB  = find(Hb ==max(Hb),1,'first');
idxBWMinB = find(Hb > Hb(idxHMaxB)-6, 1, 'first');
idxBWMaxB = find(Hb > Hb(idxHMaxB)-6, 1, 'last');
idxHCenterB = round((idxBWMaxB - idxBWMinB)/2) + idxBWMinB;
if isempty(idxBWMinB), idxBWMinB = 1; end
if isempty(idxBWMaxB), idxBWMaxB = 1; end
if isempty(idxHCenterB), idxHCenterB = 1; end
bandWidthB    = freqB(idxBWMaxB)-freqB(idxBWMinB);
relBandWidthB = 100 * bandWidthB / (freqB(idxHCenterB) + eps);
p2phB         = abs(min(hB) - max(hB));

figH = figure(107);
clf(figH)
figureName = Trans.name;
set(figH,'name',figureName,'NumberTitle','off')
subplot(311)
  hold all
    plot(plotXAxisData,RFLineA);
    plot(plotXAxisData,RFLineB,'LineStyle',':');
    V = axis; V(3)=-16384; V(4)=16383; axis(V);
    enableFlash = evalin('base','enableFlash');
    if (enableAveraging)
      nrAvg = numAcqs;
    else
      nrAvg = 1;
    end
    if (enableFlash == 0)
      title(['activeTX = ',num2str(activeTX),', activeRX = ',num2str([activeRXa activeRXb]),', RF line averaging ',num2str(nrAvg),' x, upsample ',num2str(upsampleFactor) 'x'])
    else
      title(['activeTX = ALL, activeRX = ',num2str([activeRXa activeRXb]),', RF line averaging ',num2str(numAcqs),' times, upsample ',num2str(upsampleFactor) 'x'])
    end
    plot(plotXAxisData([measureStartIndex measureStartIndex]),[V(3) V(4)], 'r');
    plot(plotXAxisData([measureStopIndex measureStopIndex]),[V(3) V(4)], 'r');
  hold off
  xlabel(plotXAxisLabel);
  axis tight;
  legend('A','B');
subplot(312)
  hold all
    plot(plotXAxisData(measureStartIndex:measureStopIndex), hA, 'b');
    plot(plotXAxisData(measureStartIndex:measureStopIndex), hA_env, 'b:');
    plot(plotXAxisData(measureStartIndex:measureStopIndex), hB,'r');
    plot(plotXAxisData(measureStartIndex:measureStopIndex), hB_env,'r:');
  hold off
  xlabel(plotXAxisLabel);
  axis tight;
  title({['RX A: peak-to-peak = ' num2str(p2phA,5) ' samples, ' num2str(hA_env_max_idx)]...
         ['RX B: peak-to-peak = ' num2str(p2phB,5) ' samples, ' num2str(hB_env_max_idx)]...     
        });
subplot(313)
  hold all
    plot(freqA, Ha)
    plot(freqB, Hb,'LineStyle',':');
    plot(freqA([idxBWMinA idxBWMaxA idxHCenterA]), Ha([idxBWMinA idxBWMaxA idxHCenterA]), 'r.');
    plot(freqB([idxBWMinB idxBWMaxB idxHCenterB]), Hb([idxBWMinB idxBWMaxB idxHCenterB]), 'm.');
  hold off
  xlabel('Frequency (MHz)')
  ylabel('Magnitude spectrum (dB)')
  title({['RX A: F_{peak} = ' num2str(freqA(idxHMaxA),2) ' MHz, F_{c} = ',num2str(freqA(idxHCenterA),2)...
          ' MHz, BW_{6dB} = ' num2str(bandWidthA,2) ' MHz, BW_{6dB} = ' num2str(relBandWidthA,3) ' %']...
         ['RX B: F_{peak} = ' num2str(freqB(idxHMaxB),2) ' MHz, F_{c} = ',num2str(freqB(idxHCenterB),2)...
          ' MHz, BW_{6dB} = ' num2str(bandWidthB,2) ' MHz, BW_{6dB} = ' num2str(relBandWidthB,3) ' %']...          
        });
 axis tight;
%EF#1
