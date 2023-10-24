%Tool for the verification of individual channels of an array.
%The array elements are pulsed one-by-one, and for each excitation the
%responses of all elements are recorded to a file for later analysis.

clearvars
usbmd_Globals

filename =  fullfile(usbmd_g_MatFilesDir, mfilename);

%% System object: transducer characteristics in the Trans object
Trans = usbmd_InitTrans(usbmd_g_TransName, usbmd_g_TransFreq);

%% Provide the input parameters for this script
enableFlash      = 0;    % if 1 then all transducers pulse at the same time
speedOfSound     = 1540;
pulseRate        = 100;
startDepthWL     = Trans.startDepthWL;
endDepthWL       = 300 + startDepthWL;
depthRangeWL     = endDepthWL - startDepthWL;
noopValue        = 20;

inputTxt = sprintf( 'Give the pulse frequency [ENTER for %.2f] in MHz: ', Trans.frequency );
f        = input( inputTxt );
if isempty(f)
    pulseFreqMHz = Trans.frequency;
else
    pulseFreqMHz = usbmd_g_SupportedTxFreqMHz(abs(usbmd_g_SupportedTxFreqMHz-f) == min(abs(usbmd_g_SupportedTxFreqMHz-f)));
end
fprintf(' You will get %.2f MHz for the pulse\n', pulseFreqMHz );

inputTxt = sprintf('Give the number of pulse cycles [ENTER for 1]: ');
nr = input( inputTxt );
if isempty(nr)
    nrPulseCycles = 1;
else
    nrPulseCycles = nr;
end
fprintf(' You will get %d pulse cycles\n', nrPulseCycles);

interleave = 1;  % todo: remove from script
sampleMode = 'NS200BW';
if (Trans.frequency ~= 15.625)
    warning(['To achieve maximum data sample rate you may want to set Trans.frequency to Fc=15.625 iso ',num2str(Trans.frequency)]);
end
samplesPerWave   = 4;

% compute nrTxEvents, the number of transmit events
if (enableFlash == 1)
    nrTxEvents = 1;
else
    nrTxEvents = Trans.numelements;
end

%% System object: parameter attributes of the Resource object
Resource.Parameters.connector             = 1;
Resource.Parameters.numTransmit           = 128;
Resource.Parameters.numRcvChannels        = 128;
Resource.Parameters.speedOfSound          = 1540;
Resource.Parameters.speedCorrectionFactor = 1.0;
Resource.Parameters.startEvent            = 1;
Resource.Parameters.simulateMode          = 0; 
Resource.Parameters.fakeScanhead          = 0;
Resource.Parameters.GUI                   = 'vsx_gui';
Resource.Parameters.verbose               = 2;
Resource.VDAS.dmaTimeout                  = 1e4; % Default: 1000 msec

%% Define medium
% pt1;
Media.MP = [40*rand(100,1), zeros(100,1), 100*rand(100,1), rand(100,1)]; % [x,y,z,reflectivity]
% Media.MP(2,:) = [0,0,25,1.0]; % [x,y,z,reflectivity]
% Media.MP(3,:) = [-10,0,20,1.0]; % [x,y,z,reflectivity]
% Gamex404GS;
Media.function = 'movePoints';

%% Specify Resources
Resource.RcvBuffer(1).datatype     = 'int16';
Resource.RcvBuffer(1).rowsPerFrame = interleave * nrTxEvents * ... 
    ceil( depthRangeWL * 2 * samplesPerWave / interleave / 128) * 128;
Resource.RcvBuffer(1).colsPerFrame = Resource.Parameters.numRcvChannels;
Resource.RcvBuffer(1).numFrames    = 1;         % minimum size is 1 frame.

%% Specify transmit wave
%TW = usbmd_ComputeTW(pulseFreqMHz,nrPulseCycles);
TW.type = 'parametric';
TW.Parameters = [pulseFreqMHz, .67, 2, 1];


%% Specify TX structure array
TX = repmat(struct(...
            'waveform', 1, ...
            'Apod'    , [], ...
            'Origin'  , [0.0, 0.0, 0.0], ...
            'focus'   , 0, ...
            'Delay'   , zeros(1, Trans.numelements)), ...
            1, nrTxEvents * interleave);
for i = 1:nrTxEvents
    for n = 1:interleave
        idx = (i-1)*interleave + n;
        TX(idx).Delay = TX(idx).Delay + (interleave-n)/samplesPerWave * ones(1,Trans.numelements);
    end
end
if enableFlash == 0
    for i = 1 : nrTxEvents
        for n = 1:interleave
            idx = (i-1)*interleave + n;
            tmpApod    = zeros(1, Trans.numelements);
            tmpApod(i) = 1;            
            TX(idx).Apod = tmpApod;
        end
    end
else
    for n = 1:interleave
        TX(n).Apod = ones(1,Trans.numelements);
    end
end

%% Specify TGC Waveform structure
TGC(1).CntrlPts = usbmd_g_TGCSetPoints;
TGC(1).rangeMax = endDepthWL;                        % maximum depth in the range, expressed in wavelengths
TGC(1).Waveform = computeTGCWaveform(TGC);

%% Specify Receive structure array
lowPassCoef = [0 0 0 0 0 0 0 0 0 0 0 1];
if interleave == 1
    inputFilter = firpm(40,[0 0.02 0.22 0.78 0.98 1],[0 0 1 1 0 0]);
else
    inputFilter = firpm(40,[0 0.02 0.22 1],[0 0 1 1]);
end
inputFilter = inputFilter(1:21);
ApodR = ones(1,Trans.numelements);
Receive = repmat(struct(...
                 'Apod'           , ApodR, ...
                 'startDepth'     , startDepthWL, ...
                 'endDepth'       , endDepthWL, ...
                 'TGC'            , 1, ...
                 'mode'           , 0, ...
                 'bufnum'         , 1, ...
                 'framenum'       , 1, ...
                 'acqNum'         , 1, ...
                 'sampleMode'     , sampleMode, ...
                 'LowPassCoef'    , lowPassCoef, ...
                 'InputFilter'    , inputFilter), ...
                 1, nrTxEvents * interleave);
% Set event specific Receive attributes
for i = 1 : nrTxEvents * interleave
    Receive(i).acqNum = i;
end

%% Set TPCHighVoltage for profile one to Trans.maxHighVoltage
UI(1).Statement = '[result,hv] = setTpcProfileHighVoltage(usbmd_g_TransVoltage,1);';
UI(2).Statement = 'hv1Sldr = findobj(''Tag'',''hv1Sldr'');';
UI(3).Statement = 'set(hv1Sldr,''Value'',hv);';
UI(4).Statement = 'hv1Value = findobj(''Tag'',''hv1Value'');';
UI(5).Statement = 'set(hv1Value,''String'',num2str(hv,''%.1f''));';
% Run callback to ensure new settings are transferred to hw.
%UI(6).Statement = 'cb=get(hv1Value, ''Callback''); cb{1}(0,0,cb{2},cb{3})';

%% Specify external process
Process(1).classname  = 'External'; % Identifies the processing as external.
Process(1).method     = 'StorePulseEchos1';
Process(1).Parameters = {'srcbuffer'  , 'receive', ...
                         'srcbufnum'  ,  1, ...      % no. of buffer to process.
                         'srcframenum', -1, ...      % process the most recent frame.  
                         'dstbuffer'  , 'none' };
EF(1).Function        = text2cell('%EF#1');          % Call an external function defined below

%% Specify SeqControl structure array
SeqControl(1).command  = 'timeToNextAcq'; 
SeqControl(1).argument = 1e6 * ( 1 / pulseRate);  % [us]
SeqControl(2).command  = 'transferToHost'; % perform DMA
SeqControl(3).command = 'waitForTransferComplete'; % wait until the hw has set the TNP flag at the end of the DMA transfer
SeqControl(3).argument = 2; % the number of the SeqControl structure that has the �transferToHost� command for which we want to wait
SeqControl(4).command = 'markTransferProcessed'; % after processing has been performed, reset the TNP flag
SeqControl(4).argument = 2; % the number of the SeqControl structure that has the �transferToHost� command for which we want to wait
SeqControl(5).command = 'sync'; % pause the hardware sequencer at this event until the software completes its processing and cancels the pause
SeqControl(6).command  = 'returnToMatlab'; %  allow Matlab to process any pending GUI actions
SeqControl(7).command  = 'triggerOut';
SeqControl(8).command  = 'noop';   % no action
SeqControl(8).argument = noopValue;% value*200ns; Note additional time constant: 3.5 us after the trigger follows TX (Verasonics intrinsic delay)
                                   % Trigger - 3.5 us Verasonics delay - value*200ns delay - TX
                                   % Set the RWN generator to: burst delay = 0ns, pulse width = 8us   
SeqControl(8).condition = 'Hw&Sw';
SeqControl(9).command  = 'noop';   % no action
SeqControl(9).argument = round((1/pulseRate)/200e-9) - SeqControl(8).argument;   % value*200ns delay; frame rate
SeqControl(9).condition = 'Hw&Sw';

%% Specify Event structure array
n = 1;    % event index
for j = 1 : 2  % we acquire the frame of interest twice and we will not record the first frame. This is needed for startup, and somehow avoids getting strange data from a "timeToNextAcq duration too short".
	for i = 1 : nrTxEvents * interleave
		% Acquires raw data (Hardware sequencer)
		Event(n).info       = 'Acquire RF Data.';
		Event(n).tx         = i;      % use i-th TX structure
		Event(n).rcv        = i;      % use i-th receive structure array
		Event(n).recon      = 0;      % no reconstruction
		Event(n).process    = 0;      % no processing
		Event(n).seqControl = [1,7];  % TTNA and trigger
		n = n + 1;
	end
end
Event(n-1).seqControl = [1,7,2];     % transfer all acquired data to host

% External processing
Event(n).info       = 'Call external function for storing data';
Event(n).tx         = 0; 	% no TX structure
Event(n).rcv        = 0; 	% no Rcv structure
Event(n).recon      = 0; 	% no reconstruction
Event(n).process    = 1; 	% call external processing function having this index
Event(n).seqControl = [3,4,5]; % return to MatLab
n = n + 1;
    
Event(n).info       = 'Return to MatLab';
Event(n).tx         = 0; 	% no TX structure
Event(n).rcv        = 0; 	% no Rcv structure
Event(n).recon      = 0; 	% no reconstruction
Event(n).process    = 0; 	% call external processing function having this index
Event(n).seqControl = 6;    % return to MatLab
n = n + 1;

%% Save all the structures to a .mat file and call VSX.
save(filename); VSX

return


%% External functions
%EF#1
StorePulseEchos1(RFData)
C = clock;

nrTxEvents              = evalin('base','nrTxEvents');
rowsPerFrame            = evalin('base','Resource.RcvBuffer(1).rowsPerFrame');
lineLength              = rowsPerFrame / nrTxEvents;
Fc                      = evalin('base','Trans.frequency') * 1e6;
samplesPerWave          = evalin('base','samplesPerWave');
Fs                      = samplesPerWave * Fc;
Trans                   = evalin('base','Trans');
TW                      = evalin('base','TW');
pulseFreqMHz            = evalin('base','pulseFreqMHz');
nrPulseCycles           = evalin('base','nrPulseCycles');
usbmd_g_DataSaveDir     = evalin('base','usbmd_g_DataSaveDir');
interleave              = evalin('base','interleave');

if (lineLength ~= size(RFData,1) / nrTxEvents)
    error('StorePulseEchos1: invalid line length.');
end

prefix = sprintf('%d%02d%02d_%02d%02d%02d_', C(1), C(2), C(3), ...
    C(4), C(5), round(C(6)));
saveFileName = fullfile(usbmd_g_DataSaveDir, [prefix, 'allResponses.mat']);

% We have "nrTxEvents" transmit pulse events.
%   if nrTxEvents==Trans.numelements we have one transmit pulse event for each element separately
%   if nrTxEvents==1    then all the Trans.numelements elements were fired simultaneously
% We have Trans.numelements receive channels per transmit pulse event
% Each receive channel contains "lineLength" samples
M = zeros(sum(Trans.numelements), lineLength, nrTxEvents);
for t = 1 : nrTxEvents
    lineIndexRange = (t-1)*lineLength+1 : t*lineLength;
    M(:,:,t) = double(RFData(lineIndexRange, Trans.ConnectorES))';
end

% Perform interleaving
if interleave > 1
    for t = 1 : nrTxEvents
        for chan = 1 : size(M,1)
            tmp = reshape(M(chan,:,t), lineLength/interleave, interleave);
            M(chan,:,t) = reshape(tmp', 1, lineLength);
        end
    end
end

readme = 'dimension M = numelements x line_length x num_tx_events';
disp(['========================================================================================']);
fprintf('Saving data to %s...\n', saveFileName);
disp([''])
disp(['  number of transmit events               : ', num2str(nrTxEvents)]);
disp(['  number of receive channels per transmit : ', num2str(sum(Trans.numelements))]);
disp(['  number of samples per response          : ', num2str(lineLength)]);
disp(['  sample rate Fs in MHz                   : ', num2str(Fs/1e6)]);
disp(['  transducer frequency in MHz             : ', num2str(Fc/1e6)]);
save(saveFileName,'M','Fs','Fc','Trans','TW','pulseFreqMHz','nrPulseCycles','readme');
disp(['Done.']);
disp(['Click on Freeze button in the GUI to repeat measurement (and save a next allResponses.mat)']);
disp(['========================================================================================']);
return
%EF#1
