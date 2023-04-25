% SetUp File for L11-4v probe - Edited by Beatrice Federici

% Acquire data and send them to server for processing and display
% Server address is defined in the VSX file and retrieved here through evalin('base', t)


clear all

filename = 'L11-4v-PlaneWaves-Beamforming-Display-Server-n';
sampleMode = 'BS100BW'; %BS100BW
returnToMatlabFreq = 1;
interAcqPeriod = 160; %160 us
interFramePeriod = 10000; %10 ms --> acquisition frame rate: 100 fps
na = 11; % no.plane waves
na_transmit = 1;

bf_type = 'DAS'
switch bf_type
    case 'RAW'
        bf_idx = 1
    case 'DAS'
        bf_idx = 2
    case 'ABLE'
        bf_idx = 3

%Nax %added by VSX automatically from Receive structure
Nel = 128; 
bytesPerElementSent = 2; % write channel data as int16
bytesPerElementRead = 8; % read updated parameters as double
numTunableParameters = 3; %[intensity, na, bf_type]

% Define Server Address and Port
%IPV4 = '131.155.34.167';
IPV4 = '131.155.34.16'; % Massimo
%IPV4 = '131.155.125.231'; % Workstation Ben
PORT = 30000;

saved_WritingTime = [];
timeExtFuncStartWrite_vect = []; % start writing
saved_UpdatingTime = [];
timeExtFuncEndUpdate_vect = []; % end reading updates and modify base env
%timeStamp_vect = [];
%timeToExtFunc_vect =[];

P.startDepth = 5;   % Acquisition depth in wavelengths
P.endDepth = 192;   % This should preferrably be a multiple of 128 samples.

% set dtheta to range over +/- 18 degrees.
if (na > 1)
    dtheta = (36*pi/180)/(na-1); 
    P.startAngle = -36*pi/180/2; 
else
    dtheta = 0;
    P.startAngle=0; 
end 

%Resource.Parameters.waitForProcessing = 0;
% Define system parameters.
Resource.Parameters.Connector = 1;
Resource.Parameters.numTransmit = 128;      % number of transmit channels.
Resource.Parameters.numRcvChannels = 128;    % number of receive channels.
Resource.Parameters.speedOfSound = 1540;    % set speed of sound in m/sec before calling computeTrans
Resource.Parameters.verbose = 2;
Resource.Parameters.initializeOnly = 0;
Resource.Parameters.simulateMode = 1;
%  Resource.Parameters.simulateMode = 1 forces simulate mode, even if hardware is present.
%  Resource.Parameters.simulateMode = 2 stops sequence and processes RcvData continuously.

% Specify Trans structure array.
Trans.name = 'L11-4v';
Trans.units = 'wavelengths'; % Explicit declaration avoids warning message when selected by default
Trans = computeTrans(Trans);  % L11-4v transducer is 'known' transducer so we can use computeTrans.
Trans.maxHighVoltage = 50;  % set maximum high voltage limit for pulser supply.
Trans.frequency = 6.25;
%disp(Trans.frequency); %Trans.frequency = 6.25; 

% Specify PData structure array.
PData(1).PDelta = [Trans.spacing, 0, 0.5];
PData(1).Size(1) = ceil((P.endDepth-P.startDepth)/PData(1).PDelta(3)); % startDepth, endDepth and pdelta set PData(1).Size.
PData(1).Size(2) = ceil((Trans.numelements*Trans.spacing)/PData(1).PDelta(1));
PData(1).Size(3) = 1;      % single image page
PData(1).Origin = [-Trans.spacing*(Trans.numelements-1)/2,0,P.startDepth]; % x,y,z of upper lft crnr.
% No PData.Region specified, so a default Region for the entire PData array will be created by computeRegions.

% Specify Media object. 'pt1.m' script defines array of point targets.
pt1;
Media.attenuation = -0.5;
Media.function = 'movePoints';

% Specify Resources.
Resource.RcvBuffer(1).datatype = 'int16';
if sampleMode == 'NS200BW'
    Resource.RcvBuffer(1).rowsPerFrame = na*512*8; % 4 samples times max round trip (2*512) in wavelengths (4 accounts for NS200BW, 4 samples per period)
elseif sampleMode == 'BS100BW'
    Resource.RcvBuffer(1).rowsPerFrame = na*512*4; % 2 samples times max round trip in wavelengths (2 accounts for BS100BW)
end
Resource.RcvBuffer(1).colsPerFrame = Resource.Parameters.numRcvChannels;
Resource.RcvBuffer(1).numFrames = 30;    % 30 frames stored in RcvBuffer.
Resource.InterBuffer(1).numFrames = 1;   % one intermediate buffer needed.
Resource.ImageBuffer(1).numFrames = 10;
Resource.DisplayWindow(1).Title = 'L11-4vFlashAngles';
Resource.DisplayWindow(1).pdelta = 0.35;
ScrnSize = get(0,'ScreenSize');
DwWidth = ceil(PData(1).Size(2)*PData(1).PDelta(1)/Resource.DisplayWindow(1).pdelta);
DwHeight = ceil(PData(1).Size(1)*PData(1).PDelta(3)/Resource.DisplayWindow(1).pdelta);
Resource.DisplayWindow(1).Position = [250,(ScrnSize(4)-(DwHeight+150))/2, ...  % lower left corner position
                                      DwWidth, DwHeight];
Resource.DisplayWindow(1).ReferencePt = [PData(1).Origin(1),0,PData(1).Origin(3)];   % 2D imaging is in the X,Z plane
Resource.DisplayWindow(1).Type = 'Verasonics';
Resource.DisplayWindow(1).numFrames = 20;
Resource.DisplayWindow(1).AxesUnits = 'mm';
Resource.DisplayWindow(1).Colormap = gray(256);

% Specify Transmit waveform structure.  
TW(1).type = 'parametric';
TW(1).Parameters = [Trans.frequency,.67,2,1];   

% Specify TX structure array.
TX = repmat(struct('waveform', 1, ...
                   'Origin', [0.0,0.0,0.0], ...
                   'Apod', kaiser(Resource.Parameters.numTransmit,1)', ...
                   'focus', 0.0, ...
                   'Steer', [0.0,0.0], ...
                   'Delay', zeros(1,Trans.numelements)), 1, na);
% - Set event specific TX attributes.
if fix(na/2) == na/2       % if na even
    P.startAngle = (-(fix(na/2) - 1) - 0.5)*dtheta;
else
    P.startAngle = -fix(na/2)*dtheta;
end
for n = 1:na   % na transmit events
    TX(n).Steer = [(P.startAngle+(n-1)*dtheta),0.0];
    TX(n).Delay = computeTXDelays(TX(n));
end

% Specify TGC Waveform structure.
TGC.CntrlPts = [0,297,424,515,627,764,871,1000];
TGC.rangeMax = P.endDepth;
TGC.Waveform = computeTGCWaveform(TGC);

% Specify Receive structure arrays. 
maxAcqLength = ceil(sqrt(P.endDepth^2 + ((Trans.numelements-1)*Trans.spacing)^2));
Receive = repmat(struct('Apod', ones(1,Trans.numelements), ...
                        'startDepth', P.startDepth, ...
                        'endDepth', maxAcqLength,...
                        'TGC', 1, ...
                        'bufnum', 1, ...
                        'framenum', 1, ...
                        'acqNum', 1, ...
                        'sampleMode', 'BS100BW', ...
                        'mode', 0, ...
                        'callMediaFunc', 0), 1, na*Resource.RcvBuffer(1).numFrames);
                    
% - Set event specific Receive attributes for each frame.
for i = 1:Resource.RcvBuffer(1).numFrames
    Receive(na*(i-1)+1).callMediaFunc = 1;
    for j = 1:na
        Receive(na*(i-1)+j).framenum = i;
        Receive(na*(i-1)+j).acqNum = j;
    end
end

% Specify Recon structure arrays.
Recon = struct('senscutoff', 0.6, ...
               'pdatanum', 1, ...
               'rcvBufFrame',-1, ...
               'IntBufDest', [1,1], ...
               'ImgBufDest', [1,-1], ...
               'RINums', 1:na);

% Define ReconInfo structures.
% We need na ReconInfo structures for na steering angles.
ReconInfo = repmat(struct('mode', 'accumIQ', ...  % default is to accumulate IQ data.
                   'txnum', 1, ...
                   'rcvnum', 1, ...
                   'regionnum', 1), 1, na);
               
% - Set specific ReconInfo attributes.
if na>1
    ReconInfo(1).mode = 'replaceIQ'; % replace IQ data
    for j = 1:na  % For each row in the column
        ReconInfo(j).txnum = j;
        ReconInfo(j).rcvnum = j;
    end
    ReconInfo(na).mode = 'accumIQ_replaceIntensity'; % accum and detect
else
    ReconInfo(1).mode = 'replaceIntensity';
end

% Specify Process structure array.
pers = 20;
Process(1).classname = 'Image';
Process(1).method = 'imageDisplay';
Process(1).Parameters = {'imgbufnum',1,...   % number of buffer to process.
                         'framenum',-1,...   % (-1 => lastFrame)
                         'pdatanum',1,...    % number of PData structure to use
                         'pgain',1.0,...            % pgain is image processing gain
                         'reject',2,...      % reject level 
                         'persistMethod','simple',...
                         'persistLevel',pers,...
                         'interpMethod','4pt',...  
                         'grainRemoval','none',...
                         'processMethod','none',...
                         'averageMethod','none',...
                         'compressMethod','power',...
                         'compressFactor',20,...
                         'mappingMethod','full',...
                         'display',1,...      % display image after processing
                         'displayWindow',1, ...
                         'srcData', 'unsignedColor'};

Process(2).classname = 'External';
Process(2).method = 'beamformingServer';
Process(2).Parameters = {'srcbuffer','receive',... % name of buffer to process.
'srcbufnum',1,... %RcvData{1}(:,:,1);
'srcframenum',-1,... %-1 only last frame
'dstbuffer','none'};

% external processing function to process time tag data
Process(3).classname = 'External';
Process(3).method = 'updateFromServer';
Process(3).Parameters = {'srcbuffer','none',... % name of buffer to process.
                         'dstbuffer','none'};


% Specify SeqControl structure arrays.
SeqControl(1).command = 'jump'; % jump back to start
SeqControl(1).argument = 1;
SeqControl(2).command = 'timeToNextAcq';  % time between synthetic aperture acquisitions
SeqControl(2).argument = interAcqPeriod;  % 160 usec
SeqControl(3).command = 'timeToNextAcq';  % time between frames
SeqControl(3).argument = interFramePeriod - (na-1)*interAcqPeriod;  % 100 frames every s
SeqControl(4).command = 'returnToMatlab';
nsc = 5; % nsc is count of SeqControl objects

% Specify Event structure arrays.
n = 1;
for i = 1:Resource.RcvBuffer(1).numFrames
    for j = 1:na  % Acquire frame
        Event(n).info = 'Full aperture.';
        Event(n).tx = j;   
        Event(n).rcv = na*(i-1)+j;   
        Event(n).recon = 0;      
        Event(n).process = 0;    
        Event(n).seqControl = 2;
        n = n+1;
    end
    Event(n-1).seqControl = [3,nsc]; % modify last acquisition Event's seqControl
      SeqControl(nsc).command = 'transferToHost'; % transfer frame to host buffer
      nsc = nsc+1;
      
    Event(n).info = 'Update.'; 
    Event(n).tx = 0;         
    Event(n).rcv = 0;        
    Event(n).recon = 0;      
    Event(n).process = 3;    
    Event(n).seqControl = 0;
    n = n+1;
    
    
    Event(n).info = 'External beamforming & display.'; 
    Event(n).tx = 0;         
    Event(n).rcv = 0;        
    Event(n).recon = 0;      
    Event(n).process = 2;    
    if floor(i/returnToMatlabFreq) == i/returnToMatlabFreq  % Exit to Matlab 
        Event(n).seqControl = 4; 
    else
        Event(n).seqControl = 0;
    end
    n = n+1;
end

Event(n).info = 'Jump back';
Event(n).tx = 0;        
Event(n).rcv = 0;       
Event(n).recon = 0;     
Event(n).process = 0; 
Event(n).seqControl = 1;


% User specified UI Control Elements
% - Sensitivity Cutoff
UI(1).Control =  {'UserB7','Style','VsSlider','Label','Sens. Cutoff',...
                  'SliderMinMaxVal',[0,1.0,Recon(1).senscutoff],...
                  'SliderStep',[0.025,0.1],'ValueFormat','%1.3f'};
UI(1).Callback = text2cell('%SensCutoffCallback');

% - Range Change
MinMaxVal = [64,300,P.endDepth]; % default unit is wavelength
AxesUnit = 'wls';
if isfield(Resource.DisplayWindow(1),'AxesUnits')&&~isempty(Resource.DisplayWindow(1).AxesUnits)
    if strcmp(Resource.DisplayWindow(1).AxesUnits,'mm');
        AxesUnit = 'mm';
        MinMaxVal = MinMaxVal * (Resource.Parameters.speedOfSound/1000/Trans.frequency);
    end
end
UI(2).Control = {'UserA1','Style','VsSlider','Label',['Range (',AxesUnit,')'],...
                 'SliderMinMaxVal',MinMaxVal,'SliderStep',[0.1,0.2],'ValueFormat','%3.0f'};
UI(2).Callback = text2cell('%RangeChangeCallback');

% Callback for saving RF data
UI(3).Control = {'UserB3', 'Style', 'VsPushButton', 'Label', 'SaveFast RF'};
UI(3).Callback = text2cell('%SaveRFCallback');

% External function definitions.
import vsv.seq.function.ExFunctionDef
EF(1).Function = vsv.seq.function.ExFunctionDef('beamformingServer', @beamformingServer);
EF(2).Function = vsv.seq.function.ExFunctionDef('updateFromServer', @updateFromServer);
%EF(3).Function = vsv.seq.function.ExFunctionDef('saveStructures', @saveStructures);


% Specify factor for converting sequenceRate to frameRate.
frameRateFactor = returnToMatlabFreq;

% Save all the structures to a .mat file.
save(strcat('MatFiles/',filename));

return




% **** Callback routines to be converted by text2cell function. ****
%SensCutoffCallback - Sensitivity cutoff change
ReconL = evalin('base', 'Recon');
for i = 1:size(ReconL,2)
    ReconL(i).senscutoff = UIValue;
end
assignin('base','Recon',ReconL);
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'Recon'};
assignin('base','Control', Control);
return
%SensCutoffCallback

%RangeChangeCallback - Range change
simMode = evalin('base','Resource.Parameters.simulateMode');
% No range change if in simulate mode 2.
if simMode == 2
    set(hObject,'Value',evalin('base','P.endDepth'));
    return
end
Trans = evalin('base','Trans');
Resource = evalin('base','Resource');
scaleToWvl = Trans.frequency/(Resource.Parameters.speedOfSound/1000);

P = evalin('base','P');
P.endDepth = UIValue;
if isfield(Resource.DisplayWindow(1),'AxesUnits')&&~isempty(Resource.DisplayWindow(1).AxesUnits)
    if strcmp(Resource.DisplayWindow(1).AxesUnits,'mm');
        P.endDepth = UIValue*scaleToWvl;    
    end
end
assignin('base','P',P);

evalin('base','PData(1).Size(1) = ceil((P.endDepth-P.startDepth)/PData(1).PDelta(3));');
evalin('base','PData(1).Region = computeRegions(PData(1));');
evalin('base','Resource.DisplayWindow(1).Position(4) = ceil(PData(1).Size(1)*PData(1).PDelta(3)/Resource.DisplayWindow(1).pdelta);');
Receive = evalin('base', 'Receive');
maxAcqLength = ceil(sqrt(P.endDepth^2 + ((Trans.numelements-1)*Trans.spacing)^2));
for i = 1:size(Receive,2)
    Receive(i).endDepth = maxAcqLength;
end
assignin('base','Receive',Receive);
evalin('base','TGC.rangeMax = P.endDepth;');
evalin('base','TGC.Waveform = computeTGCWaveform(TGC);');
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'PData','InterBuffer','ImageBuffer','DisplayWindow','Receive','TGC','Recon'};
assignin('base','Control', Control);
assignin('base', 'action', 'displayChange');
return
%RangeChangeCallback


%SaveRFCallback
if ~isempty(findobj('tag','UI')) % running VSX
    if evalin('base','freeze')==0   % no action if not in freeze
        msgbox('Please freeze VSX');
        return
    else
        Control.Command = 'copyBuffers';
        runAcq(Control); % NOTE:  If runAcq() has an error, it reports it then exits MATLAB.
    end
else % not running VSX
    if evalin('base','exist(''RcvData'',''var'');')
        RcvData = evalin('base','RcvData');
    else
        disp('RcvData does not exist!');
        return
    end
end

RFfilename = ['L11-4vPlaneWaves-Beamforming-Display-Server','_',datestr(now,'dd-mmmm-yyyy_HH-MM-SS')];

RcvLastFrame = size(RcvData,3);
if (~evalin('base','simButton'))
    RcvLastFrame = Resource.RcvBuffer(1).lastFrame;
end

RcvData = RcvData{1}(:,:,:);


[fn,pn,~] = uiputfile('*.mat','Save RF data as',RFfilename);
if ~isequal(fn,0) % fn will be zero if user hits cancel
    fn = strrep(fullfile(pn,fn), '''', '''''');
    %save(fn,'RcvData','RcvLastFrame','-v7.3');
    savefast(fn,'RcvData','RcvLastFrame');


    fn2 = [fn(1:end-4), '_param.mat']
    save(fn2, '-regexp', '^(?!Rcv.*$).')


    %save(fn,'RcvData','RcvLastFrame', '-v7.3', '-nocompression');
    fprintf('The RF data has been saved at %s \n',fn);
else
    disp('The RF data is not saved.');
end
%SaveRFCallback


function beamformingServer(RFData)
    %keyboard

    % Absolute time at which it enters the external function for writing frames
    timeExtFuncStartWrite_vect = evalin('base', 'timeExtFuncStartWrite_vect');
    dateInfo = datetime('now','TimeZone','local', 'Format' , 'yyyy-MM-dd HH:mm:ss.SSSSSS');
    timeExtFuncStartWrite_vect = [timeExtFuncStartWrite_vect, dateInfo];
    assignin('base', 'timeExtFuncStartWrite_vect', timeExtFuncStartWrite_vect);

    tTOTALstart = tic;

    info = whos('RFData');
    if info.class ~= 'int16'
        disp("Error: data format")
            return
    end

    t = evalin('base','t'); % tcp socket
    na = evalin('base', 'na');
    %sampleMode = evalin('base', 'sampleMode');
    Receive = evalin('base', 'Receive');
    na_transmit = evalin ('base', 'na_transmit');
    bytesPerElementSent = evalin('base', 'bytesPerElementSent'); % write channel data as int16
    
    % Initilization Raw Data
    % NB: ROWS PER ACQUISITION CHANGE WITH THE SAMPLE MODE 
    % (Receive(1).samplesPerWave takes that into account)
    nRowsPerAcq = 2*(Receive(1).endDepth - Receive(1).startDepth)*Receive(1).samplesPerWave;  %Receive(1).endSample
    fprintf('no. rows per acquisition equals to: %d.', nRowsPerAcq);
    disp('WARNING: Ensure no. rows per acquisition is consistent with server expected rows')
    
    % SEND DATA  
    disp('Writing ...')
           
    if na_transmit == na
        data = RFData(1 : na_transmit*nRowsPerAcq, :); %when moving to S51 specificy the elements
    elseif na_transmit == 5
        angles = [2,4,6,8,10];
        data = [];
        for k = 1:length(angles)
            data = vertcat(data, RFData(k*nRowsPerAcq : (k+1)*nRowsPerAcq - 1, :)); %when moving to S51 specificy the elements
        end
    elseif na_transmit == 1
        start = ceil(na/2);
        data = RFData(start*nRowsPerAcq : (start+1)*nRowsPerAcq - 1, :); %when moving to S51 specificy the elements
    else
        disp("Error: na_transmit can only be set to 1 or na")
        return
        keyboard
    end
    %disp(size(data)) %data = reshape(data,[1,sk(1)*sk(2)]);
    data_line = reshape(data,1,[]);
    
    filename =  strcat('saved_data_matlab',extractBefore(datestr(datetime('now')), ' '), strrep(extractAfter(datestr(datetime('now')), ' '),':',''), '.mat'); 
    save(filename, 'RFData', 'data_line')
    
    % write as int16
    write(t,data_line,'int16');

    tTOTALtime = toc(tTOTALstart);
    saved_WritingTime = evalin('base','saved_WritingTime'); %
    saved_WritingTime = [saved_WritingTime; tTOTALtime];
    assignin('base','saved_WritingTime', saved_WritingTime);

end


function updateFromServer()

    
    
    t = evalin('base','t'); % tcp socket
    na = evalin('base', 'na');
    na_transmit = evalin ('base', 'na_transmit');
    bf_idx = evalin ('base', 'bf_idx');
    TPC = evalin('base', 'TPC');
    bytesPerElementRead = evalin('base', 'bytesPerElementRead'); % read updated parameters as double
    numTunableParameters = evalin('base', 'numTunableParameters');
    saved_UpdatingTime = evalin('base','saved_UpdatingTime'); %
    timeExtFuncEndUpdate_vect = evalin('base', 'timeExtFuncEndUpdate_vect');
    saved_WritingTime = evalin('base','saved_WritingTime'); %
    timeExtFuncStartWrite_vect = evalin('base', 'timeExtFuncStartWrite_vect');
    
    %READ DATA
    disp('Reading ...')
    
    tUPDATEstart = tic;
    
    %keyboard
    total = 0;
    tsb_cat = [];
    i = 0; %used for timeout

    while  total < numTunableParameters

        while t.BytesAvailable == 0
            % disp('waiting for data');
            i=i+1;
            if i == 1e8; return; end
        end

        
        tsb_uint8 = read(t, t.BytesAvailable, 'uint8');

        tsb_cat = cat(2, tsb_cat, tsb_uint8);
        disp(tsb_cat);
        total = length(tsb_cat)/bytesPerElementRead; %normalize as we read as uint8 and not as int16
        
    end
    
    tsb = typecast(uint8(tsb_cat),'double');
    %disp(tsb)
    
    if length(tsb) ~= numTunableParameters
        disp("Error: received less data than expected")
        return
        keyboard
    end
    
    if tsb(2) ~= 0 && tsb(2) <= na
       %keyboard
       %save performance with previous number of firing angles
       saveStructures(na_transmit, bf_idx, timeExtFuncEndUpdate_vect, saved_UpdatingTime, saved_WritingTime, timeExtFuncStartWrite_vect);
       
       a = fprintf('Firing angles: %d', tsb(2));
       disp(a);
       
       assignin('base', 'na_transmit', tsb(2));
    end

    if tsb(3) ~= 0:
       %keyboard
       %save performance with previous number of firing angles
       saveStructures(na_transmit, bf_idx, timeExtFuncEndUpdate_vect, saved_UpdatingTime, saved_WritingTime, timeExtFuncStartWrite_vect);
       
       switch tsb(3)
        case 1 
            bf_idx = 1
        case 2
            bf_idx = 2
        case 3 
            bf_idx = 3

    end

    if tsb(1) ~= 0
        hvSuggested = tsb(1);
        %keyboard
        % High Voltage 1
        if TPC(1).inUse

            ntpc = 1;
            a = fprintf('Old Transmit Voltage value:',TPC(ntpc).hv);
            disp(a);
            b = fprintf('Transmit Voltage value:', hvSuggested);
            disp(b);
            
            hv = hvSuggested;

            % Attempt to set high voltage.
            % On error, setTpcProfileHighVoltage() returns voltage range minimum.
            [result, hvset] = setTpcProfileHighVoltage(hv, ntpc);
            if ~strcmpi(result, 'Success')
                % ERROR!  Failed to set high voltage.

                error('ERROR!  Failed to set Verasonics TPC high voltage for profile %d because \"%s\". hv suggested: %d \n', ntpc, result, hv);
            end
        end

    end
    
    tUPDATEtime = toc(tUPDATEstart);
    saved_UpdatingTime = [saved_UpdatingTime; tUPDATEtime];
    assignin('base','saved_UpdatingTime', saved_UpdatingTime);
    
    % Absolute time at which it finishes the external function for updating
    dateInfo = datetime('now','TimeZone','local', 'Format' , 'yyyy-MM-dd HH:mm:ss.SSSSSS');
    timeExtFuncEndUpdate_vect = [timeExtFuncEndUpdate_vect, dateInfo];
    assignin('base', 'timeExtFuncEndUpdate_vect', timeExtFuncEndUpdate_vect);
end


function saveStructures(na_transmit, bf_idx, timeExtFuncEndUpdate_vect, saved_UpdatingTime, saved_WritingTime, timeExtFuncStartWrite_vect)
    
    switch bf_idx
        case 1
            bf_type = 'RAW'
        case 2
            bf_type = 'DAS'
        case 3
            bf_type = 'ABLE'



    switch na_transmit
        
        case 1
            timeExtFuncEndUpdate_vect1 = timeExtFuncEndUpdate_vect; 
            saved_UpdatingTime1 = saved_UpdatingTime; 
            saved_WritingTime1 = saved_WritingTime;  
            timeExtFuncStartWrite_vect1 = timeExtFuncStartWrite_vect;
            filename =  strcat('L114v_na1_', bf_type, '_matlab', extractBefore(datestr(datetime('now')), ' '), strrep(extractAfter(datestr(datetime('now')), ' '),':',''), '.mat'); 
            save(filename, 'timeExtFuncEndUpdate_vect1', 'saved_UpdatingTime1', ...
                'saved_WritingTime1',  'timeExtFuncStartWrite_vect1')

        case 5
            timeExtFuncEndUpdate_vect5 = timeExtFuncEndUpdate_vect; 
            saved_UpdatingTime5 = saved_UpdatingTime; 
            saved_WritingTime5 = saved_WritingTime;  
            timeExtFuncStartWrite_vect5 = timeExtFuncStartWrite_vect;
            filename =  strcat('L114v_na5_', bf_type, '_matlab',extractBefore(datestr(datetime('now')), ' '), strrep(extractAfter(datestr(datetime('now')), ' '),':',''), '.mat'); 
            save(filename, 'timeExtFuncEndUpdate_vect5', 'saved_UpdatingTime5', ...
                'saved_WritingTime5',  'timeExtFuncStartWrite_vect5')

        case 11
            timeExtFuncEndUpdate_vect11 = timeExtFuncEndUpdate_vect; 
            saved_UpdatingTime11 = saved_UpdatingTime; 
            saved_WritingTime11 = saved_WritingTime;  
            timeExtFuncStartWrite_vect11 = timeExtFuncStartWrite_vect;
            filename =  strcat('L114v_na11_', bf_type, '_matlab', extractBefore(datestr(datetime('now')), ' '), strrep(extractAfter(datestr(datetime('now')), ' '),':',''), '.mat'); 
            save(filename, 'timeExtFuncEndUpdate_vect11', 'saved_UpdatingTime11', ...
                'saved_WritingTime11',  'timeExtFuncStartWrite_vect11')
    end
end
