%usbmd_InitTrans.m  computes the specification of the transducer type for one
%                   of the following transducers: {'CL15-7', 'S5-1', 'L15-7io',
%                   'ViewFlex', 'mTEE'}
%
%Usage : Trans = usbmd_InitTrans(transducerName, frequency, numElements);
%        Trans = usbmd_InitTrans(transducerName);
%
%Input : transducerName  : string defining the selected transducer array
%                          {'CL15-7', 'L15-7io', 'ViewFlex', 'S5-1', 'mTEE'}
%        frequency       : optional input argument for a non-default
%                          transducer frequency in MHz. Default is taken
%                          when []:
%                          - 'CL15-7'          :  8.9290 MHz
%                          - 'S5-1'            :  2.4038 MHz
%                          - 'L15-7io'         :  8.9290 MHz
%                          - 'ViewFlex'        :  6.2500 MHz
%                          - 'mTEE'            :  5.0000 MHz
%        numElements     : optional input argument for a lower-than-default
%                          amount of transducer elements. The selected
%                          transducer element positions will be the ones
%                          centered around the middle of the array. Default
%                          is taken when set to []:
%                          - 'CL15-7'          : 128
%                          - 'S5-1'            :  80
%                          - 'L15-7io'         : 128
%                          - 'ViewFlex'        :  64
%                          - 'mTEE'            :  48 chan, Fc=5 MHz, pitch, 0.151 mm
%
%Output: Trans           : structure defining the transducer type

function Trans = usbmd_InitTrans(transducerName, frequency, numElements)

    Trans = struct('name', [], ...
                   'secondName', [], ...
                   'id', [], ...
                   'frequency', [], ...
                   'type', [], ...
                   'units', [], ...
                   'numelements', [], ...
                   'elementWidth', [], ...
                   'spacingMm', [], ...
                   'spacing', [], ...
                   'ElementPos', [], ...
                   'ElementSens', [], ...
                   'startDepthWL', [], ...
                   'RxInversion', [], ... % Receive channel inversion FLICE
                   'lensCorrection', [], ...
                   'maxHighVoltage', [], ...
                   'Bandwidth', [], ...
                   'impedance', [], ...
                   'ConnectorES', [], ...
                   'shape', []); % See getIusConstants()

    if nargin > 1
        if ~isempty(frequency)
            verifyValidFrequency(frequency);
            Trans.frequency = frequency;
        end
    end
    
    if strcmp(transducerName,'CL15-7') || ...
            strcmp(transducerName,'L15-7io') || ...
            strcmp(transducerName,'ViewFlex') || ...
            strcmp(transducerName,'S5-1') || ...
            strcmp(transducerName,'mTEE')
        Trans.name = 'custom';
        Trans = computeCustomTrans(Trans, transducerName);
    else
        error(['Transducer ', transducerName, ' is not accepted.']);
    end
    
    if nargin > 2
        if ~isempty(numElements)
            Trans = lowerNumElements(Trans, numElements);
        end
    end
    
    % Check if high frequency module can be used
    VerasonicsSerialNumber = getenv('VerasonicsSerialNumber');
    needsHighFreq = 4 * Trans.frequency > 62.5 || ...
        ( length(Trans.Bandwidth) == 2 && 2 * Trans.Bandwidth(2) > 62.5 );
    if needsHighFreq && VerasonicsSerialNumber == '1'
        error('High frequency module not installed on this machine.');
    end        
end

%--------------------------------------------------------------------------
% Local function to compute the Trans structure for a custom array
%--------------------------------------------------------------------------
function Trans = computeCustomTrans(Trans, transducerName)

    getIusConstants;

    Trans.secondName = transducerName;
    Trans.type = 0;
    Trans.id = 0;
    Trans.units = 'mm';
    if strcmp( transducerName, 'S5-1')  
            Trans.lensCorrection = 0.6899;
            if isempty(Trans.frequency)
                Trans.frequency = 3.1250; %4.808; % 2.4038;  % default
            end
            Trans.connType       = 1;
            Trans.numelements    = 80;
            Trans.elementWidth   = 0.25;
            Trans.spacingMm      = 0.254;
            Trans.maxHighVoltage = 50;
            Trans.impedance      = 50;
            Trans.ConnectorES      = (49:128)';
            idx=zeros(80,1); offset=0; for n=1:2:40, idx(n)=n+offset; idx(n+1)=n+1+offset; offset=offset+2; end; offset=38; for n=41:2:80, idx(n)=n+offset+1; idx(n+1)=n+offset; offset=offset-6; end
            Trans.ConnectorES = Trans.ConnectorES(idx);
            Trans.ElementSens    = gausswin(101,pi)';
            Trans.startDepthWL   = 0;
            Trans.RxInversion    = ones(1, Trans.numelements);
        Trans.spacing = Trans.spacingMm * Trans.frequency / 1.540;
        Trans.ElementPos     = zeros(Trans.numelements,4);
        Trans.ElementPos(:,1)= Trans.spacing * ...
            (-((Trans.numelements-1)/2):((Trans.numelements-1)/2));
        Trans.shape = TRANSDUCER_SHAPE_LINE;

    elseif strcmp( transducerName, 'mTEE')  
            Trans.lensCorrection = 0.6899;
            if isempty(Trans.frequency)
                Trans.frequency = 4.808;  % default
            end
            Trans.numelements    = 48;
            Trans.elementWidth   = 0.150;
            Trans.spacingMm      = 0.151;
            Trans.maxHighVoltage = 20;
            Trans.impedance      = 50;
            Trans.ConnectorES    = (81:128)';
            % idx=zeros(80,1); offset=0; for n=1:2:40, idx(n)=n+offset; idx(n+1)=n+1+offset; offset=offset+2; end; offset=38; for n=41:2:80, idx(n)=n+offset+1; idx(n+1)=n+offset; offset=offset-6; end
            % Trans.ConnectorES = Trans.ConnectorES(idx);
            idx=zeros(48,1); offset=0; for n=1:2:24, idx(n)=n+offset; idx(n+1)=n+1+offset; offset=offset+2; end; offset=22; for n=25:2:48, idx(n)=n+offset+1; idx(n+1)=n+offset; offset=offset-6; end
            Trans.ConnectorES    = Trans.ConnectorES(idx);
            Trans.ConnectorES    = flipud(Trans.ConnectorES);
            Trans.ElementSens    = gausswin(101,pi)';
            Trans.startDepthWL   = 0;
            Trans.RxInversion    = ones(1, Trans.numelements);
        Trans.spacing = Trans.spacingMm * Trans.frequency / 1.540;
        Trans.ElementPos     = zeros(Trans.numelements,4);
        Trans.ElementPos(:,1)= Trans.spacingMm * ...
            (-((Trans.numelements-1)/2):((Trans.numelements-1)/2));
        Trans.shape = TRANSDUCER_SHAPE_LINE;
    
    elseif strcmp( transducerName, 'CL15-7')
            Trans.lensCorrection = 0.6899;
            if isempty(Trans.frequency)
                Trans.frequency = 8.9290;  % default
            end
            Trans.numelements    = 128;
            Trans.elementWidth   = 0.16;
            Trans.spacingMm      = 0.178;
            Trans.maxHighVoltage = 50;
            Trans.impedance      = 50;
            Trans.ConnectorES    = (1:128)';
            Trans.ElementSens    = gausswin(101,pi)';
            Trans.startDepthWL   = Trans.frequency * 6.3 / 1.540;
            Trans.RxInversion    = ones(1, Trans.numelements);
        Trans.spacing = Trans.spacingMm * Trans.frequency / 1.540;
        Trans.ElementPos     = zeros(Trans.numelements,4);
        Trans.ElementPos(:,1)= Trans.spacingMm * ...
            (-((Trans.numelements-1)/2):((Trans.numelements-1)/2));
        Trans.shape = TRANSDUCER_SHAPE_LINE;
        
    elseif strcmp( transducerName, 'L15-7io')
            Trans.lensCorrection = 0.6899; 
            if isempty(Trans.frequency)
                Trans.frequency = 8.9290;  % default
            end
            Trans.connType       = 1;
            Trans.numelements    = 128;
            Trans.elementWidth   = 0.16;
            Trans.spacingMm      = 0.178;
            Trans.maxHighVoltage = 40;
            Trans.impedance      = 50;
            Trans.ConnectorES = [2,1,5,6,10,9,13,14,18,17,21,22,25,26,29,30,33,34,37,38,41,42,45,46,49,50,53,54,57,58,61,62,65,66,69,70,73,74,77,78,81,82,85,86,89,90,93,94,97,98,101,102,105,106,109,110,113,114,117,118,121,122,125,126,128,127,124,123,120,119,116,115,112,111,108,107,104,103,100,99,96,95,92,91,88,87,84,83,80,79,76,75,72,71,68,67,64,63,60,59,56,55,52,51,48,47,44,43,40,39,36,35,32,31,28,27,24,23,20,19,16,15,12,11,8,7,3,4]';
            Trans.ElementSens    = gausswin(101,pi)';
            Trans.startDepthWL   = 0; %CL15-7: Trans.frequency * 6.3 / 1.540;
            Trans.RxInversion    = ones(1, Trans.numelements);
        Trans.spacing = Trans.spacingMm * Trans.frequency / 1.540;
        Trans.ElementPos     = zeros(Trans.numelements,4);
        Trans.ElementPos(:,1)= Trans.spacingMm * ...
            (-((Trans.numelements-1)/2):((Trans.numelements-1)/2));
        Trans.shape = TRANSDUCER_SHAPE_LINE;        

    elseif strcmp(transducerName, 'ViewFlex')
            Trans.lensCorrection = 0;
            if isempty(Trans.frequency)
                Trans.frequency = 6.25;  % default
            end
            Trans.numelements    = 64;
            Trans.elementWidth   = 0.2;
            Trans.spacingMm      = 0.205;
            Trans.maxHighVoltage = 50;
            Trans.impedance      = 50;
            Trans.ConnectorES    = [67 68 71 72 75 76 79 80 83 84 87 88, ...
                91 92 95 96 99 100 103 104 107 108 111 112 115 116 119, ...
                120 123 124 127 128 126 125 122 121 118 117 114 113, ...
                110 109 106 105 102 101 98 97 94 93 90 89 86 85 82 81, ...
                78 77 74 73 70 69 66 65]';
            theta = (-pi/2:pi/100:pi/2);
            X = Trans.elementWidth * pi * sin(theta);
            Trans.ElementSens    = abs(cos(theta).*(sin(X+eps)./(X+eps)));
            Trans.startDepthWL   =  0;
            Trans.RxInversion    = ones(1, Trans.numelements);
        Trans.spacing = Trans.spacingMm * Trans.frequency / 1.540;
        Trans.ElementPos     = zeros(Trans.numelements,4);
        Trans.ElementPos(:,1)= Trans.spacingMm * ...
            (-((Trans.numelements-1)/2):((Trans.numelements-1)/2));
        Trans.shape = TRANSDUCER_SHAPE_LINE;

    else
            error(['Transducer ', Trans.name, ' is not known.']);
    end
end

%--------------------------------------------------------------------------
% Local function that exits with an error if the provided transducer
% frequency is not supported.
%--------------------------------------------------------------------------
function verifyValidFrequency(frequency)
    
    validFrequencies = [41.67 31.25 25.0 20.83 17.86 15.625 13.89 12.5, 10.4167, 8.9286, 7.8125, 6.944, 6.25];
	
	if ~any(validFrequencies == frequency)
        error(['Frequency ',num2str(frequency),' MHz is not accepted']);
    end
end

%--------------------------------------------------------------------------
% Local function to lower the number of elements for any array after the
% Trans structure was set by computeTrans or by computeCustomTrans.
%--------------------------------------------------------------------------
function Trans = lowerNumElements(Trans, N)
    if (N > 0) && (N <= Trans.numelements)
        centerIdx = (Trans.numelements + 1) / 2;
        if mod(Trans.numelements,2) == mod(N,2)
            leftIdx   = ceil(centerIdx - N/2);
            rightIdx  = floor(centerIdx + N/2);
            idxRange  = leftIdx : 1 : rightIdx;
        else
            leftIdx   = centerIdx - N/2;
            rightIdx  = centerIdx + N/2 - 1;
            idxRange  = leftIdx : 1 : rightIdx;
        end
        Trans.numelements = N;
        Trans.ElementPos  = Trans.ElementPos(idxRange,:);
        Trans.ConnectorES = Trans.ConnectorES(idxRange);
        Trans.RxInversion = Trans.RxInversion(idxRange);
    else
        error(['Invalid numelements for ', Trans.name])
    end
end
