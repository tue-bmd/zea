% Copyright 2001-2021 Verasonics, Inc.  Verasonics Registered U.S. Patent and Trademark Office.
%
% VSX (Verasonics Script eXecute) for processing .mat file (filename), for
% use exclusively on Verasonics "Vantage" system products.
%
% Usage: run VSX and input 'filename'
%   where filename is the name of a .mat file contaiuning all the structures
%   needed.
%
%   To allow automated scripts to launch VSX, such as HardwareTest scripts,
%   'filename' may be set in the workspace before calling VSX.  To prevent
%   subsequently run scripts from inadvertently using that same 'filename',
%   VSX will use the detected 'filename' once and then delete the variable.
%
% The keyword "TEST" has been inserted in locations where temporary edits
% can be made to facilitate debug or testing.
%
% Program steps; (these match the "%%" section headings that can be found
% though the "Go To" command from Matlab Editor Toolbar)
%   1. initialize, clear variables, load user script from .mat file
%   2. Check and validate required structures, set defaults
%   3. Resource.System: determine intent & requirements of script
%   4. Initialize Trans structure
%   5. Determine system configuration & status
%   6. Check configuration required by script versus actual system
%   7. Check Operating Mode and set default if needed
%   8. Load external processing function(s)
%   9. Resource.VDAS initialize
%  10. Create channel mapping arrays based on UTA type
%  11. Check Resource.Parameters for valid attributes and initialize
%  12. Trans.HVMux and .Connector mapping
%  13. Media, PData, TW, TX, RcvProfile, TGC, Recon structures
%  14. TPC Profile 5 and HIFU Option Initialization
%  15. If not in simulate mode try to open the hardware.
%  16. check probe connected status and ID value
%  17. Initialize GUI and Display Window(s)
%  18. Final initialization and calls to VsUpdate and TXEventCheck
%  19. Run sequence: loop for continuously calling 'runAcq'.
%      While vsExit==0
%         Call runAcq to process events in sequence.
%         Every jump command back to 1st event will return control to Matlab
%            to check for GUI actions.
%            - for Freeze action, stop sequencer and wait in Matlab for
%              Freeze button to change state.
%  20. VsUpdate calls within the run-time while loop
%  21. Closing VSX: Copy buffers back to Matlab workspace.

% Revision History:
%   May 2021 VTS-2155, VTS-2267 Restrict probes > 23 MHz to HF systems only
%   Apr 2021 VTS-1837 add checks for indexing errors in Event structure
%   Oct 2020 VTS-1975 Expose V-512 functionality for 4.4.0; VTS-1818
%   Aug  7 2020 VTS-1796 pass initialization status to TXEventCheck
%   May 14 2020 VTS-1365 New TW waveform utilities, TW.type support
%   May 9, 2020 VTS-343 SHIAperture replaces "VDAS Aperture" to allow
%       256-byte HVMux programming tables.
%   April 23, 2020 VTS-1695 Delete pan and zoom controls
%   April 6 2020 VTS-1672 VSX initialization clears operating state left
%       from previous run of a sequence, such as HV Mux settings.
%   March 2020 VTS-1480 New UI Control package scheme; Matlab 2019b compatibility
%   Feb 2020 changes for 4.3.0: VTS-172 report frame rate for both runAcq
%       and overall software loop
%   Jan 12 2020 VTS-1583 TPC profile 5 monitor limit updates for 4.2.0
%   Nov 5 2019 VTS-1513 same initialization for Matlab and Verasonics viewers
%   Sep 1 2019: VTS-1416 add "activeClamp" to Resource.System, to support
%       managing active clamp (Gen3 Plus) acquisition module
%   July 2019, release 4.1.0: VTS-1063 add support for Volume Imaging Package and VTS-1317 bug fix
%   May 2019: VTS-902 MATLAB version check moved to activate
%   Jan 2019: VTS-1047 VSX query for name of script to run will also
%       accept name of SetUp scriptt
% Revised April-August 2018 for Vantage 4.0.0 SW release:
%   VTS-877 Minimum Matlab release 2017a
%   VTS-852 new mechanism for dynamic mux programming using
%           computeMuxAperture
%   VTS-843 use of Trans.ConnectorES and HVMux.ApertureES
%   VTS-827 Trans.HVMux.Aperture generation for dynamic mux programming
%   VTS-823 New utility "computeHvMuxVdas_Aperture" to create Trans field
%           HVMux.VDAS_Aperture for dynamic mux programming
%   VTS-808 eliminate Resource.Parameters.numTransmit and numRcvChannels as
%           required fields; VSX creates numTransmit for use by runAcq etc.
%   VTS-804 Apod mapping can be either all elements or active elements
%   VTS-802 add 'Resource.System' structure to identify script requirements
%   VTS-768 to incorporate "autoScriptTest" functionality in VSX
%   VTS-792 Connector types 6,12 are interchangeable for UTA 160-DH/32 LEMO
%   VTS-825 showEULA feature disabled but not deleted


%% *** 1. initialize, clear variables, load user script from .mat file ***
import com.verasonics.hal.hardware.*
import com.verasonics.hal.faults.*

% TEST Keep this commented out code for the duration of HAL-NG integration and testing.
%import com.verasonics.common.logger.*
%logPrefsFile = 'logPrefs.xml';
%if ~Logger.configureLoggingFromFile(logPrefsFile, 'vsx')
%    warning('Logger preferences file not found at %s', logPrefsFile)
%end
%clear logPrefsFile

% Check that path containing this file has been activated.
vsxDir = fileparts(mfilename('fullpath'));
vpfRoot = getenv('VERASONICS_VPF_ROOT');
isActivated = false;
if getenv('VERASONICS_IS_ACTIVATED') == '1'
    isActivated = true;
end
% A deployed application has VSX in a subdirectory, so a "string
% starts with" comparison is used rather than a direct comparison.
if ~isActivated || ~startsWith(vsxDir, vpfRoot)
    error('Run ''activate'' before VSX');
end
clear vsxDir vpfRoot isActivated


% clear runAcq, in case it had been left open (such as from an abnormal
% exit of a previous run due to an error condition)
clear runAcq

% temporary cleanup of any spectral Doppler processing.
try
    spectralDoppler('cleanup');
catch
end

% Clear all variables with a few exceptions.
%
% NOTE:  "Mcr_" prefixed variables are designed for use by the MATLAB Compiler
% Runtime, though you don't have to use the MCR to use the "Mcr_" prefix.
% This prefix can be used for any variable you want VSX to keep for later
% use, such as 'Mcr_AutoScriptTest' as seen below.
vars = whos;
for i = 1:size(vars,1)
    if ~(strcmp(vars(i).name,'filename')                  || ...
         strcmp(vars(i).name,'vsTempDir')                 || ...
         ~isempty(strfind(vars(i).name,'Mcr_'))           || ... % Name contains "Mcr_"
         strcmp(vars(i).name,'RcvData')                   || ...
         strcmp(vars(i).name,'vars')                      || ...
         strcmp(vars(i).name, 'isUsingOldVSXGUIcreation') || ...
         strcmp(vars(i).name, 'vsxApplication')           || ...
         strcmp(vars(i).name,'Vs_VdasDebugging'))
        clear(vars(i).name);
    end
end

clear i vars

if exist('Mcr_AutoScriptTest', 'var')
    % This enables automatic cycling and exit of VSX, for use in automated
    % SetUp script test routines.
    clear Mcr_AutoScriptTest
    autoScriptTest = 1;
else
    autoScriptTest = 0;
end

% Define temporary directory for dynamically generated files.
% The addpath function is not recommended in a compiled application and
% dynamically generated scripts cannot be used in a compiled application.
% The compiled application should define vsTempDir relative to ctfroot.
if ~isdeployed()
    if ~exist('vsTempDir', 'var')
        vsTempDir = tempdir;
    end
    addpath(vsTempDir);
    rehash path
end

% Default to not using HIFUPlex; if we load a HIFUPlex script it will
% overwrite this variable and set it to true.
usingHIFUPlex = 0;

% Default to not using usingMultiSys; if we load a usingMultiSys script it will
% overwrite this variable and set it to true.
usingMultiSys = 0;

% Initialize the flag that indicates whether or not the hardware is in a
% fault state.
isHwInFaultState = false;


%% *** 1.1 Initialize app

if ~exist('vsxApplication', 'var') || isempty(vsxApplication)
    vsxApplication = vsv.apps.VSXResearchApp( 'VSGUI' );
elseif ~isa(vsxApplication, 'vsv.apps.AbstractVSXApp' )
    error('Given application must be a vsv.apps.AbstractVSXApp');
end

% application = vsv.apps.PanelApp();
usingApplication = 1;


% Read in structures defined by user script from .mat file.
%
% Use filename variable if it is given and not empty, but only once.
% Otherwise ask the user for the filename.
if(~exist('filename', 'var') || (exist('filename', 'var') && isempty(filename)))
    % OK, the filename variable does NOT exist, so ask the user for the
    % filename.
    filename = input('Name of .mat file to process: ','s');
    if isempty(filename)
        disp('VSX exiting; no .mat file specified.')
        clear
        return
    end
end
% Check if filename is a .m file, and if so, get .mat filename from file.
if endsWith(filename,'.m'), fid = fopen(filename);
else, fid = fopen([filename '.m']); end % try to open filename with no .m extension as a .m file
if fid>=3 % fid is >= 3 if file found
    n = 0;
    txtline = fgetl(fid);
    while ischar(txtline)
       if contains(txtline,'filename =') || contains(txtline,'filename=')
           n = n+1;
           filenamex = char(extractBetween(txtline,'''',''''));
           % if txtline has mulitple expressions filenamex may have
           % multiple rows.  We want only the first row.
           filename = filenamex(1, :);
           break
       elseif contains(txtline,'save(')
           n = n+1;
           filenamex = char(extractBetween(txtline,'''',''''));
           % if txtline has mulitple expressions filenamex may have
           % multiple rows.  We want only the first row.
           filename = filenamex(1, :);
           break
       else
           txtline = fgetl(fid);
       end
    end
    fclose(fid);
    clear fid filenamex txtline
    if n == 0
        fprintf(2, 'VSX: unable to obtain name of .mat file from %s.m', filename);
        clear filename
        error(' ');
    end
end

% Use a try-catch so if file does not exist we can clear 'filename' for
% next run of VSX.
try
    load(filename);
    [~,displayWindowTitle] = fileparts(filename); % Strip leading path to get just filename
    clear filename;
catch
    % File does not exist.  Clear 'filename' and report error.
    fprintf(2, ['The file "', filename, '" was not found.\n']);
    clear filename;
    error('VSX: file not found');
end

% set system constants
sysClk = 250; % system primary oscillator clock rate in MHz
minTpcVoltage = 1.6; % minimum TPC voltage in Volts

maxADRate = 62.5; % maximum A/D sample rate in MHz
% AFE5808-5812 A/D maximum sample rate is 65 MHz.  So maximum for this
% system is 62.5 MHz (250 MHz sysClk divided by 4)
minADRate = 10; % Minimum A/D sample rate for AFE5808-5812 in MHz

rchPerBd = 64; % number of receive channels per acquisition board
tchPerBd = 64; % number of transmit channels per acquisition board

%% *** 2. Check and validate required structures, set Resource defaults

% *** Check for presence of required structures
RequiredStructs = {'Trans','Resource','TW','TX','Event','SeqControl'};
for stnum = 1:length(RequiredStructs)
    stName = char(RequiredStructs{stnum});
    if ~exist(stName, 'var')
        error('VSX: %s is a required structure for script execution but was not found.', RequiredStructs{stnum} );
    end
end






%% *** 2.1 Check and Initialize Resource structure
if isempty(Resource)
    error('VSX: Resource structure is empty.');
elseif length(Resource(:)) ~= 1
    error('VSX: An array of Resource structures is not supported.');
end

% - Resource.Parameters structure initialization
% ***verbose level***:  set the variable to control command line display of
% informative or warning messages, if not set in user's script:
% supported values:
%   verbose = 0: display error messages only
%   verbose = 1: display error and warning messages only
%   verbose = 2: display error, warning, and status messages
%   verbose = 3: display error, warning, status, and debug messages

if ~isfield(Resource, 'Parameters') || ~isfield(Resource.Parameters,'verbose') || isempty(Resource.Parameters.verbose)
    % not defined by user, so default to 2 (see above definitions)
    Resource.Parameters.verbose = 2;
end

% - Set a Resource.Parameters.Connector default, if none provided, to
% identify which connector to use on HW configurations with more than one.
% For backward compatibility, check for 'connector' and if present translate it to
% 'Connector'.
if ~isfield(Resource.Parameters,'Connector') || isempty(Resource.Parameters.Connector)
    % check for old format before setting default
    if ~isfield(Resource.Parameters,'connector') || isempty(Resource.Parameters.connector)
        Resource.Parameters.Connector = 1; % default to connector #1
        % note this default value is implicit if system only has one connector,
        % but the variable still needs to be initialized since it is used in
        % system initialization logic for setting other variables.
        if Resource.Parameters.verbose > 2
            disp(' ');
            disp('VSX status: Resource.Parameters.Connector undefined; setting default value of 1.');
            disp(' ');
        end
    else
        % 'connector' exists so use it to define new format 'Connector',
        % but check for length = 1 to make sure it was not a 'Connector'
        % format definition with the name mis-spelled.
        if length(Resource.Parameters.connector) == 1 && Resource.Parameters.connector > 0
            % only one connector being used, so copy the value
            Resource.Parameters.Connector = Resource.Parameters.connector;
        else
            % connector = 0 so select both connectors (the old 'connector'
            % format was only used with UTA modules having one or two
            % connectors)
            Resource.Parameters.Connector = [1 2];
        end
        if Resource.Parameters.verbose > 2
            disp(' ');
            disp('VSX status: Resource.Parameters.Connector undefined; for backward compatibility');
            disp('    it is being created from older format Resource.Parameters.connector.');
            disp(' ');
        end
    end
end

% put entries in ascending order, to make decoding simpler
Resource.Parameters.Connector = sort(Resource.Parameters.Connector);

% for backward compatibility, don't allow deselecting all connectors
if isequal(Resource.Parameters.Connector, 0)
    Resource.Parameters.Connector = 1; % use default behavior of selecting connector 1
end

if ~isfield(Resource.Parameters, 'speedOfSound') || ...
        isempty(Resource.Parameters.speedOfSound)
    Resource.Parameters.speedOfSound = 1540;
end

if ~isfield(Resource.Parameters, 'speedCorrectionFactor') || ...
        isempty(Resource.Parameters.speedCorrectionFactor)
    Resource.Parameters.speedCorrectionFactor = 1.0;
end

if ~isfield(Resource.Parameters,'startEvent') || ...
        isempty(Resource.Parameters.startEvent)
    Resource.Parameters.startEvent = 1;
end

if ~isfield(Resource.Parameters, 'initializeOnly') || ...
        isempty(Resource.Parameters.initializeOnly)
    Resource.Parameters.initializeOnly = 0;
end

if ~isfield(Resource.Parameters, 'waitForProcessing') || ...
        isempty(Resource.Parameters.waitForProcessing)
    Resource.Parameters.waitForProcessing = 0;
end

%% *** 2.2 Check for correct format of TW structure before using it
if isempty(TW)
    error('VSX: TW structure is empty.');
elseif length(TW) > 1 && length(TW(:)) == size(TW, 1)
    % was defined as a column vector; convert to row and warn user if verbose
    if Resource.Parameters.verbose
        fprintf(2, 'VSX warning: TW structure array was defined as a column vector;\n');
        fprintf(2, '     converting to required row vector format.\n');
    end
    TW = TW';
elseif length(TW(:)) > 1 && length(TW(:)) ~= size(TW, 2)
    error('VSX: Multidimensional TW structure array not supported.  It must be a row vector.');
end
% Check to see if TW is specifying a simulate-only waveform.  If so and if
% verbose level is set greater than 1, automatically force simulation-only
% operation.
if Resource.Parameters.verbose>1
    for i = 1:length(TW)
        if strcmp(TW(i).type, 'sampled') || strcmp(TW(i).type, 'function')
            % simulate-only waveform has been found; force SimulateOnly
            % operation
            Resource.System.Product = 'SimulateOnly';
            disp('VSX STATUS:  A simulate-only TW waveform type has been defined.');
            disp('   Switching to SimulateOnly mode to avoid errors with attempted hardware operation.');
            break
        end
    end
end

%% *** 3. Initialize Trans structure

% Verify Trans is not empty, and user has not provided an array of Trans structures
if isempty(Trans)
    error('VSX: Trans structure is empty.');
elseif length(Trans(:)) ~= 1
    error('VSX: An array of Trans structures is not supported.');
end

% Verify numelements is specified and quit with an error if not
if ~isfield(Trans, 'numelements') || isempty(Trans.numelements)
    error('VSX: Trans.numelements must be specified in user script.')
end

if (~isfield(Trans,'ConnectorES') || isempty(Trans.ConnectorES)) && (~isfield(Trans,'Connector') || isempty(Trans.Connector))
    %  create a default version with 1:1 mapping if possible
    if Trans.numelements ~= Resource.Parameters.numTransmit
        error('VSX: Cannot create default Trans.ConnectorES array since Trans.numelements ~= Resource.Parameters.numTransmit.')
    end
    Trans.ConnectorES = (1:Trans.numelements)';
    Resource.Parameters.sizeApod = Trans.numelements;
end


if ~isfield(Trans, 'connType') || isempty(Trans.connType)
    % User script did not define connector type; set default value of -1 to
    % adapt to system being used
    Trans.connType = -1;
    if Resource.Parameters.verbose>1
        disp(' ');
        disp('VSX status: Trans.connType not specified; setting a default value of -1 to adapt to system configuration.');
        disp(' ');
    end
elseif Trans.connType == 0
    % Trans structure indicates simulate-only script
    if ~isfield(Resource, 'System') || ~isfield(Resource.System, 'Product') || isempty(Resource.System.Product)
        % This is simulate-only with no connector mapping so initialize
        % trans.ConnectorES to reflect that
        Resource.System.Product = 'SimulateOnly';
        if ~(usingMultiSys)  %for multisys 5 system configuration, the primary needs to run in simulate mode 2 with Trans.ConnectorES values defined to match channel order
            Trans.ConnectorES = (1:Trans.numelements)';
        end
    elseif ~strcmpi(Resource.System.Product, 'SimulateOnly')
        % user has specified a Product other than 'SimulateOnly', which
        % requires a non-zero connType value
        error('VSX: Resource.System.Product is in conflict with simulate-only value of 0 for Trans.connType.')
    end
end

% Verify name is specified and set default if not
if ~isfield(Trans, 'name') || isempty(Trans.name)
    Trans.name = 'undefined'; % if user did not provide a name set to 'undefined' by default
    if Resource.Parameters.verbose > 1
        disp('VSX: Trans.name not specified.  Setting default name of ''undefined''.');
    end
elseif ~ischar(Trans.name)
    error('VSX: Trans.name must be a string variable.')
end

% Verify id is specified and set default if not
if ~isfield(Trans, 'id') || isempty(Trans.id)
    Trans.id = 0; % if user did not provide an id set to zero by default
    if Resource.Parameters.verbose > 1
        disp('VSX: Trans.id not specified.  Setting default value of zero.');
    end
elseif ischar(Trans.id)
    error('VSX: Trans.id must be a numeric id value, not a string variable.')
end

% Check for Trans.units and supply default if needed.  The default value of
% 'mm' is a compromise to V1 backward compatibility, but the warning
% message will alert the user to how to correct it.
if ~isfield(Trans, 'units')
    if Resource.Parameters.verbose
        fprintf(2,'VSX Warning:  Trans.units not specified.  Setting default units of mm.\n');
        fprintf(2,'If you are actually using wavelength units from the Trans structure \n');
        fprintf(2,'you must specify Trans.units = ''wavelengths'' in your script.\n');
    end
    Trans.units = 'mm';
end

% Verify units has a recognized value
if ~strcmp(Trans.units, 'wavelengths') && ~strcmp(Trans.units, 'mm')
    error('VSX: Unrecognized value for Trans.units. Must be ''mm'' or ''wavelengths''.')
end

% Verify frequency is specified and quit with an error if not
if ~isfield(Trans, 'frequency') || isempty(Trans.frequency)
    error('VSX: Trans.frequency must be specified in user script.')
end

% provide default bandwidth if not specified
if ~isfield(Trans, 'Bandwidth') || isempty(Trans.Bandwidth)
    if isfield(Trans, 'bandwidth') && ~isempty(Trans.bandwidth)
        % for backward compatibility, use the scalar bandwidth if it was
        % set by user in setup script
        Trans.Bandwidth = [-0.5, +0.5]*Trans.bandwidth + Trans.frequency;
        if Resource.Parameters.verbose > 1
            disp('VSX: Setting Trans.Bandwidth based on legacy Trans.bandwidth value.');
        end
    else
        Trans.Bandwidth = [0.7, 1.3] * Trans.frequency;  % 60% bandwidth default value
        if Resource.Parameters.verbose > 1
            disp('VSX: Trans.Bandwidth not specified. Setting default 60 percent value.');
        end
    end
end

% Verify ElementPos is specified and quit with an error if not
if ~isfield(Trans, 'ElementPos') || isempty(Trans.ElementPos)
    error('VSX: Trans.ElementPos must be specified in user script.')
end

% check for Trans.impedance and set default with warning if not provided
if ~isfield(Trans, 'impedance') || isempty(Trans.impedance)
    Trans.impedance = 20; % set default if not provided
    % The default value of 20 Ohms is absurdly low, and is intended to allow
    % the system to run but with an extremely low HV limit, thereby making
    % it easy for the user to discover the source of the restriction.
    if Resource.Parameters.verbose
        fprintf(2, 'VSX WARNING: A value of Trans.impedance has not been specified.  Default value of 20 Ohms has been set.\n');
        fprintf('This will allow script to run, but with extremely restricted transmit voltage limit.\n');
        fprintf('Actual Trans.impedance value should be added in user''s script.\n\n');
    end
end

% Check for elBias field and set default if needed
if ~isfield(Trans, 'elBias') || isempty(Trans.elBias)
    % if not specified create the field and set it to zero (element bias
    % disabled)
    Trans.elBias = 0;
end

% define default and max HV limits based on transducer
hvMax = computeTrans(Trans.name, 'maxHighVoltage');

% Adjust hvMax limit for L22-14vX if elBias has been changed (VTS-1049),
% and also for L35-16vX (VTS-1552) or report error if outside allowed range
if strcmp(Trans.name, 'L22-14vX') || strcmp(Trans.name, 'L22-14vX-LF') || strcmp(Trans.name, 'L35-16vX')
    if Trans.elBias < -35 || Trans.elBias > 0
        error(['VSX: Trans.elBias for ', Trans.name, ' not in allowed range of 0 to -35 Volts.']);
    end
    if strcmp(Trans.name, 'L35-16vX')
        hvMax = 5 - Trans.elBias; % L35-16vX: increases with bias from 5 V. at zero up to 40 V. for max bias of -35 V.
    else
        hvMax = 25 - Trans.elBias; % L22-14vX: increases with bias from 25 V. at zero up to 60 V. for max bias of -35 V.
    end
end
hvDefault = 10*round(hvMax/20);

% - Check for maximum high voltage limit provided in Trans structure.  If not given, set a default.
if ~isfield(Trans,'maxHighVoltage') || isempty(Trans.maxHighVoltage)
    Trans.maxHighVoltage = hvDefault; % Transducer-specific default limit is defined above
    if Resource.Parameters.verbose>1
        fprintf(2,['VSX:  Trans.maxHighVoltage not specified.  Setting default limit of ', num2str(hvDefault), ' Volts.\n']);
    end
elseif Trans.maxHighVoltage < minTpcVoltage || Trans.maxHighVoltage > hvMax
    % check if user-specified voltage is outside system HW or transducer maximum limits
    error(['VSX: Trans.maxHighVoltage must be within the range ' num2str(minTpcVoltage) ' to ', num2str(hvMax), ' Volts for ', Trans.name, ' transducer.'])
end

% Verify ElementSens is specified and quit with an error if not
if ~isfield(Trans, 'ElementSens') || isempty(Trans.ElementSens)
    error('VSX: Trans.ElementSens must be specified in user script.')
end

% Verify type is specified and quit with an error if not
if ~isfield(Trans, 'type') || isempty(Trans.type)
    error('VSX: Trans.type must be specified in user script.')
end

% Verify radius is specified for curved arrays and quit with an error if not
if Trans.type == 1 && (~isfield(Trans, 'radius') || isempty(Trans.radius))
    error('VSX: For curved arrays Trans.radius must be specified in user script.')
end

% VTS-343: Check for legacy HVMux.VDASAperture and convert to new
% SHIAperture; If both are present just use SHIAperture and ignore VDASAperture
% Note that SHIAperture is a Matlab double; the old VDASAperture was a
% Matlab uint8!!
if isfield(Trans, 'HVMux') && isfield(Trans.HVMux, 'VDASAperture') && ~isfield(Trans.HVMux, 'SHIAperture')
    % Only VDASAperture exists so convert it
    if isempty(Trans.HVMux.VDASAperture)
        % it was just a placeholder so add the SHI equivalent
        Trans.HVMux.SHIAperture = [];
        Trans.HVMux.SHIAperLgth = [];
    else
        % make the conversion; strip off first row with length and convert
        % to doubles
        Trans.HVMux.SHIAperture = double(Trans.HVMux.VDASAperture(2:end, :));
        Trans.HVMux.SHIAperLgth = size(Trans.HVMux.SHIAperture, 1);
    end
end

%% *** 4. Resource.System: determine intent & requirements of script
% if Resource.System exists, evaluate it first so the user's expectations
% can guide the rest of VSX initialization.  If it does not exist, create
% it with default assumptions about expected operating state.

% Default assumption is user script wants to run on the hardware system;
% this will be modified if we find Resource.System says otherwise
VDAS = 1; % we expect hardware system to be present and we want to use it
VDASupdates = 1; % create all 'VDAS' parameters needed for hw operation

Resource.System.LEsysExpected = 0; % assume not an LE system unless told otherwise

% The default values set in the following two lines mean the System structure
% either does not exist or has provided no guidance on what is expected.
% If the required values cannot be derived from other structures provided
% by the user script, VSX will report an error condition
Resource.System.AcqSlotsExpected = [1 1 1 1 1 1 1 1]; % VTS-1975 assume V-512 so we can run any 4.4 script and UTA
Resource.System.UTAexpected = [];

% Evaluate Resource.System 'Product' field
if isfield(Resource.System, 'Product') && ~isempty(Resource.System.Product)
    if strcmpi(Resource.System.Product, 'SimulateOnly')
        % user wants simulate only; Trans.connType will be used later
        % to determine which flavor of simulate-only
        VDAS = 0;
        VDASupdates = 0;
    elseif strcmpi(Resource.System.Product, 'SoftwareOnly')
        % user want to run in simulate mode ignoring the HW system, but
        % all VDAS HW constraints will still be applied
        VDAS = 0;
        VDASupdates = 1;
        Resource.System.AcqSlotsExpected = [1 1 1 1 1 1 1 1]; % VTS-1975 assume V-512 so we can run any 4.4 script and UTA
    elseif strfind(Resource.System.Product, '512')
        % user wants a Vantage 512 system
        Resource.System.AcqSlotsExpected = [1 1 1 1 1 1 1 1];
    elseif strfind(Resource.System.Product, '256')
        % user wants a Vantage 256 system
        Resource.System.AcqSlotsExpected = [1 1 1 1 0 0 0 0]; % V-256 configuration  % VTS-1600 8 AcqSlots
    elseif strfind(Resource.System.Product, '128')
        % user wants a Vantage 128 system
        Resource.System.AcqSlotsExpected = [1 0 0 1 0 0 0 0]; % V-128 configuration  % VTS-1600 8 AcqSlots
    elseif strfind(Resource.System.Product, '64')
        if strfind(Resource.System.Product, 'LE')
            % user wants a Vantage 64 LE system
            Resource.System.AcqSlotsExpected = [1 0 0 1 0 0 0 0]; % V-64 LE configuration  % VTS-1600 8 AcqSlots
            Resource.System.LEsysExpected = 1; % enable the LE receive channel constraint
        else
            % user wants a Vantage 64 system
            Resource.System.AcqSlotsExpected = [1 0 0 0 0 0 0 0]; % V-64 configuration  % VTS-1600 8 AcqSlots
        end
    elseif strfind(Resource.System.Product, '32') && strfind(Resource.System.Product, 'LE')
        % user wants a Vantage 32 LE system
        Resource.System.AcqSlotsExpected = [1 0 0 0 0 0 0 0]; % V-32 LE configuration  % VTS-1600 8 AcqSlots
        Resource.System.LEsysExpected = 1; % enable the LE receive channel constraint
    else
        % System.Product string is unrecognized
        error('VSX: Unrecognized product name string in Resource.System.Product.')
    end
end  % done evaluating the 'Product' field

% Evaluate Resource.System 'activeClamp' field
if isfield(Resource.System, 'activeClamp') && ~isempty(Resource.System.activeClamp) && Resource.System.activeClamp
    % user script expects active clamp configuration
    Resource.System.activeClampExpected = 1;
else
    % otherwise set default value of 0
    Resource.System.activeClampExpected = 0;
end

% Evaluate Resource.System 'Frequency' field or set default
if isfield(Resource.System, 'Frequency')
    % set expected value to match
    switch Resource.System.Frequency
        case {'LF', 'Low Frequency', 'Low'}
            Resource.System.FrequencyExpected = 'LF';
        case {'SF', 'Standard Frequency', 'Standard'}
            Resource.System.FrequencyExpected = 'SF';
        case {'HF', 'High Frequency', 'High'}
            Resource.System.FrequencyExpected = 'HF';
        case 'HIFU'
            Resource.System.FrequencyExpected = 'HIFU';
        otherwise
            Resource.System.FrequencyExpected = [];
            if Resource.Parameters.verbose > 1
                disp('VSX: unrecognized name string for Resource.System.Frequency.');
                disp('Default value will be set based on Trans.Frequency.');
            end
    end
end

% To avoid repeated use of "magic numbers" create a "freqRngRqd" field here
% based on the value of Trans.frequency in this scrpt
if Trans.frequency < 0.25
    % LF system is required for frequencies below 250 KHz
    freqRngRqd = 'LF';
elseif Trans.frequency < 1.7
    % LF or SF system can be used for frequencies in range 250 KHz to 1.7 MHz
    freqRngRqd = 'LFSF';
elseif Trans.frequency < 23
    % SF or HF system can be used for frequencies in range 1.7 to 23 MHz
    freqRngRqd = 'SFHF';
else
    % HF system required for frequencies above 23 MHz
    freqRngRqd = 'HF';
end

if ~isfield(Resource.System, 'Frequency') || isempty(Resource.System.FrequencyExpected)
    % user has not specified a preference, so set default based on
    % Trans.frequency using freqRngRqd as defined above
    switch freqRngRqd
        case 'LF'
            Resource.System.FrequencyExpected = 'LF';
        case 'HF'
            Resource.System.FrequencyExpected = 'HF';
        otherwise
            % for ambiguous ranges that allow SF, use SF as the default
            Resource.System.FrequencyExpected = 'SF';
    end
end

% Evaluate Resource.System 'UTA' field or set default
if isfield(Resource.System, 'UTA')
    if strfind(Resource.System.UTA, '260-S')
        Resource.System.UTAexpected = [1 1 1 0];
    elseif strfind(Resource.System.UTA, '260-D')
        Resource.System.UTAexpected = [1 1 2 0];
    elseif strfind(Resource.System.UTA, '260-M')
        Resource.System.UTAexpected = [1 1 1 2];
    elseif strfind(Resource.System.UTA, '360')
        Resource.System.UTAexpected = [1 3 1 0];
    elseif strfind(Resource.System.UTA, '408-GE')
        Resource.System.UTAexpected = [1 7 1 0];
    elseif strfind(Resource.System.UTA, '408')
        Resource.System.UTAexpected = [1 4 1 0];
    elseif strfind(Resource.System.UTA, '160-DH')
        % Note we will always expect connector type 12, but if the same
        % UTA with type 6 is present that is accepted too since they
        % are functionally interchangeable (VTS-792)
        Resource.System.UTAexpected = [1 12 3 1];
    elseif strfind(Resource.System.UTA, '160-SH')
        Resource.System.UTAexpected = [1 12 5 4];
    elseif strfind(Resource.System.UTA, '1024-M')
        Resource.System.UTAexpected = [1 8 1 3];
    elseif strfind(Resource.System.UTA, '256-D')
        Resource.System.UTAexpected = [1 8 1 0];
    elseif strfind(Resource.System.UTA, '156-U')
        Resource.System.UTAexpected = [1 9 1 0];
    elseif strfind(Resource.System.UTA, '160-SI')
        Resource.System.UTAexpected = [1 10 5 4];
    elseif strfind(Resource.System.UTA, '64-L')
        Resource.System.UTAexpected = [1 11 1 5];
    elseif strfind(Resource.System.UTA, '128-L')
        Resource.System.UTAexpected = [1 11 1 0];
    else
        error('VSX: unrecognized name string for Resource.System.UTA.');
    end

    % check to see if Trans structure is consistent
    if isfield(Trans, 'connType') && ~isempty(Trans.connType) && Trans.connType > 0
        % Trans structure has a physical connector type specified; make
        % sure it matches what user specified in Resource.System.UTA
        if (Trans.connType ~= Resource.System.UTAexpected(2)) && ...
                ~(Trans.connType == 6 && isequal(Resource.System.UTAexpected, [1 12 3 1])) % special case of older UTA 160-DH
            error('VSX: User script error- Resource.System.UTA does not match Trans.connType.');
        end
    elseif ~isfield(Trans, 'connType') || isempty(Trans.connType)
        % Script did not provide connType, so create it to match expected
        % UTA
        Trans.connType = Resource.System.UTAexpected(2);
    end
else
    % Resource.System has not specified a UTA, so attempt to derive it
    % from Trans.connType.
    % If Trans.connType undefined, create a default value
    if ~isfield(Trans, 'connType') || isempty(Trans.connType)
        % if possible, create an appropriate default value derived from the system structure
        if VDAS == 0 && VDASupdates == 0
            % User has indicated simulate only, so default connType to zero
            Trans.connType = 0; % forces simulate-only operation with no connector mapping
        else
            % System structure has provided no clues, so default to "old
            % faithful" UTA 260 HDI format connector (this is the only connector
            % available on pre-UTA systems, where the connType field did not
            % exist)
            Trans.connType = 1;
            if Resource.Parameters.verbose>1
                disp(' ');
                disp('VSX status: Trans.connType not specified; setting a default value of 1 for use with HDI-format connectors.');
                disp(' ');
            end
        end
    end
    if Trans.connType > 0
        % Trans.connType is not zero or -1
        % Determine how many Element Signals are indexed in
        % Trans.ConnectorES, to help determine which UTA is needed.
        if isfield(Trans, 'ConnectorES') && ~isempty(Trans.ConnectorES)
            maxCCH = max(Trans.ConnectorES);
        else
            maxCCH = Trans.numelements; % assign default if Trans.ConnectorES does not exist
        end
        switch Trans.connType
            case 1 % HDI format 260-pin connector
                if any(Resource.Parameters.Connector == 2)
                    % using connector 2 or both connectors, so we need dual
                    % connector UTA
                    Resource.System.UTAexpected = [1, 1, 2, 0]; % UTA 260-D
                else
                    % only using one connector; need to decide if this is
                    % Vantage 64 script requiring mux UTA or not
                    if isfield(Trans, 'HVMux') && isfield(Trans.HVMux, 'utaSF') && Trans.HVMux.utaSF == 2
                        % This is a Vantage 64 mux UTA script using the special
                        % feature 2 Trans.HVMux structure
                        Resource.System.UTAexpected = [1, 1, 1, 2]; % UTA 260-MUX
                    else
                        Resource.System.UTAexpected = [1, 1, 1, 0]; % UTA 260-S for all other configurations
                    end
                end
            case 2 % breakout board or custom adapter; only one UTA type exists
                Resource.System.UTAexpected = [1, 2, 1, 0];
            case 3 % Cannon 360 pin ZIF Connector; only one UTA type exists
                Resource.System.UTAexpected = [1, 3, 1, 0];
            case 4 % Verasonics 408 pin connector; only one UTA type exists
                % Note UTA type [1 4 4 0] is the adapter STE test fixture for
                % use with PVT, but PVT does not support simulation operation
                % so we don't allow for it here
                Resource.System.UTAexpected = [1, 4, 1, 0];
            case {6, 12} % Hypertac Connector; two UTA types exists, the
                % UTA 160-DH/32 LEMO and the UTA 160-SH/8 LEMO.  The
                % system cannot tell which one is expected from other
                % structures in the script, and thus in this case
                % Resource.System.UTA is required.  Its absence is an
                % error condition.
                % Note also that connType 6 and 12 both identify the
                % Hypertac connector; 6 provides disconnect sensing and
                % 12 does not, to avoid the spurious disconnect
                % problem. (See VTS-792)
                error('VSX: User script error- Resource.System.UTA must be specified to identify which Hypertac UTA module is being used.');
            case 7 % GE 408 pin connector; only one UTA type exists
                Resource.System.UTAexpected = [1, 7, 1, 0];
            case 8 % 1024 Mux direct connect adapters
                if isfield(Trans, 'HVMux') && isfield(Trans.HVMux, 'utaSF') && Trans.HVMux.utaSF == 3
                    % Script expects a UTA 1024-MUX script using the special
                    % feature 3 Trans.HVMux structure
                    Resource.System.UTAexpected = [1, 8, 1, 3];
                else
                    % Script expects UTA 256 Direct
                    Resource.System.UTAexpected = [1, 8, 1, 0];
                end
            case 9 % Ultrasonics connector
                Resource.System.UTAexpected = [1 9 1 0]; % UTA 156-U
            case 10
                Resource.System.UTAexpected = [1 10 5 4]; % UTA 160-SI
            case 11
                if maxCCH > 64
                    % UTA 128-LEMO is required for > 128 elements
                    Resource.System.UTAexpected = [1 11 1 0]; % UTA 128-LEMO
                else
                    % 64 or fewer connector channels so UTA 64-LEMO is
                    % expected but UTA-128 LEMO can also be used
                    Resource.System.UTAexpected = [1 11 1 5]; % UTA 64-LEMO
                end
            case 13
                % VTS-1600: V-512 SHI direct connect
                if isfield(Trans, 'HVMux') && isfield(Trans.HVMux, 'utaSF') && Trans.HVMux.utaSF == 6
                    % Script expects the V-512 SHI 2048-MUX configuration
                    % using the special feature 6 Trans.HVMux structure
                    Resource.System.UTAexpected = [1, 13, 1, 6];
                else
                    % Script expects V-512 SHI 512 direct
                    Resource.System.UTAexpected = [1, 13, 1, 0];
                end
            otherwise
                error('VSX: Unrecognized Trans.connType value of %d.', Trans.connType);
        end
    end
end % done evaluating the Resource.System.UTA field

%% *** 5. Determine system configuration & status
% call the hwConfigCheck function to determine SW configuration, and if
% hardware is going to be used also determine the actual system HW
% configuration and check it for validity:
if Resource.Parameters.verbose > 1
    hccMode = 1; % enable verbose messages from hwConfigCheck
else
    hccMode = 0; % disable verbose messages from hwConfigCheck
end

% tell hwConfigCheck whether we want to check HW configuration, based on
% VDAS variable set above.  If VDAS = 0, hwConfigCheck will only check
% software configuration and will not attempt to communicate with the
% hardware system.
Resource.SysConfig = hwConfigCheck(hccMode, VDAS); % verbose as defined above
clear hccMode

% If any configuration faults were found, display a warning message but
% ignore all configuration faults if HW test flag is set in the base
% workspace
if ~exist('hwTestFlag', 'var')
    if Resource.SysConfig.SWconfigFault
        % SW configuration fault means we cannot trust any HW or FPGA faults that
        % were detected, so ignore them and don't allow HW operation
        Resource.SysConfig.HWconfigFault = [];
        Resource.SysConfig.FPGAconfigFault = [];
        if Resource.Parameters.verbose
            fprintf(2, 'VSX WARNING: System software is not in a valid released configuration.\n');
            if VDAS
                % don't display this text if HW is not being used, making it
                % meaningless
                fprintf(2, 'In this condition, HW and FPGA configuration checks may yield incorrect\n');
                fprintf(2, 'results, so VSX will be restricted to Software-Only use.\n');
                fprintf(2, 'System can still be used with HW-SW diagnostic test and debug utilities.\n');
            end
            fprintf(2, 'Contact Verasonics Customer Support for additional information.\n');
            disp(' ');
        end
        Resource.SysConfig.VDAS = 0; % force to zero if HW was present, to prevent use of the HW
    elseif Resource.SysConfig.HWconfigFault == 6
        % An illegal UTA configuration has been detected.  Report illegal
        % configuration to the user and exit with an error
        fprintf(2, 'UTA CONFIGURATION ERROR:\n');
        fprintf(2, ['    The ', Resource.SysConfig.UTAname, ...
            ' adapter module can not be used on a ', Resource.SysConfig.System, ' System.\n']);
        fprintf(2, '    You must install a UTA module that is compatible with this system.\n');
        error('VSX: Closing VSX.');
    elseif Resource.SysConfig.HWconfigFault > 1
        % HW is present but we can't communicate with it so just run in
        % simulation only mode
        Resource.SysConfig.FPGAconfigFault = []; % ignore any FPGA status information
        Resource.SysConfig.VDAS = 0; % force simulation only
        if Resource.Parameters.verbose
            fprintf(2, 'VSX WARNING: A HW system has been detected, but system SW is unable\n');
            fprintf(2, 'to communicate with it.  VSX will be restricted to SW simulation use only.\n');
            if Resource.SysConfig.HWconfigFault == 2
                fprintf(2, 'HW system is in Recovery Mode.\n');
            elseif Resource.SysConfig.HWconfigFault == 3
                fprintf(2, 'HW system is present, but in a fault condition.\n');
                fprintf(2, 'Try running VVT to help identify the problem.\n');
            elseif Resource.SysConfig.HWconfigFault == 4
                fprintf(2, 'UTA adapter module is not installed or is in a fault condition.\n');
            elseif Resource.SysConfig.HWconfigFault == 5
                fprintf(2, 'HW system overheated, or fans not functioning.\n');
            elseif Resource.SysConfig.HWconfigFault == 6
                fprintf(2, 'The UTA adapter module that is present cannot be used with this system.\n');
            end
            if Resource.SysConfig.HWconfigFault == 7
                fprintf(2, 'This is an External Clock Required system, with no external clock source detected.\n');
            else
                % Unlike the other faults, diagnostics cannot be used for
                % no external clock on an external clock system
                fprintf(2, 'System can still be used with HW-SW diagnostic test and debug utilities.\n');
                fprintf(2, 'Contact Verasonics Customer Support for additional information.\n');
            end
            disp(' ');
        end
    elseif Resource.SysConfig.HWconfigFault == 1
        % HW is present but in unrecognized configuration
        if Resource.Parameters.verbose
            fprintf(2, 'VSX WARNING: A HW system has been detected, but it is an unrecognized\n');
            fprintf(2, 'system HW configuration.  VSX will be allowed to continue, but\n');
            fprintf(2, 'user scripts may not function properly.\n');
            fprintf(2, 'System can still be used with HW-SW diagnostic test and debug utilities.\n');
            fprintf(2, 'Contact Verasonics Customer Support for additional information.\n');
            disp(' ');
        end
    end

    if Resource.SysConfig.FPGAconfigFault
        fprintf(2, 'VSX WARNING: An out-of-date or unrecognized version of FPGA code\n');
        fprintf(2, 'has been found in the HW system.\n');
        disp('Enter "F" or just a carriage return to check and reprogram');
        disp(' all FPGA flash memories with the released code, (which may then');
        disp(' require a full power shutdown cycle before the system can be used);');
        reply = input('or enter a "Q" to exit VSX without reprogramming: ', 's');
        if strcmpi(reply, 'Q')
            % Quit VSX without doing anything
            return
        elseif strcmp(reply, 'proceed')
            % allow VSX to continue with existing FPGA code
        else
            % any other response means start FPGA code reprogramming using
            % hwConfigCheck mode 2
            [~] = hwConfigCheck(2);
            % Note that if hwConfigCheck actually reprograms the CGD or
            % ASC, it will force an exit with an error message informing
            % the user they must shutdown with a power cycle to
            % reinitialize PCIE devices.  Otherwise, it will return
            % normally and we can proceed, but to ensure that the
            % reprogrammed versions match the expected, quit VSX to force a
            % re-run.
            disp('FPGA programming completed without the need to reboot, please re-run VSX.');
            return
        end
    end
end

%% *** 6. Check configuration required by script versus actual system

% if a required minimum Vantage software version has been specified in the
% System structure, check and verify that now
if isfield(Resource.System, 'SoftwareVersion')
    reqSwv = Resource.System.SoftwareVersion;
    if ischar(reqSwv) || ~isequal(size(reqSwv), [1 3])
        error(['VSX: Resource.System.SoftwareVersion must be specified as a 1 X 3 array of integers', ...
          '(for example, set it to [ 3 5 0 ] and not the string value ''3.5.0'').']);
    end
    sysSwv = Resource.SysConfig.SWversion;
    if reqSwv(1) > sysSwv(1) || (reqSwv(1) == sysSwv(1) && ...
            (reqSwv(2) > sysSwv(2) || (reqSwv(2) == sysSwv(2) && reqSwv(3) > sysSwv(3))))
        error('VSX: System software version is %u.%u.%u, but the script requires %u.%u.%u.', ...
            sysSwv(1), sysSwv(2), sysSwv(3), reqSwv(1), reqSwv(2), reqSwv(3));
    end
end

% Replace 'wild card' Trans.connType with actual UTA value if HW is present
if Trans.connType == -1
    if Resource.SysConfig.VDAS
        % HW is present, so set connType to match the actual UTA
        Trans.connType = Resource.SysConfig.UTAtype(2);
        if isempty(Resource.System.UTAexpected)
            Resource.System.UTAexpected = Resource.SysConfig.UTAtype;
        end
    elseif ~isempty(Resource.System.UTAexpected)
        % Resource.System has specified a UTA, so use it to set connType
        Trans.connType = Resource.System.UTAexpected(2);
    else
        % no HW and no clues from System structure, so set connType to 1
        % (default to HDI connector for HW simulation)
        Trans.connType = 1;
        Resource.System.UTAexpected = [1 1 1 0]; % 260-S
    end
end

% If hardware operation was expected, check to see if it is available
if VDAS && Resource.SysConfig.VDAS == 0
    % either hardware is not available or a fault condition exists that
    % will prevent us from using the hardware.  Switch to software-only
    % operation and notify user if verbose > 1
    VDAS = 0;
    Resource.SysConfig.UTAname = 'Modified for simulation operation'; % we will be overwriting it with UTA the script wants
    if Resource.Parameters.verbose > 1
        disp('VSX status: Hardware system is not available; switching to Software-Only operation.');
    end
end

if isfield(Resource, 'SimulationDefaults')
    if ~VDAS
        disp(' ');
        fprintf(2, 'VSX error: This script includes a Resource.SimulationDefaults structure,\n');
        fprintf(2, '    which is no longer supported by the Vantage software.  You must replace it\n');
        fprintf(2, '    with a Resource.System structure providing the equivalent information\n');
        fprintf(2, '    in order to run this script properly in simulate-only modes.\n');
        disp(' ');
        error('Exiting the script due to incompatible structure listed above.');
    elseif Resource.Parameters.verbose > 1
        disp(' ');
        disp('VSX status: This script includes a Resource.SimulationDefaults structure,');
        disp('    which is no longer supported by the Vantage software.  Please replace it ');
        disp('    with a Resource.System structure providing the equivalent information.');
        disp(' ');
    end
end

if VDAS == 0 && Trans.connType > 0
    % for Software-Only operation, set switchToSim flag to trigger
    % modifying Resource.SysConfig to match the expected hardware
    % configuration, in block after this if-else-end
    switchToSim = 1;
elseif VDAS
    % script intends to use HW, check for UTA
    % compatibility and define UTA-specific mapping arrays.
    switchToSim = 0; % flag to indicate whether we need to switch to simulate mode, based on HW confiugration checks
    numBoards = nnz(Resource.SysConfig.AcqSlots); % number of boards actually present, regardless of where they are
    % At this point we know script intends to use HW and HW is present, so
    % need to check whether UTA required by script matches UTA that is
    % actually present, and if so whether system configuration matches what
    % is required by script and UTA

    % (VTS-792): For the UTA 160-DH/32 Lemo, both connector types 6 and 12
    % are functionally interchangeable so eliminate the error condition if
    % Trans and SysConfig.UTAtype are calling out connector types 6 and 12
    % or 12 and 6
    if isequal(Resource.SysConfig.UTAtype, [1 6 3 1]) && Trans.connType == 12
        Trans.connType = 6;
    elseif isequal(Resource.SysConfig.UTAtype, [1 12 3 1]) && Trans.connType == 6
        Trans.connType = 12;
    elseif Trans.connType == 11 && ~isempty(Resource.System.UTAexpected) ...
            && Resource.System.UTAexpected(4) == 5 ...
            && isequal(Resource.SysConfig.UTAtype, [1 11 1 0])
        % for the Lemo-only UTA's, if the system is expecting a 64-LEMO
        % adapter but 128-LEMO is present, change UTAexpected to match
        Resource.System.UTAexpected(4) = 0;
    end

    % Call computeUTA to determine required numCH and activeCG; use UTA
    % that is present if it appears to be compatible with UTA required by
    % script, otherwise use UTA expected by Resource.System
    if Resource.SysConfig.UTAtype(2) ~= Trans.connType ...
            || max(Resource.Parameters.Connector) > Resource.SysConfig.UTAtype(3) ...
            || ~isempty(Resource.System.UTAexpected) && (Resource.SysConfig.UTAtype(4) ~= Resource.System.UTAexpected(4))
        % UTA in system is incomapatible with script; use UTA expected by
        % script
        UTAtype = Resource.System.UTAexpected;
    else
        % looks like we can use the UTA that is present
        UTAtype = Resource.SysConfig.UTAtype;
    end
    UTA = computeUTA(UTAtype, Resource.Parameters.Connector);
    if isequal(UTAtype, [1 2 1 0])
        % special case of breakout board with no connector mapping; number
        % of active channels will be defined here to match physical number
        % of acquisition modules in the system.  This also applies when
        % Resource.VDAS.el2ChMapDisable is true; in this case the UTA numCh
        % and activeCG fields will be overwritten to enable all channels on
        % all boards
        UTA.numCh = tchPerBd * numBoards;
        UTA.activeCG = upsample(Resource.SysConfig.AcqSlots, 2) + ...
            upsample(Resource.SysConfig.AcqSlots, 2, 1);
    end
    % find the number of boards in use, and determine if compatible with HW
    % system configuration VTS-1600 expand to 8 slots
    activeBoards = min((UTA.activeCG([1 3 5 7 9 11 13 15]) + UTA.activeCG([2 4 6 8 10 12 14 16])), 1);
    
    % Check whether HW system frequency range is compatible with script
    % requirements;  report verbose level 2 warning message if not.
    if Resource.Parameters.verbose > 1
        switch freqRngRqd
            case 'LF'
                % must use LF system
                if ~strcmp(Resource.SysConfig.Frequency, 'LF')
                    disp('VSX warning: Low Frequency system is required for adequate performance with this script.');
                end
            case 'LFSF'
                % cannot use HF system
                if strcmp(Resource.SysConfig.Frequency, 'HF')
                    disp('VSX warning: High Frequency system will not provide adequate performance with this script.');
                end
            case 'SFHF'
                % cannot use LF system
                if strcmp(Resource.SysConfig.Frequency, 'LF')
                    disp('VSX warning: Low Frequency system will not provide adequate performance with this script.');
                end
        end
    end
    
    % For Trans.frequency > 23 MHz, do not allow use with LF or SF system
    % configurations.
    if strcmp(freqRngRqd, 'HF') && ~strcmp(Resource.SysConfig.Frequency, 'HF')
        if Resource.Parameters.verbose
            fprintf(2, 'VSX warning: Hardware system frequency range does not match HF range required by script.\n');
            fprintf(2, 'Enter s to continue in simulation-only mode, or just enter return to exit:  ');
            r = input(' ', 's');
            if strcmpi(r, 's')
                switchToSim = 1; % switch to simulate mode & redefine SysConfig in code block below
            else
                % user wants to quit
                return
            end
        else
            % verbose is zero; don't query for user input- just exit
            % with an error message
            error('VSX: Hardware system frequency range does not match HF range required by script.\n')
        end
    end
    
    
    % Check whether HW UTA and number of acq boards matches script
    % requirements; report error or switch to simulate mode if not.
    if (Resource.SysConfig.UTAtype(2) ~= Trans.connType ...
            || max(Resource.Parameters.Connector) > Resource.SysConfig.UTAtype(3) ...
            || ~isempty(Resource.System.UTAexpected) && (Resource.SysConfig.UTAtype(4) ~= Resource.System.UTAexpected(4)) ...
            || min(Resource.SysConfig.AcqSlots - activeBoards) < 0) ...
            && (~isfield(Resource, 'VDAS') || ~isfield(Resource.VDAS, 'el2ChMapDisable') || Resource.VDAS.el2ChMapDisable == 0)
        % UTA and system configuration do not match script requirements;
        % script cannot be run on the HW system so either quit or proceed
        % in softwate-only mode
        if Resource.Parameters.verbose
            fprintf(2, 'VSX warning: System configuration and UTA module do not match script requirements.\n');
            fprintf(2, 'Enter s to continue in simulation-only mode, or just enter return to exit:  ');
            r = input(' ', 's');
            if strcmpi(r, 's')
                switchToSim = 1; % switch to simulate mode & redefine SysConfig in code block below
            else
                % user wants to quit
                return
            end
        else
            % verbose is zero; don't query for user input- just exit
            % with an error message
            error('VSX: Hardware system UTA module does not match script requirements.\n')
        end
    end
    clear activeBoards
end

if switchToSim
    % HW system is incompatible and user has elected to switch to
    % simulation mode; first check to see if user requested simulate mode 2
    % and honor that (VTS-2467)
    if ~isfield(Resource.Parameters,'simulateMode') || isempty(Resource.Parameters.simulateMode) || Resource.Parameters.simulateMode < 2
        % user did not select mode 2, so force mode 1
        Resource.Parameters.simulateMode = 0;
    end
    VDAS = 0;
    % For simulate mode independent of HW, we need to
    % explicitly set appropriate UTA type based on expected
    % UTA from the System structure as well as other aspects of system
    % configuration
    Resource.SysConfig.UTA = 1;
    Resource.SysConfig.UTAtype = Resource.System.UTAexpected;
    Resource.SysConfig.UTAname = 'Modified for simulation operation';
    Resource.SysConfig.AcqSlots = Resource.System.AcqSlotsExpected;
    numBoards = nnz(Resource.SysConfig.AcqSlots);
    Resource.SysConfig.LEsys = Resource.System.LEsysExpected;
    Resource.System.activeClamp = Resource.System.activeClampExpected;
    Resource.SysConfig.Frequency = Resource.System.FrequencyExpected;
    if Resource.System.activeClamp
        % set TX, RX index values for active clamp system
        switch Resource.System.FrequencyExpected
            case 'LF'
                Resource.SysConfig.TXindex = 7;
                Resource.SysConfig.RXindex = 6; % production AFE5812
            case 'SF'
                Resource.SysConfig.TXindex = 5;
                Resource.SysConfig.RXindex = 6; % production AFE5812
            case 'HF'
                Resource.SysConfig.TXindex = 6;
                Resource.SysConfig.RXindex = 6; % production AFE5812
            case 'HIFU'
                Resource.SysConfig.TXindex = 5;
                Resource.SysConfig.RXindex = 6; % production AFE5812
        end
    else
        % set TX, RX index values for non-active clamp system
        switch Resource.System.FrequencyExpected
            case 'LF'
                Resource.SysConfig.TXindex = 3;
                Resource.SysConfig.RXindex = 5; % production AFE5812
            case 'SF'
                Resource.SysConfig.TXindex = 1;
                Resource.SysConfig.RXindex = 5; % production AFE5812
            case 'HF'
                Resource.SysConfig.TXindex = 4;
                Resource.SysConfig.RXindex = 5; % production AFE5812
            case 'HIFU'
                Resource.SysConfig.TXindex = 1;
                Resource.SysConfig.RXindex = 5; % production AFE5812
        end
    end
end

% Add TPC structures not specified by the user.
if ~exist('TPC','var'), TPC = []; end
for i = length(TPC)+1:5
    TPC(i).inUse = [];
end
clear i

% - Determine state of the variables used to control system operating limits.
% Check state of HW configuration options for LE system configurations, and
% use of TPC profile 5: use 'p5req' flag from hw as-is (note that the p5req
% value has already been updated by hwConfigCheck to reflect license file
% status).
LEsys = Resource.SysConfig.LEsys;
p5ena = Resource.SysConfig.p5req;



% Find which profiles are in use.
if exist('SeqControl','var')
    if isempty(SeqControl)
        error('VSX: SeqControl structure is empty.');
    elseif length(SeqControl) > 1 && length(SeqControl(:)) == size(SeqControl, 1)
        % was defined as a column vector; convert to row and warn user if verbose
        if Resource.Parameters.verbose
            fprintf(2, 'VSX warning: SeqControl structure array was defined as a column vector;\n');
            fprintf(2, '     converting to required row vector format.\n');
        end
        SeqControl = SeqControl';
    elseif length(SeqControl(:)) > 1 && length(SeqControl(:)) ~= size(SeqControl, 2)
        error('VSX: Multidimensional SeqControl structure array not supported.  It must be a row vector.');
    end
    for j=1:size(SeqControl,2)
        if strcmp(SeqControl(j).command,'setTPCProfile')
            TPC(SeqControl(j).argument).inUse = 1;
        end
    end
    clear j
else
    SeqControl = [];
end
% Profile 1 is active by default, if none other active.
if ~any([TPC.inUse])
    TPC(1).inUse = 1;
end

% Set TPC structure default values.
for i = 1:size(TPC,2)
    if TPC(i).inUse
        if ~isfield(TPC(i), 'hv') || isempty(TPC(i).hv)
            TPC(i).hv = minTpcVoltage; % Default system setting for hv at startup.
        end
        if ~isfield(TPC(i), 'maxHighVoltage') || isempty(TPC(i).maxHighVoltage)
            TPC(i).maxHighVoltage = Trans.maxHighVoltage;
        else
            if TPC(i).maxHighVoltage > Trans.maxHighVoltage
                if Resource.Parameters.verbose
                    fprintf(2,'VSX WARNING: TPC(%d).maxHighVoltage reduced to Trans.maxHighVoltage limit of %.1f volts.\n',i,Trans.maxHighVoltage);
                end
                TPC(i).maxHighVoltage = Trans.maxHighVoltage;
            end
        end
        % Always initialize highVoltageLimit to match maxHighVoltage.
        TPC(i).highVoltageLimit = TPC(i).maxHighVoltage;
    else
        TPC(i).inUse = 0;
        TPC(i).hv = minTpcVoltage;
        TPC(i).maxHighVoltage = minTpcVoltage;
        TPC(i).highVoltageLimit = minTpcVoltage;
    end
end
clear i

% VTS-1583 If TPC(5) is active, initialize monitor limits to the disabled
% state, and set both min and max limits to zero
if TPC(5).inUse
    TPC(5).HVmonitorEnabled = 0;
    TPC(5).HVmonitorLimits = [0 0];
end

if usingHIFUPlex
    hv2GUIprofile = HIFUParam.Volt;
end


%% *** 7. Check Operating Mode and set default if needed
% - The variable VDAS is used to indicate presence of hardware.
% - the variable VDASupdates is used to control matlab generation of VDAS variables.
% - When not in simulateMode = 1 (simulate acquisition data), the system
% will automatically switch to simulate if HW is not available:

if ~isfield(Resource.Parameters,'simulateMode') || isempty(Resource.Parameters.simulateMode)
    if VDAS
        Resource.Parameters.simulateMode = 0; % default to running with HW if system expects to use it
    else
        Resource.Parameters.simulateMode = 1; % default to simulation-only if HW will not be used.
    end
    if Resource.Parameters.verbose > 1
        disp('VSX status: Resource.Parameters.simulateMode was not defined.');
        disp(['Setting a default value of ', num2str(Resource.Parameters.simulateMode), '.']);
    end
elseif VDAS == 0 && Resource.Parameters.simulateMode == 0
    % Script will be running in simulate-only operation so override
    % selection of hardware operation (but note simulateMode 2 is valid
    % since it can be used with simulate-only scripts)
    Resource.Parameters.simulateMode = 1;
    if Resource.Parameters.verbose > 2
        disp(' ');
        disp('VSX status: Resource.Parameters.simulateMode is being set to 1 for');
        disp('  simulation-only operation.');
    end
end

% check simulateMode value
switch Resource.Parameters.simulateMode
    case 0 % user wants to use hardware.
        rloopButton = 0;  % used to set state of rcv data loop button on GUI.
        if VDAS
            % HW system is actually present, so get ready to use it
            simButton = 0;    % used to set state of simulate button on GUI.
        else
            % HW is not available so revert to simulate mode
            simButton = 1;    % used to set state of simulate button on GUI.
            Resource.Parameters.simulateMode = 1;
        end
    case 1 % user wants to simulate acquisition
        rloopButton = 0;
        simButton = 1;
    case 2 % user wants to run script with existing RcvData array.
        rloopButton = 1;
        simButton = 0;
    otherwise
        error('VSX: unrecognized value for Resource.Parameters.simulateMode.');
end


%% *** 8. Load external processing function(s)
% If .mat file contains an External Function definition, create the
% function in the tempDir directory. Decoding the function here means
% that it can be used for a VsUpdate or GUI function.
if exist('EF','var')
     exFncManager = vsv.seq.function.ExFunctionManager(vsTempDir);
     exFncManager.createTemporaryFunctions(EF);
end


%% *** 9. Resource.VDAS initialize
if VDASupdates
    if ~isfield(Resource,'VDAS')
        Resource.VDAS = struct();
    end
    if ~isfield(Resource.VDAS,'numTransmit')
        Resource.VDAS.numTransmit = tchPerBd * numBoards;
    end
    if ~isfield(Resource.VDAS,'numRcvChannels')
        Resource.VDAS.numRcvChannels = rchPerBd * numBoards;
    end
    if ~isfield(Resource.VDAS,'exportDelta')
        Resource.VDAS.exportDelta = 2048;
    end
    if ~isfield(Resource.VDAS,'testPattern')
        Resource.VDAS.testPattern = 0;
    end
    if ~isfield(Resource.VDAS,'testPatternDma')
        Resource.VDAS.testPatternDma = 0;
    end
    if ~isfield(Resource.VDAS,'dmaPrecompute')
        Resource.VDAS.dmaPrecompute = 1;
    end
    if ~isfield(Resource.VDAS,'dmaComputeFree')
        Resource.VDAS.dmaComputeFree = 0;
    end
    if ~isfield(Resource.VDAS,'dmaTimeout')
        Resource.VDAS.dmaTimeout = 1000;
    end
    if ~isfield(Resource.VDAS,'watchdogTimeout') || isempty(Resource.VDAS.watchdogTimeout)
        % watchdog timeout interval in milliseconds; allowed range 10:10000
        % (10 msec to 10 sec)
        Resource.VDAS.watchdogTimeout = 10000; % 10 second default value
    end
    if ~isfield(Resource.VDAS,'el2ChMapDisable')
        Resource.VDAS.el2ChMapDisable = 0;
    end
    if ~isfield(Resource.VDAS,'sysClk')
        Resource.VDAS.sysClk = sysClk;
    end
    if ~isfield(Resource.VDAS,'elBiasSel')
        Resource.VDAS.elBiasSel = 0; % default state is element bias disabled
    end
end

% Create the hardware event handler which will generate messages when a
% warning or error occurs.
hwEventHandler = com.verasonics.vantage.events.VantageHwEventHandler;
Faults.clearFaults;
hwEventHandler.installHandler();


%% *** 10. Create channel mapping arrays based on UTA type
% At this point we have already confirmed System.UTAexpected is compatible
% with the script being run; now we check if it is compatible with the
% system HW configuration, and if so create the mapping arrays to memory
% columns and VDAS channels.  For each recognized UTA type, determine if
% HW configuration is compatible and exit with error if not.
% Set VDAS to indicate presence/ absence of HW (for simulation, V128 & V256
% will be changed if needed to match UTA configuration)

if Trans.connType
    % don't call compute UTA if Trans.connType is zero
    UTA = computeUTA(Resource.SysConfig.UTAtype, Resource.Parameters.Connector);
end
if ~VDASupdates
    % simulation-only script or an unrecognized UTA, so create required
    % size of receive buffer and system channels based on Trans.numelements
    % with fixed 1:1 mapping (VTS-808)
    Resource.Parameters.numTransmit = Trans.numelements;
else
    if isequal(Resource.SysConfig.UTAtype, [1 2 1 0]) || Resource.VDAS.el2ChMapDisable
        % special case of breakout board with no connector mapping; number
        % of active channels will be defined here to match physical number
        % of acquisition modules in the system.  This also applies when
        % Resource.VDAS.el2ChMapDisable is true; in this case the UTA numCh
        % and activeCG fields will be overwritten to enable all channels on
        % all boards
        UTA.numCh = tchPerBd * numBoards;
        UTA.activeCG = upsample(Resource.SysConfig.AcqSlots, 2) + ...
            upsample(Resource.SysConfig.AcqSlots, 2, 1);
    end
    % find the number of boards in use, and determine if compatible with HW
    % system configuration
    % VTS-1600 8 AcqSlots
    activeBoards = min((UTA.activeCG([1 3 5 7 9 11 13 15]) + UTA.activeCG([2 4 6 8 10 12 14 16])), 1);
    if min(Resource.SysConfig.AcqSlots - activeBoards) < 0
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Failed to create Software-Only simulation configuration.');
    end
    % build the cgEnaDma and Cch2Vch arrays
    % VTS-1600 we now support up to 16 CG's on 8 boards
    Resource.VDAS.cgEnaDma = sum(UTA.activeCG .* 2.^(0:15));
    Resource.VDAS.Cch2Vch = [];
    % build HWactiveCG based on boards actually in HW system:
    HWactiveCG = [];
    for i=1:8 % VTS-1600 8 Acq Slots
        if Resource.SysConfig.AcqSlots(i)
            HWactiveCG = [HWactiveCG, UTA.activeCG((2*i-1):(2*i))];
        end
    end
    activeCGnums = find(HWactiveCG);
    for i=1:length(activeCGnums)
        Resource.VDAS.Cch2Vch = [Resource.VDAS.Cch2Vch, (32*(activeCGnums(i)-1) + (1:32))];
    end
    Resource.Parameters.numTransmit = UTA.numCh;
    clear i activeBoards activeCGnums HWactiveCG
end


clear rchPerBd tchPerBd


%% *** 11. Check Resource.Parameters for valid attributes and initialize

% Check for VsUpdate function handle definition, and create default if needed
if ~isfield(Resource.Parameters,'UpdateFunction')||isempty(Resource.Parameters.UpdateFunction)
    Resource.Parameters.UpdateFunction = 'VsUpdate';
end
updateh = str2func(Resource.Parameters.UpdateFunction); % create handle to update function.

if VDASupdates
    if ~isfield(Resource.Parameters,'ProbeConnectorLED') || ...
            isempty(Resource.Parameters.ProbeConnectorLED)
        defaultProbeConnectorLED = [1 1 1 1];
        if isfield(Resource.VDAS,'shiConnectorLights')
            if ~isempty(Resource.VDAS.shiConnectorLights)
                Resource.Parameters.ProbeConnectorLED = zeros(1, 4);
                for i=1:length(Resource.Parameters.ProbeConnectorLED)
                    Resource.Parameters.ProbeConnectorLED(i) = ...
                        bitget(Resource.VDAS.shiConnectorLights, i);
                end
                clear i
            else
                Resource.Parameters.ProbeConnectorLED = defaultProbeConnectorLED;
            end
            % Issue warning for use of deprecated Resource.VDAS.shiConnectorLights
            warning(['Resource.VDAS.shiConnectorLights is deprecated.\n' ...
                     'Use Resource.Parameters.ProbeConnectorLED = ' ...
                     '[%d, %d, %d, %d]\n'], Resource.Parameters.ProbeConnectorLED(:))
        else
            Resource.Parameters.ProbeConnectorLED = defaultProbeConnectorLED;
        end
        clear defaultProbeConnectorLED
    end
    if ~isfield(Resource.Parameters,'ProbeThermistor') || ...
            isempty(Resource.Parameters.ProbeThermistor)
        Resource.Parameters.ProbeThermistor = ...
            repmat(struct('enable', 0, ...
                          'threshold', 0, ...
                          'reportOverThreshold', 0), 1, 2);
        if isfield(Resource.VDAS,'shiThermistors')
            if ~all(size(Resource.VDAS.shiThermistors) == [3, 2])
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('Resource.VDAS.shiThermistors must be 3 x 2');
            end
            for i = 1:length(Resource.Parameters.ProbeThermistor)
                Resource.Parameters.ProbeThermistor(i).enable = ...
                    Resource.VDAS.shiThermistors(1, i);
                Resource.Parameters.ProbeThermistor(i).threshold = ...
                    Resource.VDAS.shiThermistors(2, i);
                Resource.Parameters.ProbeThermistor(i).reportOverThreshold = ...
                    Resource.VDAS.shiThermistors(3, i);
            end
            clear i
            warning(['Resource.VDAS.shiThermistors is deprecated. ' ...
                     'Resource.Parameters.ProbeThermistor has been populated ' ...
                     'from Resource.VDAS.shiThermistors. Inspect the ' ...
                     'Resource.Parameters.ProbeThermistor structure and ' ...
                     'use those values in your script.'])
        end
    end
    if ~isfield(Resource.Parameters,'SystemLED') || ...
            isempty(Resource.Parameters.SystemLED)
        defaultSystemLED = {'running', 'paused', 'activeTxAndOrRx', 'transferToHostComplete'};
        if isfield(Resource.VDAS, 'shiLeds')
            if ~isempty(Resource.VDAS.shiLeds)
                Resource.Parameters.SystemLED = cell(1, 4);
                for i = 1:length(Resource.Parameters.SystemLED)
                    shiLed = bitand(bitshift(Resource.VDAS.shiLeds, -8*(i-1)), hex2dec('FF'));
                    switch shiLed
                        case hex2dec('00')
                            Resource.Parameters.SystemLED{i} = 'off';
                        case hex2dec('0D')
                            Resource.Parameters.SystemLED{i} = 'running';
                        case hex2dec('13')
                            Resource.Parameters.SystemLED{i} = 'starting';
                        case hex2dec('01')
                            Resource.Parameters.SystemLED{i} = 'paused';
                        case hex2dec('02')
                            Resource.Parameters.SystemLED{i} = 'activeTxAndOrRx';
                        case hex2dec('12')
                            Resource.Parameters.SystemLED{i} = 'activeTxAndOrRxProfile5';
                        case hex2dec('0B')
                            Resource.Parameters.SystemLED{i} = 'transferToHostComplete';
                        case hex2dec('03')
                            Resource.Parameters.SystemLED{i} = 'pausedOnTriggerIn';
                        case hex2dec('20')
                            Resource.Parameters.SystemLED{i} = 'pausedDmaWaitPrevious';
                        case hex2dec('21')
                            Resource.Parameters.SystemLED{i} = 'pausedDmaWaitForProcessing';
                        case hex2dec('11')
                            Resource.Parameters.SystemLED{i} = 'pausedOnMultiSysSync';
                        case hex2dec('22')
                            Resource.Parameters.SystemLED{i} = 'pausedSync';
                        case hex2dec('10')
                            Resource.Parameters.SystemLED{i} = 'externalBoxFault';
                        case hex2dec('04')
                            Resource.Parameters.SystemLED{i} = 'missedTimeToNextAcq';
                        case hex2dec('07')
                            Resource.Parameters.SystemLED{i} = 'missedTimeToNextEB';
                        otherwise
                            % Close the hardware event handler.
                            hwEventHandler.uninstallHandler();
                            error('Unrecognized shiLed value %d.\n', shiLed);
                    end
                end
                clear i
            else
                Resource.Parameters.SystemLED = defaultSystemLED;
            end
            % Issue warning for use of deprecated Resource.VDAS.shiLeds.
            warning(['Resource.VDAS.shiLeds is deprecated.\n', ...
                     'Use Resource.Parameters.SystemLED = ' ...
                     '[''%s'', ''%s'', ''%s'', ''%s'']\n'], ...
                    Resource.Parameters.SystemLED{:})
        else
            Resource.Parameters.SystemLED = defaultSystemLED;
        end
        clear defaultSystemLED shiLed
    end
end

% - If VDASupdates are requested, validate Receive Buffer definition per
% system constraints
if VDASupdates
    if isfield(Resource, 'RcvBuffer')
        % There must be either 1 or an even number of frames in RcvBuffer
        for i=1:size(Resource.RcvBuffer,2)
            if Resource.RcvBuffer(i).numFrames ~= 1 && ...
                    mod(Resource.RcvBuffer(i).numFrames, 2)
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error(['VSX: Resource.RcvBuffer(%d).numFrames ' ...
                       'must be 1 or an even integer.'], i);
            end
        end
        clear i
    end
end
if isfield(Resource, 'RcvBuffer')
    if ~usingMultiSys %multisys has modes where there are more cols than receive chs
        % Now make column size correction to all RcvBuffers, but don't need to
        % repeat warnings. Also determine total defined RcvBuffer size and
        % report to user at verbose level 3
        rBufSize = 0;
        for i=1:size(Resource.RcvBuffer,2)
            Resource.RcvBuffer(i).colsPerFrame = Resource.Parameters.numTransmit;
            rBufSize = rBufSize + Resource.RcvBuffer(i).colsPerFrame * ...
                Resource.RcvBuffer(i).rowsPerFrame * Resource.RcvBuffer(i).numFrames;
        end
        if Resource.Parameters.verbose > 2
            disp(['VSX Debug: Total allocated Receive Buffer memory space is ' num2str(rBufSize/2^19) ' Megabytes.']);
        end
        clear rBufSize
    end
end

% Determine whether to reuse RcvData buffer
% If RcvData already exists in workspace, check for same dimensions
% as specified in Resource structure. If dimensions not equal, clear the RcvData.
if exist('RcvData','var')
    clrRcvData = 0;  % set default to not clear RcvData
    if size(RcvData,1) ~= size(Resource.RcvBuffer,2)
        clrRcvData = 1;
    else
        for i = 1:size(RcvData,1)
            if size(RcvData{i},1) ~= Resource.RcvBuffer(i).rowsPerFrame
                clrRcvData = 1;
            end
            if size(RcvData{i},2) ~= Resource.RcvBuffer(i).colsPerFrame
                clrRcvData = 1;
            end
            if size(RcvData{i},3) ~= Resource.RcvBuffer(i).numFrames
                clrRcvData = 1;
            end
            if isfield(Resource.RcvBuffer,'datatype')
                if ~strcmp(class(RcvData{i}),Resource.RcvBuffer(i).datatype)
                    fprintf('RcvData found in workplace, but datatype is not ''int16'' - clearing.\n');
                    clrRcvData = 1;
                end
            elseif ~isa(RcvData{i},'int16')
                fprintf('RcvData found in workplace, but datatype is not ''int16'' - clearing.\n');
                clrRcvData = 1;
            end
        end
    end
    if (clrRcvData == 0)
        fprintf('RcvData in workplace matches Resource.RcvBuffer specification - reusing without clearing.\n');
    else
        if Resource.Parameters.simulateMode==2
            fprintf(2,'SimulateMode 2 specified, but RcvData in workspace doesn''t match definition in Resource.RcvBuffer.\n');
        end
        clear('RcvData');
    end
    clear i clrRcvData
end

% - Check for the number of LogData records specified.  These records are use for debugging
%   purposes and are optionally generated by the mex file runAcq.  Default number is 128 records.
%   Meaning of fields in a 4 uint LogData record:
%     int32    id        This number is an identifier for the record and is
%                         the same as 'id' in LOGTIME(id).
%     int32    datatype  Identifier for type of data (0 = time, 1 = int, 2 = double);
%     int32     data1     For a time record, this field is the hi value of
%                         an AbsoluteTime number; for data record, it is
%                         the an integer value (for datatype=1) or the integer
%                         representation of a double (hi 16 bits integer,
%                         low 16 bits fraction) (for datatype=2).
%     int32     data2     For a time record, this field is the lo value of
%                         an AbsoluteTime number.

global LogData;
shouldConvertLogData = true;
if ~isfield(Resource.Parameters,'numLogDataRecs') || ...
        isempty(Resource.Parameters.numLogDataRecs)
    Resource.Parameters.numLogDataRecs = 128;
    shouldConvertLogData = false;
end
LogData = zeros(4, Resource.Parameters.numLogDataRecs, 'int32');

%% *** 12. Trans.HVMux and .Connector mapping

% Check for the special case of the Vantage 64 HV Mux adapter
if isequal(Resource.SysConfig.UTAtype, [1, 1, 1, 2])
    % The mux adapter only works with non-mux scripts
    if computeTrans(Trans.name,'HVMux')
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Probe with HVMux cannot be used with UTA 260-MUX adapter module.');
    end
    if ~isfield(Trans,'HVMux') || isempty(Trans.HVMux)
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Trans.HVMux structure required for UTA 260-MUX.  Use computeUTAMux64 in your script.');
    end
end

% Check format of TX structure before using it
if isempty(TX)
    error('VSX: TX structure is empty.');
elseif length(TX) > 1 && length(TX(:)) == size(TX, 1)
    % was defined as a column vector; convert to row and warn user if verbose
    if Resource.Parameters.verbose
        fprintf(2, 'VSX warning: TX structure array was defined as a column vector;\n');
        fprintf(2, '     converting to required row vector format.\n');
    end
    TX = TX';
elseif length(TX(:)) > 1 && length(TX(:)) ~= size(TX, 2)
    error('VSX: Multidimensional TX structure array not supported.  It must be a row vector.');
end
% VTS-804 determine Apod length for Resource.Parameters.sizeApod
% Look at length of TX(1).Apod to see if all-element mapping is being used
if ~isfield(TX(1), 'Apod') || isempty(TX(1).Apod)
    % Close the hardware event handler.
    hwEventHandler.uninstallHandler();
    error('VSX: Required Apod field in TX(1) is missing or empty.');
elseif size(TX(1).Apod, 2) == Trans.numelements
    % Apod array covers all elements
    Resource.Parameters.sizeApod = Trans.numelements;
elseif size(TX(1).Apod, 2) > Trans.numelements
    % Close the hardware event handler.
    hwEventHandler.uninstallHandler();
    error('VSX: length of TX(1).Apod exceeds Trans.numelements.');
else
    % "active aperture" element mapping; start with the size of the user's
    % Apod array.  This will be updated later to the required value derived
    % from Trans.ConnectorES or for HVMux probes Trans.HVMux.ApertureES
    Resource.Parameters.sizeApod = size(TX(1).Apod, 2);
end

noConnES = 0;
% flag to indicate Trans.Connector already has the composite mapping;
% initialize to false as default assumption

if isfield(Trans,'HVMux')
    % This is an HVMux transducer
    % now check for settling time field and create with default value if not
    % present
    if ~isfield(Trans.HVMux,'settlingTime') || isempty(Trans.HVMux.settlingTime)
        Trans.HVMux.settlingTime = 4; % default to 4 usec
    end
    if ~isfield(Trans.HVMux,'type') || isempty(Trans.HVMux.type)
        Trans.HVMux.type = 'preset'; % default to fixed, pre-programmed mux tables (disables dynamic mux programming)
    end

    if ~isfield(Trans.HVMux, 'ApertureES') || isempty(Trans.HVMux.ApertureES)
        % ApertureES field does not exist; check for legacy use of
        % Aperture
        if ~isfield(Trans.HVMux, 'Aperture') || isempty(Trans.HVMux.Aperture)
            % neither field exists; this is an error
            % Close the hardware event handler.
            hwEventHandler.uninstallHandler();
            error('VSX: Could not find required Trans.HVMux fields ApertureES or Aperture for HVMux script.')
        else
            % Aperture without ApertureES so assume Aperture is full EL
            % to CH mapping and use it as-is
            noApES = 1; % flag to indicate Aperture already has the composite mapping
            Trans.HVMux.ApertureES = Trans.HVMux.Aperture; % dummy duplicate in the ES field
        end
    else
        noApES = 0; % ApertureES actually exists
    end
    % verify columns are correct length in precomputed Trans.HVMux.ApertureES
    if size(Trans.HVMux.ApertureES, 1) ~= Trans.numelements
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Number of rows in Trans.HVMux.ApertureES must equal Trans.numelements.')
    end
    % check for illegal Element Signal index in the ApertureES array
    if max(Trans.HVMux.ApertureES(:)) > Resource.Parameters.numTransmit && Trans.HVMux.utaSF == 0
        % Note for HVMux UTA's the ApertureES values have not been
        % mapped through the UTA Mux switching; they will represent
        % input signals to the mux array and HVMux.Aperture
        % will identify the mux output system channel numbers
        % (VTS-852).
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Trans.HVMux.ApertureES is indexing a non-existent system channel number.\n')
    end
    if Resource.Parameters.sizeApod < Trans.numelements
        % active aperture Apod mapping is being used; aize of the active
        % aperture is set by number of non-zero entries in HVMux.ApertureES
        % columns
        Resource.Parameters.sizeApod = nnz(Trans.HVMux.ApertureES(:,1)); % use first column to set activeAperture, the required Apod size
        for i=1:size(Trans.HVMux.ApertureES, 2) % now check all other columns for equivalent value
            if Resource.Parameters.sizeApod ~= nnz(Trans.HVMux.ApertureES(:,i))
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: All Trans.HVMux.ApertureES columns must have the same no. of non-zero entries for active aperture Apod mapping.')
            end
        end
    end
    clear i
    % now check for illegal channel index in the ApertureES array
    if max(Trans.HVMux.ApertureES(:)) > Resource.Parameters.numTransmit && Trans.HVMux.utaSF == 0
            % Note for HVMux UTA's the ApertureES values have not been
            % mapped through the UTA Mux switching; they will represent
            % input signals to the mux array and HVMux.Aperture
            % will identify the mux output system channel numbers
            % (VTS-852).
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Trans.HVMux.ApertureES is indexing a non-existent system channel number.\n')
    end
elseif isfield(Trans,'ConnectorES') || isfield(Trans,'Connector')
    % ConnectorES (or legacy Connector) field is required for both HVMux
    % and Non-HVMux probes, but not used for active aperture mux scripts
    if isfield(Trans,'ConnectorES')
        noConnES = 0; % ConnectorES actually exists
    else
        % Connector without ConnectorES so assume Connector is full EL
        % to CH mapping and use it as-is
        noConnES = 1; % flag to indicate Connector already has the composite mapping
        Trans.ConnectorES = Trans.Connector; % dummy duplicate in the ES field
    end
    if size(Trans.ConnectorES, 1) ~= Trans.numelements
        % verify column of correct length
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Length of Trans.ConnectorES must equal Trans.numelements.')
    end
    % now check for illegal channel index in the ConnectorES array
    if max(Trans.ConnectorES(:)) > Resource.Parameters.numTransmit
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Trans.ConnectorES is indexing a non-existent system channel number.\n')
    end
    if Resource.Parameters.sizeApod < Trans.numelements
        % Not using all-element Apod, so set required active aperture Apod size
        Resource.Parameters.sizeApod = nnz(Trans.ConnectorES(:, 1)); % use first column of Trans.ConnectorES to set activeAperture
    end
else
    % No Trans.ConnectorES or Trans.Connector so create a default version with 1:1 mapping if possible
    if Trans.numelements ~= Resource.Parameters.numTransmit
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: Cannot create default Trans.ConnectorES array since Trans.numelements ~= Resource.Parameters.numTransmit.')
    end
    Trans.ConnectorES = (1:Trans.numelements)';
    noConnES = 0; % ConnectorES actually exists
    Resource.Parameters.sizeApod = Trans.numelements;
end

if VDAS == 0 && VDASupdates == 0 % SimulateOnly, with or without connector mapping
    % VTS-843 for simulate-only script we do not use a UTA structure at all
    % (there is no UTA.TransConnector mapping to apply), but still need to
    % create Trans.Connector from Trans.ConnectorES
    Trans.Connector = Trans.ConnectorES;
    if isfield(Trans,'HVMux')
        % Also remap HVMux.ApertureES into HVMux.Aperture
        Trans.HVMux.Aperture = Trans.HVMux.ApertureES;
    end
elseif VDASupdates && Resource.VDAS.el2ChMapDisable
    % when this flag is set while running with HW, overwrite
    % Trans.numelements and Trans.ConnectorES with the forced values of all
    % HW channels and a one-to-one mapping
    if ~isfield(Trans,'HVMux')
        Trans.numelements = UTA.numCh;
        Resource.Parameters.sizeApod = Trans.numelements;
        if isfield(Trans,'ConnectorES')
            Trans.ConnectorES = (1:Trans.numelements)';
        end
        Trans.Connector = Trans.ConnectorES;
    else
        Trans.HVMux.Aperture = Trans.HVMux.ApertureES;
        Trans.ConnectorES = (1:Trans.numelements)';
        Trans.Connector = Trans.ConnectorES;
    end
else
    % Create Trans.Connector by remapping Trans.ConnectorES through
    % UTA.TransConnector
    % Shift UTA.TransConnector up and add a zero as first entry; add one to
    % Trans ConnectorES and ApertureES so zero entries will be remapped to zero
    % through remapped UTA.TransConnector
    UtaTC = [0; UTA.TransConnector];
    if noConnES == 0
        % don't apply the TransConnector mapping if using legacy composite
        % Connector array
        Trans.Connector = UtaTC(Trans.ConnectorES + 1);
    end
    if isfield(Trans,'HVMux') && noApES == 0
        % Also remap HVMux.ApertureES into HVMux.Aperture if not using
        % dummy ApertureES
        Trans.HVMux.Aperture = UtaTC(Trans.HVMux.ApertureES+1);
    end
    clear UtaTC
end

%% *** 13. Media, PData, TW, TX, RcvProfile, TGC, Recon structures
% - Check for existance of 'MP' structure array. If not found, set defaults.
%       Media points are specified by number, position(x,y,z), and reflectivity.
%       MP(1,:) = [0,0,20,1.0];
if ~exist('Media','var')
    Media.model = 'PointTargets1';
    pt1;
    Media.numPoints = size(Media.MP,1);
elseif (isfield(Media, 'model'))
    if (strcmp(Media.model, 'PointTargets1'))
        pt1;
    elseif (strcmp(Media.model, 'PointTargets2'))
        pt2;
    elseif (strcmp(Media.model, 'PointTargets3'))
        mpt3;
    else
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('Unknown media model. Could not initialize MP array.\n');
    end
elseif (isfield(Media, 'program'))
    eval(Media.program);
    Media.numPoints = size(Media.MP,1);
elseif (~isfield(Media, 'MP'))
    % Close the hardware event handler.
    hwEventHandler.uninstallHandler();
    error('MP array not found.\n');
elseif (~isfield(Media, 'numPoints'))
    Media.numPoints = size(Media.MP,1);
end


% ***** PData structure *****
if exist('PData','var')
    % - Check to see if the PData structure(s) specify all pdeltas and Regions.  If not, create them.
    for i = 1:size(PData,2)
        k = 0;  % k is flag for need to call computeRegions
        if (~isfield(PData(i),'Region'))||(isempty(PData(i).Region)) % if no Regions defined
            k = 1;  % computeRegions will create and compute a single Region the size of PData
        else  % some Regions are defined, but may not be computed.
            for j = 1:size(PData(i).Region,2)  % check for all Regions computed.
                if (~isfield(PData(i).Region(j),'numPixels'))||(~isfield(PData(i).Region(j),'PixelsLA'))
                    % no compute fields defined - need to specify Shape structure.
                    if (~isfield(PData(i).Region(j),'Shape'))||(isempty(PData(i).Region(j).Shape))
                        % Close the hardware event handler.
                        hwEventHandler.uninstallHandler();
                        error('No Shape structure specified in PData(%d).Region(%d)\n',i,j);
                    end
                    k = 1;  % both computed fields not found but Shape structure provided.
                elseif (isempty(PData(i).Region(j).numPixels))||(isempty(PData(i).Region(j).PixelsLA))
                    % computed fields found but one or both empty; in this case also need Shape attribute
                    if (~isfield(PData(i).Region(j),'Shape'))||(isempty(PData(i).Region(j).Shape))
                        % Close the hardware event handler.
                        hwEventHandler.uninstallHandler();
                        error('No Shape structure specified in PData(%d).Region(%d)\n',i,j);
                    end
                    k = 1;
                end
            end
        end
        if (k==1), [PData(i).Region] = computeRegions(PData(i)); end  % compute all Regions
    end
end

% ***** TW Structure *****

% For the TW structure all checks for required veriables, generation of
% default values for optional variables, and checking for required range
% limits and format of all variables are executed in the computeTWWaveform
% function which is called automatically by VsUpdate(TW).  The VsUpdate
% function is called by VSX during initialization for the TW, TX, and
% Receive structures; VsUpdate is also called during run time whenever an
% 'update&Run' command affecting the Hardware evnet sequence is executed.

% ***** TX Structure *****
% The VsUpdate(TX) function will check for missing variables, out of range
% values, etc.  It will also add the TX VDAS parameters for running
% with the hardware, if VDASupdates is true.

if ~any(isfield(TX,{'VDASApod'}))
    % VsUpdate(TX) call will be made later during initialization, so at this
    % point we do nothing and just continue
else
    fprintf(2,'VSX: TX.VDASApod cannot be included in SetUp file. To set TX.VDASApod\n');
    fprintf(2,'manually, use Resource.Parameters.updateFunction to specify user provided function.\n');
    % Close the hardware event handler.
    hwEventHandler.uninstallHandler();
    error('VSX exiting...');
end

% ***** DMAControl Structure *****
if exist('DMAControl', 'var')
    % Close the hardware event handler.
    hwEventHandler.uninstallHandler();
    error(['%s: DMAControl structures cannot be included in SetUp file.\n', ...
           'To specify DMAControl structures, define an update functiion ', ...
           'using Resource.Parameters.updateFunction.\n'], mfilename)
end

% **** RcvProfile Structure ***

if VDASupdates % only add structure when preparing to run with HW
    % if RcvProfile structure doesn't exist, create it before calling compute
    % function
    if ~exist('RcvProfile','var')
        RcvProfile.AntiAliasCutoff = [];
    end

    RcvProfile = computeRcvProfile(RcvProfile);
    % The compute function will assign defaults for unspecified items, check
    % for valid values, etc.
end


% **** TGC Structure *****
% If a TGC(1) waveform exists, initialize the tgc slide pot variables.
if exist('TGC','var')
    tgc1 = TGC(1).CntrlPts(1);
    tgc2 = TGC(1).CntrlPts(2);
    tgc3 = TGC(1).CntrlPts(3);
    tgc4 = TGC(1).CntrlPts(4);
    tgc5 = TGC(1).CntrlPts(5);
    tgc6 = TGC(1).CntrlPts(6);
    tgc7 = TGC(1).CntrlPts(7);
    tgc8 = TGC(1).CntrlPts(8);
else
    tgc1=511; tgc2=511; tgc3=511; tgc4=511; tgc5=511; tgc6=511; tgc7=511; tgc8=511;
end
nTGC = 1;


% **** ReconInfo structures ****

% The functionality to automatically generate ReconInfo.Aperture is
% included in VsUpdate(Receive), which has already been called so nothing
% needs to be done here.  By locating this code in VsUpdate(Receive) it will
% always be executed even when the Receive structure is modified during
% system operation.


% **** Recon Structure ****
% - Check 'Recon' structure array for compatible PData and destination buffer sizes, and number
%   of ReconInfos.
%   If only one column and ReconInfo.regionnum = 0 or [], add ReconInfo structures for all regions.
if exist('Recon','var')
    nextRI = size(ReconInfo,2) + 1; % this will keep track of new ReconInfo structures created.
    for i = 1:size(Recon,2)
        j = Recon(i).pdatanum;
        if isfield(Recon(i),'IntBufDest')&&(~isempty(Recon(i).IntBufDest))&&(Recon(i).IntBufDest(1) > 0)
            k = Recon(i).IntBufDest(1); % get dest. buffer no.
            % Check for InterBuffer specified.
            if ~isfield(Resource,'InterBuffer')
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('Resource.InterBuffer(%d) specified in Recon but not defined.\n',k);
            end
        else
            % If no IntBufDest, check to verify that no IQ reconstructions are specified.
            for k = 1:numel(Recon(i).RINums)
                n = Recon(i).RINums(k);
                if (isfield(ReconInfo(n),'mode'))&&(~isempty(ReconInfo(n).mode))
                    if isnumeric(ReconInfo(n).mode)
                        if ReconInfo(n).mode > 2
                            % Close the hardware event handler.
                            hwEventHandler.uninstallHandler();
                            error('VSX: Recon(%d) specifies a ReconInfo.mode > 2, but no IntBufDest is given.\n',i);
                        end
                    elseif ischar(ReconInfo(n).mode)
                        switch ReconInfo(n).mode
                            case {'replaceIntensity','addIntensity','multiplyIntensity'}
                                continue
                            otherwise
                                % Close the hardware event handler.
                                hwEventHandler.uninstallHandler();
                                error('VSX: Recon(%d) specifies a ReconInfo.mode > 2, but no IntBufDest is given.\n',i);
                        end
                    else
                        % Close the hardware event handler.
                        hwEventHandler.uninstallHandler();
                        error('VSX: Recon(%d).mode must be numeric or string.\n',i);
                    end
                else
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Recon(%d).mode missing or empty.\n',i);
                end
            end
        end
        if isfield(Recon(i),'ImgBufDest')&&(~isempty(Recon(i).ImgBufDest))
            k = Recon(i).ImgBufDest(1); % get dest. buffer no.
            if k ~= 0
                % Check for ImageBuffer specified.
                if ~isfield(Resource,'ImageBuffer')
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('Resource.ImageBuffer(%d) specified in Recon but not defined.(%d)\n',k,j);
                end
            end
        end
        % Set the required value for Recon(i).numchannels.  Note that even
        % if a value was set by the user's script it will be overwritten
        % here.  An exception to this is when using the Volume Imaging
        % Package, where a Primary host control may be doing both partial
        % Recon over the local HW system channels and full Recon over all
        % channels of the multisystem (VTS-1288)
        if (~usingMultiSys)||(~isfield(Recon(i),'numchannels'))||(isempty(Recon(i).numchannels))
            Recon(i).numchannels = Resource.RcvBuffer.colsPerFrame;
        end
        % check for only one column and all regionnums in specified ReconInfos set to zero.
        if size(Recon(i).RINums,2)==1
            for j = 1:size(Recon(i).RINums,1) % set any missing or empty fields to zero.
                if ~isfield(ReconInfo(Recon(i).RINums(j,1)),'regionnum')||...
                        isempty(ReconInfo(Recon(i).RINums(j,1)).regionnum)
                    ReconInfo(Recon(i).RINums(j,1)).regionnum = 0;
                end
            end
            % check for all ReconInfos in col. set to specify all regions (0)
            if ~any([ReconInfo(Recon(i).RINums(:,1)).regionnum])
                for j = 1:size(Recon(i).RINums,1) % set first col RIs to region 1
                    ReconInfo(Recon(i).RINums(j,1)).regionnum = 1;
                end
            else
                if ~all([ReconInfo(Recon(i).RINums(:,1)).regionnum])
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: single col. Recon(%d).RINums must specify ReconInfo.regionnums that are all set or all missing or 0.\n',i)
                end
            end
        end
    end
end

%% *** 13.1 Event Structure
if isempty(Event)
    error('VSX: Event structure is empty.');
elseif length(Event) > 1 && length(Event(:)) == size(Event, 1)
    % was defined as a column vector; convert to row and warn user if verbose
    if Resource.Parameters.verbose
        fprintf(2, 'VSX warning: Event structure array was defined as a column vector;\n');
        fprintf(2, '     converting to required row vector format.\n');
    end
    Event = Event';
elseif length(Event(:)) > 1 && length(Event(:)) ~= size(Event, 2)
    error('VSX: Multidimensional Event structure array not supported.  It must be a row vector.');
end

try
    vsv.vsx.validate.mustBeValidEventIndices(Event);
catch EventValidateError
    error( EventValidateError.identifier, EventValidateError.message);
end

%% *** 14. TPC Profile 5 and HIFU Option Initialization
% Check for use of HIFU transmit profile 5 in setup script and initialize the workspace variable,
% 'TPC(5).inUse', which controls all HIFU or profile 5-related features in the system:
%     inUse = 0 (default) if the setup script does not make any use of Profile 5;
%     inUse = 1 for Extended Burst Option using internal auxiliary power supply for Profile 5 transmit.
%     inUse = 2 for HIFU semi-custom option using ext. power supply with remote control for Profile 5 transmit.
% Check SeqControl commands for a setTPCProfile command with an argument of 5.

% check for a user-specified substitute for TXEventCheck function & create
% handle, even if profile 5 is not in use at the moment (it may be after
% some gui activity, etc.)
if ~isfield(Resource,'HIFU') || ~isfield(Resource.HIFU,'TXEventCheckFunction') || isempty(Resource.HIFU.TXEventCheckFunction)
    Resource.HIFU.TXEventCheckFunction = 'TXEventCheck';
end
% now confirm the specified function actually exists and is on the path
if isempty(which(Resource.HIFU.TXEventCheckFunction))
    % Close the hardware event handler.
    hwEventHandler.uninstallHandler();
    error('VSX: The function specified by ''Resource.HIFU.TXEventCHeckFunction'' could not be found.');
end
TXEventCheckh = str2func(Resource.HIFU.TXEventCheckFunction); % create handle to TXEventCheck function.

% If profile 5 is going to be used, check that the hardware has the appropriate HIFU or Extended Transmit option installed.
% Check for the Resource.Parameter attribute, 'externalHifuPwr' from the
% setup script indicating that the external power supply is to be used.
% Also check and initialize other parameters related to use of profile 5.
if (TPC(5).inUse == 1)
    % Now check for the required TPC(5) max high voltage limit
    if ~exist('TPC','var')||size(TPC,2)<5||~(isfield(TPC(5),'maxHighVoltage'))||isempty(TPC(5).maxHighVoltage)
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error('VSX: A script using profile 5 transmit must specify TPC(5).maxHighVoltage.');
    end

    if isfield(Trans, 'HVMux')
        % if a probe with HV mux chips is being used, exit with an error
        % unless user has set special field to enable use
        if isfield(Trans.HVMux, 'P5allowed') && Trans.HVMux.P5allowed == 1
            if Resource.Parameters.verbose
                fprintf(2, 'VSX WARNING: A Sequence Control Command selecting TPC Profile 5 has been detected.\n');
                fprintf(2, '             Use of extended burst durations can be destructive to an HVMux probe.\n');
            end
            if Resource.Parameters.verbose > 1
                resp = input('Enter "y" to contimue, or return to exit: ', 's');
                if ~strcmpi('y', resp)
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    return
                end
                clear ans
            end
        else
            % Close the hardware event handler.
            hwEventHandler.uninstallHandler();
            error('VSX: Profile 5 transmit is not allowed when using probes with HVMux element switching.');
        end
    end

    % Now check actual HW configuration
    if VDAS
        % HW is present, so query actual HW configuration
        % Check for the 'p5ena flag. The value might be:
        %   0 for no HIFU option enabled
        %   1 for the "Extended Transmit Option" using internal auxilliary supply on the TPC
        %   2 for the "HIFU Option" using external OEM 1200 Watt power supply.
        if p5ena == 0
            % Close the hardware event handler.
            hwEventHandler.uninstallHandler();
            error('VSX: System is not licensed for use of Profile 5 transmit features.  Contact Verasonics Support for assistance.');
        end

        if p5ena == 2
            if isfield(Resource.HIFU,'externalHifuPwr') && (Resource.HIFU.externalHifuPwr >= 1)
                TPC(5).inUse = 2;
            else
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error(['VSX: System is configured for use with external HIFU power supply, ',...
                       'but Resource.HIFU.externalHifuPwr has not been set in SetUp script.']);
            end

            if Resource.SysConfig.HWconfigFault == 1
                % HIFU operation not allowed if configuration faults are
                % present
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: HIFU feature cannot be used due to HW configuration fault or licensing error.');
            end
            % At this point, the system is set to use the external power
            % supply option.  So we have to initialize the external power
            % supply and make sure it is connected and up to the initial
            % TPC(5).hv Voltage setting before we try to open the VDAS HW
            % (TPC initialization is likely to fail if HIFU capacitor
            % voltage is sitting at zero). - Initialize external supply
            % communication and set it to the minimum voltage level (or
            % user-specified TPC(5).hv value), 2 Amps

            % As part of initializing the power supply, we also determine
            % whether it is configured for series or parallel connection of
            % the two outputs and configure the control function to match.
            % This also lets us determine whether the power supply is
            % actually connected to the system and working properly, since
            % we will read back the actual voltage from the TPC.

            % get current voltage to determine if push capacitor is still
            % charged up from a previous run of the system
            [Result, prevV] = getHardwareProperty('TpcExtCapVoltage');
            if ~strcmp(Result,'Success')
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: Error from getHardwareProperty call to read push capacitor Voltage.');
            end
            % assume parallel connection for first step in initialization
            Resource.HIFU.extPwrConnection = 'parallel';
            % now try programming the supply at the minimum voltage level
            extPSstatus = extPwrCtrl('INIT', minTpcVoltage);
            if extPSstatus ~= 0
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: Cannot initialize external power supply.  Make sure it is connected and turned on.');
            end
            % wait for push cap voltage to stabilize near power supply
            % setting
            pause(1); % wait for power supply programming to take effect
            diff = 2;
            prevV = max(5, prevV); % don't trick the while loop with a very low previous reading
            msgCount = 0; % print status msg every second
            while diff > 0.1 && prevV >= 2.3
                pause(0.2)
                % now read the actual voltage at push cap. through the TPC
                [Result,newV] = getHardwareProperty('TpcExtCapVoltage');
                if ~strcmp(Result,'Success')
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Error from getHardwareProperty call to read push capacitor Voltage.');
                end
                diff = abs(prevV - newV);
                prevV = newV;
                msgCount = msgCount+1;
                if msgCount > 5 && Resource.Parameters.verbose > 1
                    msgCount = 0;
                    disp(['VSX init waiting for push cap voltage to stabilize; current value ', num2str(newV), ' Volts.']);
                end
            end

            % check the result:
            if newV < 1.1
                % less than this Voltage means we have a fault or power supply
                % not connected;
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: No output from external power supply, or it may be disconnected.');
            elseif newV >= 1.1 && newV < 2.3
                % between these levels means supply is using parallel
                % connection and is working properly, so no need to do
                % anything else
            elseif newV >= 2.3 && newV < 4.5
                % between these levels means we have a
                % series-connected supply, so reconfigure for series
                % operation and test again

                % but first we must disable supply to reprogram it
                extPwrCtrl('CLOSE', minTpcVoltage);
                pause(0.5) % give it time to close
                Resource.HIFU.extPwrConnection = 'series'; % reconfigure for series
                % now try reprogramming the supply
                extPSstatus = extPwrCtrl('INIT', minTpcVoltage);
                if extPSstatus ~= 0
                    fprintf(2, 'External power supply error code %d.\n', extPSstatus);
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Cannot initialize external power supply.  Make sure it is connected and turned on.');
                end
                pause(1); % wait for the power supply to come back on & start bleeding capacitor down
                % now monitor the actual voltage at push cap. through the
                % TPC and wait for it to stabilize
                diff = 2;
                msgCount = 0; % print status msg every second
                while diff > 0.05 && prevV >= 2.3
                    pause(0.2)
                    % now read the actual voltage at push cap. through the TPC
                    [Result,newV] = getHardwareProperty('TpcExtCapVoltage');
                    if ~strcmp(Result,'Success')
                        % Close the hardware event handler.
                        hwEventHandler.uninstallHandler();
                        error('VSX: Error from getHardwareProperty call to read push capacitor Voltage.');
                    end
                    diff = abs(prevV - newV);
                    prevV = newV;
                    msgCount = msgCount+1;
                    if msgCount > 10 && Resource.Parameters.verbose > 1
                        msgCount = 0;
                        disp(['VSX waiting for series-connected external supply to stabilize; current value ', num2str(newV), ' Volts.']);
                    end
                end
                if newV < 1.1 || newV > 2.4
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Error while initializing external supply for series connection.');
                end
            else
                % any other voltage means we have a fault.
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: Error while initializing external supply for parallel connection.');
            end

            clear newV prevV diff msgCount Result

            % power supply series or parallel initialization is complete;
            % now check for TPC(5).hv greater than the minimum voltage
            % level, and if so set the supply to the specified voltage
            if TPC(5).hv > minTpcVoltage
                extPSstatus = extPwrCtrl('INIT', TPC(5).hv);
                if extPSstatus ~= 0
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Initialization of external power supply to TPC(5).hv voltage level has failed.');
                end
            end

        else % system is configured for internal aux. supply; confirm that's what the user intended
            if isfield(Resource.HIFU,'externalHifuPwr') && (Resource.HIFU.externalHifuPwr >= 1)
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: System is not configured for use with external HIFU power supply.');
            end
        end
    else
        % No hardware is present, so set HIFU defaults for simulation.
        if isfield(Resource.HIFU,'externalHifuPwr') && (Resource.HIFU.externalHifuPwr >= 1)
            % set simulation defaults for external HIFU
            TPC(5).inUse = 2;
        end
    end
end






%% *** 15. If not in simulate mode try to open the hardware.
if VDAS
    % Try to open the Verasonics hardware.
    %
    % NOTE:  Textual output from compiled C code is not seen in MATLAB until
    % after the function returns.
    try
        errMsg = [];
        hwResult = Hardware.connect(true);
        if hwResult == HwResult.success
            % (VTS-1672): check to see if hw system has already been
            % initialized; if not proceed with hardware initialization but
            % if so don't need to repeat initialization but do need to
            % reset operating states that may have been retained from
            % previous run of a different sequence with VSX.  (Note the
            % command Hardware.resetHardware does not actually reset in the
            % sense of a power-up reset; it only clears operating states
            % created by running a sequence, such as HVMux settings.)
            if Hardware.getBool(BoolAttr.isSoftwareInitialized)
                Hardware.resetHardware(true);
                hwResult = HwResult.success;
            else
                hwResult = Hardware.initializeHardware();
            end
            if hwResult == HwResult.success
                Result = 'SUCCESS';
            else
                faultCondition = Hardware.getFaultCondition();
                if faultCondition == FaultConditionResult.none
                    errMsg = 'VSX:  Failed to initialize hardware.';
                else
                    numFaults = Faults.getNumberOfFaults;
                    if numFaults > 0
                        % Get the first fault that occurred, this is
                        % probably the root cause.
                        faultInfo = Faults.getFirstFault();
                        errMsg = sprintf('VSX: Failed to open hardware because of fault. Fault:%s', faultInfo.faultReport);
                    else
                       errMsg = 'VSX:  Failed to connect to hardware because of a hardware fault.';
                    end
                end
                Result = 'FAIL';
            end
        else
            errMsg = 'VSX:  Failed to connect to hardware.';
            Result = 'FAIL';
        end
    catch
        errMsg = 'VSX: There was an exception in connecting to the hardware.';
        Result = 'FAIL';
    end
    if ~strcmpi(Result,'SUCCESS')
        % If hardware didn't open successfully, report an error and quit
        % Close the hardware event handler.
        hwEventHandler.uninstallHandler();
        error(errMsg)
    end
end


%% *** 16. check probe connected status and ID value
% VTS-865 revised to use new getConnectorInfo query supporting connectors
% with no disconnect sensing and/or no probe ID feature

if VDAS == 1 && Resource.SysConfig.UTA
    % skip the probe ID and connector selection if there is no HW system
    % present, or if there is HW but no SHI or UTA baseboard is present
    % (which will be indicated by SysConfig.UTA set to zero)

    % Query the system to see if a probe is connected at the primary
    % connector, and if so check ID if required and ID is supported.  If no
    % problems are found, select the connector(s) identified by
    % Resource.Parameters.Connector; note it has already been initialized
    % to appropriate default value if needed

    [Presence, ~, ID] = getConnectorInfo();
    % note the second return argument (selected status) is not used here

    primaryConnector = Resource.Parameters.Connector(1);
    % The first entry in Connector will be the "primary"; this is the
    % connector for which ID will be checked.
    probeConnected = abs(Presence(primaryConnector));
    % Presence will be -1 if no disconnect sensing, so the abs converts
    % that to setting probeConnected true.

    if probeConnected
        % a probe is connected so check if ID matches the value expected by
        % the script, but only if the connector can provide an ID- if not,
        % just proceed with connector selection since there is nothing else
        % we can check.
        if ID(primaryConnector) == -1 || ID(primaryConnector) == Trans.id
            % either no ID is available or probe with the expected ID is
            % connected, so select the connector(s) and proceed to run
            result = setSelectedConnector(Resource.Parameters.Connector);
            if ~strcmpi(result, 'Success')
                % report failure and exit
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: Failed to select probe connector because "%s".\n', result);
            end
        else
            % a probe is connected with an ID that does not match Trans.id;
            % check if script wants to ignore ID to decide what to do next
            if strcmpi(Trans.name, 'custom')  || Trans.id == -1
                % Either setting Trans.name to 'custom' or Trans.id to -1
                % means user wants to ignore the actual ID value read from
                % the connected probe.  Before allowing the script to run
                % we have to see if the connected probe uses HVMux chips;
                % if so we will exit with an error condition
                probeName = computeTrans(ID(primaryConnector));
                if strcmp(probeName, 'Unknown')
                    hvMux = 0;
                else
                    % this is a recognized transducer so find HVMux status
                    hvMux = computeTrans(probeName, 'HVMux');
                end

                if hvMux
                    % an HVMux probe is connected so override the 'custom'
                    % script and exit with an error.
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    errorstr = ['VSX: Connected probe is ', probeName, ' using HVMux chips.  ID mismatch not allowed for HVMux probes.'];
                    error(errorstr);
                else
                    % not an hvMux probe, so go ahead and run the script
                    % normally, ignoring the ID mismatch
                    result = setSelectedConnector(Resource.Parameters.Connector);
                    if ~strcmpi(result, 'SUCCESS')
                        % report failure and exit
                        % Close the hardware event handler.
                        hwEventHandler.uninstallHandler();
                        error('VSX: Failed to select probe connector because "%s".\n', result);
                    end
                    if Resource.Parameters.verbose>1
                        disp(['VSX status: ID of probe specified by script doesn''t match ID of connected probe (', probeName, '),']);
                        disp('but Trans.name in script is set to ''custom'' or Trans.id is set to minus one so ID mismatch is being ignored.');
                    end
                end
            else
                % Script did not ask to ignore ID so just exit with an error
                % message:
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: ID of probe specified by script doesn''t match ID of probe at connector %d.\n', primaryConnector);
            end
        end
    else
        % No probe is connected at identified connector; determine if
        % fakeScanhead is set
        if isfield(Resource.Parameters,'fakeScanhead') && Resource.Parameters.fakeScanhead
            % fakeScanhead mode, so select connector and run
            if Resource.Parameters.verbose > 1 && Resource.Parameters.simulateMode == 0
                % display status message
                disp('VSX Status: No probe has been detected at the specified connector, and ');
                disp('Resource.Parameters.fakeScanhead is set so proceeding to run the script');
                disp('on the HW system (this is live acquisition, not simulation mode).');
            end
            result = setSelectedConnector(Resource.Parameters.Connector);
            if ~strcmpi(result, 'SUCCESS')
                % report failure and exit
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: Failed to select probe connector because "%s".\n', result);
            end
        else
            if autoScriptTest
                % For autoScriptTest, automatically run a script when no
                % probe is selected.  The normal VSX code is after the
                % 'else' below.
                Resource.Parameters.fakeScanhead = 1;
                % select connector and run with "inverted
                % monitoring" to detect a probe connect event and quit
                result = setSelectedConnector(Resource.Parameters.Connector);
                if ~strcmpi(result, 'SUCCESS')
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Failed to select probe because "%s".\n', result);
                end
            elseif Resource.Parameters.simulateMode == 0
                % Hardware acquistion mode is selected, but nothing is
                % connected and fakeScanhead not set, so ask user if they
                % want to quit or automatically switch to fake mode.
                if Resource.Parameters.verbose
                    disp(' ');
                    disp('No probe detected at specified connector - ');
                    disp('Do you want to run the script in fake scanhead mode for ');
                    disp('live acquisition with nothing connected?');
                    Reply = input('Enter Y or return to proceed, N to exit:', 's');
                    if strcmpi(Reply, 'N')
                        % user says exit, so just return at this point
                        % Close the hardware event handler.
                        hwEventHandler.uninstallHandler();
                        return
                    end
                    % user wants script to run, so set the flag to allow associated
                    % logic to function normally
                    Resource.Parameters.fakeScanhead = 1;
                    result = setSelectedConnector(Resource.Parameters.Connector);
                    if ~strcmpi(result, 'SUCCESS')
                        % report failure and exit
                        % Close the hardware event handler.
                        hwEventHandler.uninstallHandler();
                        error('VSX: Failed to select probe connector because "%s".\n', result);
                    end
                else
                    % Resource.Parameters.verbose is false, so just exit with error message
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: No probe detected at connector number %d.\n',primaryConnector);
                end
            else
                % Initial simulate mode setting is either simulate or loop
                % playback, but nothing is connected and fakeScanhead not
                % set.  Allow script to start running in simulate mode, but
                % automatically set fake scanhead and select connector so
                % use will be able to toggle into live HW acquisition if
                % they want to.
                Resource.Parameters.fakeScanhead = 1;
                result = setSelectedConnector(Resource.Parameters.Connector);
                if ~strcmpi(result, 'SUCCESS')
                    % report failure and exit
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Failed to select probe connector because "%s".\n', result);
                end
            end
        end
    end


    % If this scanhead is a HVMux scanhead, set the HVMux attributes.
    if isfield(Trans,'HVMux')
        if UTA.elBiasEna == 1 && Trans.elBias ~= 0
            % Close the hardware event handler.
            hwEventHandler.uninstallHandler();
            error('VSX: HVMux-based probes cannot use Trans.elBias with this UTA module or SHI.');
        end
    end

    % Check for Element Bias use
    if Trans.elBias ~= 0
        % Element bias is requested; check status of UTA.elBiasEna
        switch UTA.elBiasEna
            case 0
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: Trans.elBias can not be used with this UTA module.');
            case 1
                % check allowed range for HVMux power supply when used for
                % element bias
                if abs(Trans.elBias) < 10 || abs(Trans.elBias) > 100
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Trans.elBias setting is outside the supported range of 10 to 100 Volts for HVMux power supply.');
                end
                Resource.VDAS.elBiasSel = 1;
            otherwise
                % note case 2 for UTA baseboard bias source will be added
                % in a future release
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error('VSX: Unrecognized UTA.elBiasEna value of "%d".\n', UTA.elBiasEna);
        end
    end
end

%% *** 17. Initialize GUI and Display Window(s)
% VTS-1317: For 4.1.0, changing default display window Type to Verasonics,
% and initializing DisplayWindow before calling vsx_gui since vsx_gui makes
% use of the Type value.
% Set up DisplayWindow(s) for output image(s). DisplayWindow(s) can be created for displaying
% processed ImageBuffers or user data.  If no DisplayWindow structure is specified in the user script,
% this step is skipped.


if usingApplication

    vsxApplication.importSeqComponents();
    % an application can use its own display window. Therefore it need to
    % be started here before the display windows are gettig initialized
    vsxApplication.startApplication();
    displaySettings = vsv.seq.display.DisplaySettings( );

end

if isfield(Resource,'DisplayWindow')
    for i = 1:size(Resource.DisplayWindow,2)
        % Set defaults for DisplayWindow structure
        if (~isfield(Resource.DisplayWindow(i),'Title'))||isempty(Resource.DisplayWindow(i).Title)
            Resource.DisplayWindow(i).Title = displayWindowTitle;
        end
        if (~isfield(Resource.DisplayWindow(i),'Position'))||isempty(Resource.DisplayWindow(i).Position)
            % Close the hardware event handler.
            hwEventHandler.uninstallHandler();
            error('VSX: DisplayWindow(%d) has no ''Position'' attribute.',i);
        else
            imWidth = Resource.DisplayWindow(i).Position(3); % Position(3) is imWidth, not figure width.
            imHeight = Resource.DisplayWindow(i).Position(4); % Position(4) is imHeight, not figure height.
        end
        DisplayData = zeros(imHeight,imWidth,'uint8');
        if (~isfield(Resource.DisplayWindow(i),'Colormap'))||isempty(Resource.DisplayWindow(i).Colormap)
            Resource.DisplayWindow(1).Colormap = gray(256);  % default colormap is greyscale.
        end
        if (~isfield(Resource.DisplayWindow(i),'Type'))||isempty(Resource.DisplayWindow(i).Type)
            Resource.DisplayWindow(i).Type = 'Verasonics';
        end

% VTS-1513 following initialization steps apply to both viewers; removed
% from DisplayWindow(i).Type case statements below
        if ~isfield(Resource.DisplayWindow(i),'splitPalette')||isempty(Resource.DisplayWindow(i).splitPalette)
            Resource.DisplayWindow(i).splitPalette = 0;
        end
        if ~isfield(Resource.DisplayWindow(i),'pdelta')||isempty(Resource.DisplayWindow(i).pdelta)
            Resource.DisplayWindow(i).pdelta = 0.5;
        end
        if ~isfield(Resource.DisplayWindow(i),'clrWindow')||isempty(Resource.DisplayWindow(i).clrWindow)
            Resource.DisplayWindow(i).clrWindow = 0;
        end
        % Determine whether a 2D DisplayWindow is x,z (normal 2D scan), x,z or x,y (3D C-scan) oriented.
        % The DisplayWindow.ReferencePt is a point in the x,z plane for normal 2D scans, or a point in
        % the x,y,z volume for a 3D volume.  For 3D scans and DisplayWindow.mode = '2D', the
        % DisplayWindow.orientation attribute determines the orientation of the 2D slice.
        %  - check for ReferencePt specified and convert to 3 dimensions if necessary.
        if ~isfield(Resource.DisplayWindow(i),'ReferencePt')||isempty(Resource.DisplayWindow(i).ReferencePt)
            Resource.DisplayWindow(i).ReferencePt = [PData(1).Origin(1),0,PData(1).Origin(3)];
        end
        if length(Resource.DisplayWindow(i).ReferencePt)==2
            Resource.DisplayWindow(i).ReferencePt(3) = Resource.DisplayWindow(i).ReferencePt(2);
            Resource.DisplayWindow(i).ReferencePt(2) = 0;
        end
        if ~isfield(Resource.DisplayWindow(i),'mode')||isempty(Resource.DisplayWindow(i).mode)
            Resource.DisplayWindow(i).mode = '2d';  % default mode to '2d'
        end

        if Trans.type < 2
            % DisplayWindow is x,z oriented
            vmin = Resource.DisplayWindow(i).ReferencePt(3); % vertical minimum is DisplayWindow.ReferencePt(3).
            vmax = vmin + imHeight*Resource.DisplayWindow(i).pdelta;
            xmin = Resource.DisplayWindow(i).ReferencePt(1);
            xmax = xmin + imWidth*Resource.DisplayWindow(i).pdelta;
        else
            if ~isfield(Resource.DisplayWindow(i),'Orientation')
                % Close the hardware event handler.
                hwEventHandler.uninstallHandler();
                error(['VSX: Resource.DisplayWindow(%d).Orientation must be defined for a 2D Displaywindow',...
                       'with Trans.type = 2\n'],i);
            end
            switch Resource.DisplayWindow(i).Orientation
                case 'xz'
                    % DisplayWindow is x,z oriented
                    vmin = Resource.DisplayWindow(i).ReferencePt(3);
                    vmax = vmin + imHeight*Resource.DisplayWindow(i).pdelta;
                    xmin = Resource.DisplayWindow(i).ReferencePt(1);
                    xmax = xmin + imWidth*Resource.DisplayWindow(i).pdelta;
                case 'yz'  % DisplayWindow is y,z oriented
                    vmin = Resource.DisplayWindow(i).ReferencePt(3);
                    vmax = vmin + imHeight*Resource.DisplayWindow(i).pdelta;
                    xmin = Resource.DisplayWindow(i).ReferencePt(2);
                    xmax = xmin + imWidth*Resource.DisplayWindow(i).pdelta;
                case 'xy'  % DisplayWindow is x,y oriented
                    vmin = Resource.DisplayWindow(i).ReferencePt(2);
                    vmax = vmin + imHeight*Resource.DisplayWindow(i).pdelta;
                    xmin = Resource.DisplayWindow(i).ReferencePt(1);
                    xmax = xmin + imWidth*Resource.DisplayWindow(i).pdelta;
            end
        end

        % Set limits for x axis.
        scale = 1.0;  % default is no scaling (wavelengths)
        axisname = 'wavelengths';
        if isfield(Resource.DisplayWindow(i),'AxesUnits')&&~isempty(Resource.DisplayWindow(i).AxesUnits)
            if strcmpi(Resource.DisplayWindow(i).AxesUnits,'mm')
                axisname = 'mm';
                if ~isfield(Resource.Parameters,'speedOfSound')||isempty(Resource.Parameters.speedOfSound)
                    scale = 1.54/Trans.frequency; % use default speed of sound.
                else
                    scale = (Resource.Parameters.speedOfSound/1000)/Trans.frequency;
                end
            end
        end

        % in a future release this section will replace the entire section
        % that determines the display initial settings as well as the
        % initialization of the display window.
        if usingApplication && vsxApplication.hasDisplay()

            % @ToDo this needs to be optimized and will be replaced in the
            % future with a common scheme for displays
            displaySettings.importStruct( Resource.DisplayWindow(i) );
            displaySettings.Scale = scale;
            if Trans.type < 2
                displaySettings.Orientation = 'xz';
            end

            % display @type vsv.vsx.display.AbstractDisplay
            display = vsxApplication.getDisplay(i, displaySettings);
            display.initializeDisplay(displaySettings);

            Resource.DisplayWindow(i).figureHandle = display.getFigureHandle();
            Resource.DisplayWindow(i).figureHandle.CurrentAxes = display.getAxesHandle();
            Resource.DisplayWindow(i).imageHandle  = display.getImageHandle();
            Resource.DisplayWindow(i).Type         = display.getDisplayType();
        else
            switch Resource.DisplayWindow(i).Type
                case 'Matlab'  % DisplayWindow is Matlab figure window.
                    % Create the figure window.

                    if usingHIFUPlex
                        axesBmode = findobj('tag','axesBmode');
                        axesHIFU = findobj('tag','axesHIFU');
                        loadAnnotation;
                        delete(get(axesBmode,'Children'));
                    else
                        Resource.DisplayWindow(i).figureHandle = figure( ...
                            'Name',Resource.DisplayWindow(i).Title,...
                            'NumberTitle','off',...
                            'Position',[Resource.DisplayWindow(i).Position(1), ... % left edge
                            Resource.DisplayWindow(i).Position(2), ... % bottom
                            imWidth + 100, imHeight + 150], ...            % width, height + border
                            'Colormap',Resource.DisplayWindow(i).Colormap, ...
                            'Visible','off');
                        axes('Units','pixels','Position',[60,90,imWidth,imHeight]);
                        set(gca, 'Units','normalized');  % restore normalized units for re-sizing window.
                    end

                    if strcmp(Resource.DisplayWindow(i).mode,'2d')
                        set(gca,'YDir','reverse');
                    else
                        % Close the hardware event handler.
                        hwEventHandler.uninstallHandler();
                        error('VSX: Resource.DisplayWindow.modes other than ''2d'' not currently supported.');
                    end
                    if usingHIFUPlex
                        Resource.DisplayWindow(i).imageHandle = image('Parent',axesBmode,...
                            'CData',DisplayData, ...
                            'XData',scale*[xmin,xmax], ...
                            'YData',scale*[vmin,vmax], ...
                            'Tag','imgBmode');
                        set(axesBmode,'FontSize',12); xlabel(axesBmode,axisname,'FontSize',12);
                        title(axesBmode,'Real-time B-mode','FontSize',14)
                        drawnow
                    else
                        Resource.DisplayWindow(i).imageHandle = image('CData',DisplayData, ...
                            'XData',scale*[xmin,xmax], ...
                            'YData',scale*[vmin,vmax]);
                        set(gca,'FontSize',12); xlabel(axisname,'FontSize',14);
                        axis equal tight;
                        drawnow
                        set(Resource.DisplayWindow(i).figureHandle,'Visible','on');
                    end
                case 'Verasonics'  % DisplayWindow is Verasonics Java window.

                    % Import and initialize the Java libaries that are needed for the Verasonics viewer.
                    import com.verasonics.viewer.ui.components.*
                    import com.verasonics.viewer.ui.overlays.*
                    import com.verasonics.viewer.ui.panels.*
                    import com.verasonics.viewer.ui.*
                    import com.verasonics.viewer.*
                    import com.verasonics.viewer.tools.mouseclicktool.*
                    import com.verasonics.vantage.image.*
                    import com.verasonics.vantage.image.events.*
                    import java.awt.*

                    % Uncomment the line below to print image viewer events (enable debug mode).
                    % MatlabVantageWindow.setDebugMode(true);

                    % Define the background color of the viewer window.
                    viewerBgColor = '0xCDCDCD';

                    % Create the Verasonics Window.
                    vantageWindow = MatlabVantageWindow.create(Attr('xPos',Resource.DisplayWindow(i).Position(1)), ... % left edge
                                                               Attr('yPos',Resource.DisplayWindow(i).Position(2)), ... % bottom
                                                               Attr('width',imWidth + VantageWindow.PAD_WIDTH), ...    % width of image + padding
                                                               Attr('height',imHeight + VantageWindow.PAD_HEIGHT), ...  % height of image + padding
                                                               Attr('resizeable', true), ...
                                                               Attr('bkgColor', viewerBgColor), ...
                                                               Attr('title',Resource.DisplayWindow(i).Title));

                    % Get the Unique ID of this window. Used to communicate with this specific window.
                    Resource.DisplayWindow(i).figureHandle = vantageWindow;

                    % Create the image viewer that is displayed in the window.
                    imageViewer(i) = ImageViewerPanel(Attr('label',Resource.DisplayWindow(i).Title), ...
                                                   Attr('showAxisAnnotation', true), ...
                                                   Attr('bkgColor', viewerBgColor), ...
                                                   Attr('scaleToWindowSize', true), ...
                                                   Attr('acqRateMethod', 'synchConstantRate30Hz'));%#ok

                    % Add the image viewer to the window.
                    vantageWindow.addComponent(imageViewer(i), ... % component
                                       0, 0, ... % X|Y grid placement
                                               Attr('anchor', 'north'), ...
                                               Attr('weightX', 1.0), ...
                                       Attr('weightY', 1.0));

                    % Add the menubar to the image viewer window.
                    imageViewerMenu = MatlabImageViewerMenu(vantageWindow, imageViewer(i));
                    vantageWindow.addMenuBar(Attr('menuBarObject', imageViewerMenu));

                    % Get the Unique ID of this image viewer. Used to send events to this specific image viewer.
                    Resource.DisplayWindow(i).imageHandle = imageViewer(i).uid;

                    % set unitsType for verasonics viewer
                    if strcmp(axisname, 'mm')
                        unitsType = SpatialUnitsType.millimeters;
                    else
                        unitsType = SpatialUnitsType.wavelengths;
                    end

                    % Generate an event to update the colormap.
                    colorMapEvent = ColorMapEvent(256, ColorMapType.planarNoramlizedDoubles, Resource.DisplayWindow(i).Colormap);
                    VantageImageEvent.generateColorMapEvent(imageViewer(i).uid, colorMapEvent);

                    % Generate an event that the spatial units have changed.
                    spatialUnitsEvent = SpatialUnitsEvent(unitsType, scale);
                    VantageImageEvent.generateSpatialUnitsEvent(imageViewer(i).uid, spatialUnitsEvent);

                    % Generate an event that the spatial position has changed.
                    spatialPositionEvent = SpatialPositionEvent(xmin, xmax, vmin, vmax);
                    VantageImageEvent.generateSpatialPositionEvent(imageViewer(i).uid, spatialPositionEvent);

                otherwise
                    % Close the hardware event handler.
                    hwEventHandler.uninstallHandler();
                    error('VSX: Unrecognized Type for DisplayWindow(%d).',i);
            end
        end
    end

    clear imWidth imHeight
end

% ***** Initialize GUI and add UI objects. *****
% Set up GUI; first check for user-defined replacement
if ~isfield(Resource.Parameters, 'GUI') || isempty(Resource.Parameters.GUI)
    Resource.Parameters.GUI = 'vsx_gui';
end
% now confirm the specified function actually exists and is on the path
if isempty(which(Resource.Parameters.GUI))
    % Close the hardware event handler.
    hwEventHandler.uninstallHandler();
    error('VSX: The GUI function specified by ''Resource.Parameters.GUI'' could not be found.');
end
if ~usingHIFUPlex && ~usingApplication
    vsxGUIh = str2func(Resource.Parameters.GUI); % create the function call
    vsxGUIh(); % and run it
end
% - Set state of simulate and rcvdataloop buttons (initial state is 0).
set(findobj('String','Rcv Data Loop'),'Value',rloopButton);
set(findobj('String','Simulate'),'Value',simButton);

% Make UI window the current figure, unless the caller has requested to hide it.
if exist('Mcr_GuiHide', 'var') && 1 == Mcr_GuiHide
    % Caller has requested that we do NOT show the GUI window.
    if Resource.Parameters.verbose>1
        disp('NOTE:  Caller has requested that the VSX GUI window be hidden.  Use ctrl-c to abort VSX if necessary.');
    end
else
    % OK, DO make the GUI window active.
    set(0, 'CurrentFigure', findobj('tag','UI'));
end

isUsingOldVSXGUIcreation = exist('isUsingOldVSXGUIcreation','var') && isUsingOldVSXGUIcreation;

% - If .mat file contains a UI object, add it to UI window.
if exist('UI','var') && strcmp('vsx_gui',Resource.Parameters.GUI) && ~isUsingOldVSXGUIcreation

    mainGUIFigure = findobj('tag','UI');


    userControlApp = vsxApplication.getUserControlManager();

    if ~isempty(userControlApp)
        userControlApp.parseUIControls( UI );
        userControlApp.buildUIControls(  );
        drawnow;
        pause(0.1);
        if exist('Control', 'var')
            clear( 'Control');
        end    
    end
else
    % do the old way
    vsv.seq.view.deprecated.vsxGUIcreation();
end

if (usingMultiSys)
    [RDMAconfig, RDMAcontrol, RDMAsystem] = vsv.multi.updateNaming(RDMAconfig, RDMAcontrol, RDMAsystem);
    import com.verasonics.common.socket.*
    import com.verasonics.common.util.*
    Net.initialize;
    if strcmp(RDMAconfig.rdmaRole,'primary')
        vsv.multi.generateSecondaryEcho(mainGUIFigure); %generate listeners for multisystem echoing of commands to secondarys
    end
    com.verasonics.common.util.Prop.deleteProperty('remoteCmd');
end


%% *** 18. Final initialization and calls to VsUpdate and TXEventCheck

% 'runAcq' is called with the single input, 'Control', which is a two attribute structure
%    array where the first attribute, 'Command', is a command string,  and the second attribute,
%    'Parameters', is a cell array whose interpretation depends on the command given.
% Valid commands are:
%    Control.Command =
%      'set' - set an attribute of an existing object (structure) and return.
%          Control.Parameters = {'Object', objectNumber, 'attribute', value, 'attribute', value, ...}
%      'set&Run' - set an attribute of an existing object (structure) and run sequence.
%          Control.Parameters = {'Object', objectNumber, 'attribute', value, 'attribute', value, ...}
%      'update&Run' - update the entire object(s) by re-reading from Matlab, then run sequence.
%          Control.Parameters = {'InterBuffer','ImageBuffer','DisplayWindow',
%                                'Parameters', 'Trans','HVMux','Media','PData',
%                                'TW','TX','Receive','ReceiveProfiles','TGC',
%                                'Recon','Process','SeqControl','Event'}*
%                                * provide one or more of this list of objects
%      'copyBuffers' - Copies back to Matlab IQData,ImgData,ImgDataP and LogData
%          buffers, without running the Event sequence.
%      'setBFlag' - set the BFlag used for triggering the conditional branch sequence control.
%      'imageDisplay' - display an image using the specified Image parameters.
%          Control.Parameters = {'Image attribute, value, ...}
%      'debug' = turns on or off debug output.;
action = 'none';  % First main loop action must be 'none'.
vsExit = 0;   % set to 1 when UI window is closed.
freeze = 0; % freeze will get set to 1 when 'Freeze' togglebutton is pressed.
softwarePeriod = 1;   % set initial frame period to 1 second
runAcqPeriod = 1;
initialized = 0;  % will be set to one by runAcq after hardware initialization

if autoScriptTest
    % Initialize the runcount variable, for automatic exit of VSX
    % after 25 runAcq return to Matlab iterations
    runcount = 0;
end

% ***** final initialization step: calls to VsUpdate and TXEventCheck
% note these initialization calls must be made in the order listed here
% an error from any of these function calls will result in VSX exiting
% with an error
updateh('TW');
updateh('TX');
updateh('Receive');
updateh('SeqControl');
if isfield(Trans, 'HVMux')
    % create SHIAperture table for HVMux probes or UTA's VTS-823, 343
    Trans=computeHvMuxSHIAperture(Trans);
end
if VDASupdates
    TXEventCheckh(true); % VTS-1796 pass initialization status to TXEventCheck
end

if ~exist('Control', 'var')
    Control.Command = [];  % Set Control.Command to 'empty' for no control action.
end
%Control.Command = 'debug';  % Control.Command to turn on or off debug output.
%Control.Parameters = {'on'}

% VTS-1583 We do not need to initialize the TPC profile 5 monitor limits at
% this point; they will have been left in the disabled state as a result of
% a power-up cycle or upon exit from the previous run of VSX (in the
% cleanup function)

% return % TEST: uncomment this line to examine workspace after VSX initialization, but before calling runAcq

% Check to see if user only wants runAcq to initialize.
if Resource.Parameters.initializeOnly > 0
    runAcq(Control);
    % Close the hardware event handler.
    hwEventHandler.uninstallHandler();
    return
end

% final import to make sure the sequence components
if usingApplication
    vsxApplication.importSeqComponents();
    vsxApplication.acquisitionStarted();
end

%% *** 19. Run sequence: loop for continuously calling 'runAcq'.


nloop = 1; % loop this time is considered as time between two matlab exit (depends on sequence)
nloopUpdate = 1; 
saved_interfaceElapsedTime = [];
saved_tElapsedRunAcq = zeros(1e6, 1);
saved_tElapsedSoftware = zeros(1e6, 1);
saved_updateFrameIndex = zeros(1e6, 1); % keep track when updating occurs as this affect loop time

t = tcpclient(IPV4, PORT, 'Timeout' , 300); %or 'localhost'


% send parameters to python for initialization 
disp('Writing settings over network')
Nax = Receive(1).endSample;
%NOTE: bytesPerElementRead and Sent is opposed to the server 
% values as what is wirtten in matl is read in python
msg = [Nax, Nel, na_transmit, returnToMatlabFreq, bytesPerElementRead, bytesPerElementSent, numTunableParameters, bf_idx];

% write as int16
write(t,msg,'int16');

%read acknowledgement
disp('Reading ...')
T = 1;
bytesPerElementRead_ack = 1; %uint8
total = 0;
i = 0; %used for timeout

while  total < T

    while t.BytesAvailable == 0
        % disp('waiting for data');
        i=i+1;
        if i == 1e8; return; end
    end


    tsb = read(t, t.BytesAvailable, 'uint8');
    total = length(tsb)/bytesPerElementRead_ack; %normalize as we read as uint8 and not as int16

end
    
if tsb ~= 1 
   disp("Error: settings initialization went wrong")
    return
    keyboard
end





tStartLoop = tic; % loop timer for initial pass through to runAcq call

while vsExit==0

    if(hwEventHandler.didErrorEventOccur)
        vsv.util.events.EventReporter(hwEventHandler.getFirstErrorEvent);
        vsExit = 1;
    elseif (hwEventHandler.didWarnEventOccur)
        vsv.util.events.EventReporter(hwEventHandler.getLastWarnEvent);
        hwEventHandler.clearEvents();
    end
    drawnow
    try
        if freeze == 1
            % If running with the hardware, stop the hardware sequencer and
            % wait for freeze togglebutton to be pressed off.
            if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
            Control(n).Command = 'stopSequence';
            runAcq(Control); % NOTE:  If runAcq() has an error, it reports it then exits MATLAB.
            Control = struct('Command', [], 'Parameters', []);
            % We are about to enter a wait-for-unfreeze.  During that time, no commands besides
            % callbacks are available.  For callers that require the ability to send a command during
            % the time VSX is frozen, we implement a MATLAB Timer.
            %
            % If the caller sets the variable Mcr_FreezeTimerFunction to the name of their callback
            % function, then a timer will be created to periodically call that callback, thus allowing
            % the injection of commands while frozen, such as unfreeze.  (This would be the case for a
            % caller who has hidden the VSX GUI control panel, and thus the unfreeze button, with their
            % own custom GUI control panel that uses only "pull" technology rather than VSX GUI control
            % panel's "push" technology.  "Pull" here means that case where VSX is running separately
            % from a custom control panel and VSX is asking the custom control panel for any commands
            % to execute.
            %
            % NOTE:  Because the timer works within the MATLAB single-threaded
            % environment, it cannot guarantee execution times or execution rates.
            if((true == exist('Mcr_FreezeTimerFunction', 'var')) && (0 == strcmp('', Mcr_FreezeTimerFunction)))
                % OK, we DO have a callback hook to call from a Timer.
                %
                % We want to call the user's callback SCRIPT, but Timers want to
                % call a FUNCTION that accepts (object,event).  So, we make an
                % anonymous function here that calls the user's script.
                functionHandle = @(object, event)evalin('base', Mcr_FreezeTimerFunction);
                freezeTimer = timer ...
                    ( ...
                    'BusyMode',      'queue', ...        % So communication is not lost
                    'ExecutionMode', 'fixedRate', ...
                    'StartDelay',    0, ...              % Call callback immediately
                    'Period',        0.25, ...           % Call again every 1/4 seconds
                    'TimerFcn',      functionHandle ... % User's callback
                    );

                start(freezeTimer);
            end
            
            % Wait for unfreeze
            if usingHIFUPlex
                waitfor(findobj('Tag','toggleFreeze'),'Value',0);
            elseif usingMultiSys
                if strcmp(RDMAconfig.rdmaRole,'secondary')
                    socket = Net.getSocketById(syncSocketId(1));
                    while (freeze==1)
                        disp('frozen')
                        try
                            cmd=socket.readString;
                            socket.writeString('C');
                            if ~isempty(cmd)
                                fprintf('Remote Cmd: %s',char(cmd));
                                cmd=socket.readString;
                                eval(char(cmd));%listen for unfreeze command
                                if (freeze==0)
                                    disp('let it go!'); %message of unfreeze
                                end
                            end
                        catch
                            disp('no message from primary')
                        end
                    end
                else  % Primary System
                    waitfor(findobj('String','Freeze'),'Value',0);
                    disp('unfreeze')
                end

            elseif usingApplication && vsxApplication.hasFreezeControl
                freezeControl = vsxApplication.getFreezeControl();
                freezeControl.waitForUnfreeze();
            else
                waitfor(findobj('String','Freeze'),'Value',0);
            end
            if exist('exit', 'var') % backward-compatibility
                vsExit = exit;
                clear exit
            end
            if vsExit==1 % might get set while in freeze.
                break;
            end

            % If we used a timer above, stop it and delete it.
            if(true == exist('freezeTimer', 'var'))
                % OK, we DO have a timer to stop and delete.
                stop(freezeTimer);
                delete(freezeTimer);
                clear freezeTimer;
            end

            % Reset startEvent, in case sequence contains return-to-Matlab
            % points prior to the end of sequence, where 'freeze' may have
            % been pressed. Sequence will be re-started after updating startEvent.
            if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
            Control(n).Command = 'startSequence';
        end
        % The following switch statement implements actions set by the default GUI controls.
        switch action
            case 'none'
            case 'tgc'
                TGC(nTGC).CntrlPts = [tgc1,tgc2,tgc3,tgc4,tgc5,tgc6,tgc7,tgc8];
                TGC(nTGC).Waveform = computeTGCWaveform(TGC(nTGC));
                if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                Control(n).Command = 'set&Run';
                Control(n).Parameters = {'TGC', nTGC, 'Waveform', TGC(nTGC).Waveform};
            case 'persist'
                % Set the value of persistence for the first Process structure with class = 'image, method =
                %  'imageDisplay' and 'displayWindow' = 1;
                flag = 0;
                if exist('Process','var')
                    % Find the first 'Image/imageDisplay' Process structure for displayWindow 1
                    for i = 1:size(Process,2)
                        if (strcmp(Process(i).classname,'Image'))&&(strcmp(Process(i).method,'imageDisplay'))
                            for j = 1:2:size(Process(i).Parameters,2)
                                if (strcmp(Process(i).Parameters{j},'displayWindow'))&&(Process(i).Parameters{j+1}==1)
                                    for k = 1:2:size(Process(i).Parameters,2)
                                        if strcmp(Process(i).Parameters{k},'persistLevel')
                                            Process(i).Parameters{k+1} = persist;
                                            if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                                            Control(n).Command = 'set&Run';
                                            Control(n).Parameters = {'Process',i,...
                                                                     'persistLevel',persist};
                                            flag = 1;
                                        end
                                    end
                                end
                            end
                        end
                        if flag==1, break, end
                    end
                end
            case 'pgain'
                % change processing gain factor for 1st 'image' Process structure with method "imageDisplay".
                for i = 1:size(Process,2)
                    if (strcmp(Process(i).classname,'Image'))&&(strcmp(Process(i).method,'imageDisplay'))
                        break;
                    end
                end
                if i <= size(Process,2)
                    if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                    Control(n).Command = 'set&Run';
                    Control(n).Parameters = {'Process',i,...
                                             'pgain',pgain};
                end
            case 'speed'
                % change speed of sound correction factor.
                Resource.Parameters.speedCorrectionFactor = speedCorrect;
                if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                Control(n).Command = 'set&Run';
                Control(n).Parameters = {'Parameters',1,...
                                         'speedCorrectionFactor',speedCorrect};
            case 'displayChange'
                % @Refactor this needs to change to make this more generic
                % vsxResearch GUi needs to implement the getDisplay
                % function
                if usingApplication && vsxApplication.hasDisplay()

                    for i = 1:numel(Resource.DisplayWindow)

                        % @ToDo this needs to be optimized and will be replaced in the
                        % future with a common scheme for displays
                        displaySettings.importStruct( Resource.DisplayWindow(i) );
                        displaySettings.Scale = scale;

                        % display @type vsv.vsx.display.AbstractDisplay
                        display = vsxApplication.getDisplay(i, displaySettings);
                        display.updateDisaplay(displaySettings);

                        % update resource parameter
                        Resource.DisplayWindow(i).imageHandle  = display.getImageHandle();
                        Resource.DisplayWindow(i).clrWindow    = 1;
                        Resource.ImageBuffer(i).lastFrame      = 0; % this clears the ImageBuffer
                    end

                else
                    for i = 1:numel(Resource.DisplayWindow)
                        % adjust Matlab display window for size, position or aspect ratio change.
                        if ~isempty(Resource.DisplayWindow(i).Type)&&strcmp(Resource.DisplayWindow(i).Type,'Matlab')
                            if i > 1, continue; end  % Don't modify any Matlab windows other than first.
                            axesHandle = get(Resource.DisplayWindow(i).figureHandle,'CurrentAxes');
                            cla(axesHandle); % this clears any objects attached to the old axes.
                            oldPos = get(Resource.DisplayWindow(i).figureHandle,'Position');
                            set(Resource.DisplayWindow(i).figureHandle, ...
                                'Position',[oldPos(1), ... % left edge
                                            oldPos(2), ... % bottom
                                            Resource.DisplayWindow(i).Position(3) + 100, ... % width
                                            Resource.DisplayWindow(i).Position(4) + 150]);    % height
                            set(axesHandle, 'Units','pixels', ...
                                            'Position',[60,90,Resource.DisplayWindow(i).Position(3),Resource.DisplayWindow(i).Position(4)]);
                            set(axesHandle, 'Units','normalized');  % restore normalized units for re-sizing window.
                            ymin = Resource.DisplayWindow(i).ReferencePt(3);
                            ymax = ymin + Resource.DisplayWindow(i).Position(4)*Resource.DisplayWindow(i).pdelta;
                            xmin = Resource.DisplayWindow(i).ReferencePt(1);
                            xmax = xmin + Resource.DisplayWindow(i).Position(3)*Resource.DisplayWindow(i).pdelta;
                            DisplayData = zeros(Resource.DisplayWindow(i).Position(4),Resource.DisplayWindow(i).Position(3),'uint8');
                            Resource.DisplayWindow(i).imageHandle = image(DisplayData,...
                                    'Parent',axesHandle, ...
                                    'Xdata',scale*[xmin,xmax], ...
                                    'YData',scale*[ymin,ymax]);
                            Resource.DisplayWindow(i).clrWindow = 1;
                            Resource.ImageBuffer(i).lastFrame = 0; % this clears the ImageBuffer
                            axis(axesHandle, 'equal', 'tight');
                            set(axesHandle,'FontSize',12); xlabel(axesHandle,axisname,'FontSize',14);
                        elseif strcmp(Resource.DisplayWindow(i).Type,'Verasonics')
                            % Set axes limits base on Resource.DisplayWindow(i).Orientation
                            if Trans.type < 2
                                % DisplayWindow is x,z oriented
                                vmin = Resource.DisplayWindow(i).ReferencePt(3); % vertical minimum is DisplayWindow.ReferencePt(3).
                                vmax = vmin + Resource.DisplayWindow(i).Position(4)*Resource.DisplayWindow(i).pdelta;
                                xmin = Resource.DisplayWindow(i).ReferencePt(1);
                                xmax = xmin + Resource.DisplayWindow(i).Position(3)*Resource.DisplayWindow(i).pdelta;
                            else
                                if ~isfield(Resource.DisplayWindow(i),'Orientation')
                                    % Close the hardware event handler.
                                    hwEventHandler.uninstallHandler();
                                    error(['VSX: Resource.DisplayWindow(%d).Orientation must be defined for a 2D Displaywindow',...
                                           'with Trans.type = 2\n'],i);
                                end
                                switch Resource.DisplayWindow(i).Orientation
                                    case 'xz'
                                        % DisplayWindow is x,z oriented
                                        vmin = Resource.DisplayWindow(i).ReferencePt(3);
                                        vmax = vmin + Resource.DisplayWindow(i).Position(4)*Resource.DisplayWindow(i).pdelta;
                                        xmin = Resource.DisplayWindow(i).ReferencePt(1);
                                        xmax = xmin + Resource.DisplayWindow(i).Position(3)*Resource.DisplayWindow(i).pdelta;
                                    case 'yz'  % DisplayWindow is y,z oriented
                                        vmin = Resource.DisplayWindow(i).ReferencePt(3);
                                        vmax = vmin + Resource.DisplayWindow(i).Position(4)*Resource.DisplayWindow(i).pdelta;
                                        xmin = Resource.DisplayWindow(i).ReferencePt(2);
                                        xmax = xmin + Resource.DisplayWindow(i).Position(3)*Resource.DisplayWindow(i).pdelta;
                                    case 'xy'  % DisplayWindow is x,y oriented
                                        vmin = Resource.DisplayWindow(i).ReferencePt(2);
                                        vmax = vmin + Resource.DisplayWindow(i).Position(4)*Resource.DisplayWindow(i).pdelta;
                                        xmin = Resource.DisplayWindow(i).ReferencePt(1);
                                        xmax = xmin + Resource.DisplayWindow(i).Position(3)*Resource.DisplayWindow(i).pdelta;
                                end
                            end

                            % Generate an event that the spatial position has changed.
                            spatialPositionEvent = SpatialPositionEvent(xmin, xmax, vmin, vmax);
                            VantageImageEvent.generateSpatialPositionEvent(Resource.DisplayWindow(i).imageHandle, spatialPositionEvent);
                        end
                    end
                end
            case 'rcvloop'
                % switch to/from receive data loop mode (Resource.Parameters.simulateMode = 2)
                if rloopButton == 1
                    Resource.Parameters.simulateMode = 2;
                    if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                    Control(n).Command = 'set&Run';
                    Control(n).Parameters = {'Parameters',1,'simulateMode',2,'startEvent',Resource.Parameters.startEvent};
                    simButton = 0;
                    set(findobj('String','Simulate'),'Value',0);
                else
                    if (VDAS==1) % restart sequence
                        Resource.Parameters.simulateMode = 0;
                        if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                        Control(n).Command = 'set&Run';
                        Control(n).Parameters = {'Parameters',1,'simulateMode',0,'startEvent',Resource.Parameters.startEvent};
                        simButton = 0;
                        set(findobj('String','Simulate'),'Value',0);
                    else % if no hardware, go back to simulate mode 1
                        Resource.Parameters.simulateMode = 1;
                        if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                        Control(n).Command = 'set&Run';
                        Control(n).Parameters = {'Parameters',1,'simulateMode',1,'startEvent',Resource.Parameters.startEvent};
                        simButton = 1;
                        set(findobj('String','Simulate'),'Value',1);
                    end
                end
            case 'simulate'
                % switch mode to/from simulate.
                if (simButton == 1)
                    Resource.Parameters.simulateMode = 1;
                    if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                    Control(n).Command = 'set&Run';
                    Control(n).Parameters = {'Parameters',1,'simulateMode',1,'startEvent',Resource.Parameters.startEvent};
                    if rloopButton == 1
                        rloopButton = 0;
                        set(findobj('String','Rcv Data Loop'),'Value',0);
                    end
                else
                    if (VDAS==1)
                        Resource.Parameters.simulateMode = 0;
                        if isempty(Control(1).Command), n=1; else n=length(Control)+1; end
                        Control(n).Command = 'set&Run';
                        Control(n).Parameters = {'Parameters',1,'simulateMode',0,'startEvent',Resource.Parameters.startEvent};
                    else
                        simButton = 1; % if no hardware, change state back to simulate.
                        set(findobj('String','Simulate'),'Value',1);
                        if Resource.Parameters.verbose>1
                            fprintf('VSX: Entering simulate mode since hardware is not present.\n');
                        end
                    end
                end
            case 'setTPC5hvMonitorLimits'
                import com.verasonics.hal.tpc.* % enable communication with Hal TPC functions
                com.verasonics.common.logger.Logger.logInfo('VSXsettingProfile5VoltageLimits()');
                if TPC(5).HVmonitorLimits(1) < TPC(5).HVmonitorLimits(2)
                    % monitoring is to be enabled so update limits first
                    % and then enable
                    status = Tpc.setProfile5MonitoredVoltageStartLimits(TPC(5).HVmonitorLimits(1), TPC(5).HVmonitorLimits(2));
                    if status ~= TpcResult.success
                        % we've had an error; shut down and exit with error
                        % message
                        rc = char(status);
                        vsv.hal.VantageHardware.cleanUp(Resource, TPC, Hardware, minTpcVoltage);
                        error(['VSX: error when setting TPC(5) HV monitor limits: ', rc, ' \n']);
                    end
                    TPC(5).HVmonitorEnabled = 1;
                else
                    % monitoring is to be disabled so don't bother updating
                    % limits
                    TPC(5).HVmonitorEnabled = 0;
                end
                % now update the enabled state
                status = Tpc.setProfile5MonitoredVoltageStartLimitsEnabled(TPC(5).HVmonitorEnabled); % set the enable state
                if status ~= TpcResult.success
                    % we've had an error; shut down and exit with error
                    % message
                    rc = char(status);
                    vsv.hal.VantageHardware.cleanUp(Resource, TPC, Hardware, minTpcVoltage);
                    error(['VSX: error when setting TPC(5) HV monitor enabled: ', rc, ' \n']);
                end
        end

        action = 'none';  % this prevents continued execution of same action.
        % if ~isempty(Control(n).Command), disp(Control), drawnow, end % TEST: Uncomment this line to print out Control commands.

        %% *** 20. VsUpdate calls within the run-time while loop

        % @Refactor this needs to be removed in future releases. but for
        % now we need to inform the application that some of the parameters
        % may have changed. Some scripts can set some parameters in the
        % workspace that the application is not aware of
        vsxApplication.updateControlParameter(Control);

        % Make a copy of Control, and clear it so any new commands will
        % be processed next time.
        % final import to make sure the sequence components
        if usingApplication
            % will merge the controls with the controls defined else where
            vsxApplication.exportControlStruct();
        end


        VSX_Control = Control;
        Control = struct('Command', [], 'Parameters', []);
        if ~isempty(VSX_Control) && ~isempty(VSX_Control(1).Command)
            if nloop < 1e6
                saved_updateFrameIndex(nloopUpdate) = nloop;
                nloopUpdate = nloopUpdate + 1;
            end
            % Note that on the first call to runAcq after VSX initialization,
            % Control.Command will be empty and so this code will not execute;
            % VsUpdate and TXEventCheck have already been called just before
            % entering the run-time while loop.

            % There is a Control.Command to be processed.
            % Create flags to track which update calls are needed, so they
            % will be done in the correct order and only once.
            updateTW = false;
            updateTX = false;
            updateReceive = false;
            updateSeqControl= false;
            checkTXLimits = false;
            computeHVMux = false;
            for ctlnum = 1:length(VSX_Control)
                if strcmp(VSX_Control(ctlnum).Command, 'update&Run')
                    for parnum = 1:length(VSX_Control(ctlnum).Parameters)
                        switch VSX_Control(ctlnum).Parameters{parnum}
                            case 'TW'
                                updateTW = true;
                                updateTX = true;
                                checkTXLimits = true;
                            case 'TX'
                                updateTX = true;
                                checkTXLimits = true;
                                computeHVMux = true;
                            case 'Receive'
                                updateReceive = true;
                                checkTXLimits = true;
                                computeHVMux = true;
                            case 'SeqControl'
                                updateSeqControl = true;
                                checkTXLimits = true;
                            case {'Event', 'Parameters', 'Trans'}
                                checkTXLimits = true;
                        end
                    end
                elseif strcmp(VSX_Control(ctlnum).Command, 'set&Run')
                    for parnum = 3:4:length(VSX_Control(ctlnum).Parameters)
                        if strcmp(VSX_Control(ctlnum).Parameters(parnum), 'startEvent')
                            checkTXLimits = true;
                        end
                    end
                end
            end
            % Now process the updates that have been identified.
            if updateTW; updateh('TW'); end
            if updateTX; updateh('TX'); end
            if updateReceive; updateh('Receive'); end
            if updateSeqControl; updateh('SeqControl'); end
            if VDASupdates && checkTXLimits
                % call TXEventCheck to process the updated structures in the
                % matlab workspace.  If TXEventCheck decides it has to change
                % any of the TPC.highVoltageLimit values, it will append the
                % update&Run(TPC) command to VSX_Control (VTS-787)
                TXEventCheckh(false); % VTS-1796 pass initialization status to TXEventCheck
            end
            if VDASupdates && isfield(Trans, 'HVMux') && computeHVMux
                % update Trans.HVMux.SHIAperture for HVMux scripts if needed
                Trans=computeHvMuxSHIAperture(Trans);
                assignin('base','Trans',Trans); % return modified Trans to base workspace
            end
            clear updateTW updateTX updateReceive updateSeqControl checkTXLimits ctlnum parnum
        end
    catch oopsy
        if (VDAS == 1)
           % This will ensure the hardware sequencer has been stopped,
           % disable external power supply if it was in use, and clear
           % runAcq
           vsv.hal.VantageHardware.cleanUp(Resource, TPC, Hardware, minTpcVoltage);
        end
        rethrow(oopsy)
    end
        
    % Call runAcq and time its execution.
    tElapsedSoftware = toc(tStartLoop);
    if nloop < 1e6
        saved_tElapsedSoftware(nloop) = tElapsedSoftware;
    end
    
    tStartLoop = tic;
    % NOTE:  If runAcq() has an error, try/catch below will allow closing
    % HW before reporting error and exiting.
    try
        if (~vsExit) %added by BWC for multisys compatibility
            runAcq(VSX_Control);
        end
    catch msg

        % Set the flag that says an exception occurred.
        isHwInFaultState = true;

        % Be sure the TPC power is turned off because an error occurred.
        if (VDAS == 1)
           % This will ensure the hardware sequencer has been stopped,
           % disable external power supply if it was in use, and clear
           % runAcq
           vsv.hal.VantageHardware.cleanUp(Resource, TPC, Hardware, minTpcVoltage);
        end
        % Display the error message to the user.
        if hwEventHandler.didErrorEventOccur
            vsv.util.events.EventReporter(hwEventHandler.getFirstErrorEvent);
            vsExit = 1;
        elseif (hwEventHandler.didWarnEventOccur)
            vsv.util.events.EventReporter(hwEventHandler.getLastWarnEvent);
            hwEventHandler.clearEvents();
        else
            fprintf(2, 'RUNACQ Has reported the following error:\n\n');
            fprintf(2, "%s\n", getReport(msg));
            disp(' ');
            % Close the hardware event handler.
            hwEventHandler.uninstallHandler();
            return
        end
    end
    tElapsedRunAcq = toc(tStartLoop);
    if nloop < 1e6
        saved_tElapsedRunAcq(nloop) = tElapsedRunAcq;
    end
    
    % Perform low pass filter on loop and runAcq Period times.
    % VTS-1818 Shorten time constant for quicker response with typical
    % scripts
    softwarePeriod = softwarePeriod * 0.94 + tElapsedSoftware * 0.06;
    runAcqPeriod = runAcqPeriod * 0.94 + tElapsedRunAcq * 0.06;

    if autoScriptTest
        try
            % While running auto script test increment the runcount variable,
            % and set exit to one to force VSX to quit when runcount reaches 25
            % iterations.  Also automatically implement some GUI control
            % changes when runcount reaches 10
            runcount = runcount+1;
            astUI = 1; % will be replaced by autoScriptTest, if desired
            changeAction = 0;

            % All sliders will be executed when runcount reaches 10 iterations
            if runcount == 10 && astUI == 1
                ControlNew = [];
                ControlNew.Command = 'update&Run';
                ControlNew.Parameters = {'PData','InterBuffer','ImageBuffer','DisplayWindow','Parameters',...
                    'Trans','Media','TX','Receive','TGC','TPC','Recon','Process'};

                SldrUI = findobj('Style','Slider');
                % Only VsStyle Sliders need to be tested.
                % Skip other sliders of vsx_gui
                SldrUI = SldrUI(contains({SldrUI.Tag},'User'));

                m = 2; % Control(1) is 'update&Run'
                for i = 1:length(SldrUI)
                    h = SldrUI(i);
                    if strcmp(get(h,'Visible'),'on')
                        % VTS-1513 Move slider to max for worst case testing
                        hMax = get(h,'Max');
                        set(h,'Value', hMax);
                        feval(get(h,'Callback'),h);
                    end
                    for k = 1:length(Control)
                        if strcmpi(Control(k).Command,'set&Run')
                            ControlNew(m) = Control(k);
                            m = m+1;
                        end
                    end
                    Control = struct('Command', [], 'Parameters', []); % for next callback
                end

                TGC(1).CntrlPts = [tgc1,tgc2,tgc3,tgc4,tgc5,tgc6,tgc7,tgc8];
                TGC(1).Waveform = computeTGCWaveform(TGC(1));

                % VsButtonGroup UI control
                if exist('UI','var')
                    % test all UI controls just once
                    for i = 1:length(UI)
                        if isfield(UI(i),'Control')
                            if ~isempty(UI(i).Control) && ~isempty(UI(i).Callback)
                                VsStyle = UI(i).Control{3};
                                switch VsStyle
                                    case 'VsSlider'
                                        if strfind(get(UI(i).handle(1),'String'),'Range')
                                            changeAction = 1;
                                        end
                                    case 'VsButtonGroup'
                                        event.NewValue = UI(i).handle(3);
                                        set(UI(i).handle(2),'Value',0);
                                        set(UI(i).handle(3),'Value',1);
                                        feval([UI(i).Control{1},'Callback'],UI(i).handle(1),event);
                                        %                             eval([UI(i).Control{1},'Callback(UI(',num2str(i),').handle(1),event);'])
                                end
                            end
                        end
                        for k = 1:length(Control)
                            if strcmpi(Control(k).Command,'set&Run')
                                ControlNew(m) = Control(k);
                                m = m+1;
                            end
                        end
                        Control = struct('Command', [], 'Parameters', []); % for next callback
                    end
                    Control = ControlNew;
                end

                % in case the action is changed by other callbacks. In the
                % autoscript, only displayChange is required.
                if changeAction
                    action = 'displayChange';
                end
            end
            if runcount > 25
                vsExit = 1;
            end
        catch oopsy
            % Set the flag that says an exception occurred.
            isHwInFaultState = true;

            if (VDAS == 1)
               % This will ensure the hardware sequencer has been stopped,
               % disable external power supply if it was in use, and clear
               % runAcq
               vsv.hal.VantageHardware.cleanUp(Resource, TPC, Hardware, minTpcVoltage);
            end
            rethrow(oopsy)
        end
    end % End of the automatic control changes used only for autoScriptTest

    if freeze==1    % if freeze got set by runAcq, set state of togglebutton.
        if usingHIFUPlex
            set(findobj('Tag','toggleFreeze'),'Value',1,'String','RESTART','enable','on');
        else
            set(findobj('String','Freeze'),'Value',1);
        end
    end
    if exist('exit', 'var') % backward-compatibility
        vsExit = exit;
        clear exit
    end
    
    nloop = nloop + 1;
    
end  % end of the VSX runtime while loop

%% *** 21. Closing VSX: Copy buffers back to Matlab workspace.

% Attempt to copy buffers UNLESS there was a hardware fault.
% If a hardware fault occurred, then runAcq is no longer running and cannot
% perform the copyBuffers function.
if ~isHwInFaultState
    % The copyBuffers command also stops the hardware sequencer.
    Control(1).Command = 'copyBuffers';
    runAcq(Control); % NOTE: If runAcq() has an error, it reports it then exits MATLAB.
end

% - Multisystem Cleanup
if usingMultiSys
    if strcmp(RDMAconfig.rdmaRole,'primary')
        % Save rcvData Buffer from primary system
        save(['RcvData_M' num2str(RDMAconfig.nodeIndex) '.mat'],'RcvData','-V7.3');
        
    elseif strcmp(RDMAconfig.rdmaRole,'secondary')
        % Save rcvData Buffer from secondary systems
        fname = ['RcvData_S' num2str(RDMAconfig.nodeIndex) '.mat'];
        save(['RcvData_S' num2str(RDMAconfig.nodeIndex) '.mat'],'RcvData','-V7.3');
    end
    Net.release; %close tcp sockets used in multisys
end

if (VDAS == 1)
   % This will ensure the hardware sequencer has been stopped,
   % disable external power supply if it was in use, and clear
   % runAcq
   vsv.hal.VantageHardware.cleanUp(Resource, TPC, Hardware, minTpcVoltage);
end

if Resource.Parameters.verbose>1
    % Print out frame rate estimate.
    if ~exist('frameRateFactor','var')
        fprintf('runAcq sequence rate Hz = %3.2f\n', 1/runAcqPeriod)
        fprintf('Overall software sequence rate Hz = %3.2f\n', 1/softwarePeriod)
    else
        fprintf('runAcq frame rate Hz = %3.2f\n', frameRateFactor/runAcqPeriod)
        fprintf('Overall software frame rate Hz = %3.2f\n', frameRateFactor/softwarePeriod)
    end
    if Resource.Parameters.verbose>2
        fprintf('runAcq sequence period msec = %3.1f\n', 1000 * runAcqPeriod)
        fprintf('Overall software sequence period msec = %3.1f\n', 1000 * softwarePeriod)
    end
end
% If a LogData record size was specified, convert the LogData file.
% *** Note: The called routine is platform specific.
if (ismac && shouldConvertLogData), convertLogData; end

% Close the hardware event handler.
hwEventHandler.uninstallHandler();

% If logging was enabled, check for warnings and print the results.
vsxLogger = vsv.common.util.Logger;
numWarnings = vsxLogger.getNumWarnings();
if(numWarnings > 0)
    if (numWarnings  <= 25 )
        fprintf('There were %d runtime warnings:\n', numWarnings);
    else
        fprintf('There were %d runtime warnings. The first 25 are:\n', numWarnings);
    end
    vsxLogger.printWarnings();
end

% indicate to the vsx application that acquisition finished
if usingApplication
    vsxApplication.acquisitionFinished();
end

return
