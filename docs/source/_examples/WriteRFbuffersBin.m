%WriteRFbuffersBin  Save Verasonics RF buffers as raw .bin files + one .mat.
%
% Replaces the slow '-v7.3' save of the full RcvData: each RF buffer is written
% to a raw int16 binary file, and only the (small) dimensions and metadata
% needed to reconstruct it are stored in a single Parameters.mat next to the
% binaries.
%
% OUTPUT FORMAT (read by zea's Verasonics converter,
% zea/data/convert/verasonics.py):
%   <RF_data_datetime>/
%     Parameters.mat            all base-workspace variables (Trans, TX, TW,
%                               Receive, Event, SeqControl, Resource, TGC, ...)
%                               EXCEPT the bulky RcvData/IQ/image buffers, plus:
%       RF_rows   (1,nBuf)      fast-time samples per buffer (RcvData rows)
%       RF_cols   (1,nBuf)      number of SAVED channels per buffer
%       RF_frames (1,nBuf)      frames per buffer
%       NonzeroRFcolumns {1,nBuf}  per-buffer logical channel mask; its length
%                               is the full hardware-channel count and it is
%                               true for every channel written to the binary.
%       Resource, ImgDataP      re-added (the converter reads frame order from
%                               Resource and the reference image from ImgDataP).
%     RF_data_<k>.bin           buffer k (1-based, matching RcvData{k}) as int16
%                               in MATLAB column-major order
%                               (rows x savedChannels x frames).
%   A buffer that is entirely zero (allocated but never acquired) is NOT written,
%   so the converter treats it as "present but not saved" rather than reading a
%   zero buffer.
%
% When to call:
%   1. After the acquisition and after VSX exits - all buffers are then in the
%      base workspace.
%   2. From a UI pushbutton during runtime.
%   3. From an external-function call in the event loop.
%   In cases 2/3 the buffers are copied into this function's workspace first via
%   runAcq('copyBuffers'). We detect case 1 by checking whether RcvData already
%   exists in the base workspace.
%
% Wiring a save button in a Verasonics setup script:
%   import vsv.seq.uicontrol.VsButtonControl
%   UI(n).Control  = VsButtonControl('LocationCode','UserB6','Label','Save RF');
%   UI(n).Callback = @WriteRFbuffersBin;

function [] = WriteRFbuffersBin(varargin)
% varargin absorbs the UI action a Verasonics pushbutton passes to its callback.

SaveIQ = false;   % also store IData/QData (larger files); RF is always saved.

tic;

% --- Obtain the buffers -------------------------------------------------------
% After VSX exits the buffers are in the base workspace; during a live
% acquisition (UI button / external function) they must be copied here first.
% ImgData/ImgDataP/IData/QData are optional: not every sequence produces them.
if evalin('base', 'exist(''RcvData'',''var'')')
    RcvData  = evalin('base', 'RcvData');
    Resource = evalin('base', 'Resource');
    [ImgData, hasImgData] = getBaseVar('ImgData');
    [ImgDataP, hasImgDataP] = getBaseVar('ImgDataP');
    if SaveIQ
        [IData, hasIData] = getBaseVar('IData');
        [QData, hasQData] = getBaseVar('QData');
    end
else
    % copyBuffers populates RcvData/IData/QData/ImgData/ImgDataP in THIS
    % function's workspace (not the base workspace).
    Control.Command = 'copyBuffers';
    runAcq(Control);
    % Resource exists both here and in base with different fields; merge them.
    Resource = MergeResourceStructs(Resource);
    hasImgData = exist('ImgData', 'var') == 1;
    hasImgDataP = exist('ImgDataP', 'var') == 1;
    if SaveIQ
        hasIData = exist('IData', 'var') == 1;
        hasQData = exist('QData', 'var') == 1;
    end
end

% --- Create the save folder (named by date-time of the save) ------------------
CurrentDatestr = datestr(now, 'dd-mmmm-yyyy_HH-MM-SS'); %#ok<TNOW1,DATST>
save_folder_path = fullfile(pwd, sprintf('RF_data_%s', CurrentDatestr));
if ~exist(save_folder_path, 'dir')
    mkdir(save_folder_path);
end
RFfilename_data = fullfile(save_folder_path, 'Parameters.mat');

% --- Save all other workspace variables (metadata) ----------------------------
% Everything in base EXCEPT the bulky data buffers is written with '-v7.3' (HDF5,
% required by the converter). The buffers themselves go to the .bin files and are
% re-added below. For dynamic acquisitions you could instead save only the
% structures that change; saving all is the simplest, most robust default.
assignin('base', 'RFfilename_data', RFfilename_data);
evalin('base', ...
    "save(RFfilename_data,'-regexp','^(?!(RcvData|IData|QData|ImgData|ImgDataP|Resource)$).*$','-v7.3')");

% --- Write each RF buffer to its own binary file ------------------------------
nBuffers = numel(RcvData);
RF_rows = zeros(1, nBuffers);
RF_cols = zeros(1, nBuffers);
RF_frames = zeros(1, nBuffers);
NonzeroRFcolumns = cell(1, nBuffers);

for k = 1:nBuffers
    % A probe with fewer elements than channels leaves zero-padded columns; only
    % channels that contain data are saved. The mask also records the full
    % hardware-channel count for reconstruction.
    NonzeroRFcolumns{k} = squeeze(any(any(RcvData{k}, 1), 3));

    [RF_rows(k), ~, RF_frames(k)] = size(RcvData{k});
    RF_cols(k) = sum(NonzeroRFcolumns{k});

    % Skip buffers that were allocated but never acquired (all zeros): writing a
    % 0-column binary would only confuse the converter.
    if RF_cols(k) == 0
        warning('WriteRFbuffersBin:emptyBuffer', ...
            'RF buffer %d is empty (all zeros); not writing a .bin for it.', k);
        continue
    end

    FN = fullfile(save_folder_path, sprintf('RF_data_%d.bin', k));
    [fid, msg] = fopen(FN, 'w');
    if fid < 0
        error('WriteRFbuffersBin:fopen', 'Could not open %s for writing: %s', FN, msg);
    end
    fwrite(fid, RcvData{k}(:, NonzeroRFcolumns{k}, :), 'int16');
    fclose(fid);
end

% --- Append the per-buffer dimensions and re-add the kept buffers --------------
saveVars = {'RF_rows', 'RF_cols', 'RF_frames', 'NonzeroRFcolumns', 'Resource'};
if hasImgData,  saveVars{end + 1} = 'ImgData';  end %#ok<*AGROW>
if hasImgDataP, saveVars{end + 1} = 'ImgDataP'; end
if SaveIQ && hasIData, saveVars{end + 1} = 'IData'; end
if SaveIQ && hasQData, saveVars{end + 1} = 'QData'; end
save(RFfilename_data, saveVars{:}, '-append');

fprintf('The RF data has been saved at %s in %.2f seconds\n', RFfilename_data, toc);
end


function [val, present] = getBaseVar(name)
%getBaseVar  Fetch a base-workspace variable, reporting whether it exists.
present = evalin('base', sprintf('exist(''%s'',''var'')', name)) == 1;
if present
    val = evalin('base', name);
else
    val = [];
end
end


function Resource = MergeResourceStructs(Resource)
%MergeResourceStructs  Merge this function's Resource with the base one.
% After copyBuffers the function-workspace Resource carries the buffer frame
% bookkeeping (first/last frame per buffer) while the base Resource carries the
% acquisition definition; the converter needs both.
Resource_func = Resource;
Resource = evalin('base', 'Resource');

mergestructs = @(x, y) cell2struct( ...
    [struct2cell(x); struct2cell(y)], [fieldnames(x); fieldnames(y)]);

if isfield(Resource_func, 'InterBuffer')
    Resource.InterBuffer = mergestructs(Resource.InterBuffer, Resource_func.InterBuffer);
end
if isfield(Resource_func, 'ImageBuffer')
    Resource.ImageBuffer = mergestructs(Resource.ImageBuffer, Resource_func.ImageBuffer);
end
if isfield(Resource_func, 'DisplayWindow')
    Resource.DisplayWindow = mergestructs(Resource.DisplayWindow, Resource_func.DisplayWindow);
end
if isfield(Resource_func, 'RcvBuffer')
    Resource.RcvBuffer = mergestructs(Resource.RcvBuffer, Resource_func.RcvBuffer);
end
end
