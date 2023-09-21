% Description:
% This file is supplementary to the ultrasound BM/d toolbox (usbmd).
% Used to save recorded Verasonics data (for instance using the
% SetUpL11_4vFlashAngles.m script) to a .hdf5 file in usbmd format.
% The dataset can later be read using the usbmd toolbox. Just run
% this script after acquisition in the same workspace in Matlab.
%
% If output_dir is defined in the workspace the script will save to this
% directory.
%
% If filename is defined in the workspace the script will save with this
% filename.
%
% The script will generate a unique filename if the output directory by
% appending a number to the filename.
%
% The script assumes that every frame has the same transmit and receive
% events.

% Define the output directory
if exist('output_dir', 'var') && exist(output_dir, 'dir')
    disp('output_dir already defined in workspace')
else
    output_dir = input('Output directory: ', 's');
end

if exist('filename', 'var') && exist(filename, 'dir')
    disp('filename already defined in workspace');
else
    % Prompt the user for the filename
    filename = input('Enter the filename (without extension): ', 's');
end

output_dir = 'D:\Vincent_van_de_schaft\verasonics';

% Prompt the user for the filename
filename = input('Enter the filename (including extension): ', 's');

save_to_usbmd_format(output_dir, filename)