% Description:
% This file is supplementary to the ultrasound BM/d toolbox (usbmd).
% Used to save recorded Verasonics data (for instance using the
% SetUpL11_4vFlashAngles.m script) to a .hdf5 file in usbmd format.
% The dataset can later be read using the usbmd toolbox. Just run
% this script after acquisition in the same workspace in Matlab.
%
% Note that the dimensions of the arrays are all in reversed order. This is
% needed because MATLAB stores data in column-major order, while numpy
% works with row-major order. When viewed in numpy the order will be
% correct.
%
% The script assumes that every frame has the same transmit and receive
% events.

% Define the output directory
output_dir = 'D:\Vincent_van_de_schaft\verasonics';

% Prompt the user for the filename
filename = input('Enter the filename (including extension): ', 's');

save_to_usbmd_format(output_dir, filename)