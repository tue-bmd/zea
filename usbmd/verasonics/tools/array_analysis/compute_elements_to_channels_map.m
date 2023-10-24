% compute_elements_to_channels_map.m   uses data produced by the script SetUp_record_pulse_echos.m
%                                      to compute the mapping (i.e. Trans.ConnectorES) between array
%                                      elements and Verasonics channels.
%
% todo: think about migrating to the usbmd format

usbmd_Globals

filename = uigetfile(fullfile(usbmd_g_DataSaveDir,'*_allResponses.mat'));
load(fullfile(usbmd_g_DataSaveDir, filename));

num_elements  = size(M,1);
num_samples   = size(M,2);
num_transmits = size(M,3);
assert(num_elements == num_transmits, 'Expected as many transmit events as receivers');

% Ask user for a start and a stop index of the first acoustic reflection
T = sum(M,3);
figure, plot(T')
title('Click left and right around the first acoustic reflection')
drawnow
[xx,yy] = ginput(2);
xx = round(xx);

% Compute all envelopes
M_env = zeros(num_elements, xx(2)-xx(1)+1, num_transmits);
for t = 1 : num_transmits
    for e = 1 : num_elements
        ir = squeeze( M(e, xx(1):xx(2), t) );
        M_env(e,:,t) = abs(hilbert(ir));
    end
end

% Look for the first transducer element (that is the element with the
% earliest response to its own transmit)
intra_transducer_delays = zeros(size(M_env,1),1);
for n = 1 : num_elements
  ir_n = squeeze(M_env(n,:,n));
  N = length(ir_n);
  ir_n_2 = interp1(0:N-1, ir_n, 0:0.1:N-1,'spline');
  [~, intra_transducer_delays(n)] = max(ir_n_2);
end

min_delay = min(intra_transducer_delays);
idx_first_element = find(intra_transducer_delays == min_delay);
assert(length(idx_first_element)==1, 'More than one first element found, reposition the probe and try again.');

A = M_env(:,:,idx_first_element);
[~,max_indices] = max(A,[],2);
[~,sorted_indices] = sort(max_indices,'ascend');
assert(length(unique(sorted_indices)) == size(M,3),'number of unique indices does not correspond to the number of array elements')

NewConnectorES = Trans.ConnectorES(sorted_indices);

fprintf('Trans.ConnectorES = [');
for n = 1 : length(NewConnectorES)
    fprintf('%d ', NewConnectorES(n));
end
fprintf(']'';\n');

% BEFORE
Mdiag = zeros(num_elements,num_samples);
for n = 1:num_elements
    Mdiag(n,:) = M(n,:,n);
end
usbmd_PlotSignalsStack(Mdiag')
axis([0 num_samples 0 num_elements+1])

% AFTER
usbmd_PlotSignalsStack(Mdiag(sorted_indices,:)')
axis([0 num_samples 0 num_elements+1])
