% usbmd_PulseEchoStatistics.m
%
% Reading a pulse-echo file made by SetUp_record_pulse_echos.m, and then
% computes and returns some basic pulse-echo statistics in a structure.
%
% Usage:
%
%    statistics = usbmd_PulseEchoStatistics( inputFile, depthROI, showPlots );
%
%    inputFile  : Optional full filename of input .mat file. When omitted
%                 or [], a dialog box will appear to select the mat file.
%    depthROI   : Two-element array with start and end index corresponding 
%                 to the depth range of interest. When omitted or when []
%                 the user will get an image and the request to click on
%                 the start and end depth range.
%    showPlots  : Optional boolean. When true, plots are created. When
%                 omitted or [], default value is false.

function statistics = usbmd_PulseEchoStatistics( inputFile, depthROI, showPlots )

    usbmd_Globals
    
    %======================================================================
    % Deal with input arguments
    %======================================================================
    if ~exist('showPlots','var')
        showPlots = [];
    end
    if ~exist('depthROI','var')
        depthROI = [];
    end
    if ~exist('inputFile','var')
        inputFile = [];
    end
    
    if isempty(showPlots)
        showPlots = false;
    end
    if isempty(inputFile)
        [inputFile, inputPath] = uigetfile(fullfile(usbmd_g_DataSaveDir,'*_allResponses.mat'));
    end
    
    %======================================================================
    % Settings (customizable)
    %======================================================================
    usbmd_Globals
    plot_main_diagonal_only = false;
    plot_3D_data_using_bar3 = false;
    
    %======================================================================
    % Read input data and perform a few assertions.plot_main_diagonal_only
    %======================================================================
    % inputFile = 'D:\Verasonics\Vantage-4.8.4-2211151000\Harm\data\20231016_142738_allResponses.mat';
    % inputFile = 'D:\Verasonics\Vantage-4.8.4-2211151000\Harm\data\20231016_144244_allResponses.mat';
    load(fullfile(inputPath,inputFile), 'M', 'Fs', 'Trans');
    num_elements  = size(M,1);
    num_transmits = size(M,3);
    assert(num_elements >= 2, 'You cannot apply usbmd_PulseEchoStatistics.m on recordings with less than 2 elements.');
    assert(num_elements == Trans.numelements, 'First dimension of matrix M is not equal to Trans.numelements.');
    assert(num_elements == num_transmits, 'Expected as many transmit events as receivers');
    
    exampleChannels = round([1/num_elements, 0.25, 0.5, 0.75, 1] * num_elements);
exampleChannels = [1 21 41 50 
    if strcmp(Trans.name,'custom')
        imageCaption = sprintf('probe = %s', Trans.secondName);
    else
        imageCaption = sprintf('probe = %s', Trans.name);
    end
    
    %======================================================================
    % Ask the user for the depth range of interest
    %======================================================================
    if isempty(depthROI)
        T = sum(M,3);
        figure, plot(T')
        title('Click left and right around the first acoustic reflection')
        drawnow
        [xx,~] = ginput(2);
        depthROI = sort(round(xx));
        depthROI(2) = depthROI(1) + 350;
        close
    end
    
    %======================================================================
    % Compute all envelopes
    %======================================================================
    M_env = zeros(num_elements, depthROI(2)-depthROI(1)+1, num_transmits);
    for t = 1 : num_transmits
        for e = 1 : num_elements
            ir = squeeze( M(e, depthROI(1):depthROI(2), t) );
            M_env(e,:,t) = abs(hilbert(ir));
        end
    end
    
    %======================================================================
    % Compute all statistics
    %======================================================================
    
    % peak-to-peak based on interpolated signal
    resultP2P = zeros(num_transmits, num_elements);  % peak-to-peak
    resultVar = zeros(num_transmits, num_elements);  % variance
    for t = 1 : num_transmits
        for e = 1 : num_elements
    
            s = M(e, depthROI(1):depthROI(2), t);   % signal
            s2 = interp(s, 8);                      % interpolated signal
    
            resultP2P(t, e) = max(s2) - min(s2);
            resultVar(t, e) = var(s2);
        end
    end
    
    % compute for all responses the (normalized) center frequency within
    % the user-indicated depth range
    resultPeakFreqMHz   = zeros(num_transmits, num_elements);
    resultCenterFreq6dB = zeros(num_transmits, num_elements);
    resultCenterFreq3dB = zeros(num_transmits, num_elements);
    resultBandWidth6dB  = zeros(num_transmits, num_elements);
    resultBandWidth3dB  = zeros(num_transmits, num_elements);
    Nfft = 8192;
    f    = linspace(0, Fs/2, 1+Nfft/2)';
    for t = 1 : num_transmits
        for e = 1 : num_elements
            s = M(e, depthROI(1):depthROI(2), t);  % signal
            S = fft(s, Nfft);
            S = S(1:1+Nfft/2);
            S = abs(S);
            idxPeak = find(S == max(S), 1, 'first');
            peakFreq = f(idxPeak);
            resultPeakFreqMHz(t,e) = peakFreq / 1e6;
            
            SdB = 20*log10(S+eps);
            maxVal = SdB(idxPeak);
            idx1 = find(SdB > maxVal-6, 1, 'first');
            idx2 = find(SdB > maxVal-6, 1, 'last');
            idx3 = find(SdB > maxVal-3, 1, 'first');
            idx4 = find(SdB > maxVal-3, 1, 'last');
            resultBandWidth6dB(t,e) = f(idx2)-f(idx1);
            resultBandWidth6dB(t,e) = resultBandWidth6dB(t,e) / 1e6;
            resultBandWidth3dB(t,e) = f(idx4)-f(idx3);
            resultBandWidth3dB(t,e) = resultBandWidth3dB(t,e) / 1e6;     
            
            centerFreq6dB            = f( round((idx1 + idx2) / 2) );
            resultCenterFreq6dB(t,e) = centerFreq6dB / 1e6;
            centerFreq3dB            = f( round((idx3 + idx4) / 2) );
            resultCenterFreq3dB(t,e) = centerFreq3dB / 1e6;
        end
    end
    
    statistics.inputFile          = inputFile;
    statistics.Trans              = Trans;
    statistics.sampleRateHz       = Fs;
    statistics.speed_of_sound     = usbmd_g_SpeedOfSound;
    statistics.start_index        = depthROI(1);
    statistics.end_index          = depthROI(2);
    statistics.start_depth_m      = usbmd_g_SpeedOfSound * depthROI(1) /Fs;
    statistics.end_depth_m        = usbmd_g_SpeedOfSound * depthROI(2) /Fs;
    statistics.peakToPeak         = resultP2P;
    statistics.peakToPeakdB       = 20*log10(resultP2P+eps);
    statistics.variance           = resultVar;
    statistics.variancedB         = 10*log10(resultVar+eps);
    statistics.peakFrequencyMHz   = resultPeakFreqMHz;
    statistics.centerFrequency3dB = resultCenterFreq3dB;
    statistics.centerFrequency6dB = resultCenterFreq6dB;
    statistics.bandWidth3dB       = resultBandWidth3dB;
    statistics.bandWidth6dB       = resultBandWidth6dB;
    
    %======================================================================
    % Plots
    %======================================================================
    if showPlots
        hPPvalueLin = figure;
        set(hPPvalueLin, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(1:num_elements, diag(statistics.peakToPeak))
            mean_val = mean(diag(statistics.peakToPeak));
            std_val = std(diag(statistics.peakToPeak));
            hold on
                line([1, num_elements], mean_val * [1 1], 'Color', ...
                    'red', 'LineStyle', '--');
            hold off
            xlabel('element index')
            ylabel('peak-to-peak value (lin)')
            ylim([0 2.5e4])
            title_txt = sprintf('intra-element peak-to-peak value, ($\\mu, \\sigma$) = (%.0f, %.1f)', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.peakToPeak), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; V(5) = 0; V(6) = 2^16; axis(V);
                zlabel('peak-to-peak value (lin)')
            else
                imagesc(statistics.peakToPeak)
                colorbar
                ax = gca;
                ax.CLim = [0.9e4 2.4e4];
            end
            xlabel('transmit event index')
            ylabel('element index')
            title('peak-to-peak value (lin)')        
        end
    
        hPPvalueLog = figure;
        set(hPPvalueLog, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(1:num_elements, diag(statistics.peakToPeakdB))
            mean_val = mean(diag(statistics.peakToPeakdB));
            std_val = std(diag(statistics.peakToPeakdB));
            hold on
                line([1, num_elements], mean_val * [1 1], 'Color', ...
                    'red', 'LineStyle', '--');
            hold off
            xlabel('element index')
            ylabel('peak-to-peak value (dB)')
            ylim([0 90])
            title_txt = sprintf('intra-element peak-to-peak value, ($\\mu, \\sigma$) = (%.1f, %.1f) dB', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.peakToPeakdB), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; V(5) = 0; V(6) = 100; axis(V);
                zlabel('peak-to-peak value (dB)')
            else
                imagesc(statistics.peakToPeakdB)
                colorbar
                ax = gca;
                ax.CLim = [75 90];
            end
            xlabel('transmit event index')
            ylabel('receive channel index')
            title('peak-to-peak value (dB)');
        end
    
        hVarianceLin = figure;
        set(hVarianceLin, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(1:num_elements, diag(statistics.variance));
            mean_val = mean(diag(statistics.variance));
            std_val = std(diag(statistics.variance));
            hold on
                line([1, num_elements], mean_val * [1 1], 'Color', 'red', 'LineStyle', '--');
            hold off
            V = axis; V(1) = 0; V(2) = num_elements; axis(V);
            xlabel('element index');
            ylabel('intra-element variance (lin)');
            title_txt = sprintf('intra-element variance, ($\\mu, \\sigma$) = (%.1f, %.1f)', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.variance), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; axis(V);
                zlabel('variance (lin)')
            else
                imagesc(statistics.variance)
                colorbar
            end
            xlabel('transmit event index')
            ylabel('receive channel index')
            title('variance (lin)')
        end
        
        hVarianceLog = figure;
        set(hVarianceLog, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(1:num_elements, diag(statistics.variancedB));
            mean_val = mean(diag(statistics.variancedB));
            std_val = std(diag(statistics.variancedB));
            hold on
                line([1, num_elements], mean_val * [1 1], 'Color', ...
                    'red', 'LineStyle', '--');
            hold off
            V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0; axis(V);
            xlabel('element index');
            ylabel('intra-element variance (dB)');
            title_txt = sprintf('intra-element variance, ($\\mu, \\sigma$) = (%.1f, %.1f) dB', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.variancedB), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; V(5) = 0; axis(V);
                zlabel('variance (dB)')
            else
                imagesc(statistics.variancedB)
                colorbar
            end
            xlabel('transmit event index')
            ylabel('receive channel index')
            title('intra-element variance (dB)')
        end
        
        hPeakFreq = figure;
        set(hPeakFreq, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(1:num_elements, diag(statistics.peakFrequencyMHz));
            mean_val = mean(diag(statistics.peakFrequencyMHz));
            std_val = std(diag(statistics.peakFrequencyMHz));
            hold on
                line([1,num_elements], mean_val * [1 1], 'Color', ...
                    'red', 'LineStyle', '--');
            hold off
            V = axis; V(1) = 0; V(2) = num_elements; axis(V);
            xlabel('element index');
            ylabel('peak frequency (MHz)');
            ylim([0 3])
            title_txt = sprintf('peak frequency, ($\\mu, \\sigma$) = (%.1f, %0.1f) MHz', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.peakFrequencyMHz), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; axis(V);
                zlabel('peak frequency (MHz)')
            else
                imagesc(statistics.peakFrequencyMHz)
                colorbar
                ax = gca;
                ax.CLim = [0 3];
            end
            xlabel('transmit event index')
            ylabel('receive channel index')
            title('peak frequency (MHz)')
        end
        
        hCenterFreq3dB = figure;
        set(hCenterFreq3dB, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(1:num_elements, diag(statistics.centerFrequency3dB));
            mean_val = mean(diag(statistics.centerFrequency3dB));
            std_val = std(diag(statistics.centerFrequency3dB));
            hold on
                line([1,num_elements], mean_val * [1 1], 'Color', 'red', 'LineStyle', '--');
            hold off
            V = axis; V(1) = 0; V(2) = num_elements; axis(V);
            xlabel('element index')
            ylabel('3dB center frequency (MHz)')
            ylim([0 3])
            title_txt = sprintf('3dB center frequency, ($\\mu, \\sigma$) = (%.1f, %.1f) MHz', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.centerFrequency3dB), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; axis(V);
                zlabel('3dB center frequency (MHz)')
            else
                imagesc(statistics.centerFrequency3dB)
                colorbar
                ax = gca;
                ax.CLim = [2.5 3];
            end
            xlabel('transmit event index')
            ylabel('receive channel index')
            title('3dB center frequency (MHz)');
        end
        
        hCenterFreq6dB = figure;
        set(hCenterFreq6dB, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(1:num_elements, diag(statistics.centerFrequency6dB));
            mean_val = mean(diag(statistics.centerFrequency6dB));
            std_val = std(diag(statistics.centerFrequency6dB));
            hold on
                line([1,num_elements], mean_val * [1 1], 'Color', ...
                    'red', 'LineStyle', '--');
            hold off
            V = axis; V(1) = 0; V(2) = num_elements; axis(V);
            xlabel('element index')
            ylabel('6dB center frequency (MHz)')
            ylim([0 3])
            title_txt = sprintf('6dB center frequency, ($\\mu, \\sigma$) = (%.1f, %.1f) MHz', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.centerFrequency6dB), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; axis(V);
                zlabel('6dB center frequency (MHz)')
            else
                imagesc(statistics.centerFrequency6dB)
                colorbar
                ax = gca;
                ax.CLim = [2.5 3];
            end
            xlabel('transmit event index')
            ylabel('receive channel index')
            title('6dB center frequency (MHz)');
        end
        
        hBandWidth3dB = figure;
        set(hBandWidth3dB, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(diag(statistics.bandWidth3dB));
            mean_val = mean(diag(statistics.bandWidth3dB));
            std_val = std(diag(statistics.bandWidth3dB));
            hold on
                line([1,num_elements], mean_val * [1 1], 'Color', 'red', 'LineStyle', '--');
            hold off
            V = axis; V(1) = 0; V(2) = num_elements; axis(V);
            xlabel('element index')
            ylabel('3dB bandwidth (MHz)')
            ylim([0 2])
            title_txt = sprintf('3dB bandwidth, ($\\mu, \\sigma$) = (%.1f MHz, %.1f)', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.bandWidth3dB), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; axis(V);
                zlabel('3dB bandwidth (MHz)')
            else
                imagesc(statistics.bandWidth3dB)
                colorbar
                ax = gca;
                ax.CLim = [0.5 2];
            end
            xlabel('transmit event index')
            ylabel('receive channel index')
            title('3dB bandwidth (MHz)')
        end
        
        hBandWidth6dB = figure;
        set(hBandWidth6dB, 'Name', imageCaption);
        if plot_main_diagonal_only
            bar(diag(statistics.bandWidth6dB));
            mean_val = mean(diag(statistics.bandWidth6dB));
            std_val = std(diag(statistics.bandWidth6dB));
            hold on
                line([1,num_elements], mean_val * [1 1], 'Color', ...
                    'red', 'LineStyle', '--');
            hold off
            V = axis; V(1) = 0; V(2) = num_elements; axis(V);
            xlabel('element index')
            ylabel('6dB bandwidth (MHz)')
            ylim([0 2])
            title_txt = sprintf('6dB bandwidth, ($\\mu, \\sigma$) = (%.1f MHz, %.1f)', mean_val, std_val);
            title(title_txt, 'Interpreter', 'latex')
        else
            if plot_3D_data_using_bar3
                bar3(statistics.bandWidth6dB), view([230 30])
                V = axis; V(1) = 0; V(2) = num_elements; V(3) = 0;
                V(4) = num_elements; axis(V);
                zlabel('6dB bandwidth (MHz)')
            else
                imagesc(statistics.bandWidth6dB)
                colorbar
                ax = gca;
                ax.CLim = [0.5 2];
            end
            xlabel('transmit event index')
            ylabel('receive channel index')
            title('6dB bandwidth (MHz)')
        end
    end
    
    hRFline = figure;
    set(hRFline, 'Name', imageCaption');
    for i = 1 : length(exampleChannels)
        idx = exampleChannels(i);
        s = squeeze(M(idx, depthROI(1):depthROI(2), idx));  % signal
        E = abs(hilbert(s));
        
        subplot(length(exampleChannels), 2, 2*i-1)
        plot(depthROI(1):depthROI(2),s)
        hold on
            plot(depthROI(1):depthROI(2),E,'g');
        hold off
        xlim([depthROI(1) depthROI(2)]);
        ylim([-1e4 1e4])
        title(['Example RF line within indicated depth range, rx=tx=', num2str(idx)])
        if i==length(exampleChannels)
           xlabel('sample index $n$','Interpreter','Latex')
        end
        txt = sprintf('$r_{%d}[n]$', idx);
        ylabel(txt, 'Interpreter', 'Latex');
        %print('-dpng',['im',num2str(rrr)]);
        
        f = linspace(0, Fs*1e-6, Nfft+1);
        f = f(1:Nfft/2+1);
        S = fft(s,Nfft);
        S = S(1:1+Nfft/2);
        S = abs(S);
        SdB = 20*log10(S+eps);

        idxPeak  = find(S == max(S), 1, 'first');
        peakFreq = f(idxPeak);
        maxVal   = SdB(idxPeak);
        idx1 = find(SdB > maxVal-6, 1, 'first');
        idx2 = find(SdB > maxVal-6, 1, 'last');
        bandWidth6dB = (f(idx2)-f(idx1));
        idx3 = round((idx1 + idx2) / 2);
        centerFreq6dB = f(idx3);
        
        subplot(length(exampleChannels), 2, 2*i)
        plot(f, SdB);
        hold on
            plot([f(idx1) f(idx1)], [0, SdB(idx1)], 'r:');
            plot(f(idx1), SdB(idx1), 'ro');
            plot([f(idx2) f(idx2)], [0, SdB(idx2)], 'r:');
            plot(f(idx2), SdB(idx2), 'ro');
            plot([f(idx3) f(idx3)], [0, SdB(idx3)], 'b:');
            plot(f(idx3), SdB(idx3), 'bo' )
        hold off
        ylim([-20 100])
        title_txt = sprintf('RF line spectrum, -6dB bandwidth = %.1f MHz, Fc = %.1f MHz, rx=tx=%d', bandWidth6dB, centerFreq6dB, idx);
        title(title_txt);
        if i==length(exampleChannels)
            txt = sprintf('frequency $\\Omega$ [MHz]');
            xlabel(txt, 'Interpreter', 'Latex');
        end
        txt = sprintf('$20 \\log_{10} |R_{%d}[\\Omega]|$', idx);
        ylabel(txt, 'Interpreter', 'Latex');
        xlim([0 Fs/2*1e-6]);
        %print('-dpng',['im',num2str(rrr)]);++
    end
    
    % plot RF line stack
    % iusPlotSignalsStack(M,chanIdxs,[],[ymin ymax],revision);

end