
% M [NxN] double             : intput data
% ChannelLabels [1xN] double : channel numbers
% sampleRange [1x2] integer  : cropp off the rows of the M matrix
% linePlotIdxs [1x2] integer : sample numbers for a line overlay

function [] = iusPlotSignalsStack( M, ChannelLabels, sampleRange, linePlotIdxs, revision )
% IUSPLOTSIGNALSTACK plots normalized signal sensor traces in one figure

if nargin < 2
    ChannelLabels = 1:size(M,2);
end

if nargin < 3 || isempty(sampleRange)
    sampleRange = 1:size(M,1);
end

if nargin < 4
    linePlotIdxs = [];
end

if nargin < 5
    revision = [];
end

if numel(sampleRange) == 2
    sampleRange = min(sampleRange):max(sampleRange);
end

sM = M(sampleRange,:);

hRFlinePlot  = figure; set(hRFlinePlot,'Name',['Signal plot (stacked) ' '[' revision ']']);
offsetFactor = 2;
hold all;
    sMplot = sM/max(sM(:)) + ones(size(sM,1),1)*offsetFactor*(1:size(sM,2));
    plot(sMplot);
hold off;
set(get(hRFlinePlot,'Children'),'YTickMode','manual');
set(get(hRFlinePlot,'Children'),'YTickLabelMode','manual');

yticks = (1:size(sM,2))*offsetFactor;
if sum(size(sM,2)) > 20
	yticks         = yticks(1:5:end);
    yChannelLabels = ChannelLabels(1:5:end);
else
    yChannelLabels = ChannelLabels;
end

set(get(hRFlinePlot,'Children'),'YTick', yticks);
set(get(hRFlinePlot,'Children'),'YTickLabel', yChannelLabels);

if ~isempty(linePlotIdxs)
    hold on;
    xx = [1; 1]*linePlotIdxs(:)';
    yy = repmat([min(sMplot(:)); max(sMplot(:))],1,numel(linePlotIdxs));
    plot(xx,yy,'Color','black','LineStyle','--'); 
    ylabel('channel index');
    xlabel('samples');
    axis tight;
    title(['Channels: ' num2str(size(sMplot,2))]);
end