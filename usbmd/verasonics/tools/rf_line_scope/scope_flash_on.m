% scope_flash_on.m   this script is called by SetUp_rf_line_scope.m
%                    it sets the apodization to 1 for all elements

function [] = scopeFlashOn()

% get stuff from base workspace
TX       = evalin( 'base', 'TX' );
Trans    = evalin( 'base', 'Trans' );

% set transmit aperture to all elements
ApodT      = ones(1,Trans.numelements);
TX(1).Apod = ApodT;

%-update and run
assignin( 'base', 'TX', TX );
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'TX'};
assignin('base','Control', Control);

end