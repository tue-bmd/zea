% scope_flash_off.m  this script is called by SetUp_rf_line_scope.m
%                    it sets the apodization to 1 for the single active
%                    element and to 0 for all other elements

function [] = scopeFlashOff()

% get stuff from base workspace
TX       = evalin( 'base', 'TX' );
Trans    = evalin( 'base', 'Trans' );
activeTX = evalin( 'base', 'activeTX' );

% set transmit aperture to single element
ApodT           = zeros(1,Trans.numelements);
ApodT(activeTX) = 1;
TX(1).Apod      = ApodT;

%-update and run
assignin( 'base', 'TX', TX );
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'TX'};
assignin('base','Control', Control);

end