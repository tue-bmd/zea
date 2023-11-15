%VERA_TOOLS Various ultrasound lab tools for measurements using Verasonics
%
%   Before using any of the tools make sure to set the probe name and your
%   data directory name in ./common/usbmd_Globals.m
%
%   Sub-directories and tools:
%
%   ./rf_line_scope/   scope for live monitoring of RF-line data
%       SetUp_rf_line_scope.m      : call this script on the Verasonics to
%                                    get live RF line images plus spectra
%                                    for selected transmit - receive
%                                    element pairs. A good experiment
%                                    involves a hard-reflecting flat
%                                    surface in a water tank with
%                                    ultrasound probe above it at some
%                                    distance.
%
%   ./array_analysis/  tools to record and analyse pulse echo statistics
%       SetUp_record_pulse_echos.m : call this script on the Verasonics to
%                                    get on disk a recorded data file
%                                    containing pulse-echo signals from any
%                                    element to any element.
%       usbmd_PlotSignalStack.m    : plot the recorded pulse echo signals
%       compute_elements_to_channels_map: determine the Verasonics channel
%                                         map (in Trans.ConnectorES) for a
%                                         custom probe of which the channel
%                                         ordering is unknown. Input is the
%                                         data recorded by the first
%                                         script.
%
%   ./common./         contains globas and the usbmd_InitTrans.m

