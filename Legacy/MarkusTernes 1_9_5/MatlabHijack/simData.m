close all
clear all
clc
% dIdVexperiment - A class to represent a dI/dV experiment with predefined parameters and values.
%HIJACKS MARKUS TERNES CODE AND ALLOWS TO RUN IT IN MATLAB!!

%In order to not click enter when Spin-Gui is called you need to change
%the line 4028: answ=x_dialog('Script for execution',scrpt); to
%answ=scrpt;

%The following code expamples

muBohr=0.057883818;        %[meV/T]
kBoltzmann=0.0861733;    %[meV/K]

%%%%INITIALIZATION OF THE dIdV CLASS
exp = dIdVexperiment(1);
exp.xrange = 30; exp.B = [0 0 10]; exp.T = 0.01.*kBoltzmann;

%%%SETS THE ATOM SPIN PARAMETERS
exp.atom = struct('S', 1, 'g', 2, ...
    'D', -3, 'E', 0, ...
    'J', -0.04, 'U', 0.2, 'w', 20);

%%%%OPENS SPIN-GUI AND IMPORTS THE DATA
[exp.Bias, exp.dIdV] = exp.execSpinGui();

%%% PLOT THE RESULTS
exp.plotdIdV()
%%
% Brange = 0:1:10; i = 0;
% for B = Brange
%     i = i + 1;
%     exp.B = [0 0 B];
%     [Bias, dIdV(:,i)] = exp.execSpinGui();
% end
%
% exp.plot2Ddata(Bias,Brange,dIdV',"B (T)")

%%
close all
clear all
clc

muBohr=0.057883818;        %[meV/T]
kBoltzmann=0.0861733;    %[meV/K]
%%%%INITIALIZATION OF THE dIdV CLASS
exp = dIdVexperiment(2);
exp.xrange = 30; exp.B = [0 0 0]; exp.T = 0.5.*kBoltzmann;

exp.atom(1).S = 1; exp.atom(1).D = -1.5; exp.atom(1).U = 0.2;
exp.atom(2).S = 0.5; exp.atom(2).U = 0.4;
exp.heisenberg_coupling = 14;

[exp.Bias, exp.dIdV] = exp.execSpinGui();
exp.plotdIdVAll()
