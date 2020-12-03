clear; clf; clc;
%% A1
load('calibration.mat')
close all
% a)
% Average over all vectors
y_avg = sum(y,2)/7;
% Subtract average from measurements
e = y- y_avg;

% b)
% E[y_m,k] = E[\tau_k] + E[d/c] + E[b_m] + E[e_(m,k)]
% E[y_m,k] - E[\tau_k] - E[d/c] =  E[b_m] + 0
% E[e_m,k] = E[b_m];
biases = mean(e); % Expected Values, E[y], for each microphone based on error.

%c) Since b_m is deterministic, the variance of the error is
%going to be the variance of e_(m_k)
var_eps_m = var(e);
stds = diag(std(e));
% Notes: When we eventually need to choose wheights for each microphone,
% we should base them on the variance of each of them. Higher variances
% need to be weighed less than higher ones.

%d) choose microphone 1
[N, l] = hist(e(:,1),20);
Wb=l(2)-l(1); % Bin width
Ny = length(e); % Nr of samples
% bar(l, N/(Ny*Wb));


%% A2
load('experiment.mat')
maxiter = 100;
th_hat = zeros(length(y),3);
diagP = zeros(length(y),3);
% initial estimate of theta
th_hat0 = [.1 .6 0];
for k = 1:size(y,1)
    % non linear LS estimate 
    [th_hat(k,:),diagP(k,:)] = nls(y(k,:),biases,stds,th_hat0,maxiter,mic_locations);
end
plotresults(th_hat(:,1:2)',diagP(:,1:2),mic_locations')

%% A3

%% A4

%% Functions
function [th_hat, diagP] = nls(yk,biases,stds,th_hat0,maxiter,mic_locations)

W = inv(stds*stds');
th_hat = th_hat0';
i = 0;
while i <= maxiter   
ftheta = f(th_hat,mic_locations);
e = yk'-biases'-ftheta;

% Linearise the model around the current estimate
dF = Jacobian(th_hat, mic_locations);

% Solve a least squares problem to compute delta_theta
del_theta = (dF'*W*dF)\dF'*W*e;

% Update the estimate th_hat
th_hat = th_hat+ del_theta;

% Set i = i + 1 and check for convergence
i = i+1; 
%if del_theta/th_hat <= 10^(-9)
 %  break
% end
end
diagP = diag(inv(dF'*W*dF));
end

function dF = Jacobian(theta,mic_locations)
    c = 343; % speed of sound in [m/s]
    dF = [(theta(1)-mic_locations(:,1))./vecnorm(theta(1:2) -  mic_locations')'/c, ...
    (theta(2)-mic_locations(:,2))./vecnorm(theta(1:2) -  mic_locations')'/c, ...
    ones(7,1)];
end

function ftheta = f(theta,mic_locations)
    c = 343; % speed of sound in [m/s]
    ftheta =(theta(3) + vecnorm(theta(1:2)-mic_locations')/c)';
end
