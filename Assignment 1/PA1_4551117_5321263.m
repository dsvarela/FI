clear; close all; clc;
%% A1
load('calibration.mat')
close all
% a)
% Average over all vectors
y_avg = sum(y,2)/7;
% Subtract average from measurements
e = y - y_avg;

% b)
% E[y_m,k] = E[\tau_k] + E[d/c] + E[b_m] + E[e_(m,k)]
% E[y_m,k] - E[\tau_k] - E[d/c] =  E[b_m] + 0
% E[e_m,k] = E[b_m];
biases = mean(e); % Expected Values, E[y], for each microphone based on error.

%c) Since b_m is deterministic, the variance of the error is
%going to be the variance of e_(m_k)
vars = diag(var(e));
stds = diag(std(e));
% Notes: When we eventually need to choose wheights for each microphone,
% we should base them on the variance of each of them. Higher variances
% need to be weighed less than higher ones.

%d) choose microphone 1
[N, l] = hist(e(:,1),20);
Wb=l(2)-l(1); % Bin width
Ny = length(e); % Nr of samples
bar(l, N/(Ny*Wb));
title('Histogram Values for Microphone 1')

%% A2
load('experiment.mat')
maxiter = 100;
th_hat = zeros(length(y),3);
diagP = zeros(length(y),4);
% initial estimate of theta
th_hat0 = [.1 .6 0];
for k = 1:size(y,1)
    % non linear LS estimate
    [th_hat(k,:),diagP(k,:)] = nls(y(k,:),biases,vars,th_hat0,maxiter,mic_locations);
end
plotresults(th_hat(:,1:2)',diagP(:,1:2),mic_locations')

%% A3
[x_K,diagP_K] = Kalman(eye(2),eye(2),th_hat(:,1:2)',zeros(2),[0.1; 0.6],1.5,7*10^(-7),diagP);

figure(1)
plotresults(x_K,diagP_K,mic_locations')

figure(2)
subplot(211)
plot(x_K(1,:))
hold on
plot(th_hat(:,1))
legend({'KF','NLS'},'Location','northeast')
title('x Positions for the Robot')
hold off
subplot(212)
plot(x_K(2,:))
hold on
plot(th_hat(:,2))
legend({'KF','NLS'},'Location','northeast')
title('y Positions for the Robot')
hold off

%% A4
A = eye(3); B = [0 0 1]'; u = 0.5;
Q = diag([3*10^(-7), 2*10^(-7), 10^(-6)]);
R = vars;
diagPc = zeros(length(y),2);
diagPp = zeros(length(y),2);

% Predicted States
th_p4 = zeros(3,length(y));
% Corrected States
th_c4 = zeros(3,length(y));
% Vector to store values of the Kalman Filter for Question Five.
K = zeros(3,7);

th_0 = [.10;.60;0];
% Pc = zeros(3,3);
Pc = diag([1, 1, 1]);
% First Prediction
th_p4(:,1) = A*th_0 + B*u; % Predicted States
Pp = A*Pc*A' + Q; % Predicted Covariance Matrices
for k = 1:length(y)
    yk_ub = y(k,:)' - biases';
    [th_p4(:,k+1), th_c4(:,k), Pp, Pc] = ekf(yk_ub,Q,R,th_p4(:,k), Pp,mic_locations);
    diagPc(k,:) = [Pc(1,1), Pc(2,2)];
    diagPp(k,:) = [Pp(1,1), Pp(2,2)];
end

figure;
plotresults(th_p4(1:2,1:end-1),diagPp,mic_locations')

figure;
subplot(3,1,1)
plot(1:117, th_p4(1,1:end-1)', 1:117, th_hat(:,1), 'linewidth', 2);
title('x Positions for the Robot', 'fontsize', 16)
legend('EKF', 'NLS', 'fontsize', 14)
xlabel('Time Step [k]','fontsize', 14)
ylabel('x Position [m]', 'fontsize', 14)
subplot(3,1,2)
plot(1:117, th_p4(2,1:end-1)', 1:117, th_hat(:,2), 'linewidth', 2);
title('y Positions for the Robot', 'fontsize', 16)
legend('EKF', 'NLS', 'fontsize', 14)
xlabel('Time Step [k]','fontsize', 14)
ylabel('y Position [m]', 'fontsize', 14)
subplot(3,1,3)
plot(1:117, th_p4(3,1:end-1)', 1:117, th_hat(:,3), 'linewidth', 2);
title('Beep Time Estimates', 'fontsize', 16)
legend('EKF', 'NLS', 'fontsize', 14)
xlabel('Time Step [k]','fontsize', 14)
ylabel('Beep Time [s]','fontsize', 14)
%% Functions
function [x_K,diagP_K] = Kalman(A,C,y,P0,x0,Rscale,Qscale,diagP)
    x = zeros(length(x0),length(y));
    x(:,1) = x0;
    P = P0;
    Q = Qscale*eye(2);
    diagP_K = zeros(length(y),length(x0));
    for n = 2:length(y)
        R = Rscale*[diagP(n,1), 0; 0, diagP(n,2)];
        % predicted state estimate:
        x_pred = A*x(:,n-1);
        % predicted error covariance:
        P_pred = A*P*A'+Q;
        % Kalman gain:
        K = P_pred*C'/(R+C*P_pred*C');
        % updated error covariance:
        P = (eye(2)-K*C)*P_pred;
        diagP_K(n,:) = [P(1,1),P(2,2)];
        % updated state estimate:
        x(:,n) = x_pred+K*(y(:,n)-C*x_pred);
    end
    x_K = x;
end

function [th_p4_, th_c4, Pp_, Pc] = ekf(yk_ub,Q,R,th_p4,Pp,mic_locations)
A = eye(3); B = [0 0 1]'; u = 0.5;

% Compute Kalman gain:
H = Jacobian(th_p4, mic_locations);
K = Pp*H'/(H*Pp*H' + R); % K is 3x7, R is mic uncertainty = stds

% Correction of Predicted State (Measurement Update)
th_c4 = th_p4 + K*(yk_ub - f(th_p4, mic_locations));
Pc = Pp - K*H*Pp;

% Prediction For the Next Step(Time Update)
th_p4_ = A*th_c4 + B*u';
Pp_ = A*Pc*A' + Q;
end

function [th_hat, diagP] = nls(yk,biases,vars,th_hat0,maxiter,mic_locations)
th_hat = th_hat0';
i = 0;
while i <= maxiter
    ftheta = f(th_hat,mic_locations);
    e = yk'-biases'-ftheta;
    
    % Linearise the model around the current estimate
    dF = Jacobian(th_hat, mic_locations);
    
    % Solve a least squares problem to compute delta_theta
    del_theta = (dF'/vars*dF)\dF'/vars*e;
    
    % Update the estimate th_hat
    th_hat = th_hat + del_theta;
    
    % Set i = i + 1 and check for convergence
    i = i+1;
    if del_theta/th_hat <= 10^(-6)
        break
    end
end
diagP = [diag(inv(dF'/vars*dF));i];
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
