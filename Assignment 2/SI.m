%% Setting up a signal over here.
SN = 4551117;
ts = .01; % s
fs = 1/ts; % Hz
tin = 0:ts:20;
len = round(length(tin)/2);
in_amp = 0;
fin_amp = 100;

r_up = linspace(in_amp,fin_amp, round(length(tin)/3)-1)';
r_down = -linspace(-fin_amp,-in_amp, round(length(tin)/3)-1)';
zed = zeros(round(length(tin)/6), 1);

amp = [zed;r_up;r_down;zed;0];
% uin = amp.*sin(5.*tin)';
 uin = 500*[ones((len-1)/2, 1); -ones((len-1)/2, 1); ones((len-1)/2, 1); -ones((len-1)/2+1, 1);];
% uin  = 10000*rand(length(tin),1);
% uin = 100*[zeros(len-1, 1); ones(len, 1)];
% uin = 100*[zeros(len-1, 1); zeros(len, 1)];
% uin  = 10*(sin(3*pi*tin) + sin(10*pi*tin) + sin(100*pi*tin) + sin(50*pi*tin));
%% Data Collection
y_raw = exciteSystem(SN,uin,fs);
ys = y_raw;  % spiked data
ylen = length(y_raw);  % length of data 
spike_idx = find(y_raw == max(y_raw));  % indices of the spikes (assumed contant and max);
good_idx = find(y_raw ~= max (y_raw));  % good indexes, without spikes
y_ns = y_raw(good_idx);  % delete spikes from data.
y_ds = interp1(good_idx,y_ns,1:ylen,'spline')'; % interpolate missing data

mean(y_ds); % seems to be around 5.95;
var(y_ds); % seems to be around 1.29e+04.
plot(tin,y_ds, tin, max(y_ds)/max(uin)*uin)
%%
% Remove Delay (0.7451s, by visual inspection) %7451 %1490
idx = 74;
y = y_ds(idx:end) - 5.95;
u = uin(1:end-idx+1);
t = tin(1:end-idx+1);
plot(t, y, t, max(y)/max(u)*u);
%% 
s = 20; m = 1; l = 1; ndat = length(y); N = ndat; 
U = hankel(u(1:s*m),u(s*m:end));
Su = svd(U);    
semilogy(1:s, Su);
%%
% figure(1);
n = 2;
[A,B,C,D,x0,sv,~] = subspaceID(u,y,s,n, 'po-moesp');
[yhat, ~] = simsystem(A,B,C,D,x0,u);
plot(t,yhat, t,y);
%%
N_ = len; n = 2;
yt = y(1:N_); yv = y(N_+1:end);
ut = u(1:N_); uv = u(N_+1:end);

[At,Bt,Ct,Dt,x0,sv,Phi] = subspaceID(ut,yt,s,n,'po-moesp');
[yhat, xhat] = simsystem(At,Bt,Ct,Dt,x0,ut);
vaft = max(0, (1-norm(yt-yhat)^2/norm(yt)^2)*100);
xf = xhat(:,end);
[yhatv, ~]= simsystem(At,Bt,Ct,Dt,xf,uv);
vafv = max(0, (1-norm(yv-yhatv)^2/norm(yv)^2)*100);

figure(5);
plot(1:ndat,y, 'b' ,1:N_,yhat, 'r', N_+1:ndat,yhatv,'g');
title('Training and Validation Data.')
fprintf('VAF of training set: %2.5f \n', vaft);
fprintf('VAF of validation set: %2.5f \n', vafv);

function [A,B,C,D,x0,sv, Phi] = subspaceID(u,y,s,n,method)
%% Instructions:
% Implement your subspace ID methods here.
% Avoid duplicate code!
% Write the method specific code into the switch case!
% Use the following function inputs and outputs.

% Function INPUT
% u         system input (matrix of size N x m)
% y         system output (matrix of size N x l)
% s         block size (scalar)
% n         model order (scalar)
% method    method (string e.g. 'moesp')
%
% Function OUTPUT
% A         System matrix A (matrix of size n x n)
% B         System matrix B (matrix of size n x m)
% C         System matrix C (matrix of size l x n)
% D         System matrix D (matrix of size l x m)
% x0        Initial state (vector of size n x one)
% sv        Singular values (vector of size n x one)
[N, l] = size(y);
[~, m] = size(u);
switch method
    case 'moesp'
        U=hankel(u(1:s),u(s:N));
        Y=hankel(y(1:s),y(s:N));
        L =(triu(qr([U;Y]',0)))';
        L22 = L(s+1:2*s,s+1:2*s);
        [Un,Sn,~]=svd(L22);
    case 'pi-moesp'
        Y = hankel(y(1:l*s+s),y(l*s+s:end));
        U = hankel(u(1:l*s+s),u(l*s+s:end));
        U0s = U(1:s*l,:); Uss = U(s*l+1:end,:);
        Yss = Y(s*l+1:end,:);
        L = triu(qr([Uss;U0s;Yss]',0))';
        L32 =  L(2*s+1:3*s,s+1:2*s);
        [Un,Sn,~] = svd(L32);
    case 'po-moesp'
        Y = hankel(y(1:l*s+s),y(l*s+s:end));
        U = hankel(u(1:l*s+s),u(l*s+s:end));
        U0s = U(1:s*l,:); Uss = U(s*l+1:end,:);
        Y0s = Y(1:s*l,:); Yss = Y(s*l+1:end,:);
        L = triu(qr([Uss;U0s;Y0s;Yss]',0))';
        L32 =  L(3*s+1:4*s,s+1:3*s);
        [Un,Sn,~] = svd(L32);
end
sv = diag(Sn);

A=Un(1:s*l-l,1:n)\Un(l+1:s,1:n);
C=Un(1:l,1:n);
Phi=zeros(l*N, n + n*m + l*m);
for k=0:N-1
    Phi_1 = C*mypower(A,k);
    Phi_2 = zeros(1,n);
    for j = 0:k-1
        % Phi_2 = Phi_2 + kron(u(j+1)', C*A^(k-j-1));
        Phi_2 = Phi_2 + (u(j+1)*C*mypower(A,k-1-j));
    end
    % Phi_3 = kron(u(k+1)', eye(l));
    Phi_3 = u(k+1);
    Phi(k+1,:) = [Phi_1, Phi_2, Phi_3];
end
xBD = Phi\y;
x0 = xBD(1:n);
vecB = xBD(n+1:n+m*n);
vecD = xBD(n+m*n+1:end);
B = vecB; D = vecD;
end

function [Abar,Bbar,C,D,K,x0] = mypem(A,B,C,D,K,x0,y,u,maxiter)
%% Instructions:
% Implement your PEM method here.
% Use the following function inputs and outputs.

% Function INPUT
% A0        Initial guess for system matrix A (matrix of size n x n)
% B0        Initial guess for system matrix B (matrix of size n x m)
% C0        Initial guess for system matrix C (matrix of size l x n)
% D0        Initial guess for system matrix D (matrix of size l x m)
% K0        Initial guess for system matrix K (matrix of size n x l)
% x00       Initial guess for initial state (vector of size n x one)
% u         System input (matrix of size N x m)
% y         System output (matrix of size N x l)
% maxiter   Maximum number of iterations (scalar)
%
% Function OUTPUT
% Abar      Estimate of system matrix A (matrix of size n x n)
% Bbar      Estimate of system matrix B (matrix of size n x m)
% C         Estimate of system matrix C (matrix of size l x n)
% D         Estimate of system matrix D (matrix of size l x m)
% K         Estimate of system matrix K (matrix of size n x l)
% x0        Estimate of initial state (vector of size n x one)

n = size(A,1); l = size(C,1); m = size(B,2);
Abar = A - K*C;
Bbar = B - K*D;

for iter = 1:maxiter
% Simulate the system:
n = length(x0); N= length(u);
xhat = zeros(n,N+1);
xhat(:,1) = x0;
yhat = zeros(N,1);
for i = 1:N
    xhat(:,i+1) = Abar*xhat(:,i) + Bbar*u(i) + K*y(i);
    yhat(i)= C*xhat(:,i) + D*u(i);
end
% Find the error vector:
E = y-yhat;
J = sum(norm(E).^2);
if iter == 1
    fprintf('The Initial Cost Function Value is J = %2.4f \n', J);
end
if iter == maxiter
    fprintf('The Final Cost Function Value is J = %2.4f \n', J);
end
theta = matrices2theta(Abar,Bbar,C,D,K,x0);
% Partial Derivatives (For this full parameterization);
Psi = zeros(length(y), length(theta));
for r = 1: length(theta)
    dx0 = zeros(size(x0)); dBbar = zeros(size(Bbar)); dC = zeros(size(C));
    dD = zeros(size(D)); dK = zeros(size(K)); dAbar = zeros(size(Abar));
    if r <= n*n % Parameters of Abar
        col = ceil(r/n); row = rem(r,n);
        if row == 0
            row = n;
        end
        dAbar(row,col)=1;
    elseif r <= n*n + l*n % Parameters of C
        r_ = r-n*n;
        dC(r_) = 1;
    elseif r <= n*n + l*n + n % Parameters of x0
        r_ = r-n*n-l*n;
        dx0(r_) = 1;
    elseif r <= n*n + l*n + n + m*n % Parameters of Bbar
        r_ = r-n*n-l*n-n;
        dBbar(r_) = 1;
    elseif r <= n*n + l*n + n + m*n + l*m % Parameters of D
        r_ = r-n*n-l*n-n-n*m;
        dD(r_) = 1;
    else  % Parameters of K
        r_ = r-n*n-l*n-n-n*m-l*m;
        dK(r_) = 1;
    end
    
    n = length(x0); N= length(u);
    dxhat = zeros(n,N+1);
    dxhat(:,1) = dx0;
    dyhat = zeros(N,1);
    for i = 1:N
        dxhat(:,i+1) = Abar*dxhat(:,i) + dAbar*xhat(:,i) + ...
            dBbar*u(i) + dK*y(i);
        dyhat(i)= C*dxhat(:,i) + dC*xhat(:,i)+ dD*u(i);
    end   
   Psi(:,r) = -dyhat;
end
dJ = 2/N*Psi'*E;
Hhat = 2/N*(Psi'*Psi);
lambda = 100;
theta_ = theta - (Hhat + lambda*eye(length(theta)))\dJ;

[Abar,Bbar,C,D,K,x0] = theta2matrices(theta_,size(Abar),size(Bbar), ...
    size(C),size(D),size(K),size(x0));
end
end

function Z = mypower(X,y)
if y == 0
    Z = eye(size(X),'like',X([])); %always real
else
    % X and y can be sparse
    % Z = X^y for integer y. Use repeated squaring.
    % For example: A^13 = A^1 * A^4 * A^8
    p = abs(y);
    R = X;
    first = true;
    while p > 0
        if rem(p,2) == 1 %if odd
            if first
                Z = R;  % hit first time. D*I
                first = false;
            else
                Z = R*Z;
            end
        end
        p = fix(p/2);
        if p ~= 0
            R = R*R;
        end
    end
    if y < 0
        Z = inv(Z);
    end
end
end