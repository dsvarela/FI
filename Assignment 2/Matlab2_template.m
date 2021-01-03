clear; close all; clc;
load 'flutter.mat'
s = 100; m = 1; l = 1; N = 1014; ndat = 1014;
%% Exercise 1: Percistency of Excitation
Y = hankel(y(1:s*l),y(s*l:end));
U = hankel(u(1:s*m),u(s*m:end));

% Generate values from a normal distribution with mean avg and standard deviation sigma.
sigma = 1; %So that the 1st svd value is around the 1st svd of Hu
avg = 0;
wn = avg + sigma.*randn(1014,1);
WN = hankel(wn(1:s), wn(s:end));

Su = svd(U);
Swn = svd(WN);
figure(1);
semilogy(1:s, Su, 1:s, Swn)

% Your comments and answers:
%
%  Comparing both singular values, we see a far harsher drop in the singular values of the input signal compared to those of white noise. In fact, as
% the number of samples goes to infinity, it's to be expected that the SV of white noise would remain constant. 
%  The singular values of the input tend to drop rapidly, suggesting this
% signal is less persistently exciting than white noise. Still, for s = 100, the singular values have not dropped completely in the input
% signal, so we can safely say that the persistency of excitation of it is at least 100. Further confirmation for this comes in the fact that the
% Block Hankel Matrix U has rank n=100, which is in line with the definition of persistency of excitation.
%  However, even though the rank of the matrix is 100, we see from the SV plot that, at higher orders the information present in the signal
% becomes fairly low. In practice, this probably means that , even though the persistency of excitation, if viewed from the rank, is higher than 
% 100, the signal probably doesn't contain enough information in it to actually exicite the system for higher orders.

%% Exercise 2+3: MOESP, PI-MOESP, PO-MOESP
meth = {'moesp'; 'pi-moesp'; 'po-moesp'};
n = 20;
for i = 1:3
    
[A,B,C,D,x0,sv,~] = subspaceID(u,y,s,n, meth{i});
if i == 1 
    figure(2);
    semilogy(sv);
end
figure(3);
subplot(3,1,i);
[yhat, ~] = simsystem(A,B,C,D,x0,u);
plot(y, 'linewidth', 1.5); hold on;
plot(yhat, 'linewidth', 1.5); hold off;
title(sprintf('%s', upper(meth{i})), 'Fontsize', 20);
legend('y','yhat', 'Fontsize', 16)
vafb = max(0, (1-norm(y-yhat)^2/norm(y)^2)*100);
fprintf('For n = %d, %s yeilds a model with VAF = %2.5f. \n', n, upper(meth{i}), vafb)
end

% Your comments and answers:
%  Looking at the singular values plot, we find that the identifying a drop here can be quite dificult. The information present in the singular
% values seems to drop rather quickly (and to drop below 1) after about order 20, so that's the order chosen for this exercise.
%  Alternatively, we can look at a plot of the VAF as a function of the order n (not shown here, because it would take to long to recompute).
% In this case, we find that we get VAF values as high as 90% as soon as order n=3, though these seem to oscillate quite a bit up until
% n=30. After that, and up until order 50, we get a plateau. After n=50, we start running into numerical issues, likely caused by our limited s
% value, s=100, which ruin the fit (and VAF) of higher order models.
%  Nevertheless, the results seem quite good. All three methods yeild acceptable results, With the methods PI-MOESP and PO-MOESP consistently
% delivering better results than the standard MOESP Algorithm. This makes sense, since these two methods use more information than the standard
% MOESP algorithm, so it's so be expected that they deliver better results. Still, all methods are quite close to each other for basically
% any order, since, in the case of a simulation without noise measurement noise, the MOESP estimates will not be biased, and therefore are expected
% to be close to the results delivered by the other 2 methods.
%  On the other hand, it's odd that the PI-MOESP algorithm is delivering
% consistently better results than the PO-MOESP algotithm, since this last one uses information on the past outputs as well. It could be an error in
% the code, or it could be just a quirk of this particular data set.
%  The values we obtained for the VAF of all models are really good at this order, since we're getting around or over 90% for all models. This could
%  in practise suggest that n=20 will result in an overfitting of the data, something we'll confirm in question 5 when we do data validation.

%% Exercise 4: PEM

iter = 50;
K = zeros(n,1);
[Abar,Bbar,C,D,K,x0] = mypem(A,B,C,D,K,x0,y,u,iter);
A = Abar + K*C;
B = Bbar + K*D;
[yhat, ~] = simsystem(A,B,C,D,x0,u);
figure(4); plot(y, 'linewidth', 2); hold on;
plot(yhat, 'linewidth', 2); hold off;
title('Plot of Model after PEM vs original data.')
vaf = max(0, (1-norm(y-yhat)^2/norm(y)^2)*100);
fprintf('VAF before optimization: %2.5f \n', vafb);
fprintf('VAF after optimization: %2.5f \n', vaf);

% Your comments and answers:
%
%  For this exercise, we used a inital guess form K0 of zeros, and all other parameters were the results from the PO-MOESP model. The number of
% iterations chosen here was 50, but in practice the improvements were, as expected, of only about a couple of percentage points (partially also
% because our model had a VAF of 96% to begin with.
%  We'd expect that, if the number of iterations were to increase, we would get even better model fits. However, after some point we hit a point of
% diminishing returns.
%  A practical aspect of this implementation was the need for regularization using the Levenberg Marquadt method, since the Hessian
%  was really close to singularity. This is likely because our model was already really good, so we can expect the cost function to be quite flat
%  for the order we chose. The choice was made to set the lambda parameter to 100 in order to avoid numerical issues. This, however, is a really
%  high value which essentially makes the algorithm be close to a simple steepest descent algorithm.

%% Exercise 5: Training and Validation data
N_ = 580; n = 3;
yt = y(1:N_); yv = y(N_+1:end);
ut = u(1:N_); uv = u(N_+1:end);

[At,Bt,Ct,Dt,x0,sv,Phi] = subspaceID(ut,yt,s,n,'po-moesp');
[yhat, xhat] = simsystem(At,Bt,Ct,Dt,x0,ut);
vaft = max(0, (1-norm(yt-yhat)^2/norm(yt)^2)*100);
xf = xhat(:,end);
[yhatv, ~]= simsystem(At,Bt,Ct,Dt,xf,uv);
vafv = max(0, (1-norm(yv-yhatv)^2/norm(yv)^2)*100);


% Your comments and answers:
%
%  Trying to do model training and validation for the order initially chosen, n=20, proved quite difficult, as I got very poor results for the
% validation data set. This implies that this order is very high, and we're overfitting our model to the specific data set we had available.
% With this in mind we lowered (considerably) the model order of our system to n=3. Playing around with different ratios, I found that a value of
% close to 0.6 worked best here. 
%  The results for the validation data set aren't amazing, and the training model also has a VAF worse than for n=20 but they seem to generally follow the 
% output itself, with the highest validation VAF I was able to achieve being VAF = 65%. This VAF, despite being quite low, shows that higher order models likely
% overfit the data to the point where doing any validation on them would result in poor results.
%  In the plot, the blue line is the original output data, red is the training set results and green is the validation set results.

%% Plots of Question 5
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