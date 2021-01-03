function [yhat, xhat]= simsystem(A,B,C,D,x0,u)
%% Instructions:
% Implement a function that simulates the system here!
% Use the following function inputs and outputs.

% Function INPUT 
% A         System matrix A (matrix of size n x n)
% B         System matrix B (matrix of size n x m)
% C         System matrix C (matrix of size l x n)
% D         System matrix D (matrix of size l x m)
% x0        Initial state (vector of size n x one)
% u         system input (matrix of size N x m)

% Function OUTPUT
% yhat      predicted output (vector of size l x one)
n = length(x0); N= length(u);
xhat = zeros(n,N+1);
xhat(:,1) = x0;
yhat = zeros(N,1);
for i = 1:N
    xhat(:,i+1) = A*xhat(:,i) + B*u(i);
    yhat(i)= C*xhat(:,i) + D*u(i);
end
end