function [Abar,Bbar,C,D,K,x0] = theta2matrices(theta,Asize,Bsize,Csize,Dsize,Ksize,xsize)
%%
% Function INPUT
% theta     Paramter vector (vector of size n*n+n*m+l*n+l*m+n*l+n)
% Asize     Size of Abar 
% Bsize     Size of Bbar 
% Csize     Size of C 
% Dsize     Size of D 
% Ksize     Size of K 
% xsize     Size of x0

% Function OUTPUT
% Abar      System matrix A (matrix of size n x n)
% Bbar      System matrix B (matrix of size n x m)
% C         System matrix C (matrix of size l x n)
% D         System matrix D (matrix of size l x m)
% K         System matrix K (matrix of size n x l)
% x0        Initial state (vector of size n x one)


theta_Abar = theta(1:Asize(1)*Asize(2));  theta(1:Asize(1)*Asize(2)) = [];
theta_C = theta(1:Csize(1)*Csize(2)); theta(1:Csize(1)*Csize(2)) = []; 
theta_x0 = theta(1:xsize(1)*xsize(2)); theta(1:xsize(1)*xsize(2)) = []; 
theta_Bbar = theta(1:Bsize(1)*Bsize(2)); theta(1:Bsize(1)*Bsize(2)) = []; 
theta_D = theta(1:Dsize(1)*Dsize(2)); theta(1:Dsize(1)*Dsize(2)) = []; 
theta_K = theta(1:Ksize(1)*Ksize(2));

Abar = reshape(theta_Abar, Asize(1), Asize(2));
C = reshape(theta_C, Csize(1), Csize(2));
x0 = reshape(theta_x0, xsize(1), xsize(2));
Bbar = reshape(theta_Bbar, Bsize(1), Bsize(2));
D = reshape(theta_D, Dsize(1), Dsize(2));
K = reshape(theta_K, Ksize(1), Ksize(2));
end