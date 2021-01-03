function theta = matrices2theta(Abar,Bbar,C,D,K,x0) 
%%
% Function INPUT
% Abar      System matrix Abar (matrix of size n x n)
% Bbar      System matrix Bbar (matrix of size n x m)
% C         System matrix C (matrix of size l x n)
% D         System matrix D (matrix of size l x m)
% K         System matrix K (matrix of size n x l)
% x0        Initial state (vector of size n x one)

% Function OUTPUT
% theta     Paramter vector (vector of size n*n+n*m+l*n+l*m+n*l+n)
theta = [Abar(:); C(:); x0(:); Bbar(:); D(:); K(:)];
end