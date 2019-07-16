function [x,w] = gauss(N)
% GAUSS - nodes x (Legendre points) and weights w for Gauss quad on [-1,1]
%
% [x,w] = gauss(N) return x and w as N-by-1 vectors. From Trefethen 2000 book.
beta = .5./sqrt(1-(2*(1:N-1)).^(-2));
T = diag(beta,1) + diag(beta,-1);
[V,D] = eig(T);
x = diag(D); [x,i] = sort(x);
w = 2*V(1,i).'.^2;             % transpose not in Trefethen
