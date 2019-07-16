function u = evalCFIEhelm_src_r(t,t_r, G,sigma,k,eta)
% EVALCFIEHELM - evaluate 2D Helmholtz CFIE (D-i.eta.S) potential at targets
%
% u = evalCFIEhelm(t,G,sigma,k,eta) where t is a list of target points
%  (points in the complex plane), G is a curve struct (see curvquad.m) with
%  N quadrature nodes, sigma is a N-element column vector, k is the wavenumber,
%  and eta (~k) controls the amount of SLP in the CFIE, returns the potential
%  at the target points evaluated using the quadrature rule in G.
%  A direct sum is used, vectorized over target points only.
%
% Barnett 6/8/14

N = numel(G.x);
u = 0.0*t; % initialize output
for j=1:N
  d = t-G.x(j);  % displacement of targets from jth src pt
  d_r = t_r - G.x(j);
  kr = k*abs(d);
  kr_r = k * abs(d_r);
  if ( abs(d) > 1e-10 )
      costhetan = real(conj(G.nx(j)).*d)./abs(d);   % theta angle between x-y & ny
      costhetan_r = real(conj(G.nx(j)).*d_r)./abs(d_r);
      u = u + ((k.* (costhetan .* besselh(1,1,kr) -costhetan_r .* besselh(1,1,kr_r) )- 1i*eta.*(besselh(0,1,kr)- besselh(0,1,kr_r) )) ) .* ...
          ((1i/4.0) * G.w(j) * G.sp(j) * sigma(j)) ;

      %begin mng
      %u = u + (1i/4*eta*besselh(0,1,kr)) * ...
      %    ( G.w(j) * G.sp(j) * sigma(j));
      %end mng
  end
end
