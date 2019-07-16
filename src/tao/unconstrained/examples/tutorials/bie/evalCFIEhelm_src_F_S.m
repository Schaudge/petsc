function u = evalCFIEhelm_src_F_S(t,G,sigma,k,eta)
% Evaluate the off diagonals of F
%
% Barnett 6/8/14

N = numel(G.x);
u = 0.0*t; % initialize output
for j=1:N
  d = t-G.x(j);  % displacement of targets from jth src pt
  kr = k*abs(d);

  if (abs(d) > 1e-10)
 

      costhetan = real(conj(G.nx(j)).*d)./abs(d);   % theta angle between x-y & ny
      
      u = u + ((1i/4.0) * k*costhetan.*besselh(1, 1,kr) - (1i/4.0) *1i*eta*besselh(0,1,kr)) * ...
          ( G.w(j) * G.sp(j) * sigma(j)); 
      %begin mng
      %u = u + (1i/4*eta*besselh(0,1,kr)) * ...
      %    ( G.w(j) * G.sp(j) * sigma(j));
      %end mng
  

  end
end

