function [ux,uy] = evalCFIEhelm_src_derF_S2_r(t,G,sigma,k,eta)
% Evaluate the derivative of the column of the matrix of F without the
% diagonal element
%
% Barnett 6/8/14

N = numel(G.x);
ux = 0.0*t; % initialize output
uy = ux;
for j=1:N
  d = t-G.x(j);  % displacement of targets from jth src pt
  if (abs(d) >1e-10)
    kr =k*abs(d);
    costhetan = real(conj(G.nx(j)).*d)./abs(d);   % theta angle between x-y & ny
    costheta = -real(d)./abs(d);   
    sintheta = -imag(d)./abs(d);
    %dcosthetan = real( conj(G.nx(j)) .* ( ( ( (-1).*abs(d) - d.*costheta ) + ( -1i.*abs(d) - d.* sintheta ) )) ./...
    %                (abs(d)).^2  );  
    dcosthetanx = real( conj(G.nx(j)) .* ( ( ( (-1.0).*abs(d) - d.*costheta )  )) ./...
                    (abs(d)).^2  );
    dcosthetany = real( conj(G.nx(j)) .* ( (  ( -1i.*abs(d) - d.* sintheta ) )) ./...
                    (abs(d)).^2  );                
       %%%%rajoute +
    ux = ux + ( k*k*costhetan.*( besselh(1,1,kr)./kr - besselh(2,1,kr)).*(costheta) +...
           (k*dcosthetanx.*(besselh(1,1,kr))) -...
           1i*eta*k*(costheta).*(-besselh(1,1,kr))) * ...
          ((1i/4.0) * G.w(j) * G.sp(j) * sigma(j));
    uy = uy + ( k*k*costhetan.*( besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sintheta) +...
           (k*dcosthetany.*(besselh(1,1,kr))) -...
           1i*eta*k*(sintheta).*(-besselh(1,1,kr))) * ...
          ((1i/4.0) * G.w(j) * G.sp(j) * sigma(j));      
  end
end
