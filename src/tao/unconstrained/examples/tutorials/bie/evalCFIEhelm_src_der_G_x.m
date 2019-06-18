function [ux, uy] = evalCFIEhelm_src_der_G_x(t,G,sigma,k,eta)

%evaluate the derivative of u wrt to x
%
% Barnett 6/8/14

N = numel(G.x);
u = 0*t; % initialize output
ux = u;
uy = u;
for j=1:N
  d = t-G.x(j);  % displacement of targets from jth src pt
  kr = k*abs(d);
  if (abs(d)>1e-5) 
    costhetan = real(conj(G.nx(j)).*d)./abs(d);   % theta angle between x-y & ny
    %sintheta = imag(conj(G.nx(j)).*d)./abs(d);

    costheta = -real(d)./abs(d);   
    sintheta = -imag(d)./abs(d);
    %dcosthetan = real( conj(G.nx(j)) .* ( ( ((-1) .*abs(d) - d .* costheta) + (-1i.*abs(d) - d.* sintheta) )) ./ (abs(d)).^2  );
    %dcosthetan = real( conj(G.nx(j)) .* ( ( ((-1) .*abs(d) - d .* costheta) + (-1i.*abs(d) - d.* sintheta) )) ./ (abs(d)).^2  ); 
    dcosthetanx = real( conj(G.nx(j)) .* ( ( ( (-1).*abs(d) - d.*costheta )  )) ./...
                    (abs(d)).^2  );
    dcosthetany = real( conj(G.nx(j)) .* ( (  ( -1i.*abs(d) - d.* sintheta ) )) ./...
                    (abs(d)).^2  ); 
    
    
    % u = u + (k*costhetan./besselh(1,kr) - 1i*eta*besselh(0,kr)) * ...
      %    ((1i/4) * G.w(j) * G.sp(j) * sigma(j));  
    ux = ux + ( k*k*costhetan.*(besselh(1,1,kr)./kr - besselh(2,1,kr)).*(costheta) +...
           (k*dcosthetanx.*(besselh(1,1,kr))) -...
           1i*eta*k*(costheta).*(-besselh(1,1,kr))) * ...
          ((1i/4) * G.w(j) * G.sp(j) * sigma(j));
    uy = uy + ( k*k*costhetan.*(besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sintheta) +...
           (k*dcosthetany.*(besselh(1,1,kr))) -...
           1i*eta*k*(sintheta).*(-besselh(1,1,kr))) * ...
          ((1i/4) * G.w(j) * G.sp(j) * sigma(j));
  end
  %begin mng
%   u = u + (-1i*k/4*besselh(1,1,kr).*-(costheta + 1i*sintheta))*... % + 1/4*eta*besselh(0,1,kr)) * ...(dr+di)/(abs(d))
%       ( G.w(j)* G.sp(j)  * sigma(j));
  %end mng
end