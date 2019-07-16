function [uxx, uyy, uxy] = evalHessian(t,G,sigma,k,eta)

%evaluate the derivative of u wrt to x
%
% Barnett 6/8/14

N = numel(G.x);
u = 0*t; % initialize output
uxx = u;
uyy = u;
uxy = u;
for j=1:N
  d = t-G.x(j);  % displacement of targets from jth src pt
  kr = k*abs(d);
  if (abs(d)>1e-10) 
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
    
    %%second derivatives
    costhetax = -(-1 * abs(d) - costheta .* real(d)) ./ (abs(d)).^2;
    costhetay = -(-0 * abs(d) - sintheta .* real(d)) ./ (abs(d)).^2;

    sinthetay = -(-1 * abs(d) - sintheta .* imag(d)) ./ (abs(d)).^2;
    sinthetax = -(-0 * abs(d) - costheta .* imag(d)) ./ (abs(d)).^2;

    dcosthetanxx = real( conj(G.nx(j)) .* (  ( (-1).*costheta - ((-1).*costheta + d.* costhetax) ).*abs(d).^2 -2*( (-1).*abs(d) - d.*costheta ).*costheta .*abs(d) ) ./...
                    (abs(d)).^4  );
    dcosthetanxy = real( conj(G.nx(j)) .* ( ((-1i).*sintheta - ((-1i).* sintheta + d.* costhetay) ).*abs(d).^2 -2*( (-1i).*abs(d) - d.*costheta ).*sintheta .*abs(d) ) ./...
                    (abs(d)).^4  );
    dcosthetanyy = real( conj(G.nx(j)) .* (  ( (-1).*sintheta - ((-1).*sintheta + d.* sinthetay) ).*abs(d).^2 -2*( (-1).*abs(d) - d.*sintheta ).*sintheta .*abs(d) ) ./...
                    (abs(d)).^4  );




    
    % u = u + (k*costhetan./besselh(1,kr) - 1i*eta*besselh(0,kr)) * ...
      %    ((1i/4) * G.w(j) * G.sp(j) * sigma(j));  
%     ux = ux + ( k*k*costhetan.*(besselh(1,1,kr)./kr - besselh(2,1,kr)).*(costheta) +...
%            (k*dcosthetanx.*(besselh(1,1,kr))) -...
%            1i*eta*k*(costheta).*(-besselh(1,1,kr))) * ...
%           ((1i/4) * G.w(j) * G.sp(j) * sigma(j));
    uxx = uxx + ( k*k*dcosthetanx.*(besselh(1,1,kr)./kr - besselh(2,1,kr)).*(costheta)+...
                  k*k*costhetan.*((besselh(1,1,kr)./kr -...
             besselh(2,1,kr)).*(costhetax) +...
            (( k.*costheta.*( (besselh(1,1,kr)./kr -besselh(2,1,kr)).*kr -...
             besselh(1,1,kr).*k.*costheta)./kr.^2) -...
            (besselh(2,1,kr)./kr - besselh(3,1,kr)).*k.*costheta).*(costheta))   +...
           (k*dcosthetanx.*k.* ((besselh(1,1,kr)./kr - besselh(2,1,kr)).*(costheta)) + k*dcosthetanxx.*(besselh(1,1,kr))) -...
           1i*eta*k*k*costheta.*-((besselh(1,1,kr)./kr -besselh(2,1,kr)).*(costheta) )+...
           1i*eta*k.*costhetax.*(-besselh(1,1,kr)) ) * ...
          ((1i/4) * G.w(j) * G.sp(j) * sigma(j));

    uyy = uyy + ( k*k*dcosthetany.*(besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sintheta)+...
                  k*k*costhetan.*((besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sinthetax) + (( k.*sintheta.*( (besselh(1,1,kr)./kr -besselh(2,1,kr)).*kr - besselh(1,1,kr).*k.*sintheta)./kr.^2) - (besselh(2,1,kr)./kr - besselh(3,1,kr)).*k.*sintheta).*(sintheta))   +...
           (k*dcosthetany.*k.* ((besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sintheta)) + k*dcosthetanyy.*(besselh(1,1,kr))) -...
           1i*eta*k.*(sintheta).*k.* -((besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sintheta))+1i*eta*k.*sinthetay.*(-besselh(1,1,kr)) ) * ...
          ((1i/4) * G.w(j) * G.sp(j) * sigma(j));

    uxy = uxy + ( k*k*dcosthetany.*(besselh(1,1,kr)./kr - besselh(2,1,kr)).*(costheta)+...
                  k*k*costhetan.*((besselh(1,1,kr)./kr - besselh(2,1,kr)).*(costhetay) + (( k.*sintheta.*( (besselh(1,1,kr)./kr -besselh(2,1,kr)).*kr - besselh(1,1,kr).*k.*sintheta)./kr.^2) - (besselh(2,1,kr)./kr - besselh(3,1,kr)).*k.*sintheta).*(costheta))   +...
           (k*dcosthetanx.*k.* ((besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sintheta)) + k*dcosthetanxy.*(besselh(1,1,kr))) -...
           1i*eta*k.*(costheta).*k.* -((besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sintheta))+1i*eta*k.*costhetay.*(-besselh(1,1,kr)) ) * ...
          ((1i/4) * G.w(j) * G.sp(j) * sigma(j));
  end
  %begin mng
%   u = u + (-1i*k/4*besselh(1,1,kr).*-(costheta + 1i*sintheta))*... % + 1/4*eta*besselh(0,1,kr)) * ...(dr+di)/(abs(d))
%       ( G.w(j)* G.sp(j)  * sigma(j));
  %end mng
end
uxx = uxx';
uyy = uyy';
uxy = uxy';