function [ux,uy] = evalCFIEhelm_src_derF_S(t,G,sigma,k,eta)
% Evaluate the derivative wrt x of the row of the matrix of F without the
% diagonal element
%
% Barnett 6/8/14
g6 = [4.967362978287758 -16.20501504859126 25.85153761832639 ...
      -22.22599466791883 9.930104998037539 -1.817995878141594]; % 6th order
N = numel(G.x);
ux = 0*t; % initialize output
uy = 0*t;
for j=1:N
  d = t-G.x(j);  % displacement of targets from jth src pt
  if (abs(d) >1e-5)
%     sw = G.sp(j)*G.w(j);                        % speed weight
%     N = numel(G.x); l = mod(i-j,N); if l>N/2, l=N-l; end   % index distance i to j
%     if l>0 && l<=6, sw = sw * (1 + g6(l)); end   % apply correction  
    kr =k*abs(d);
    costhetan = real(conj(G.nx(j)).*d)./abs(d);   % theta angle between x-y & ny
    costheta = real(d)./abs(d);   
    sintheta = imag(d)./abs(d);
    %dcosthetan = real( conj(G.nx(j)) .* ( ( ( (1).*abs(d) - d.*costheta ) + ( 1i.*abs(d) - d.* sintheta ) )) ./...
    %                (abs(d)).^2  );  
    dcosthetanx = real( conj(G.nx(j)) .* ( ( ( (1).*abs(d) - d.*costheta )  )) ./...
                    (abs(d)).^2  );
    dcosthetany = real( conj(G.nx(j)) .* ( (  ( 1i.*abs(d) - d.* sintheta ) )) ./...
                    (abs(d)).^2  );                
       
    ux = ux + ( k*k*costhetan.*( besselh(1,1,kr)./kr - besselh(2,1,kr)).*(costheta) +...
           (k*dcosthetanx.*(besselh(1,1,kr))) -...
           1i*eta*k*(costheta).*(-besselh(1,1,kr))) * ...
          ((1i/4) * G.w(j) * G.sp(j) * sigma(j));
    uy = uy + ( k*k*costhetan.*( besselh(1,1,kr)./kr - besselh(2,1,kr)).*(sintheta) +...
           (k*dcosthetany.*(besselh(1,1,kr))) -...
           1i*eta*k*(sintheta).*(-besselh(1,1,kr))) * ...
          ((1i/4) * G.w(j) * G.sp(j) * sigma(j));      
  end
end
