function a = CFIEnystKR_src(G,GG,i,j,k,eta)
% CFIENYSTKR - element of 2D Helmholtz CFIE Nystrom matrix, Kapur-Rokhlin corr
%
% Aij = CFIEnystKR(G,i,j,k,eta) returns A_{ij} given i,j, and a curve struct G
%  whose boundary quadrature must be PTR, wavenumber k, and mixing parameter
%  eta. Barnett 6/8/14.

g6 = [4.967362978287758 -16.20501504859126 25.85153761832639 ...
      -22.22599466791883 9.930104998037539 -1.817995878141594]; % 6th order

if isequal(G.x,GG.x) 
 sw = G.sp(j)*G.w(j);                        % speed weight
 N = numel(G.x); l = mod(i-j,N); if l>N/2, l=N-l; end   % index distance i to j
 if l>0 && l<=6, sw = sw * (1 + g6(l)); end   % apply correction
 if i==j, a = 0; return; end                 % kill diagonal
 d = G.x(i)-G.x(j); kr = k*abs(d);           % CFIE kernel...
 costhetan = real(conj(G.nx(j)).*d)./abs(d);  % theta angle between x-y & ny
%a = (1i/4) * (k*costheta*besselh(1,1,kr) - 1i*eta*besselh(0,1,kr)) * sw;
%Marieme
 a = 1/2 + ((1i/4) * (k*costhetan*besselh(1,1,kr) - 1i*eta*besselh(0,1,kr)) ) * sw;
else
    %disp('blah');
    %sw=GG.w(j);
    sw = GG.w(j)*GG.sp(j); %Marieme  
  %   N = numel(G.x); l = mod(i-j,N); if l>N/2, l=N-l; end   % index distance i to j
   %  if l>0 && l<=6, sw = sw * (1 + g6(l)); end   % apply correction
   % if i==j, a = 0; return; end
    d = G.x(i)-GG.x(j); kr = k*abs(d);           % CFIE kernel...
    if (abs(d) > 1e-5)
    costhetan = real(conj(GG.nx(j)).*d)./abs(d);  % theta angle between x-y & ny
    a = ((1i/4) * (k*costhetan*besselh(1,1,kr) - 1i*eta*besselh(0,1,kr))) *sw;
    else
        a = 0;
    end
end