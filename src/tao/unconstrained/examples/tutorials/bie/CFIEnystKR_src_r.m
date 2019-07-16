function a = CFIEnystKR_src_r(G, G_r, GG,i,j,k,eta)
% CFIENYSTKR - element of 2D Helmholtz CFIE Nystrom matrix, Kapur-Rokhlin corr
%
% Aij = CFIEnystKR(G,i,j,k,eta) returns A_{ij} given i,j, and a curve struct G
%  whose boundary quadrature must be PTR, wavenumber k, and mixing parameter
%  eta. Barnett 6/8/14.

g6 = [4.967362978287758 -16.20501504859126 25.85153761832639 ...
      -22.22599466791883 9.930104998037539 -1.817995878141594]; % 6th order
% g10 = [7.832432020568779e+00 -4.565161670374749e+01 1.452168846354677e+02 -2.901348302886379e+02 ...
%  3.870862162579900e+02 -3.523821383570681e+02 2.172421547519342e+02 -8.707796087382991e+01 ...
%    2.053584266072635e+01 -2.166984103403823e+00];
if isequal((G.x),(GG.x)) 
    sw = GG.sp(j)*GG.w(j);
    sw1 = sw;                   % speed weight
    N = numel(GG.x); l = mod(i-j,N); if l>N/2, l=N-l; end   % index distance i to j
    if l>0 && l<=6, sw = sw * (1 + g6(l)); end   % apply correction
    d = G.x(i)-GG.x(j); kr = k*abs(d);           % CFIE kernel...
    d_r = G_r.x(i) - GG.x(j); kr_r = k*abs(d_r) ;

    costhetan = real(conj(GG.nx(j)).*d)./abs(d);
    costhetan_r = real(conj(GG.nx(j)).*d_r)./abs(d_r);

 
    if i==j
        a = 1i/4.0*( 0 - k *costhetan_r * besselh(1, 1, kr_r) -...
                               1i * eta * (0 - besselh(0,1,kr_r))) *sw1;

        return; 
    end  
                % kill diagonal

    a = (1i/4.0) * (k*costhetan*besselh(1,1,kr)- 1i*eta*besselh(0,1,kr)) * sw -...
       1i/4.0*(k *costhetan_r * besselh(1,1,kr_r)- 1i*eta* besselh(0,1,kr_r)) * sw1;

else
    
    %disp('blah');
    %sw=GG.w(j);
    sw = GG.w(j)*GG.sp(j); %Marieme  
    d = G.x(i)-GG.x(j); kr = k*abs(d);           % CFIE kernel...
    d_r = G_r.x(i) - GG.x(j); kr_r = k*abs(d_r);
    if (abs(d) > 1e-10)
    costhetan = real(conj(GG.nx(j)).*d)./abs(d);  % theta angle between x-y & ny
    costhetan_r = real(conj(GG.nx(j)).*d_r)./abs(d_r);
    a = (1i/4.0) * (k*(costhetan*besselh(1,1,kr) - costhetan_r * besselh(1,1,kr_r))- 1i*eta*(besselh(0,1,kr)- besselh(0,1,kr_r)) )  * sw;
    else
       
        a = 0.0;
    end
end