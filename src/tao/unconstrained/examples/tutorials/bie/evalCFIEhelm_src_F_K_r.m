function u = evalCFIEhelm_src_F_K_r(t,t_r,G,sigma,k,eta,i,j)
%Evaluate the derivative of the part of F corresponding to the point of
% interest x_i on the circle C_i (diagonal part of the matrix of F)
%
% Barnett 6/8/14

%u = 0*t_r(i); % initialize output
g6 = [4.967362978287758 -16.20501504859126 25.85153761832639 ...
-22.22599466791883 9.930104998037539 -1.817995878141594]; % 6th order
sw = G.sp(j)*G.w(j); % speed weight
sw1 = sw;
N = numel(G.x); l = mod(i-j,N); if l>N/2, l=N-l; end % index distance i to j
if l>0 && l<=6, sw = sw * (1 + g6(l)); end
%for j=1:N
% displacement of targets from jth src pt
d_r = t_r(i) - G.x(j);
kr_r = k*abs(d_r);
d = t(i) - G.x(j);
kr = k*abs(d);
if i == j

    costhetan_r = real(conj(G.nx(j)).*d_r)./abs(d_r); % theta angle between x-y & ny
    %real(d)./abs(d);

    u = (0-k*costhetan_r.*(besselh(1,1,kr_r)) -...
    (0-1i*eta.*(besselh(0,1,kr_r)))) * ...
    ((1i/4) * sw1 * sigma(j));
else

    costhetan = real(conj(G.nx(j)).*d)./abs(d);   % theta angle between x-y & ny
    costhetan_r = real(conj(G.nx(j)).*d_r)./abs(d_r);
    u = (1i/4.0) * (k*costhetan*besselh(1,1,kr)- 1i*eta*besselh(0,1,kr)) * sw *sigma(j)-...
       1i/4.0*(k *costhetan_r * besselh(1,1,kr_r)- 1i*eta* besselh(0,1,kr_r)) * sw1 * sigma(j);


end