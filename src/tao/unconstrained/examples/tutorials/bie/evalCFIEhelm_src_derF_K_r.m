function u = evalCFIEhelm_src_derF_K_r(t_r,G,sigma,k,eta,i,j)
%Evaluate the derivative of the part of F corresponding to the point of
% interest x_i on the circle C_i (diagonal part of the matrix of F)
%
% Barnett 6/8/14

u = 0*t_r(i); % initialize output
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
if i == j

    costhetan_r = real(conj(G.nx(j)).*d_r)./abs(d_r); % theta angle between x-y & ny
    %real(d)./abs(d);
    sintheta = -2*imag(d_r)./abs(d_r);
    dcosthetan_r = real( conj(G.nx(j)) .* ( ( ( -2*1i.*abs(d_r) - d_r.* sintheta ) )) ./...
    (abs(d_r)).^2 );
    u = ( k*k*costhetan_r.*( besselh(1,1,kr_r)./kr_r - besselh(2,1,kr_r)).*(sintheta) +...
    (k*dcosthetan_r.*(besselh(1,1,kr_r))) -...
    1i*eta*k*(sintheta).*(-besselh(1,1,kr_r))) * ...
    ((1i/4) * sw1 * sigma(j));
else
    costhetan_r = real(conj(G.nx(j)).*d_r)./abs(d_r); % theta angle between x-y & ny
    %real(d)./abs(d);
    sintheta = -2*imag(d_r)./abs(d_r);
    dcosthetan_r = real( conj(G.nx(j)) .* ( ( ( -2*1i.*abs(d_r) - d_r.* sintheta ) )) ./...
    (abs(d_r)).^2 );
    u = ( k*k*costhetan_r.*( besselh(1,1,kr_r)./kr_r - besselh(2,1,kr_r)).*(sintheta) +...
    (k*dcosthetan_r.*(besselh(1,1,kr_r))) -...
    1i*eta*k*(sintheta).*(-besselh(1,1,kr_r))) * ...
    ((1i/4) * sw1 * sigma(j));


end