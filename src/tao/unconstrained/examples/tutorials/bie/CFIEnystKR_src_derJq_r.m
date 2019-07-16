function a = CFIEnystKR_src_derJq_r(t, t_r,GG,i,j,k,eta)
% elements of the derivative of the matrix of u wrt sigma


  
sw = GG.sp(j)*GG.w(j);    


                       % speed weight
%N = numel(GG.x);   % apply correction                 % kill diagonal
d = t(i)-GG.x(j);   
d_r = t_r(i)- GG.x(j);
kr = k*abs(d);
kr_r = k*abs(d_r);% CFIE kernel...
a=0;
if (abs(d) > 1e-10)
    costhetan = real(conj(GG.nx(j)).*d)./abs(d);
    costhetan_r = real(conj(GG.nx(j)).*d_r)./abs(d_r);
% theta angle between x-y & ny
    %a = (1i/4) * (k*costheta*besselh(1,1,kr) - 1i*eta*besselh(0,1,kr)) * sw;
    %Marieme
    a = (k* (costhetan*besselh(1,1,kr) - costhetan_r * besselh(1,1,kr_r))- 1i*eta*(besselh(0,1,kr)- besselh(0,1,kr_r) ))  * ...
          ((1i/4.0) * sw ) ;
end
