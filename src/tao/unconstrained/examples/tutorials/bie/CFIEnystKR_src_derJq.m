function a = CFIEnystKR_src_derJq(t,GG,i,j,k,eta)
% elements of the derivative of the matrix of u wrt sigma
g6 = [4.967362978287758 -16.20501504859126 25.85153761832639 ...
      -22.22599466791883 9.930104998037539 -1.817995878141594];

  
sw = GG.sp(j)*GG.w(j);    


                       % speed weight
%N = numel(GG.x);   % apply correction                 % kill diagonal
d = t(i)-GG.x(j); kr = k*abs(d);           % CFIE kernel...
%a=0;
%if (abs(d) > 1e-10)
    costhetan = real(conj(GG.nx(j)).*d)./abs(d);  % theta angle between x-y & ny
    %a = (1i/4) * (k*costheta*besselh(1,1,kr) - 1i*eta*besselh(0,1,kr)) * sw;
    %Marieme
    a = ((1i/4.0) * (k*costhetan.*besselh(1,1,kr) - 1i*eta*besselh(0,1,kr)) ) * sw;
%end
