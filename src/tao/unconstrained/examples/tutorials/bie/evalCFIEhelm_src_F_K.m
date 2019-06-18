function u = evalCFIEhelm_src_F_K(t,G,sigma,k,eta)
% evaluate the diagonal of F
%
% Barnett 6/8/14

N = numel(G.x);
ctrl = 0;
u = 0*t; % initialize output
vv = u;
v = u(1:end-1);
for j=1:N
  d = t-G.x(j);  % displacement of targets from jth src pt

  if (t(j) == G.x(j))
      jjj = j;
      %index = find(abs(d) > 1e-10);
      d = d(abs(d) > 1e-10);
      ctrl = 1;
  end
  kr = k*abs(d);
  if (abs(d) > 1e-10)


    if (ctrl == 1)
     costhetan = real(conj(G.nx(j)).*d)./abs(d);   % theta angle between x-y & ny
 
    %sintheta = imag(conj(G.nx(j)).*d)./abs(d);

    costheta = real(d)./abs(d);   
    sintheta = imag(d)./abs(d);
    %dcosthetan = real( conj(G.nx(j)) .* ( ( ((1) .*abs(d) - d .* costheta) + (1i.*abs(d) - d.* sintheta) )) ./...
                  %(abs(d)).^2); 
       v = v +1/2*sigma(j)+((1i/4) * k*costhetan.*besselh(1, 1,kr) - (1i/4) *1i*eta*besselh(0,1,kr)) * ...
          ( G.w(j) * G.sp(j) * sigma(j));
      if (jjj == 1)
        vv = vv + [0;v];
      else
        vv = vv + [v(1:jjj-1);0;v(jjj:end)];  
      end
    else
      costhetan = real(conj(G.nx(j)).*d)./abs(d);   % theta angle between x-y & ny
      
      u = u + 1/2+((1i/4) * k*costhetan.*besselh(1, 1,kr) - (1i/4) *1i*eta*besselh(0,1,kr)) * ...
          ( G.w(j) * G.sp(j) * sigma(j)); 
    end
      %begin mng
      %u = u + (1i/4*eta*besselh(0,1,kr)) * ...
      %    ( G.w(j) * G.sp(j) * sigma(j));
      %end mng
  end
end
u = u +vv;
