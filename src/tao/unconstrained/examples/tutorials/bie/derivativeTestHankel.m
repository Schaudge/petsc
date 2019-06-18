%%% test derivative of hankel functions
xd = (1+1i)-(.5+.5*1i);
x =abs((1+1i)-(.5+.5*1i));
epsilon = 1e-8;
x_eps =abs( (1+1i)-(.5+.5*1i+epsilon+1i*epsilon));

dx_eps = real((x_eps - x) )/ real(epsilon) + 1i*imag((x_eps - x) )/ imag(epsilon);



costheta = -real(xd)./x;
sintheta = -imag(xd)./x;
dx = costheta + sintheta;



F = besselh(1,1,2*x);
F_eps = besselh(1,1,2*(x_eps));
dF = 2*(besselh(1,1,2*x)/(2*x)-1*besselh(2,1,2*x)).*(costheta+sintheta);
dF_eps = (F_eps-F)/epsilon;
%dF_eps = real((F_eps - F) )/ real(epsilon) + 1i*imag((F_eps - F) )/ imag(epsilon);

err = norm(dF - dF_eps);
X = sprintf('derivative error is %s for step size %s' , err, epsilon);
disp(X)