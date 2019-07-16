function G = cylinder_r(r,x0,y0)
% SMOOTHSTAR - set up [0,2pi) parametrization of smooth star closed curve
G.Z = @(t) x0-1i*y0+r*exp(-1i*t); G.Zp = @(t) ((r.*-1i*exp(-1i*t)));
%G.Zp1 = @(t) ((r.*-1i*exp(1i*t)));
G.Zpp = @(t)  -r*exp(-1i*t);
