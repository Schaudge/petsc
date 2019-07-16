%function [X0, rc] = initialize()
yp0 = 45;     %21 %45;
xp0 =-64;      %-16;
lnx1 = linspace(0,32,5); 
lny = linspace(0,72,10);
[xp,yp] = ndgrid(lnx1,lny);

 
xp = xp + xp0; 

yp = yp + yp0;
x = xp(:);
y = yp(:);

nc = length(x);


theta = -pi/7;
for ii = 1:nc
    x(ii) = x(ii) * cos(theta) - y(ii) *sin(theta);
    y(ii) = sin(theta) * x(ii) + cos(theta)*y(ii) ;
%     x(ii+nc/2) = x(ii+nc/2) * cos(-theta) - y(ii+nc/2) *sin(-theta);
%     y(ii+nc/2) = sin(-theta) * x(ii+nc/2) + cos(-theta)*y(ii+nc/2) ;
end
xp = x;
yp = y;
X0 =[xp' yp'];
%mu0 = .01 * ones(1,nn/2*(nn/2 -1 )/2);
%setrc(nn/2);
%directization points on the surface of a circle


%end

