%function [X0, rc] = initialize()

rc = 6; 
yp0 =45;      %45;
xp0 =-16;      %-16;
ln = linspace(0,32,1);
[xp,yp] = ndgrid(ln,ln);

%xp = 0; %0 10 10]; % -5 25]  %[-20 0];% 20 -20 0 20 ]; % 3 -3 3 ]
%yp= 0;  %10 0 10]; % 10 30];  %[-3 -3]; %-3 6 6 6]; %, -20 -26 -26]

xp = xp + xp0; 
yp = yp + yp0;
X0 = [(xp(:))'  (yp(:))'];
%directization points on the surface of a circle


%end

