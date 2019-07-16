%%Test optimizer
clear all;
close all;

maxiter = 4000;
fact_x = 1;
fact_y = 1;
 
yp0 = 21;     %21 %45;
xp0 =-16;      %-16;
ln = linspace(0, 32, 5); %ln = linspace(0, 8, 2)
[xp,yp] = ndgrid(ln,ln);

%xp = 0; %0 10 10]; % -5 25]  %[-20 0];% 20 -20 0 20 ]; % 3 -3 3 ]
%yp= 0;  %10 0 10]; % 10 30];  %[-3 -3]; %-3 6 6 6]; %, -20 -26 -26]

xp = xp + xp0; 
%xp = [0 0 0 0];
yp = yp + yp0;
%yp = [21 29 37 45];
X0 = [(xp(:))'  (yp(:))'];
nn = length(X0);
X0 = [-8.95034
0.244405
5.97714
-11.4491
3.35226
2.42985
-5.69126
3.92769
6.42274
17.642
22.5684
20.6793
26.911
26.3657
31.2276
38.287
33.9943
35.926];
X0 = X0';
[xp, yp, obj, vobj,verror,count] = optimizerMirror(X0, fact_x, fact_y, maxiter);
opt = -obj;
X = sprintf('max value is %f' , opt);
disp(X)
figure(3)
plot(vobj)