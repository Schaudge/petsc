clear all;
close all;

r=20;
th=0:0.01:pi/2;

%first horizontal line on the left side of domain
x1=linspace(-66.5,-20-3.5,100);
y1=40*ones(size(x1));

tt=[linspace(-66.5,-20-3.5,100); ones(size(x1))] ;
xx1=[x1(1), y1(1)];
%xs=xx1+tt;
%break
%quarter circle for left horn
x0l=3.5+20;
y0l=40-20;
xl=fliplr(-x0l+r*cos(th));
yl=fliplr(y0l+r*sin(th));

%vertical line for the inner side of the horns
t=20:-0.1:0;
ym1=t;
xm1=-3.5*ones(size(ym1));

%middle region where the source is placed
t=0:0.1:7;
xmid=-3.5+t;
ymid=zeros(size(xmid));

%the following is the mirror side of the left horn
ym2=0:0.1:20;
xm2=3.5*ones(size(ym1));
x0r=x0l;
y0r=y0l;
xr=x0r-r*cos(th);
yr=y0r+r*sin(th);
x2=-x1;
y2=y1;

%gather the entire domain for the horns
xg=[x1 xl xm1 xmid xm2 xr x2];
yg=[y1 yl ym1 ymid ym2 yr y2];

figure(99);
plot(xg(1:end),yg(1:end),'LineWidth',2)
axis equal
hold on

%circle for the target domain
x0c=0.0;
y0c=40+23;
rc=60;
thn=0:0.1:2*pi;
plot(x0c+rc*sin(thn),y0c+rc*cos(thn),'k','LineWidth',2)
%in thick red the objective region
plot(x0c+rc*sin(pi/6:0.1:pi/3),y0c+rc*cos(pi/6:0.1:pi/3),'r','LineWidth',6)

xs=0;
ys=13;

%source of incident wave
plot(xs,ys,'ks','MarkerSize',10,'LineWidth',2)

%lattice of dielectric scatterers
rc=0.1;
yp0=45;
xp0=-16; 
ln=linspace(0,32,7);
[xp,yp]=ndgrid(ln,ln);
theta=0:0.1:2*pi;

xx=xp(:);
yy=yp(:);

for n=1:7*7
    for j=1:length(theta)
xpc(j,n)=xp(n)+xp0+rc*cos(theta(j));
ypc(j,n)=yp(n)+yp0+rc*sin(theta(j));
    end
    xpc(j+1,n)=NaN;
    ypc(j+1,n)=NaN;
end

plot(xpc(:),ypc(:));%,'ro','MarkerSize',2,'LineWidth',1)


%break
%shift the lattice to be centered as in the experiment
xp=xp+xp0;
yp=yp+yp0;

%lattice, for now just points
plot(xp,yp,'ro','MarkerSize',2,'LineWidth',1)

