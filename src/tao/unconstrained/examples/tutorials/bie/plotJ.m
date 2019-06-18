clear all
close all

%lattice of dielectric scatterers
rc = 6;
yp0 =45;      %45;
xp0 =0;      %-16;
ln = linspace(0,0,1);
[xp,yp] = ndgrid(ln,ln);

xp =-60:60;
yp=-60:60;

xp = xp + xp0; 
yp = yp + yp0;

%directization points on the surface of a circle
N = 50;

%objective region
x0c = 0.0;
y0c = 40+23;
rj = 60;
thn = 0:0.1:2*pi;

%location of the source
ys = 13; xs = 0; src= xs + 1i * ys;

nc=length(xp(:));
%break
Nt=N*nc;

tol=1.0;
targold=0.0; targ=0.0;
trestold=0.0; trest=0.0;
count=0;
fact_x = 1e-2;
fact_y =1e-2;
dt = [];
dty = dt;
dt_x =1; %[1, 1, 1, 1];
dt_y = 1;
ttarg = [];
thrsh = 1;
     
    for ii=1:nc
        GG{ii} = cylinder(rc,xp(ii),yp(ii));
        GG{ii} = curvquad(GG{ii},'ptr',N,10);
    end

     %pause
    %uinc=exp(1i*k*abs(G.x-src))./abs(G.x-src)*1/4/pi;

    k = 2*pi/8; eta = k;                           % wavenumber, SLP mixing amount

        f =  @(z) sum(1i*1*besselh(0,1,k*abs(z-src))/4.0,2);   % known soln: interior source
        fgradx = @(z) sum(-1i/4.0*1*k*besselh(1,1,k*abs(z-src)).*(real(z-src))./abs(z-src),2);
        fgrady = @(z) sum(-1i/4.0*1*k*besselh(1,1,k*abs(z-src)).*(imag(z-src))./abs(z-src),2);


    unic=[]; rhs=[]; 
    for ii=1:nc
        rhs = [rhs -2*f(GG{ii}.x)];
    end
    
    A = nan(N*nc,N*nc);
    for ii=1:nc
        for jj=1:nc
            for i=1:N
                for j=1:N
                    A(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG{ii},GG{jj},i,j,k,eta);
                end
            end
        end
    end

    sigma =( A) \ rhs(:);
    dFdq =  A;
    
    %targetarea: obj=x0c+rj*sin(thn)+1i*(y0c+rj*cos(thn));
    tobj=pi/6:0.02:pi/3; %remets pi/3
    trest=[0:0.02:pi/6 pi/3:0.02:2*pi];
    obj=x0c+rj*sin(tobj)+1i*(y0c+rj*cos(tobj));
    objrest=x0c+rj*sin(trest)+1i*(y0c+rj*cos(trest));
    nobj=rj*(1i*sin(tobj)-cos(tobj));
   

    
    targold=targ;
    uobj = zeros(length(obj), nc);
    %uobjj = zeros(length(obj));
    for ii=1:nc
        for jj=1:length(obj)
                %d  = [d; obj(jj)-GG{ii}.x];  % displacement of targets from jth src pt        
            uobj(jj,ii)= evalCFIEhelm_src(obj(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);         % evaluate soln               
        end 
    targ = -sum((abs(uobj(:,ii))).^2);
    ttarg = [ttarg, -targ];    
    end
    plot(ttarg)
    
    