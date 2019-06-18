%%% MNG 05/29/2019
%Test the derivative of J wrt x or wrt to y for x_c = x+iy

clear all
close all

%lattice of dielectric scatterers
rc = 3;
yp0 = 45;
xp0 = -16;
ln = linspace(0,32,2);
[xp,yp] = ndgrid(ln,ln);
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


epsilon = 1e-8;
 
for ii=1:nc
    GG{ii} = cylinder(rc,xp(ii),yp(ii));
    GG{ii} = curvquad(GG{ii},'ptr',N,10);
    GG_eps{ii} = cylinder(rc,xp(ii)+epsilon,yp(ii)); %to check wrt to y add epsilon to y
    GG_eps{ii} = curvquad(GG_eps{ii},'ptr',N,10);
end
k = 10; eta = k;                           % wavenumber, SLP mixing amount
f = @(z) 1i*besselh(0,1,k*abs(z-src))/4.0;   % known soln: interior source
fgradx = @(z) -1i/(4.0)*k*besselh(1,1,k*abs(z-src)).*(real(z-src))./abs(z-src);   % known soln: interior source
fgrady = @(z) -1i/(4.0)*k*besselh(1,1,k*abs(z-src))/4.0.*(imag(z-src))./abs(z-src); 
unic=[]; rhs=[]; %rhs_eps = rhs; drhs = [];
for ii=1:nc
   rhs = [rhs -2*f(GG{ii}.x)];
   %rhs_eps = [rhs_eps -2*f(GG_eps{ii}.x)];
end

A = nan(N*nc,N*nc);
A_eps = A;
dA = A;
for ii=1:nc
    for jj=1:nc
        for i=1:N
            for j=1:N
                 A(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG{ii},GG{jj},i,j,k,eta);
            end
         end
     end
end

    sigma = ( A ) \ rhs(:);

for ii=1:nc
    ps=zeros(size(GG{ii}.x));
    ps_eps = ps;
    for jj=1:nc
        %if (ii == 1)
        %ps= ps+evalCFIEhelm_srcH1(GG{ii}.x,GG{jj},sigma,k,eta);         % evaluate soln
            if (ii == jj)
                ps_eps = ps_eps + 2/epsilon*( evalCFIEhelm_src_F_K(GG_eps{ii}.x,GG_eps{jj},sigma(1+(jj-1)*N:jj*N),k,eta)-...
                         evalCFIEhelm_src_F_K(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta) );

               % ps = ps+evalCFIEhelm_src_derF_K(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta,0*dsigma(1+(jj-1)*N:jj*N));



            else

                ps_eps = ps_eps + 2/epsilon*( evalCFIEhelm_src_F_S(GG_eps{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta)-...
                         evalCFIEhelm_src_F_S(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta) ) ;  
                %ps = ps + evalCFIEhelm_src_derF_S(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta,0*dsigma(1+(jj-1)*N:jj*N));
                dFdxc_eps(N*(jj-1)+1:N*jj,ii) = 2/epsilon*( evalCFIEhelm_src_F_S(GG{jj}.x,GG_eps{ii},sigma(1+(ii-1)*N:ii*N),k,eta)-...
                         evalCFIEhelm_src_F_S(GG{jj}.x,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta) ) ;
                %pause

            end




    end
   % dFdxc(N*(ii-1)+1:N*ii,ii) = ps+2*fgrad(GG{ii}.x);
    dFdxc_eps(N*(ii-1)+1:N*ii,ii) = ps_eps+2*(f(GG_eps{ii}.x)-f(GG{ii}.x))/epsilon;
end


for ii=1:nc
    pff_x=zeros(size(GG{ii}.x));
    for jj=1:nc
        if (ii == jj)
            %[u,v] = evalCFIEhelm_src_derF_K(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta);
            pff_x = pff_x + 2*0;
        end
        if (ii ~= jj) 
            [u,v] = evalCFIEhelm_src_derF_S(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta);
            pff_x = pff_x+2*u;   
            [uu,vv] = evalCFIEhelm_src_derF_S2(GG{jj}.x,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
            pf_x(N*(jj-1)+1:N*jj,ii) = 2*uu;
        end              
    end
    pf_x(N*(ii-1)+1:N*ii,ii) = pff_x+2*fgradx(GG{ii}.x);

end
dFdxc = pf_x;        

err = norm (dFdxc - dFdxc_eps);
X = sprintf('derivative error is %s for step size %s', err, epsilon);
disp(X)
for ii=1:nc
    error(ii) = norm(2*fgradx(GG{ii}.x) - 2*(f(GG_eps{ii}.x)-f(GG{ii}.x))/epsilon)
end
