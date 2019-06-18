%%% MNG 05/29/2019
%Checking the derivatives of J wrt x=x_r+1ix_i

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
    GG_eps{ii} = cylinder(rc,xp(ii),yp(ii)+epsilon);
    GG_eps{ii} = curvquad(GG_eps{ii},'ptr',N,10);
    %quiver(real(GG{ii}.x),imag(GG{ii}.x),real(GG{ii}.nx),imag(GG{ii}.nx))
   % plot(real(GG{ii}.x),imag(GG{ii}.x),'k','LineWidth',2)
   % hold on
end
k = 10; eta = k;                           % wavenumber, SLP mixing amount
f = @(z) 1i*besselh(0,1,k*abs(z-src))/4.0;   % known soln: interior source
fgradx = @(z) -1i*k*besselh(1,1,k*abs(z-src))/4.0.*(real(z-src))./abs(z-src);   % known soln: interior source
fgrady = @(z) -1i*k*besselh(1,1,k*abs(z-src))/4.0.*(imag(z-src))./abs(z-src); 
unic=[]; rhs=[]; rhs_eps = rhs; drhs = [];
for ii=1:nc
    rhs = [rhs -2*f(GG{ii}.x)];
    drhs = [drhs -2*fgradx(GG{ii}.x)]; 
    rhs_eps = [rhs_eps -2*f(GG_eps{ii}.x)];
end

A = nan(N*nc,N*nc);
B=nan(N*nc, N*nc*nc);
dA = A;
A_eps=A;

for ii=1:nc
    for jj=1:nc
        for i=1:N
            for j=1:N
                A(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG{ii},GG{jj},i,j,k,eta); %remultiply by 2
                %A_eps(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG_eps{ii},GG_eps{jj},i,j,k,eta);
                %dA(i+(ii-1)*N,j+(jj-1)*N) = 2*dCFIEnystKR_src(GG{ii},GG{jj},i,j,k,eta);
            end
        end
    end   
end
for ll = 1:nc
    for ii=1:nc
        for jj=1:nc
            for i=1:N
                for j=1:N
                    if (ii == ll)
                        if (ii == jj)
                            A_eps(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG_eps{ii},GG_eps{jj},i,j,k,eta); %remultiply by 2
                        else
                            A_eps(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG_eps{ii},GG{jj},i,j,k,eta);
                        end
                        %A_eps(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG_eps{ii},GG_eps{jj},i,j,k,eta);
                        %dA(i+(ii-1)*N,j+(jj-1)*N) = 2*dCFIEnystKR_src(GG{ii},GG{jj},i,j,k,eta);
                    else
                        A_eps(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG{ii},GG{jj},i,j,k,eta);
                    end
                end
            end
        end   
    end
 B(:,N*nc*(ll-1)+1:N*nc*ll) = A_eps;   
end

%sigma = (eye(Nt) + A) \ rhs(:);                % dense solve
%MNG
sigma = (A) \ rhs(:);
for ii =1:nc
    rrhs_eps = rhs;
    rrhs_eps(:,ii) = rhs_eps(:,ii);
    sigma_eps(:,ii) = (B(:,N*nc*(ii-1)+1:N*nc*ii)) \ rrhs_eps(:);
    dsigma(:,ii) = (sigma_eps(:,ii) - sigma)/epsilon;
end
%         dsigma2_temp = dA*sigma;
%         dsigma2 = -(eye(Nt) + A) \dsigma2_temp + (eye(Nt) + A) \ drhs(:);
% % 
%          norm(dsigma_eps-dsigma2)
%          pause()

tobj=pi/6:0.02:pi/3;
trest=[0:0.02:pi/6 pi/3:0.02:2*pi];
obj=x0c+rj*sin(tobj)+1i*(y0c+rj*cos(tobj));
objrest=x0c+rj*sin(trest)+1i*(y0c+rj*cos(trest));




% uobj=zeros(size(obj));
% uobj_eps = uobj;
Jx = 0;
Jx_eps = 0;
for ii=1:nc
    ps_eps = 0;
    for jj=1:length(obj)
        %d  = [d; obj(jj)-GG{ii}.x];  % displacement of targets from jth src pt

        uobj(jj,ii)= evalCFIEhelm_src(obj(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);         % evaluate soln    
        uobj_eps(jj,ii) = evalCFIEhelm_src(obj(jj), GG_eps{ii}, sigma(1+(ii-1)*N:ii*N), k, eta);

    end 
    %ps_eps  = (sum(abs((uobj_eps)).^2)-sum(abs((uobj)).^2) )/epsilon;
    %dJdxc_eps(ii) = ps_eps;

end

J = sum(abs(sum(uobj,2)).^2);
for ii =1:nc
  if ii==1 
    uuobj_eps=[uobj_eps(:,ii),uobj(:,ii+1:nc)];
    J_eps(ii) =  sum(abs(sum(uuobj_eps,2)).^2);

  elseif ii==nc 
    uuobj_eps=[uobj(:,1:ii-1),uobj_eps(:,nc)];
    J_eps(ii) =  sum(abs(sum(uuobj_eps,2)).^2);

  else 
    uuobj_eps=[uobj(:,1:ii-1),uobj_eps(:,ii), uobj(:,ii+1:nc)];
    J_eps(ii) =  sum(abs(sum(uuobj_eps,2)).^2);
  end
 dJdxc_eps(ii) = 1/epsilon*(J_eps(ii)-J); 
end

%missing here for nor the 2*u(x)
dJdxc=[];
%missing here for nor the 2*u(x)
ps = [];
 for ii=1:nc 
    for jj=1:length(obj)
            uobjj(jj) = sum(uobj(jj,:));
            [u,v]= ((evalCFIEhelm_src_der_G_x(obj(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta)));
            ps(jj,ii) = 2*real(conj(uobjj(jj))*v);
    end
    dJdxc(ii) =  sum(ps(:,ii)); %+0.2*((xp(ii)-x0c) + (yp(ii)-y0c))./sqrt((xp(ii)-x0c).^2+(yp(ii)-y0c).^2);
 end 
err = norm(dJdxc - dJdxc_eps);
X = sprintf('derivative error is %s for step size %s' , err, epsilon);
disp(X)