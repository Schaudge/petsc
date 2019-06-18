%%% MNG 05/29/2019
%Test the derivative of J wrt real(sigma) or imag(sigma)

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
end
k = 10; eta = k;                           % wavenumber, SLP mixing amount
f = @(z) 1i*besselh(0,1,k*abs(z-src))/4.0;   % known soln: interior source
fgrad = @(z) 1i*k*besselh(1,1,k*abs(z-src))/4.0;   % known soln: interior source
unic=[]; rhs=[];
for ii=1:nc
    rhs = [rhs -2*f(GG{ii}.x)];
end

A = nan(N*nc,N*nc);
for ii=1:nc
    for jj=1:nc
        for i=1:N
            for j=1:N
                A(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG{ii},GG{jj},i,j,k,eta); %remultiply by 2
            end
        end
    end
end

%sigma = (eye(Nt) + A) \ rhs(:);                % dense solve
%MNG
sigma = (A) \ rhs(:);


tobj=pi/6:0.02:pi/3;
obj=x0c+rj*sin(tobj)+1i*(y0c+rj*cos(tobj));
nobj=rj*(1i*sin(tobj)-cos(tobj));

xcirc=xp+1i*yp;



uobj=zeros(size(obj));
uobj_eps = uobj;
J=0;
J_eps=0;

for ii=1:size(sigma)
   sigma_eps =sigma;
   sigma_eps(ii) = sigma_eps(ii) + 1i*epsilon; %currently doing derivative wrt imag(sigma)
   kk = ceil(ii/N);
   for jj=1:length(obj) %%to be fixed
        %d  = [d; obj(jj)-GG{ii}.x];  % displacement of targets from jth src pt
        uobj(jj,ii)= evalCFIEhelm_src(obj(jj),GG{kk},sigma(1+(kk-1)*N:kk*N),k,eta);         % evaluate soln    
        uobj_eps(jj,ii) = evalCFIEhelm_src(obj(jj), GG{kk}, sigma_eps(1+(kk-1)*N:kk*N), k, eta);
   end
end


J = sum(abs(sum(uobj(:,1:50:200),2)).^2);

for ii =1:size(sigma)
  kk = ceil(ii/N);  
  if kk==1 
    uuobj_eps=[uobj_eps(:,ii),uobj(:,ii+N:N:N*nc)];
    J_eps(ii) =  sum(abs(sum(uuobj_eps,2)).^2);

  elseif kk==nc %size(sigma) 
    uuobj_eps=[uobj(:,1:N:(ii-N)),uobj_eps(:,ii)];
    J_eps(ii) =  sum(abs(sum(uuobj_eps,2)).^2);

  else 

    uuobj_eps=[uobj(:,1:N:(ii-N)),uobj_eps(:,ii), uobj(:,ii+N:50:size(sigma))];
    J_eps(ii) =  sum(abs(sum(uuobj_eps,2)).^2);
  end
 dJdq_eps(ii) = 1/epsilon*(J_eps(ii)-J); 
end




B_r = zeros(length(obj), N*nc); %derivative wrt real(sigma)
B_i = B_r; %derivative wrt imag(sigma)
for ii=1:size(sigma)
kk = ceil(ii/N);
for jj=1:length(obj)
    uobjj(jj) = sum(uobj(jj,1:50:200));
    for i=1:N
            B_r(jj,i+(kk-1)*N) = 2 * real(conj(uobjj(jj)) * CFIEnystKR_src_derJq(obj,GG{kk},jj,i,k,eta));
            B_i(jj,i+(kk-1)*N) = 2 * real(conj(uobjj(jj)) *1i* CFIEnystKR_src_derJq(obj,GG{kk},jj,i,k,eta));
    end
end
end
dJdq_r = sum(B_r,1) ; 
dJdq_i = sum(B_i,1);    
%error = norm(dJdq_r-dJdq_eps);
error =norm(dJdq_i-dJdq_eps);
X = sprintf('derivative error is %s for step size %s' , error, epsilon);
disp(X)
