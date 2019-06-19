function [targ, grad] = ObjectiveAndGradient(X)
    nn = length(X);
    xp = X(1:nn/2);
    yp = X(nn/2+1:end);
    rc = 6;
    N = 50;
    nc=length(xp);


    for ii=1:nc
        GG{ii} = cylinder(rc,xp(ii),yp(ii));
        GG{ii} = curvquad(GG{ii},'ptr',N,10);
    end

    %directization points on the surface of a circle
    N = 50;

    %objective region
    x0c = 0.0;
    y0c = 40+23;
    rj = 60;

    %location of the source
    ys = 13; xs = -25:25;  src= xs + 1i * ys; %-30:30;
    
    k = 2*pi/8; eta = k;                           % wavenumber, SLP mixing amount 
    f =  @(z) sum(1i*1*besselh(0,1,k*abs(z-src))/4.0,2);   % known soln: interior source
    fgradx = @(z) sum(-1i/4.0*1*k*besselh(1,1,k*abs(z-src)).*(real(z-src))./abs(z-src),2);
    fgrady = @(z) sum(-1i/4.0*1*k*besselh(1,1,k*abs(z-src)).*(imag(z-src))./abs(z-src),2);


    rhs=[]; 
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
    obj=x0c+rj*sin(tobj)+1i*(y0c+rj*cos(tobj));


    uobj = zeros(length(obj), nc);
    %uobjj = zeros(length(obj));
    for ii=1:nc
        for jj=1:length(obj)       
            uobj(jj,ii)= evalCFIEhelm_src(obj(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);         % evaluate soln               
        end 

    end
    targ = -sum((abs(sum(uobj,2))).^2);
      
    %derivative of J wrt \sigma 
    B_r = zeros(length(obj), N*nc);
    B_i = B_r;
    uobjj = zeros(length(obj),1);
    for ii=1:nc
        for jj=1:length(obj)
            uobjj(jj) = sum(uobj(jj,:));
            for i=1:N
                a_r =CFIEnystKR_src_derJq(obj,GG{ii},jj,i,k,eta);
                a_i = 1i*a_r;
                B_r(jj,i+(ii-1)*N) = 2 * real(conj(uobjj(jj)) * a_r);
                B_i(jj,i+(ii-1)*N) = 2 * real(conj(uobjj(jj)) * a_i);
            end
        end
    end
    dJdq_r = sum(B_r,1);
    dJdq_i = sum(B_i,1);
    AA = [real(dFdq),-imag(dFdq); imag(dFdq), real(dFdq)];
    dJdq = [dJdq_r, dJdq_i];
    [lambda] = dJdq / AA;

    % pause


    dFdxc=[];
    for ii=1:nc
        pff=zeros(size(GG{ii}.x));
        pff_x = pff;
        pff_y = pff;
        for jj=1:nc
                if (ii ~= jj) 
                    [uu, vv] = evalCFIEhelm_src_derF_S(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta);
                    pff_x = pff_x+2*uu;  
                    pff_y = pff_y+2*vv;
                    [uuu, vvv] = evalCFIEhelm_src_derF_S2(GG{jj}.x,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
                    pf_x(N*(jj-1)+1:N*jj,ii) = 2*uuu;
                    pf_y(N*(jj-1)+1:N*jj,ii)  = 2*vvv;
                end              
        end
        pf_x(N*(ii-1)+1:N*ii,ii) = pff_x+2*fgradx(GG{ii}.x);
        pf_y(N*(ii-1)+1:N*ii,ii) = pff_y+2*fgrady(GG{ii}.x);
    end
    dFdxc = pf_x;
    dFdyc = pf_y;    



    dJdxc=[];
    dJdyc=[];
    for ii=1:nc 
        for jj=1:length(obj)
           [uu, vv] = ((evalCFIEhelm_src_der_G_x(obj(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta)));
            pj_x(jj,ii) = 2*real(conj(uobjj(jj))*uu);
            pj_y(jj,ii) = 2*real(conj(uobjj(jj))*vv);
        end
        dJdxc(ii) =  sum(pj_x(:,ii)); %-.3*( (xp(ii)-x0c) )./((xp(ii)-x0c).^2+(yp(ii)-y0c).^2-60^2);
        dJdyc(ii) =  sum(pj_y(:,ii)); %-.3*( (yp(ii)-y0c))./((xp(ii)-x0c).^2+(yp(ii)-y0c).^2 - 60^2);

    end



    dt_x = lambda*[(real(dFdxc));imag(dFdxc)] - dJdxc;
    dt_y = lambda*[real(dFdyc);imag(dFdyc)] - dJdyc;
    grad = [dt_x; dt_y];   




end

