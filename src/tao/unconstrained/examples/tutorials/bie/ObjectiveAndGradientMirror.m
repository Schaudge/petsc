function [targ, grad] = ObjectiveAndGradientMirror(X)
   %global rc
    %X = X(:);
%     rc = [1.585408096146205 1.586410414142497 1.599870032193204 1.594382797243828 1.582325014968503   1.579336250977204   1.579303885214841   1.589998101697224 1.595915032789817   1.594898776903782   1.596472823949763...
%    1.596493409383315 1.595692137165520 1.590842724679065 1.594925299623017...
%    1.592274533511911 1.598414560778699 1.575303206398311 1.578784932086548 1.58827795696953... 
% 1.598683865550920 1.581351791020520 1.590412681803260 1.586742278128158 1.594278945074662];
    nn = length(X);
   %ls = 1000;
    xp = X(1:nn/2);
    yp = X(nn/2+1:end);
    nc=length(xp);
    rc = 1.6 * ones(1,nc);
  %  lambda0 = 8;



        %directization points on the surface of a circle
    N = 10;

    %objective region
    x0c = 0.0;
    y0c = 40.0+23.0;
    rj = 60.0;
   %%weight for mutiobjectives
   alpha = 1.0;
  % beta = .01;
    %location of the source
    ys = 13.0; xs = 0;   src = xs + 1i * ys; %src_r = xs - 1i * ys;%-30:30;-3.5:0.1:3.5;
    yr = 13.0;
    thn = 0:0.01:2*pi;
  
    for ii = 1:nc
        GG{ii} = cylinder(rc(ii),xp(ii),yp(ii));
        GG{ii} = curvquad(GG{ii},'ptr',N,10);
        GG_r{ii} = cylinder_r(rc(ii),xp(ii),(yp(ii)-2*yr));
        GG_r{ii} = curvquad(GG_r{ii},'ptr',N,10);
    end
    k = sqrt(((2*pi*37.5e9)^2*2.05)/ ((2997.92458e8)^2 *sqrt(2) )); eta = k;       %2.0*pi/8.0;                    % wavenumber, SLP mixing amount 
    f =  @(z) sum(1i*100.0*besselh(0,1,k*abs(z-src))/4.0,2); %.*(-3.5<= real(z) & real(z) <= 3.5);   % known soln: interior source
        fgradx = @(z) sum(-1i/4.0*100.0*k*besselh(1,1,k*abs(z-src)).*(real(z-src))./abs(z-src),2); %.*(-3.5<= real(z) & real(z) <= 3.5);
        fgrady = @(z) sum(-1i/4.0*100.0*k*besselh(1,1,k*abs(z-src)).*(imag(z-src))./abs(z-src),2); %.*(-3.5<= real(z) & real(z) <= 3.5); %+...
                   %-1i/4.0*1*k*besselh(1,1,k*abs(z-src_r)).*(imag(z-src_r))./abs(z-src_r);
%    f =  @(z) 1.0*((-3.5 <= real(z)) & (real(z) <= 3.5)) .* exp(1i*k * imag(z));
%    fgradx = @(z) 0;
%    fgrady = @(z) 1.0 *1i * k * (-3.5 <= real(z) & real(z) <= 3.5) .*exp(1i * k * imag(z));

    rhs=[]; 
    for ii=1:nc
        
        rhs = [rhs -2.0*f(GG{ii}.x)];
       
    end

    A = nan(N*nc,N*nc);
    for ii=1:nc
        for jj=1:nc
            for i=1:N
                for j=1:N
                    A(i+(ii-1)*N,j+(jj-1)*N) = 2.0*CFIEnystKR_src_r(GG{ii}, GG_r{ii},GG{jj},i,j,k,eta);
                end
            end
        end
    end
     AA = (eye(N*nc)+ A);
    sigma = AA\rhs(:);

    dFdq =(AA);

    %targetarea: obj=x0c+rj*sin(thn)+1i*(y0c+rj*cos(thn));
    tobj=pi/6:0.02:pi/3; %remets pi/3
 %   trest = [0:0.02:pi/6 pi/3:.02:2*pi];
    obj=x0c+rj*sin(tobj)+1i*(y0c+rj*cos(tobj));
  %  objrest=x0c+rj*sin(trest)+1i*(y0c+rj*cos(trest));
    obj_r =conj(obj - 2*1i * yr);
    %objrest_r =conj(objrest - 2*1i * ys);
    

    uobj = zeros(length(obj), nc);
    %uobjj = zeros(length(obj));
    for ii=1:nc
        for jj=1:length(obj)       
            uobj(jj,ii)= evalCFIEhelm_src_r(obj(jj),obj_r(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
        end
    end
    ttarg = -sum((abs(sum(uobj,2))).^2);
    targ = ttarg; % + beta * ttargrest;



    %derivative of J wrt \sigma 
    B_r = zeros(length(obj), N*nc);
    B_i = B_r;
    uobjj = zeros(length(obj),1);
    for ii=1:nc
        for jj=1:length(obj)
            uobjj(jj) = sum(uobj(jj,:));
            for i=1:N
                a_r = CFIEnystKR_src_derJq_r(obj,obj_r,GG{ii},jj,i,k,eta) ;
                a_i = 1i*a_r;
                B_r(jj,i+(ii-1)*N) = 2.0 * real(conj(uobjj(jj)) * a_r);
                B_i(jj,i+(ii-1)*N) = 2.0 * real(conj(uobjj(jj)) * a_i);
            end
        end
    end

    dJdq_r = sum(B_r,1); % + beta *sum(Brest_r,1);
    dJdq_i =  sum(B_i,1); % + beta* sum(Brest_i,1);
    AA_q = [real(dFdq),-imag(dFdq); imag(dFdq), real(dFdq)];
    dJdq = [dJdq_r, dJdq_i];
    [lambda] = dJdq / AA_q;

    % pause


    dFdxc = [];
    for ii = 1:nc
        pff = zeros(size(GG{ii}.x));
        pff_x = pff;
        pff_y = pff;
        for jj = 1:nc
            if (ii == jj)
                for i = 1:N
                    for j = 1:N
                    vv_r_diag(i,j) = evalCFIEhelm_src_derF_K_r(GG_r{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta,i,j);
                    end
                 end
                 pff_x = pff_x + 0;
                 pff_y = pff_y -2.0 * sum(vv_r_diag,2);
            elseif (ii ~= jj) 
                [uu, vv] = evalCFIEhelm_src_derF_S(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta);
                [uu_r, vv_r] = evalCFIEhelm_src_derF_S_r(GG_r{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta);
                pff_x = pff_x+2.0*(uu-uu_r);
                pff_y = pff_y+2.0*(vv-vv_r);
                [uuu, vvv] = evalCFIEhelm_src_derF_S2(GG{jj}.x,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
                [uuu_r, vvv_r] = evalCFIEhelm_src_derF_S2_r(GG_r{jj}.x,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
                pf_x(N*(jj-1)+1:N*jj,ii) = 2.0 * (uuu - uuu_r);
                pf_y(N*(jj-1)+1:N*jj,ii)  = 2.0 * (vvv - vvv_r);
            end              
        end
        pf_x(N*(ii-1)+1:N*ii,ii) = pff_x+2.0*fgradx(GG{ii}.x);
        pf_y(N*(ii-1)+1:N*ii,ii) = pff_y+2.0*fgrady(GG{ii}.x);
    end
    dFdxc = pf_x;
    dFdyc = pf_y;
    




    dJdxc=[];
    dJdyc=[];
    %pos = 1;
    %pp = 1;
    
    for ii=1:nc 
   %ctrl = 1;
        for jj=1:length(obj)
            [uu, vv] = ((evalCFIEhelm_src_der_G_x(obj(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta)));
            [uu_r, vv_r] = ((evalCFIEhelm_src_der_G_x(obj_r(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta)));
            ppj_x(jj,ii) = uu - uu_r;
            ppj_y(jj, ii) = vv - vv_r;
            pj_x(jj,ii) = 2.0*real(conj(uobjj(jj))*(uu - uu_r)); % + .2 *(xp(ii)-xp(jj)) / ((xp(ii) - xp(jj))^2 + (yp(ii) - yp(jj))^2- (rc(ii) + rc(jj))^2);   ;
            pj_y(jj,ii) = 2.0*real(conj(uobjj(jj))*(vv - vv_r)); % + .2 * (yp(ii) - yp(jj))/((xp(ii) - xp(jj))^2 + (yp(ii) - yp(jj))^2- (rc(ii) + rc(jj))^2); ;
        end 
        dJdxc(ii) =  sum(pj_x(:,ii)); 
        dJdyc(ii) =  sum(pj_y(:,ii)); 

    end


% 
%     dJdxc = alpha *dJdxc; % + beta * dJdxcrest;
%     dJdyc = alpha *dJdyc; % + beta * dJdycrest;
    dt_x = lambda*[(real(dFdxc)); imag(dFdxc)] - dJdxc;
    dt_y = lambda*[real(dFdyc); imag(dFdyc)] - dJdyc;


    grad = [dt_x dt_y];


end
 
    %for jj = 1:length(obj)  
%   H = zeros(2*nc, 2*nc);
%     for ii = 1:nc
%         for kk = 1:nc
%            if (kk ~=ii)
%             H(ii,kk) = sum(2*real(conj(ppj_x(:,kk)).* ppj_x(:,ii))); %
%             H(ii, kk+nc) = sum(2*real(conj(ppj_y(:,kk)).* ppj_x(:,ii)));
%            else
%            
%            [uxx, uyy, uxy] = evalHessian(obj,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
%            [uxx_r, uyy_r, uxy_r] = evalHessian(obj_r,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
%            
%             H(ii,kk) = sum (2*real(conj(ppj_x(:,kk)).* ppj_x(:,ii)) +...
%                        2*real((sum(conj(uobj),2)) .* (uxx-uxx_r)) );
%             H(ii,kk+nc) = sum(2*real(conj(ppj_y(:,kk)).* ppj_x(:,ii)) +...
%                        2*real( (sum(conj(uobj),2)) .* (uxy-uxy_r)));
%            H(ii+nc,kk+nc) = sum(2*real(conj(ppj_y(:,kk)).* ppj_x(:,ii)) +...
%                        2*real((sum(conj(uobj),2)) .* (uyy-uyy_r)));
%            end
%         end
%     end
    %end 
%     dJdxcrest=[];
%     dJdycrest=[];
%     for ii=1:nc 
%         for jj=1:length(objrest)
%             [uu, vv] = ((evalCFIEhelm_src_der_G_x(objrest(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta)));
%             [uu_r, vv_r] = ((evalCFIEhelm_src_der_G_x(objrest_r(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta)));
%             pjrest_x(jj,ii) = 2.0*real(conj(uobjjrest(jj))*(uu - uu_r));
%             pjrest_y(jj,ii) = 2.0*real(conj(uobjjrest(jj))*(vv - vv_r));
%         end
%         dJdxcrest(ii) =  sum(pjrest_x(:,ii)); 
%         dJdycrest(ii) =  sum(pjrest_y(:,ii)); 
%     end


%%%%%%rest

%     Brest_r = zeros(length(objrest), N*nc);
%     Brest_i = Brest_r;
%     uobjjrest = zeros(length(objrest),1);
%     for ii=1:nc
%         for jj=1:length(objrest)
%             uobjjrest(jj) = sum(uobjrest(jj,:));
%             for i=1:N
%                 arest_r = CFIEnystKR_src_derJq_r(objrest,objrest_r,GG{ii},jj,i,k,eta) ;
%                 arest_i = 1i*arest_r;
%                 Brest_r(jj,i+(ii-1)*N) = 2.0 * real(conj(uobjjrest(jj)) * arest_r);
%                 Brest_i(jj,i+(ii-1)*N) = 2.0 * real(conj(uobjjrest(jj)) * arest_i);
%             end
%         end
%     end



%     uobjrest = zeros(length(objrest), nc);
%     for ii=1:nc
%         for jj=1:length(objrest)       
%             uobjrest(jj,ii)= evalCFIEhelm_src_r(objrest(jj),objrest_r(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
%         end
%     end
%     ttargrest = sum((abs(sum(uobjrest,2))).^2);
%    targ = alpha * ttarg+ beta * ttargrest;
%     
  %targ1 = targ;



%%bounds
    %ctrl = 0;
%     for ii = 1:nc
%         if (yp(ii) <= 13)
%            yp(ii) = 13+lambda0/8;
%         end
%         if (yp(ii) >= 123)
%             yp(ii) = 123 - lambda0/8;
%         end
%        if (xp(ii) <= -60)
%            xp(ii) = -60+lambda0/8;
%         end
%         if (xp(ii) >= 60)
%             xp(ii) = 60 - lambda0/8;
%         end
%     end


% ctrl = 0;  
% index = 1:nc;
% %[~, index] = sort(xp);
%  while (ctrl == 0) 
%     for ii = 1:nc
%         for jj = ii+1:nc            
%                dist = sqrt((xp(index(ii)) - xp (index(jj))) ^2 + (yp(index(ii)) -yp(index(jj)))^2);
%                 if ( dist <= rc( index(ii) ) + rc( index(jj) ) +1e-10 )
%                   yp(index(jj)) = yp(index(jj)) + ((rc(index(ii)) + rc(index(jj))   - dist) +  1 * lambda0/(8.0)) * (yp(index(jj)) - yp(index(ii)))/dist;
%                   xp(index(jj)) = xp(index(jj)) + ((rc(index(ii)) + rc(index(jj))   - dist) +  1 * lambda0/(8.0)) * (xp(index(jj)) - xp(index(ii)))/dist;                   
%                end            
%         end
%     end
%    %[~, index] = sort(xp);
%     ctrl2 = 0;
%     for ii = 1:nc
%         for jj = ii+1:nc
%              dist = sqrt((xp(index(ii)) - xp (index(jj))) ^2 + (yp(index(ii)) -yp(index(jj)))^2);
%                 if ( dist <= rc( index(ii) ) + rc( index(jj) ) +1e-10)
%                     ctrl2 = 1;
%                 end
%         end
%     end
%   if (ctrl2 == 1)
%      ctrl = 0;
%   else
%      ctrl =1;
%   end
% end