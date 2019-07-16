%global rc;
close all;
lambda0 = 8;
%r = rc
% X = [10.8386
% 17.5658
% 9.90837
% 23.697
% 46.482
% 45.5083
% 54.5803
% 48.7436];




%xp = 0; %0 10 10]; % -5 25]  %[-20 0];% 20 -20 0 20 ]; % 3 -3 3 ]
%yp= 0;  %10 0 10]; % 10 30];  %[-3 -3]; %-3 6 6 6]; %, -20 -26 -26]


% yp0 = 21;     %21 %45;
% xp0 =0;      %-16;
% ln = linspace(0, 8, 2); %ln = linspace(0, 8, 2)
% [xp,yp] = ndgrid(ln,ln);
% 
% %xp = 0; %0 10 10]; % -5 25]  %[-20 0];% 20 -20 0 20 ]; % 3 -3 3 ]
% %yp= 0;  %10 0 10]; % 10 30];  %[-3 -3]; %-3 6 6 6]; %, -20 -26 -26]
% 
% xp = xp + xp0; 
% %xp = [0 0 0 0];
% yp = yp + yp0;
% %yp = [21 29 37 45];
% X = [(xp(:))'  (yp(:))'];
  X =[-17.5317
-10.5545
5.58774
10.3226
18.7967
-15.2141
-10.5326
-8.99636
-2.34064
30.0552
-15.7879
-15.7333
-0.0361481
-3.47565
29.4847
-15.3244
-12.7272
1.71507
-3.7387
28.6912
-18.1995
-10.2636
2.16098
-1.07204
17.8433
24.0103
16.8443
24.3595
18.9953
27.7415
29.3056
27.4957
36.8556
36.9777
10.6514
38.4062
37.4534
37.0375
38.1569
21.7754
43.5395
45.4001
50.5346
54.6873
35.0082
53.0339
54.2002
56.9149
64.1901
48.2374
];

    nn = length(X);
    xp = X(1:nn/2);
    yp = X(nn/2+1:end);
    %rc = 1.5875;
    nc=length(xp);
  rc = 1.6 *ones(1,nc);
        %directization points on the surface of a circle
    N = 10;
   
        %objective region
    x0c = 0.0;
    y0c = 40+23;
    rj = 60.0;
    thn = 0:0.1:2*pi;
    %location of the source
    ys = 13; xs = 0;  src= xs + 1i * ys; %-30:30;
ctrl = 0;
%rc = 1.6;
% index = 1:nc ;
% [~, index] = sort(xp);
%  while (ctrl == 0) 
%     for ii = index
%         for jj = index
%             if (ii ~= jj)
%                dist = sqrt((xp(ii) - xp (jj)) ^2 + (yp(ii) -yp(jj))^2);
%                 if (dist <= rc(ii) + rc(jj))
%                   pause
%                   yp(jj) = yp(jj) + ((rc(ii) + rc(jj)  - dist) +  1* lambda0/(8.0)) * (yp(jj) - yp(ii))/dist;
%                   xp(jj) = xp(jj) + ((rc(ii) + rc(jj) - dist) +  1 * lambda0/(8.0)) * (xp(jj) - xp(ii))/dist;
%     
%                    
%                end
%             end
%         end
%     end
%    %[~, index] = sort(xp);
%     ctrl2 = 0;
%     for ii = index
%         for jj = index
%             if (ii ~= jj)
%                 if (sqrt((xp(ii) - xp (jj)) ^2 + (yp(ii) -yp(jj))^2) <= rc(ii) + rc(jj))
%                     ctrl2 = 1;
%                end
%             end
% 
%         end
%     end
%   if (ctrl2 == 1)
%      ctrl = 0;
%   else
%      ctrl =1;
%   end
%    
% end
   figure(1)
   
    for ii=1:nc
        GG{ii} = cylinder(rc(ii),xp(ii),yp(ii));
        GG{ii} = curvquad(GG{ii},'ptr',N,10);
        GG_r{ii} = cylinder_r(rc(ii),xp(ii),(yp(ii) - 2*ys));
        GG_r{ii} = curvquad(GG_r{ii},'ptr',N,10);
       quiver(real(GG{ii}.x),imag(GG{ii}.x),real(GG{ii}.nx),imag(GG{ii}.nx));
        hold on
        plot(real(GG{ii}.x),imag(GG{ii}.x),'k','LineWidth',2)
        hold on
       quiver(real(GG_r{ii}.x),imag(GG_r{ii}.x),real(GG_r{ii}.nx),imag(GG_r{ii}.nx));
        hold on
        plot(real(GG_r{ii}.x),imag(GG_r{ii}.x),'k','LineWidth',2)
        hold on
    end
   
    plot(x0c+rj*sin(thn),y0c+rj*cos(thn),'k','LineWidth',4)
    %in thick red the objective region
    plot(x0c+rj*sin(pi/6:0.1:pi/3),y0c+rj*cos(pi/6:0.1:pi/3),'r','LineWidth',3) %remets pi/3
    %source of incident wave
    plot(xs,ys,'ks','MarkerSize',8,'LineWidth',2)
    hold off
    axis equal
   
    
     k = 2*pi/8;  eta = k;
    % wavenumber, SLP mixing amount 
        f =  @(z) sum(1i*1*besselh(0,1,k*abs(z-src))/4.0,2);   % known soln: interior source
        fgradx = @(z) sum(-1i/4.0*1*k*besselh(1,1,k*abs(z-src)),2);
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
                    A(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src_r(GG{ii}, GG_r{ii},GG{jj},i,j,k,eta);
                end
            end
        end
    end
       AA = (eye(N*nc)+ A);
%     %PP = diag(diag(AA));
%    [LL, UU] = lu(AA);
%     CC = inv(LL); %  AA * inv(PP);
%     y = CC \ rhs(:);
%     sigma = UU\y;  %PP \ y;
    dFdq =  (eye(N*nc)+ A);
   sigma = AA \ rhs(:);

    %targetarea: obj=x0c+rj*sin(thn)+1i*(y0c+rj*cos(thn));
    tobj=pi/6:0.02:pi/3; %remets pi/3
    obj=x0c+rj*sin(tobj)+1i*(y0c+rj*cos(tobj));
    obj_r =conj(obj - 2*1i*ys);

    uobj = zeros(length(obj), nc);
    %uobjj = zeros(length(obj));
    for ii=1:nc
        for jj=1:length(obj)
            uobj(jj,ii) = evalCFIEhelm_src_r(obj(jj), obj_r(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
            %pause% evaluate soln               
        end 

    end
    targ = -sum((abs(sum(uobj,2))).^2)
       
    xg=-60:0.5:60;
    yg=13:0.5:125;
    [xx yy] = meshgrid(xg,yg); t = xx + 1i*yy; t_r = conj(t -2 *1i *ys);    % targets for plot
    u=zeros(size(xx));
    for ii=1:nc
        u = u+evalCFIEhelm_src_r(t,t_r,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);         % evaluate soln
    end
    figure(2);
    pcolor(xx,yy,log(sqrt(real(u).^2+imag(u).^2))); shading interp;
    hold on
    for ii=1:nc
        plot(real(GG{ii}.x),imag(GG{ii}.x),'k','LineWidth',2)
    end
    plot(x0c+rj*sin(thn),y0c+rj*cos(thn),'k','LineWidth',4)
    %in thick red the objective region
    plot(x0c+rj*sin(pi/6:0.1:pi/3),y0c+rj*cos(pi/6:0.1:pi/3),'r','LineWidth',6)
    
    axis image
    colorbar
    hold off