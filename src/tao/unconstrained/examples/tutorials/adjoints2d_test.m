close all;

fontsize_labels = 14;
fontsize_grid   = 12;
fontname = 'Times';

P=8;
Ex=6;
Ey=6;

N=(P-1)*Ex;
M=(P-1)*Ey;

run('tomesh.m');
x=grid(1:2:end);
y=grid(2:2:end);

%build grid
xg=reshape(x,N,M);
yg=reshape(y,N,M);

%choose iteration of the optimization to check variables
ii=3;
file=sprintf('PDEadjoint/optimize%02d.m',ii);  
run(file)
%inital conditions for TS
u_init=reshape(Init_ts(1:2:end),N,M);
v_init=reshape(Init_ts(2:2:end),N,M);
% %Current solution.. not needed
% u_cur=reshape(Curr_sol(1:2:end),N,M);
% v_cur=reshape(Curr_sol(2:2:end),N,M);

%gradient
u_g=reshape(adj(1:2:end),N,M);
v_g=reshape(adj(2:2:end),N,M);
figure(1)
mesh(xg,yg,v_g)
title('adj')
%inital condition of adjoints !!!!!!this one breaks
u_adj=reshape(Init_adj(1:2:end),N,M);
v_adj=reshape(Init_adj(2:2:end),N,M);
figure(2)
mesh(xg,yg,u_adj)
title('Init_adj')
%solution of the forward solve
u_fwd=reshape(fwd(1:2:end),N,M);
v_fwd=reshape(fwd(2:2:end),N,M);
figure(3)
mesh(xg,yg,v_fwd)
title('fwd')
%objective
u_obj=reshape(obj(1:2:end),N,M);
v_obj=reshape(obj(2:2:end),N,M);
figure(4)
mesh(xg,yg,v_obj)
title('obj')
%compute difference between forward and ojective to see how far off one is
err=v_obj-v_fwd;
figure(5)
mesh(xg,yg,err)
title('err')
disp('Norm of error in 2-norm:')
norm(err)

