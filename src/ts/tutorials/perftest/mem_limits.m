max_dofs_cpu=2^26;
max_dofs_gpu=3*2^22;

%mesh has a fixed number of elements but poly order may change
poly=4; Nel=125*10^3;

mesh_dofs=Nel*poly^3;

%this has to be less that 1 to fit entirely in DDR
mesh_dofs/max_dofs_cpu
%this has to be less that 1 to fit entirely in HBM
mesh_dofs/max_dofs_gpu

