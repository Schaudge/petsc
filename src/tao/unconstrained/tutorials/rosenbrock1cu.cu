#include <petscdevice_cuda.h>
#include <petscdevice.h>
#include "rosenbrock1.h"
#include <cuda.h>

__global__ void Rosenbrock1ObjAndGradCUDA_Kernel(const PetscScalar x[], PetscScalar g[], PetscReal f[], PetscReal alpha, PetscInt nn)
{
  PetscReal t1, t2;
  int i;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;//1D grid
  PetscInt tid = threadIdx.x;

  __shared__ double f_array[1024];
  f_array[tid] = 0.0;

  if (idx >= nn) return;

  int total = blockDim.x * gridDim.x;
  for (i = tid; i< nn; i+=total) {
    t1 = x[2*i+1] - x[2*i]*x[2*i];
    t2 = 1 - x[2*i];
  
    g[2*i] = -4*alpha*(t1)*x[2*i] - 2.*(t2);
    g[2*i+1] = 2*alpha*(t1);
    f_array[tid] += alpha*t1*t1 + t2*t2;
  }

  // Reduction on f_array
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      f_array[tid] += f_array[tid+s];
    }
    __syncthreads();
  }

  if (tid ==0) atomicAdd(f,f_array[0]);
}

PetscErrorCode Rosenbrock1ObjAndGradCUDA(Vec X, Vec G, PetscReal *f, PetscReal alpha, PetscInt nn)
{
  PetscScalar *g;
  PetscMemType memtype_x, memtype_g;
  const PetscScalar *x;  

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayAndMemType(G, &g, &memtype_g));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));

  // n_threads is hardware dependant... Chose 32 for test case. 
  Rosenbrock1ObjAndGradCUDA_Kernel<<<1,32>>>(x, g, f, alpha, nn);
  // reduce all ff values together 

  PetscCall(VecRestoreArrayAndMemType(G, &g));
  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}


