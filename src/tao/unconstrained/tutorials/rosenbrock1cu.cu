#include <petscdevice_cuda.h>
#include <petscdevice.h>
#include "rosenbrock1.h"

__global__ void Rosenbrock1ObjAndGradCUDA_Internal(const PetscScalar x[], PetscScalar g[], PetscReal f[], PetscReal alpha)
{
  PetscReal t1, t2, ff;
  size_t idx = 0;

  t1 = x[idx + 1] - x[idx]*x[idx];
  t2 = 1 - x[idx];
  ff = alpha*(t1)*(t1) + (t2)*(t2);

  g[idx] = -4*alpha*(t1)*x[idx] - 2*(t2);
  g[idx+1] = 2*alpha*(t1);

  f[idx] = ff;
}

PetscErrorCode Rosenbrock1ObjAndGradCUDA(Vec X, Vec G, PetscReal *f, PetscReal alpha, PetscInt nn)
{
  PetscScalar *t1, *t2, *ff, *g;
  PetscMemType memtype_x, memtype_g;
  const PetscScalar *x;  

  PetscFunctionBeginUser;
  PetscCallCUDA(cudaMalloc((void **)&t1, 1*sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&t2, 1*sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&ff, 1*sizeof(PetscScalar)));
  PetscCall(VecGetArrayAndMemType(G, &g, &memtype_g));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));

  Rosenbrock1ObjAndGradCUDA_Internal<<<1,1>>>(x, g, f, alpha);

  // reduce all ff values together 

  PetscCall(VecRestoreArrayAndMemType(G, &g));
  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscCallCUDA(cudaFree(t1));
  PetscCallCUDA(cudaFree(t2));
  PetscCallCUDA(cudaFree(ff));

  PetscFunctionReturn(PETSC_SUCCESS);
}


