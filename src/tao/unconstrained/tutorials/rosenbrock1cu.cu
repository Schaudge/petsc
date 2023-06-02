#include <petscdevice_cuda.h>
#include <petscdevice.h>
#include "rosenbrock1.h"

__global__ void Rosenbrock1ObjAndGradCUDA_Internal(const PetscScalar *x, PetscScalar *g, PetscReal *f, PetscReal alpha, PetscInt nn, PetscScalar *t1, PetscScalar *t2, PetscScalar *ff)
{

  *t1 = x[1] - x[0]*x[0];
  *t2 = 1 - x[0];
  *ff = alpha*(*t1)*(*t1) + (*t2)*(*t2);

  g[0] = -4*alpha*(*t1)*x[0] - 2*(*t2);
  g[1] = 2*alpha*(*t1);

  *f = *ff;
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

  Rosenbrock1ObjAndGradCUDA_Internal<<<1,1>>>(x, g, f, alpha, nn, t1,t2,ff);

  PetscCall(VecRestoreArrayAndMemType(G, &g));
  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscCallCUDA(cudaFree(t1));
  PetscCallCUDA(cudaFree(t2));
  PetscCallCUDA(cudaFree(ff));

  PetscFunctionReturn(PETSC_SUCCESS);
}


