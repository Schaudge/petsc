#include <petscdevice_cuda.h>
#include <petscdevice.h>
#include "rosenbrock1.h"

__global__ void Rosenbrock1ObjAndGradCUDA_Internal(const PetscScalar x[], PetscScalar g[], PetscReal f[], PetscReal alpha)
{
  PetscReal t1, t2;

  t1 = x[1] - x[0]*x[0];
  t2 = 1 - x[0];

  g[0] = -4*alpha*(t1)*x[0] - 2*(t2);
  g[1] = 2*alpha*(t1);
  //TODO somehow it gives me zero even though it should be 1??
  f[0] = (alpha*t1*t1) + (t2*t2);
}

PetscErrorCode Rosenbrock1ObjAndGradCUDA(Vec X, Vec G, PetscReal *f, PetscReal alpha, PetscInt nn)
{
  PetscScalar *g;
  PetscMemType memtype_x, memtype_g;
  const PetscScalar *x;  

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayAndMemType(G, &g, &memtype_g));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));

  Rosenbrock1ObjAndGradCUDA_Internal<<<1,1>>>(x, g, f, alpha);
  // reduce all ff values together 

  PetscCall(VecRestoreArrayAndMemType(G, &g));
  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}


