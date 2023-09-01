const char help[] = "Copy of rosenbrock1.c\n";

/* ------------------------------------------------------------------------

  Copy of rosenbrock1.c.
  Once petsc test harness supports conditional linking, we can remove this duplicate.
  See https://gitlab.com/petsc/petsc/-/issues/1173
  ------------------------------------------------------------------------- */

#include "rosenbrock1.h"

int main(int argc, char **argv)
{
  /* Initialize TAO and PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(RosenbrockMain());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex cuda

  test:
    output_file: output/rosenbrock1cu_1.out
    args: -mat_type aijcusparse -tao_smonitor -tao_type nls -tao_gatol 1.e-4
    requires: !single

  test:
    suffix: chained_bfgs
    output_file: output/rosenbrock1cu_chained_bfgs.out
    args: -mat_type aijcusparse -tao_smonitor -chained -n 10000 -tao_type lmvm -tao_max_it 20 -tao_lmvm_mat_lmvm_scale_type none

  test:
    suffix: bfgs_timings
    output_file: output/rosenbrock1cu_bfgs_timings.out
    args: -tao_type lmvm -tao_lmvm_mat_lmvm_scale_type none -tao_lmvm_mat_type lmvmbfgs 

  test:
    suffix: cdbfgs_inplace_timings
    output_file: output/rosenbrock1cu_cdbfgs_inplace_timings.out
    args: -tao_type lmvm -tao_lmvm_mat_lmvm_scale_type none -tao_lmvm_mat_type lmvmcdbfgs -mat_lbfgs_type cd_inplace

  test:
    suffix: cdbfgs_reorder_timings
    output_file: output/rosenbrock1cu_cdbfgs_reorder_timings.out
    args: -tao_type lmvm -tao_lmvm_mat_lmvm_scale_type none -tao_lmvm_mat_type lmvmcdbfgs -mat_lbfgs_type cd_reorder

TEST*/
=======
#include <petscdevice_cuda.h>
#include <petscdevice.h>
#include "rosenbrock1.h"
#include <cuda.h>
#include <thrust/reduce.h>

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
  for (unsigned int s=blockDim.x/2; s>0 ; s>>=1) {
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

  // ObjGrad Together
  Rosenbrock1ObjAndGradCUDA_Kernel<<<1,256>>>(x, g, f, alpha, nn);
  

  PetscCall(VecRestoreArrayAndMemType(G, &g));
  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}
