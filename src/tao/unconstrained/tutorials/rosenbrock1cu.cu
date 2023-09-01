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
