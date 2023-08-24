const char help[] = "Copy of rosenbrock1.c\n";

/* ------------------------------------------------------------------------

  Copy of rosenbrock1.c.
  Once petsc test harness supports conditional linking, we can remove this duplicate.
  See https://gitlab.com/petsc/petsc/-/issues/1173
  ------------------------------------------------------------------------- */

#include "rosenbrock1.h"

#include <cuda_profiler_api.h>

int main(int argc, char **argv)
{
  Vec           x;    /* solution vector */
  Vec           g;    /* gradient vector */
  Mat           H;    /* Hessian matrix */
  Tao           tao;  /* Tao solver context */
  AppCtx        user; /* user-defined application context */

  /* Initialize TAO and PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(AppCtxCreate(PETSC_COMM_WORLD, &user));
  PetscCall(CreateHessian(user, &H));
  PetscCall(CreateVectors(user, H, &x, &g));

  /* The TAO code begins here */

  /* Create TAO solver with desired solution method */
  PetscCall(TaoCreate(user->comm, &tao));
  PetscCall(TaoSetFromOptions(tao));

  /* Set solution vec and an initial guess */
  PetscCall(VecZeroEntries(x));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routines for function, gradient, hessian evaluation */
  PetscCall(TaoSetObjective(tao, FormObjective, user));
  PetscCall(TaoSetObjectiveAndGradient(tao, g, FormObjectiveGradient, user));
  PetscCall(TaoSetGradient(tao, g, FormGradient, user));
  PetscCall(TaoSetObjectiveAndGradient(tao, g, FormObjectiveGradient, user));
  PetscCall(TaoSetHessian(tao, H, H, FormHessian, user));

  /* SOLVE THE APPLICATION */
  cudaProfilerStart();
  PetscCall(TaoSolve(tao));
  cudaProfilerStop();

  PetscCall(TestLMVM(tao));

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&H));
  PetscCall(AppCtxDestroy(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex cuda

  test:
    output_file: output/rosenbrock1_1.out
    args: -mat_type aijcusparse -tao_smonitor -tao_type nls -tao_gatol 1.e-4
    requires: !single

TEST*/
