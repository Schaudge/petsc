/* Program usage: mpiexec -n 1 rosenbrock1 [-help] [all TAO options] */

/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>
#include "rosenbrock1.h" // defines AppCtx, AppCtxFormFunctionGradient(), and AppCtxFormHessian()

static char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem on a single processor.  We \n\
minimize the extended Rosenbrock function: \n\
   sum_{i=0}^{n/2-1} (alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2) \n\
or the chained Rosenbrock function:\n\
   sum_{i=0}^{n-1} alpha*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\n";

/* -------------- User-defined routines ---------- */
static PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  PetscReal   zero = 0.0;
  Vec         x; /* solution vector */
  Mat         H;
  Tao         tao; /* Tao solver context */
  PetscBool   flg, test_lmvm = PETSC_FALSE;
  PetscMPIInt size; /* number of processes running */
  AppCtx      user; /* user-defined application context */
  KSP         ksp;
  PC          pc;
  Mat         M;
  Vec         in, out, out2;
  PetscReal   mult_solve_dist;
  MPI_Comm    comm;

  /* Initialize TAO and PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  /* Initialize problem parameters */
  user.comm    = comm;
  user.n       = 2;
  user.alpha   = 99.0;
  user.chained = PETSC_FALSE;
  /* Check for command line arguments to override defaults */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &user.alpha, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-chained", &user.chained, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_lmvm", &test_lmvm, &flg));

  /* Allocate vectors for the solution and gradient */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
  PetscCall(AppCtxCreateHessianMatrices(&user, &H, NULL));

  /* The TAO code begins here */

  /* Create TAO solver with desired solution method */
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));

  /* Set solution vec and an initial guess */
  PetscCall(VecSet(x, zero));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routines for function, gradient, hessian evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, &user));
  PetscCall(TaoSetHessian(tao, H, H, FormHessian, &user));

  /* Test the LMVM matrix */
  if (test_lmvm) PetscCall(PetscOptionsSetValue(NULL, "-tao_type", "bqnktr"));

  /* Check for TAO command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  /* Test the LMVM matrix */
  if (test_lmvm) {
    PetscCall(TaoGetKSP(tao, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCLMVMGetMatLMVM(pc, &M));
    PetscCall(VecDuplicate(x, &in));
    PetscCall(VecDuplicate(x, &out));
    PetscCall(VecDuplicate(x, &out2));
    PetscCall(VecSet(in, 1.0));
    PetscCall(MatMult(M, in, out));
    PetscCall(MatSolve(M, out, out2));
    PetscCall(VecAXPY(out2, -1.0, in));
    PetscCall(VecNorm(out2, NORM_2, &mult_solve_dist));
    if (mult_solve_dist < 1.e-11) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Inverse error of LMVM MatMult and MatSolve: < 1.e-11\n"));
    } else if (mult_solve_dist < 1.e-6) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Inverse error of LMVM MatMult and MatSolve: < 1.e-6\n"));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Inverse error of LMVM MatMult and MatSolve is not small: %e\n", (double)mult_solve_dist));
    }
    PetscCall(VecDestroy(&in));
    PetscCall(VecDestroy(&out));
    PetscCall(VecDestroy(&out2));
  }

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&H));

  PetscCall(PetscFinalize());
  return 0;
}

/*
  FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

  Input Parameters:
+ tao  - the Tao context
. X    - input vector
- ptr  - optional user-defined context, as set by TaoSetFunctionGradient()

  Output Parameters:
+ f - function value
- G - vector containing the newly evaluated gradient

  Note:
  Some optimization methods ask for the function and the gradient evaluation
  at the same time.  Evaluating both at once may be more efficient that
  evaluating each separately.
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(AppCtxFormFunctionGradient(user, X, f, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormHessian - Evaluates Hessian matrix.

  Input Parameters:
+ tao   - the Tao context
. x     - input vector
- ptr   - optional user-defined context, as set by TaoSetHessian()

  Output Parameters:
+ H     - Hessian matrix
- Hpre  - Preconditiong matrix

  Note:  Providing the Hessian may not be necessary.  Only some solvers
  require this matrix.
*/
PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(AppCtxFormHessian(user, X, H, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex

   test:
     requires: !single
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4

   test:
     suffix: 2
     requires: !single
     args: -tao_monitor_short -tao_type lmvm -tao_gatol 1.e-3

   test:
     suffix: 3
     requires: !single
     args: -tao_monitor_short -tao_type ntr -tao_gatol 1.e-4

   test:
     suffix: 4
     requires: !single
     args: -tao_monitor_short -tao_type ntr -tao_mf_hessian -tao_ntr_pc_type none -tao_gatol 1.e-4

   test:
     suffix: 5
     requires: !single
     args: -tao_monitor_short -tao_type bntr -tao_gatol 1.e-4

   test:
     suffix: 6
     requires: !single
     args: -tao_monitor_short -tao_type bntl -tao_gatol 1.e-4

   test:
     suffix: 7
     requires: !single
     args: -tao_monitor_short -tao_type bnls -tao_gatol 1.e-4

   test:
     suffix: 8
     requires: !single
     args: -tao_monitor_short -tao_type bntr -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4

   test:
     suffix: 9
     requires: !single
     args: -tao_monitor_short -tao_type bntl -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4

   test:
     suffix: 10
     requires: !single
     args: -tao_monitor_short -tao_type bnls -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4

   test:
     suffix: 11
     requires: !single
     args: -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmbroyden

   test:
     suffix: 12
     requires: !single
     args: -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmbadbroyden

   test:
     suffix: 13
     requires: !single
     args: -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbroyden

   test:
     suffix: 14
     requires: !single
     args: -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmbfgs

   test:
     suffix: 15
     requires: !single
     args: -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdfp

   test:
     suffix: 16
     requires: !single
     args: -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmsr1

   test:
     suffix: 17
     requires: !single
     args: -tao_monitor_short -tao_gatol 1e-4 -tao_type bqnls

   test:
     suffix: 18
     requires: !single
     args: -tao_monitor_short -tao_gatol 1e-4 -tao_type blmvm

   test:
     suffix: 19
     requires: !single
     args: -tao_monitor_short -tao_gatol 1e-4 -tao_type bqnktr -tao_bqnk_mat_type lmvmsr1

   test:
     suffix: 20
     requires: !single
     args: -tao_monitor -tao_gatol 1e-4 -tao_type blmvm -tao_ls_monitor

   test:
     suffix: 21
     requires: !single
     args: -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbadbroyden

   test:
     suffix: 22
     requires: !single
     args: -tao_max_it 1 -tao_converged_reason

   test:
     suffix: 23
     requires: !single
     args: -tao_max_funcs 0 -tao_converged_reason

   test:
     suffix: 24
     requires: !single
     args: -tao_gatol 10 -tao_converged_reason

   test:
     suffix: 25
     requires: !single
     args: -tao_grtol 10 -tao_converged_reason

   test:
     suffix: 26
     requires: !single
     args: -tao_gttol 10 -tao_converged_reason

   test:
     suffix: 27
     requires: !single
     args: -tao_steptol 10 -tao_converged_reason

   test:
     suffix: 28
     requires: !single
     args: -tao_fmin 10 -tao_converged_reason

   test:
     suffix: snes
     requires: !single
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtontr -snes_atol 1.e-4 -pc_type none -tao_mf_hessian -ksp_type cg

   test:
     suffix: snes_ls_armijo
     requires: !single
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtonls -snes_atol 1.e-4 -pc_type none -tao_mf_hessian -snes_linesearch_monitor -snes_linesearch_order 1

   test:
     suffix: snes_tr_cgnegcurve_kmdc
     requires: !single
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtontr -snes_atol 1.e-4 -pc_type none -ksp_type cg -snes_tr_kmdc 0.9 -ksp_converged_neg_curve -ksp_converged_reason

   test:
     suffix: snes_ls_lmvm
     requires: !single
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtonls -snes_atol 1.e-4 -pc_type lmvm -tao_mf_hessian

TEST*/
