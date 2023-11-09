#include <petsctao.h>
#include <petsctaoregularizer.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates various ways to use TaoRegularizer. \n";

typedef struct {
  PetscBool y_bool;
  PetscInt  n;       /* dimension */
  PetscInt  problem; /* Types of problems to solve. */
  PetscInt  g_type;  /* How to formulate regularizer */
  PetscReal stepsize;
  PetscReal mu1; /* Parameter for soft-threshold */
  Mat       A;
  Vec       b, workvec, y;
} AppCtx;

/* Objective
 *
 * f(x) = 0.5 x.T A x - b.T x */
PetscErrorCode UserObj(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecScale(user->workvec, user->stepsize));
  PetscCall(VecAXPY(user->workvec, -1., user->b));
  PetscCall(VecTDot(user->workvec, X, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Gradient: grad f = Ax - b */
PetscErrorCode UserGrad(Tao tao, Vec X, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, G));
  PetscCall(VecAXPY(G, -1., user->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 x.T A x - b.T x
 * grad f = A x - b                         */
PetscErrorCode UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecWAXPY(G, -1., user->b, user->workvec));
  PetscCall(VecTDot(user->workvec, X, f));
  *f *= user->stepsize;

  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Full Obj and Grad for iterative refinement.
 *
 * f(x) = 0.5 x.T A x - b.T x + 0.5 \|x-y\|_2^2
 * grad f = A x + x  - b - y                       */
PetscErrorCode FullUserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp, reg_val, stepsize;

  PetscFunctionBegin;
  stepsize = user->stepsize;
  PetscCall(VecWAXPY(user->workvec, -1, X, user->y));
  PetscCall(VecNorm(user->workvec, NORM_2, &reg_val));
  reg_val *= stepsize;
  /* Ax + x */
  PetscCall(MatMult(user->A, X, G));
  PetscCall(VecAXPY(G, 1., X));
  /* 0.5 * x^T (Ax + x) */
  PetscCall(VecTDot(G, X, f));
  *f *= stepsize;
  /* Full grad f(x) */
  PetscCall(VecAXPY(G, -1., user->b));
  /* b^T x */
  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* L2 Metric. step*\|X\|_2^2
 * This is to set Tao routines for TaoRegularizer  */
PetscErrorCode L2_ObjGrad_Tao(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscReal      temp, step;
  TaoRegularizer reg;

  PetscFunctionBegin;
  PetscCall(TaoGetRegularizer(tao, &reg));
  PetscCall(TaoRegularizerGetScale(reg, &step));
  PetscCall(VecCopy(X, G));
  PetscCall(VecScale(G, step * 2));
  PetscCall(VecNorm(X, NORM_2, &temp));
  temp = PetscPowReal(temp, 2);
  *f   = step * temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* L2 Metric. step*\|X-y\|_2^2 */
PetscErrorCode L2_ObjGrad(TaoRegularizer reg, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx   *user = (AppCtx *)ptr;
  PetscReal temp, step;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetScale(reg, &step));
  PetscCall(VecCopy(X, G));
  if (user->y_bool) {
    Vec y;
    PetscCall(TaoRegularizerGetCentralVector(reg, &y));
    PetscCall(VecAXPY(G, -1., y));
  }
  PetscCall(VecTDot(G, G, &temp));
  PetscCall(VecScale(G, step * 2));
  *f   = step * temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Obj(TaoRegularizer reg, Vec X, PetscReal *f, void *ptr)
{
  AppCtx   *user = (AppCtx *)ptr;
  PetscReal temp, step;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetScale(reg, &step));
  if (user->y_bool) {
    Vec y;
    PetscCall(TaoRegularizerGetCentralVector(reg, &y));
    PetscCall(VecWAXPY(user->workvec, -1., y, X));
    PetscCall(VecTDot(user->workvec, user->workvec, &temp));
  } else {
    PetscCall(VecTDot(X, X, &temp));
  }
  *f   = step * temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Grad(TaoRegularizer reg, Vec X, Vec G, void *ptr)
{
  AppCtx   *user = (AppCtx *)ptr;
  PetscReal step;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetScale(reg, &step));
  if (user->y_bool) {
    Vec y;
    PetscCall(TaoRegularizerGetCentralVector(reg, &y));
    PetscCall(VecWAXPY(G, -1., y, X));
  } else {
    PetscCall(VecCopy(X, G));
  }
  PetscCall(VecScale(G, step * 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao            tao, tao_full, reg_tao;
  TaoRegularizer reg;
  Vec            x, x_full;
  Mat            temp_mat;
  PetscMPIInt    size;
  AppCtx         user;
  PetscBool      flg;
  PetscRandom    rctx;
  PetscReal      vec_dist;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.problem  = 0;
  user.g_type   = 0;
  user.n        = 10;
  user.stepsize = 0.5;
  user.mu1      = 1;
  user.y_bool   = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  /* Types of problems to solve */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-problem", &user.problem, &flg));
  /* Determines whether y is zeros, or random */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-central_vec", &user.y_bool, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user.stepsize, &flg));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x_full));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.workvec));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.y));
  /* x: zero vec */
  PetscCall(VecZeroEntries(x));

  /* A,b data */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, user.n, user.n, NULL, &temp_mat));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetSeed(rctx, 1234));
  PetscCall(MatSetRandom(temp_mat, rctx));
  PetscCall(MatAssemblyBegin(temp_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(temp_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatTransposeMatMult(temp_mat, temp_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user.A));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.b));
  PetscCall(PetscRandomSetSeed(rctx, 5678));
  PetscCall(VecSetRandom(user.b, rctx));
  PetscCall(MatDestroy(&temp_mat));
  /* y: random vec */
  PetscCall(PetscRandomSetSeed(rctx, 9012));
  PetscCall(VecSetRandom(user.y, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  /* tao      = 0.5 x^T A x - b^T x
   * tao_full = 0.5 x^T A x - b^T x + 0.5 \|x-y\|_2^2 */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao_full));
  //TODO reg_tao - is this legal?
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &reg_tao));
  PetscCall(TaoSetFromOptions(reg_tao));//im not even setting solution vec, nor type... will this work, or do i need to set something?

  PetscCall(TaoSetType(tao, TAOCG));
  PetscCall(TaoSetType(tao_full, TAOCG));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetSolution(tao_full, x_full));
  PetscCall(TaoSetOptionsPrefix(tao, "reg_"));
  PetscCall(TaoSetOptionsPrefix(tao_full, "normal_"));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetFromOptions(tao_full));
  PetscCall(TaoSetObjectiveAndGradient(tao_full, NULL, FullUserObjGrad, (void *)&user));

  /* problem:
   *
   * 0: f: ObjGrad
   * 1: f: Obj and Grad  */
  switch (user.problem) {
  case 0:
    PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *)&user));
    break;
  case 1:
    PetscCall(TaoSetObjective(tao, UserObj, (void *)&user));
    PetscCall(TaoSetGradient(tao, NULL, UserGrad, (void *)&user));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
  }

  /* Create Regularizer, g(x,y) = 0.5 \|x-y\|_2^2
   *
   * g_type:
   *
   * 0: ObjGrad
   * 1: Obj and Grad
   * 2: Built-in type
   * 3: Tao              */
  PetscCall(TaoRegularizerCreate(PetscObjectComm((PetscObject)tao), &reg));

  /* Set Regularizers */
  switch (user.g_type) {
  case 0:
    PetscCall(TaoRegularizerSetObjectiveAndGradient(reg, L2_ObjGrad, (void *)&user));
    break;
  case 1:
    PetscCall(TaoRegularizerSetObjective(reg, L2_Obj, (void *)&user));
    PetscCall(TaoRegularizerSetGradient(reg, L2_Grad, (void *)&user));
    break;
  case 2:
    PetscCall(TaoRegularizerSetType(reg, TAOREGULARIZERL2));
    break;
  case 3:
    PetscCall(TaoSetObjectiveAndGradient(reg_tao, NULL, L2_ObjGrad_Tao, (void *)&user));
    PetscCall(TaoRegularizerUseTaoRoutines(reg, reg_tao));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Regularizer formulation type.");
  }
  /* SetScale needs to be called after SetType */
  PetscCall(TaoRegularizerSetScale(reg, user.stepsize));


  /* Solve full version */
  /* Solve Regularizer version */
  PetscCall(TaoSetRegularizer(tao, reg));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoSolve(tao_full));

  /* Testing Regularizer version vs Full version */
  PetscCall(VecAXPY(x, -1., x_full));
  PetscCall(VecNorm(x, NORM_2, &vec_dist));
  if (vec_dist < 1.e-6) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Ful TaoSolve: < 1.e-6\n"));
  } else {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Full TaoSolve: %e\n", (double)vec_dist));
  }

  PetscCall(TaoRegularizerDestroy(&reg));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoDestroy(&tao_full));
  PetscCall(TaoDestroy(&reg_tao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x_full));
  PetscCall(VecDestroy(&user.b));
  PetscCall(VecDestroy(&user.workvec));
  PetscCall(MatDestroy(&user.A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 0
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0 -g_type 0 -central_vec 0
      requires: !single

   test:
      suffix: 1
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0 -g_type 1 -central_vec 0
      requires: !single

   test:
      suffix: 2
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0 -g_type 2 -central_vec 0
      requires: !single

   test:
      suffix: 3
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0 -g_type 3 -central_vec 0
      requires: !single

   test:
      suffix: 4
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1 -g_type 0 -central_vec 0
      requires: !single

   test:
      suffix: 5
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1 -g_type 1 -central_vec 0
      requires: !single

   test:
      suffix: 6
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1 -g_type 2 -central_vec 0
      requires: !single

   test:
      suffix: 7
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1 -g_type 3 -central_vec 0
      requires: !single

   test:
      suffix: 8
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0 -g_type 0 -central_vec 1
      requires: !single

   test:
      suffix: 9
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0 -g_type 1 -central_vec 1
      requires: !single

   test:
      suffix: 10
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0 -g_type 2 -central_vec 1
      requires: !single

   test:
      suffix: 11
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0 -g_type 3 -central_vec 1
      requires: !single

   test:
      suffix: 12
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1 -g_type 0 -central_vec 1
      requires: !single

   test:
      suffix: 13
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1 -g_type 1 -central_vec 1
      requires: !single

   test:
      suffix: 14
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1 -g_type 2 -central_vec 1
      requires: !single

   test:
      suffix: 15
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1 -g_type 3 -central_vec 1
      requires: !single
TEST*/
