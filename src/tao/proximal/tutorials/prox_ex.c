/* DMApplyProximalMap example */

#include <petsctao.h>
#include <petscdm.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates various ways to use DM to solve proximal algorithms.. \n";

typedef enum {
  PROBLEM_L1,
  PROBLEM_SIMPLEX
} ProblemType;

typedef struct {
  ProblemType problem;
  PetscScalar lb, ub;
  PetscInt    n;              /* dimension */
  PetscReal   stepsize, simp; /* simp: size of simplex */
  PetscReal   mu1;            /* Parameter for soft-threshold */
  PetscReal   tol;
  PetscBool   l2_null;
  Vec         x, x_test, y;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *probTypes[2] = {"l1", "simplex"};

  PetscInt probtype;

  PetscFunctionBeginUser;
  user->n        = 10;
  user->stepsize = 1;
  user->mu1      = 1;
  user->lb       = -1;
  user->ub       = 1;
  user->simp     = 1.;
  user->tol      = 1.e-12;
  user->problem  = PROBLEM_L1;
  user->l2_null  = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "DMTaoApplyProximalMap example", "DMTAO");
  probtype = user->problem;

  PetscCall(PetscOptionsEList("-problem", "Decide whether to solve L1 or simplex.", "prox_ex.c", probTypes, 2, probTypes[user->problem], &probtype, NULL));

  user->problem = (ProblemType)probtype;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user->n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lb", &user->lb, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ub", &user->ub, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user->stepsize, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-simplex", &user->simp, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &user->mu1, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-l2_null", &user->l2_null, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscRandom rctx;

  PetscFunctionBeginUser;
  /* If stepsize ==1, default case. Else, adaptive version */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user->n, &user->x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user->n, &user->y));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(PetscRandomSetInterval(rctx, -10, 10));
  /* x : Random vec, from -10 to 10 */
  PetscCall(PetscRandomSetSeed(rctx, 1234));
  PetscCall(VecSetRandom(user->x, rctx));
  PetscCall(VecDuplicate(user->x, &user->x_test));
  PetscCall(VecCopy(user->x, user->x_test));
  /* y : Random vec, from -10 to 10 */
  PetscCall(PetscRandomSetSeed(rctx, 5678));
  PetscCall(VecSetRandom(user->y, rctx));
  PetscCall(VecZeroEntries(user->y));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->x_test));
  PetscCall(VecDestroy(&user->y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm0, dm1;
  PetscMPIInt size;
  PetscReal   vec_dist;
  AppCtx      user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DataCreate(&user));

  switch (user.problem) {
  case PROBLEM_L1:
    PetscCall(DMCreate(PETSC_COMM_SELF, &dm0));
    PetscCall(DMCreate(PETSC_COMM_SELF, &dm1));
    PetscCall(DMTaoSetType(dm0, DMTAOL1));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    PetscCall(DMTaoSetCentralVector(dm1, user.y));
    /* Try built-in Soft-Threshold */
    PetscCall(TaoSoftThreshold(user.y, user.lb, user.ub, user.x_test));
    break;
  case PROBLEM_SIMPLEX:
    PetscCall(DMCreate(PETSC_COMM_SELF, &dm0));
    PetscCall(DMCreate(PETSC_COMM_SELF, &dm1));
    PetscCall(DMTaoSetType(dm0, DMTAOSIMPLEX));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    PetscCall(DMTaoSetCentralVector(dm1, user.y));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
  }

  if (user.l2_null) {
    PetscCall(DMTaoApplyProximalMap(dm0, NULL, user.stepsize, user.y, user.x, PETSC_FALSE));
  } else {
    PetscCall(DMTaoApplyProximalMap(dm0, dm1, user.stepsize, user.y, user.x, PETSC_FALSE));
  }

  switch (user.problem) {
  case PROBLEM_L1:
    /* Testing Regularizer version vs Full version */
    PetscCall(VecAXPY(user.x, -1., user.x_test));
    PetscCall(VecNorm(user.x, NORM_2, &vec_dist));
    if (vec_dist < 1.e-11) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between DMTaoApplyProximalMap and SoftThreshold: < 1.e-11\n"));
    } else if (vec_dist < 1.e-6) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between DMTaoApplyProximalMap and SoftThreshold: < 1.e-6\n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between DMTaoApplyProximalMap and SoftThreshold: %e\n", (double)vec_dist));
    }
    break;
  case PROBLEM_SIMPLEX: {
    PetscReal sum, min;
    PetscCall(VecSum(user.x, &sum));
    PetscCall(VecMin(user.x, NULL, &min));
    vec_dist = PetscAbsReal(sum - user.simp);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Smallest element of solution: %e\n", (double)min));
    if (vec_dist < 1.e-11) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: < 1.e-11\n"));
    } else if (vec_dist < 1.e-6) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: < 1.e-6\n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: %e\n", (double)vec_dist));
    }
  } break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
  }

  PetscCall(DMDestroy(&dm0));
  PetscCall(DMDestroy(&dm1));
  PetscCall(DataDestroy(&user));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: soft1
      args: -problem l1 -l2_null 0
      output_file: output/prox_ex_soft00.out
      requires: !single

   test:
      suffix: soft2
      args: -problem l1 -l2_null 1
      output_file: output/prox_ex_soft00.out
      requires: !single

   test:
      suffix: simplex1
      args: -problem simplex -l2_null 0
      output_file: output/prox_ex_simplex00.out
      requires: !single

   test:
      suffix: simplex2
      args: -problem simplex -l2_null 1
      output_file: output/prox_ex_simplex00.out
      requires: !single

TEST*/
