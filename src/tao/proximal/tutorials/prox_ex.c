/* DMApplyProximalMap example */

#include <petsctao.h>
#include <petscdm.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates various ways to use DM to solve proximal algorithms.\n";

typedef enum {
  PROBLEM_L1,
  PROBLEM_SIMPLEX,
  PROBLEM_BOX,
  PROBLEM_ZERO,
} ProblemType;

typedef struct {
  ProblemType problem;
  PetscScalar lb, ub;
  PetscInt    n;              /* dimension */
  PetscReal   stepsize, simp; /* simp: size of simplex */
  PetscReal   tol;
  PetscBool   l2_null;
  PetscBool   lb_use_vec, ub_use_vec;
  Vec         x, x_test, y;
  Vec         lb_vec, ub_vec;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *probTypes[4] = {"l1", "simplex", "box", "zero"};

  PetscInt probtype;

  PetscFunctionBeginUser;
  user->n          = 10;
  user->stepsize   = 1;
  user->lb         = -1;
  user->ub         = 1;
  user->simp       = 1.;
  user->tol        = 1.e-12;
  user->problem    = PROBLEM_L1;
  user->l2_null    = PETSC_FALSE;
  user->lb_use_vec = PETSC_FALSE;
  user->ub_use_vec = PETSC_FALSE;
  user->lb_vec     = NULL;
  user->ub_vec     = NULL;

  PetscOptionsBegin(comm, "", "DMTaoApplyProximalMap example", "DMTAO");
  probtype = user->problem;

  PetscCall(PetscOptionsEList("-problem", "Decide which problem to solve.", "prox_ex.c", probTypes, 4, probTypes[user->problem], &probtype, NULL));

  user->problem = (ProblemType)probtype;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user->n, NULL));
  /* stepsize of L1 case */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user->stepsize, NULL));
  /* Four cases for Box:
   * Case 1: lb_real, ub_real
   * Case 2: lb_real, ub_vec
   * Case 3: lb_vec,  ub_real
   * Case 4: lb_vec,  ub_vec  */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lb", &user->lb, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ub", &user->ub, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-lb_use_vec", &user->lb_use_vec, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ub_use_vec", &user->ub_use_vec, NULL));
  /* size of simplex */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-simplex", &user->simp, NULL));
  /* whether to use NULL for regularizer */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-l2_null", &user->l2_null, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscRandom rctx;

  PetscFunctionBeginUser;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->x));
  PetscCall(VecSetSizes(user->x, PETSC_DECIDE, user->n));
  PetscCall(VecSetFromOptions(user->x));
  PetscCall(VecDuplicate(user->x, &user->y));
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

  if (user->lb_use_vec) {
    PetscCall(VecDuplicate(user->x, &user->lb_vec));
    PetscCall(VecSet(user->lb_vec, user->lb));
  }
  if (user->ub_use_vec) {
    PetscCall(VecDuplicate(user->x, &user->ub_vec));
    PetscCall(VecSet(user->ub_vec, user->ub));
  }
  PetscCall(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->x_test));
  PetscCall(VecDestroy(&user->y));
  if (user->lb_vec) PetscCall(VecDestroy(&user->lb_vec));
  if (user->ub_vec) PetscCall(VecDestroy(&user->ub_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm0, dm1;
  PetscReal   vec_dist;
  AppCtx      user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DataCreate(&user));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm0));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm1));

  switch (user.problem) {
  case PROBLEM_L1:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMTAOL1 Case\n"));
    PetscCall(DMTaoSetType(dm0, DMTAOL1));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    /* Try built-in Soft-Threshold */
    PetscCall(TaoSoftThreshold(user.y, -user.stepsize, user.stepsize, user.x_test));
    break;
  case PROBLEM_SIMPLEX:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMTAOSIMPLEX Case\n"));
    PetscCall(DMTaoSetType(dm0, DMTAOSIMPLEX));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    PetscCall(DMTaoSimplexSetContext(dm0, user.simp, user.tol));
    break;
  case PROBLEM_BOX:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMTAOBOX Case\n"));
    PetscCall(DMTaoSetType(dm0, DMTAOBOX));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    PetscCall(DMTaoBoxSetContext(dm0, user.lb, user.ub, user.lb_vec, user.ub_vec));
    break;
  case PROBLEM_ZERO:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMTAOZERO Case\n"));
    PetscCall(DMTaoSetType(dm0, DMTAOZERO));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
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
  case PROBLEM_SIMPLEX:
  {
    PetscReal sum, min, max;
    PetscCall(VecSum(user.x, &sum));
    PetscCall(VecMin(user.x, NULL, &min));
    PetscCall(VecMax(user.x, NULL, &max));
    vec_dist = PetscAbsReal(sum - user.simp);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Smallest element of solution: %e\n", (double)min));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Largest element of solution: %e\n", (double)max));
    if (vec_dist < 1.e-11) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: < 1.e-11\n"));
    } else if (vec_dist < 1.e-6) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: < 1.e-6\n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: %e\n", (double)vec_dist));
    }
  }
    break;
  case PROBLEM_BOX:
  case PROBLEM_ZERO:
  {
    PetscReal min, max;

    PetscCall(VecMin(user.x, NULL, &min));
    PetscCall(VecMax(user.x, NULL, &max));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Smallest element of solution: %e\n", (double)min));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Largest element of solution: %e\n", (double)min));
  }
    break;
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
      suffix: soft
      args: -problem l1 -l2_null {{0 1}}
      output_file: output/prox_ex_soft00.out
      requires: !single

   test:
      suffix: simplex
      args: -problem simplex -l2_null {{0 1}}
      output_file: output/prox_ex_simplex00.out
      requires: !single

   test:
      suffix: box
      args: -problem box -l2_null {{0 1}}

TEST*/
