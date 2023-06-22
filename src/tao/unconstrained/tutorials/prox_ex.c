/* TAOPROX example */

#include <petsctao.h>

static char help[] = "This example demonstrates various ways to use TAOPROX. \n";

typedef struct {
  PetscScalar lb, ub;        
  PetscInt  n; /* dimension */
  PetscInt  problem; /* Types of problems to solve. */
  PetscReal stepsize;
  PetscReal mu1; /* Parameter for soft-threshold */
  Mat A;
  Vec y, b, workvec, workvec2;
} AppCtx;

/* Types of problmes to solve :
 * See Beck et al, 2017. Chapter 6, SIAM.
 * 1: L1        : \|x\|_1
 *.....  */

/*------------------------------------------------------------*/

static PetscErrorCode Shell_Solve(Tao tao)
{
  Tao     prox_tao;
  AppCtx *user;
  Vec     out;

  PetscFunctionBegin;
  PetscCall(TaoGetPROXParentTao(tao, &prox_tao));
  PetscCall(TaoShellGetContext(tao, &user));
  PetscCall(TaoGetSolution(tao, &out));
  PetscCall(TaoSoftThreshold(user->y, user->lb, user->ub, out));
  PetscFunctionReturn(PETSC_SUCCESS);  
}


/* Obj and Grad for iterative refinement.
 *
 * f(x) = 0.5 x.T A x - b.T x
 * grad f = A x - b                         */
PetscErrorCode UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;
  PetscScalar temp;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecWAXPY(G, -1., user->b, user->workvec));
  PetscCall(VecTDot(user->workvec, X, f));
  *f *= 0.5;

  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao       tao;
  Vec       x, x2;
  PetscInt  size;
  AppCtx    user;
  PetscBool flg, shell = PETSC_FALSE;
  PetscRandom rctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.problem = 0;
  user.n = 10;
  user.stepsize = 1;
  user.mu1 = 1;
  user.lb  = -1;
  user.ub  = 1;


  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  /* Types of problems to solve */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-problem", &user.problem, &flg));
  /* If stepsize ==1, default case. Else, adaptive version */
//  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user.stepsize, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-shell", &shell, &flg));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x2));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.y));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.workvec));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.workvec2));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(PetscRandomSetInterval(rctx, -10, 10));
  /* y : Random vec, from -10 to 10 */
  PetscCall(VecSetRandom(user.y, rctx));
  /* x: zero vec */
  PetscCall(VecZeroEntries(x));
  PetscCall(VecZeroEntries(x2));
  PetscCall(PetscRandomDestroy(&rctx));

  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));

  /* Testing default case */
  switch (user.problem) {
  case 0:
    /* user version */
    /* Solving iterative refinement (/
     * f(x) = 0.5 x.T A x - b.T x, where A is spsd 
     * sol = (A + I)^-1 (b + y)*/
    {
      PetscCall(TaoSetType(tao, TAOCG));
      PetscCall(TaoSetSolution(tao, x));
      PetscCall(TaoSetFromOptions(tao));
      /* Creating data */
      Mat temp_mat;
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, user.n, user.n, 0, NULL, &temp_mat));
      PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
      PetscCall(PetscRandomSetFromOptions(rctx));
      PetscCall(PetscRandomSetInterval(rctx, -10, 10));
      PetscCall(MatSetRandom(temp_mat, rctx));
      PetscCall(MatAssemblyBegin(temp_mat,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(temp_mat,MAT_FINAL_ASSEMBLY));
      PetscCall(MatTransposeMatMult(temp_mat, temp_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user.A));
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.b));
      PetscCall(VecSetRandom(user.b, rctx));
      PetscCall(MatDestroy(&temp_mat));
      PetscCall(PetscRandomDestroy(&rctx));

      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *) &user));
    }
    break;
  case 1:
    /* L1 */
    {
      PetscCall(TaoSetType(tao, TAOPROX));
      PetscCall(TaoPROXSetInitialVector(tao, user.y));
      PetscCall(TaoSetFromOptions(tao));
      if (shell) {
        //Shell Version
        Tao subsolver;
        PetscCall(TaoPROXGetSubsolver(tao, &subsolver));
        PetscCall(TaoSetType(subsolver, TAOSHELL));
        PetscCall(TaoShellSetContext(subsolver, (void *)&user));
        PetscCall(TaoShellSetSolve(subsolver, Shell_Solve));
      } else {
        PetscCall(TaoPROXSetSoftThreshold(tao, user.lb, user.ub));
      }
    }
    break;
  default:
    break;
  }

  PetscCall(TaoApplyProximalMap(tao, 1., NULL, user.y, x));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
//  PetscCall(TaoSolve(tao));

  if (user.problem == 0) {
    /* workvec2 = (A + I)^-1 (b + y)*/          
    PetscCall(MatShift(user.A, 1));
    PetscCall(MatSetOption(user.A, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(user.A, NULL, NULL));
    PetscCall(VecWAXPY(user.workvec, 1., user.b, user.y));
    PetscCall(MatSolve(user.A, user.workvec, user.workvec2));
    PetscCall(VecView(user.workvec2, PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&user.A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&user.y));
  PetscCall(VecDestroy(&user.workvec));
  PetscCall(VecDestroy(&user.workvec2));

  PetscCall(PetscFinalize());
  return 0;
}


