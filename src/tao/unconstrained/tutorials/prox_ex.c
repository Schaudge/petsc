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
  Vec y, b, workvec;
} AppCtx;

/* Types of problmes to solve :
 * See Beck et al, 2017. Chapter 6, SIAM.
 * 1: L1        : \|x\|_1
 * 2: Constant  : c
 * 3: Affine    : a^T x + b
 * 4: cvx quad  : 0.5 x^T A x + b^T x
 * 5: a x, x>=0 
 *    inf, x<0
 * 6: a x^3, x>=0
 *     inf,  x<0
 * 7: -a log(x), x>0
 *      inf,     x<=0
 * 8: delta[0,a] \union \R(x) : 
 * 9: L_0 norm
 * PROJECTIONS:
 * 10: non-negative orthant
 * 11: box
 * 12: affine set
 * 13: l2 ball
 * 14: half-space
 * 15: intersection of hyperplane and box
 * 16: unit simplex
 *  ..... 
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
  PetscCall(TaoPROXSetSoftThreshold(user->y, user->lb, user->ub, out));
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

  user.n = 10;
  user.stepsize = 1;
  user.mu1 = 1;
  user.lb  = -1;
  user.lb  = 1;


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
  PetscCall(TaoSetType(tao, TAOPROX));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoPROXSetInitialVector(tao, user.y));
  PetscCall(TaoSetFromOptions(tao));

  /* Testing default case */
  switch (user.problem) {
  case 0:
    /* user version */
    /* Solving iterative refinement (/
     * f(x) = 0.5 x.T A x - b.T x, where A is spsd 
     * sol = (A + I)^-1 (b + y)*/
    {
      Tao subsolver;
      Mat temp_mat;
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, user.n, user.n, 0, NULL, &temp_mat));
      PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
      PetscCall(PetscRandomSetFromOptions(rctx));
      PetscCall(PetscRandomSetInterval(rctx, -10, 10));
      PetscCall(PetscRandomDestroy(&rctx));
      PetscCall(MatTransposeMatMult(temp_mat, temp_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user.A));
      PetscCall(VecSetRandom(user.b, rctx));
      PetscCall(MatDestroy(&temp_mat));

      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *) &user));
      PetscCall(TaoPROXGetSubsolver(tao, &subsolver));
      PetscCall(TaoSetType(subsolver, TAOCG));
    }
    break;
  case 1:
    /* L1 */
    {
      if (shell) {
        //Shell Version
        Tao subsolver;
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

  PetscCall(TaoSolve(tao));

  if (user.problem == 0) {
    Tao cg_tao;
    PetscCall(TaoCreate(PETSC_COMM_SELF, &cg_tao));
    PetscCall(TaoSetType(cg_tao, TAOCG));
    PetscCall(TaoSetSolution(cg_tao, x2));
    PetscCall(TaoSetFromOptions(cg_tao));
    PetscCall(TaoSetObjectiveAndGradient(cg_tao, NULL, UserObjGrad, (void *) &user));
    PetscCall(TaoSolve(cg_tao));

    PetscPrintf(PETSC_COMM_WORLD, "PROX solver\n "); 
    PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF)); 
    PetscPrintf(PETSC_COMM_WORLD, "CG solver\n "); 
    PetscCall(VecView(x2, PETSC_VIEWER_STDOUT_SELF)); 
    PetscCall(TaoDestroy(&cg_tao));
  }

  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&user.A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&user.y));
  PetscCall(VecDestroy(&user.b));
  PetscCall(VecDestroy(&user.workvec));

  PetscCall(PetscFinalize());
  return 0;
}


