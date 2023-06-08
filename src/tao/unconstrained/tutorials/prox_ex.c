/* TAOPROX example */

#include <petsctao.h>

static char help[] = "This example demonstrates various ways to use TAOPROX. \n";

typedef struct {
  PetscInt  n; /* dimension */
  PetscInt ver; /* Types of problems to solve. */
  PetscReal stepsize;
  PetscReal mu1; /* Parameter for soft-threshold */
  Mat vm_mat;
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

static PetscErrorCode Solve_UserObj1(Tao tao)
{
  AppCtx *user;
  Vec     x;

  PetscFunctionBegin;
  PetscCall(TaoShellGetContext(tao, &user));
  PetscCall(TaoGetSolution(tao, &x));

  PetscCall(TaoSoftThreshold(x, -user->mu1, user->mu1, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UserObj1(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  PetscFunctionBegin;
  /* Objective ||x||_1 */
  PetscCall(VecNorm(X, NORM_1, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}



/* TODO scaling? T6.11, Beck*/
int main(int argc, char **argv)
{
  Tao       tao;        
  Vec       x;        
  PetscInt  size;        
  AppCtx    user;
  PetscBool flg, vm = PETSC_FALSE, shell = PETSC_FALSE;
  PetscRandom rctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.n = 10;
  user.stepsize = 1;
  user.mu1 = 1;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  /* Types of problems to solve */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ver", &user.ver, &flg));
  /* If stepsize ==1, default case. Else, adaptive version */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user.stepsize, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-vm", &vm, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-shell", &shell, &flg));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(PetscRandomSetInterval(rctx, -10, 10));
  /* Random vec, from -10 to 10 */
  PetscCall(VecSetRandom(x, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOPROX));
  PetscCall(TaoSetSolution(tao, x));

  if (vm) {
    /* Testing vm case */
    MatCreate(PETSC_COMM_WORLD, &user.vm_mat);
//    TaoPROXSetVM(tao, user.vm_mat);    
  
  } else {
    if (user.stepsize == 1) {
      /* Testing default case */
      switch (user.ver) {
      case 0:
        /* user version */
        {
          /* In this case, TAOPROX is merely adding Moreau regularization, and possibly play with adaptive stepsize or VM... */
          /* But MR is actually not "query-able", as MR is making f(x) -> inf_x f(x) + MR(x,y)...  */
          Tao subsolver;
          if (shell) {
            //Shell Version
            PetscCall(TaoSetType(subsolver, TAOSHELL));
            PetscCall(TaoShellSetContext(subsolver, (void *)shell_ctx));
            PetscCall(TaoShellSetSolve(subsolver, Shell_Solve));
          } else {
            // Set objective. this obj is w/o moreau reg
            PetscCall(TaoSetObjective(tao, UserObj1, (void *) &user));
            PetscCall(TaoPROXGetSubsolver(tao, &subsolver));
            PetscCall(TaoSetObjective(tao, UserObj1, (void *) &user));
            PetscCall(TaoSetType(subsolver, TAOCG));
          }
        }
        break;
      case 1:
        /* L1 */
        {
          //how should i do it? v
          // This will automatically add obj, and prox solution...
          // That means we need prox objective implementation, 
          TaoPROXSetFunc(tao, TAO_FUNC_L1);
        }      
        break;
      case 2:
        break;
      default:
        break;
      }      
    
    } else {
      /* Testing adaptive version */
    }
  }

  PetscCall(PetscFinalize());
  return 0;
}


