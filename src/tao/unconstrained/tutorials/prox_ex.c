/* TAOPROX example */

#include <petsctao.h>
#include <petsc/private/taoimpl.h> 

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

PetscErrorCode TaoDestroy_Magic(Tao tao)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoSetUp_Magic(Tao tao)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoSetFromOptions_Magic(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoView_Magic(Tao tao, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_Magic(Tao tao)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Magic_Create(Tao tao)
{
  PetscFunctionBegin;
  tao->ops->destroy        = TaoDestroy_Magic;
  tao->ops->setup          = TaoSetUp_Magic;
  tao->ops->setfromoptions = TaoSetFromOptions_Magic;
  tao->ops->view           = TaoView_Magic;
  tao->ops->solve          = TaoSolve_Magic;
  PetscFunctionReturn(PETSC_SUCCESS);
}

//KL divergnce. sum x_i log (x/y)
//grad: log (x/y) + 1. Ignoring Hessian.
PetscErrorCode Magic_ObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user;
  Vec    y, workv;

  PetscFunctionBegin;
  PetscCall(TaoMetricGetContext(tao, &user));
  PetscCall(TaoMetricGetCentralVector(tao, &y)); 
  workv = user->workvec;

  PetscCall(VecPointwiseDivide(workv, X, y));
  PetscCall(VecLog(workv));
  PetscCall(VecCopy(workv, G));
  PetscCall(VecShift(workv, 1.));

  PetscCall(VecPointwiseMult(workv, X, workv));
  PetscCall(VecSum(workv, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao       tao, metric_tao;
  Vec       x;
  Mat       temp_mat;
  PetscInt  size;
  AppCtx    user;
  PetscBool flg;
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

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
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
  PetscCall(PetscRandomDestroy(&rctx));

  /* A,b data */
  PetscCall(VecView(user.y, PETSC_VIEWER_STDOUT_WORLD));
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

  /* Registering KL */
  PetscCall(TaoMetricRegister("TAOMETRIC_KL_USER", Magic_Create));
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoCreate(PETSC_COMM_SELF, &metric_tao));

  /* Cases that we want to try:
   *
   *  0: Built-in TAOPROX solve for Soft-Threshold TODO do register version of TAOPROX too?
   *  1: Dispatch version for TAOCG - via TaoApplyProximalMap_CG, with L2 metric, via TaoMetricSetType
   *  2: TAOCG, L2 Metric, with TaoMetricCreate
   *  3: TAOCG with KL metric 
   *  4: TAOCG with user-set KL metric */

  /* Testing default case */
  switch (user.problem) {
  case 0:
    /* L1 */
    {
      /* Stepsize of 1 */
      PetscCall(TaoSetType(tao, TAOPROX));
      PetscCall(TaoProxSetType(tao, TAOPROX_L1));
      PetscCall(TaoMetricSetType(tao, TAO_METRIC_L2));
      PetscCall(TaoSetFromOptions(tao));
      //Note: TAOPROX isnt' meant for TaoSolve. no implicit use of metric in taosolve.
    }
    break;
  case 1:
  case 2:
  case 3:
    /* user version */
    /* Solving iterative refinement
     * f(x) = 0.5 x.T A x - b.T x, where A is spsd 
     * sol = (A + lambda I)^-1 (b + y) .
     * just TaoSolve will give you A^-1 b
     * Will be solved via TaoApplyProximalMap_CG */
    {
      PetscCall(TaoSetType(tao, TAOCG));
      PetscCall(TaoSetSolution(tao, x));
      PetscCall(TaoSetFromOptions(tao));
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *) &user));
      PetscCall(TaoMetricSetCentralVector(tao, user.y));

      if (user.problem == 2) {
        /* L2 metric. Built-in */
        PetscCall(TaoCreate(PETSC_COMM_SELF, &metric_tao));
        PetscCall(TaoMetricSetType(metric_tao, TAO_METRIC_L2));
      } else if (user.problem == 2){
        /* L2 metric, via TaoMetricCreate */
        PetscCall(TaoMetricCreate(PETSC_COMM_WORLD, &metric_tao, TAO_METRIC_L2));
      } else if (user.problem == 3){
        /* User-Registered Metric */
#if 0 
        PetscCall(TaoCreate(PETSC_COMM_SELF, &metric_tao));
        PetscCall(TaoMetricSetType(metric_tao, "TAOMETRIC_KL_USER"));
        PetscCall(TaoMetricSetContext(metric_tao, (void *) &user));
#endif
      } else if (user.problem == 4){
        /* User-set (not registered) metric */
        PetscCall(TaoCreate(PETSC_COMM_SELF, &metric_tao));
        PetscCall(TaoSetObjectiveAndGradient(tao, NULL, Magic_ObjGrad, (void *) &user));
        //This will make it metric_user

        //Unclear how P3 is better than P4.... Unclear how to implement create_metric(tao).
        //if you have access to objgrad of your metric, just use this version???
        //
        // metric_tao is needed, only for obj and grad... then lets just use P4, not register version...
      } else {       
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Problem not 0,1,2,3, or 4. ");
      }
      PetscCall(TaoSetMetricTao(tao, metric_tao));
    }
    break;
  default:
    break;
  }

  PetscCall(TaoApplyProximalMap(tao, 1., user.y, x));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  if (user.problem == 0) {
    /* workvec2 = (A + I)^-1 (b + y)*/          
    PetscCall(MatShift(user.A, 1));
    PetscCall(MatSetOption(user.A, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(user.A, NULL, NULL));
    PetscCall(VecWAXPY(user.workvec, 1., user.b, user.y));
    PetscCall(MatSolve(user.A, user.workvec, user.workvec2));
    PetscCall(VecView(user.workvec2, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatDestroy(&user.A));
  }

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&user.y));
  PetscCall(VecDestroy(&user.workvec));
  PetscCall(VecDestroy(&user.workvec2));

  PetscCall(PetscFinalize());
  return 0;
}


