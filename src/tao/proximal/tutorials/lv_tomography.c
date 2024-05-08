#include <petsctao.h>
#include <petscdm.h>

/*
Description:   Loris-Verhoeven (PAPC) tomography reconstruction example .
               0.5*||Ax-b||^2 + lambda*g(x)
Reference:     None
*/

static char help[] = "Finds the forward-backward solution to the under constraint linear model Ax = b, with regularizer. \n\
            A is a M*N real matrix (M<N), x is sparse. A good regularizer is an L1 regularizer. \n\
            We find the sparse solution by solving 0.5*||Ax-b||^2 + lambda*||D*x||_1, where lambda (by default 1e-4) is a user specified weight.\n\
            D is the K*N transform matrix so that D*x is sparse. By default D is identity matrix, so that D*x = x.\n";

/* User-defined application context */
typedef struct {
  PetscReal lambda, eps;
  Vec       workM, workN, workN2, workN3, xGT, x, b;
  PetscInt  M, N, K;   /* Problem dimension: A is M*N Matrix, D is K*N Matrix */
  PetscInt  reg;       // dictionary matrix type. 0: Identity, 1: Dictionary
  Mat       A, ATA, D; /* Coefficients, Dictionary Transform of size M*N and K*N respectively. For linear least square, Jacobian Matrix J = A. For nonlinear least square, it is different from A */
} AppCtx;

/* User provided Routines */
PetscErrorCode InitializeUserData(AppCtx *);
PetscErrorCode MisfitObjectiveAndGradient(DM, Vec, PetscReal *, Vec, void *);
PetscErrorCode EvaluateResidual(Tao, Vec, Vec, void *);
PetscErrorCode EvaluateRegularizerObjectiveAndGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode EvaluateRegularizerHessian(Tao, Vec, Mat, void *);
PetscErrorCode EvaluateRegularizerHessianProd(Mat, Vec, Vec);

/*--------------------------------------------------------------------*/
int main(int argc, char **argv)
{
  Vec         x;
  Tao         tao;
  DM          f_dm, g_dm;
  PetscReal   hist[100], resid[100], v1, v2;
  PetscInt    lits[100];
  AppCtx      user; /* user-defined work context */
  PetscViewer fd;   /* used to save result to file */
  char        resultFile[] = "tomographyResult_x";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOFB));
  PetscCall(DMCreate(PETSC_COMM_SELF, &f_dm));
  PetscCall(DMCreate(PETSC_COMM_SELF, &g_dm));

  /* User set application context: A, D matrice, and b vector. */
  PetscCall(InitializeUserData(&user));

  /* Allocate solution vector x,  and function vectors Ax-b, */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.N, &x));

  PetscCall(VecSet(x, 0.0));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetType(tao, TAOFB));
  PetscCall(TaoFBSetType(tao, TAO_FB_LV));


  PetscReal norm;
  PetscCall(MatNorm(user.D, NORM_FROBENIUS, &norm));//TODO technically just for sanitiy check?

  /* Setting DMTaos */
  PetscCall(DMTaoSetObjectiveAndGradient(f_dm, MisfitObjectiveAndGradient, (void *)&user));
  PetscCall(TaoFBSetSmoothTerm(tao, f_dm));
  PetscCall(DMTaoSetType(g_dm, DMTAOL1));
  switch (user.reg) {
  case 0:
    PetscCall(TaoFBSetNonSmoothTermWithLinearMap(tao, g_dm, NULL, norm));
    break;
  case 1:
    PetscCall(TaoFBSetNonSmoothTermWithLinearMap(tao, g_dm, user.D, norm)); //TODO maybe two version of 0, and actual L norm?
    break;
  }

  /* Check for any TAO command line arguments */
  PetscCall(TaoSetFromOptions(tao));

  PetscCall(TaoSetConvergenceHistory(tao, hist, resid, 0, lits, 100, PETSC_TRUE));

  /* Perform the Solve */
  PetscCall(TaoSolve(tao));

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from MATLAB and convert to a 2D image for comparison. */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, resultFile, FILE_MODE_WRITE, &fd));
  PetscCall(VecView(x, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* compute the error */
  PetscCall(VecAXPY(x, -1, user.xGT));
  PetscCall(VecNorm(x, NORM_2, &v1));
  PetscCall(VecNorm(user.xGT, NORM_2, &v2));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1 / v2)));

  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));

  /* Free PETSc data structures */
  PetscCall(VecDestroy(&x));
  /* Free user data structures */
  PetscCall(MatDestroy(&user.A));
  PetscCall(MatDestroy(&user.D));
  PetscCall(VecDestroy(&user.b));
  PetscCall(VecDestroy(&user.xGT));
  PetscCall(PetscFinalize());
  return 0;
}

/*------------------------------------------------------------*/

PetscErrorCode MisfitObjectiveAndGradient(DM dm, Vec X, PetscReal *f, Vec g, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 */
  PetscCall(MatMult(user->A, X, user->workM));
  PetscCall(VecAXPY(user->workM, -1, user->b));
  PetscCall(VecDot(user->workM, user->workM, f));
  *f *= 0.5;
  /* Gradient. ATAx-ATb */
  PetscCall(MatMult(user->ATA, X, user->workN));
  PetscCall(MatMultTranspose(user->A, user->b, user->workN2));
  PetscCall(VecWAXPY(g, -1., user->workN2, user->workN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode InitializeUserData(AppCtx *user)
{
  PetscInt    k, n;                                  /* indices for row and columns of D. */
  char        dataFile[] = "tomographyData_A_b_xGT"; /* Matrix A and vectors b, xGT(ground truth) binary files generated by MATLAB. Debug: change from "tomographyData_A_b_xGT" to "cs1Data_A_b_xGT". */
  PetscInt    dictChoice = 1;                        /* choose from 0:identity, 1:gradient1D, 2:gradient2D, 3:DCT etc */
  PetscViewer fd;                                    /* used to load data from file */
  PetscReal   v;

  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, dataFile, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &user->A));
  PetscCall(MatSetType(user->A, MATSEQAIJ));
  PetscCall(MatLoad(user->A, fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->b));
  PetscCall(VecLoad(user->b, fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->xGT));
  PetscCall(VecLoad(user->xGT, fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(VecDuplicate(user->xGT, &user->x));
  PetscCall(MatTransposeMatMult(user->A, user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user->ATA));

  /* Specify the size */
  PetscCall(MatGetSize(user->A, &user->M, &user->N));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->x));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->workM));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->workN));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->workN2));
  PetscCall(VecSetSizes(user->x, PETSC_DECIDE, user->N));
  PetscCall(VecSetSizes(user->workM, PETSC_DECIDE, user->M));
  PetscCall(VecSetSizes(user->workN, PETSC_DECIDE, user->N));
  PetscCall(VecSetSizes(user->workN2, PETSC_DECIDE, user->N));
  PetscCall(VecSetFromOptions(user->x));
  PetscCall(VecSetFromOptions(user->workM));
  PetscCall(VecSetFromOptions(user->workN));
  PetscCall(VecSetFromOptions(user->workN2));

  user->reg = 1; //Dictionary as default

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "lv_tomography.c");
  PetscCall(PetscOptionsInt("-reg", "Choice of dictionary matrix, (0,1)", "lv_tomography.c", user->reg, &user->reg, NULL));
  PetscCall(PetscOptionsReal("-lambda", "The regularization multiplier. 1.e-4 default", "lv_tomography.c", user->lambda, &user->lambda, NULL));
  PetscOptionsEnd();

  /* Specify D */
  /* (1) Specify D Size */
  switch (dictChoice) {
  case 0: /* 0:identity */
    user->K = user->N;
    break;
  case 1: /* 1:gradient1D */
    user->K = user->N - 1;
    break;
  }

  PetscCall(MatCreate(PETSC_COMM_SELF, &user->D));
  PetscCall(MatSetSizes(user->D, PETSC_DECIDE, PETSC_DECIDE, user->K, user->N));
  PetscCall(MatSetFromOptions(user->D));
  PetscCall(MatSetUp(user->D));

  /* (2) Specify D Content */
  switch (dictChoice) {
  case 0: /* 0:identity */
    break;
  case 1: /* 1:gradient1D.  [-1, 1, 0,...; 0, -1, 1, 0, ...] */
    for (k = 0; k < user->K; k++) {
      v = 1.0;
      n = k + 1;
      PetscCall(MatSetValues(user->D, 1, &k, 1, &n, &v, INSERT_VALUES));
      v = -1.0;
      PetscCall(MatSetValues(user->D, 1, &k, 1, &k, &v, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(user->D, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(user->D, MAT_FINAL_ASSEMBLY));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_max_it 1000 -tao_brgn_regularization_type l1dict -tao_brgn_regularizer_weight 1e-8 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-8

   test:
      suffix: 2
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_regularization_type l2prox -tao_brgn_regularizer_weight 1e-8 -tao_gatol 1.e-6

   test:
      suffix: 3
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_regularization_type user -tao_brgn_regularizer_weight 1e-8 -tao_gatol 1.e-6

TEST*/
