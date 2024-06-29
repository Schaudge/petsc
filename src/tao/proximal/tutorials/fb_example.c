/* TAOFB example. Solves 0.5 |Ax-b|_2^2 + \lambda |x|_1,
 * with A is Gaussian random with size M*N, and b is a measurement vector of size M. */

#include <petsctao.h>
#include <petscdm.h>

static char help[] = "This example demonstrates TaoFB to solve proximal algorithm. \n";

/* https://github.com/tomgoldstein/fasta-matlab/blob/master/test_sparseLeastSquares.m */

typedef enum {
  USE_TAO,
  USE_DM
} ProblemType;

typedef struct {
  ProblemType probType;
  PetscInt    m, n, k; //A : m x n, k : signal sparsity
  Mat         A;
  Vec         x0, x, workvec, workvec2, workvec3, b;
  PetscReal   scale, optimum;
} AppCtx;

/* Objective and Gradient
 *
 * f(x) = 0.5 |Ax-b|_2^2
 * grad f = A^T (A x - b)               */
PetscErrorCode UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecAXPY(user->workvec, -1., user->b));
  PetscCall(MatMultTranspose(user->A, user->workvec, G));
  PetscCall(VecTDot(user->workvec, user->workvec, f));
  *f *= 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 |Ax-b|_2^2
 * grad f = A^T (A x - b)               */
PetscErrorCode UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecAXPY(user->workvec, -1., user->b));
  PetscCall(MatMultTranspose(user->A, user->workvec, G));
  PetscCall(VecTDot(user->workvec, user->workvec, f));
  *f *= 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscRandom rctx;
  PetscReal   norm, *array, *array2, *array3, randreal;
  PetscInt    i, *indices, p, temp, temp2;

  PetscFunctionBegin;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));

  p = PetscCeilInt(user->n , user->k);

  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n, &user->x));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n, &user->x0));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n, &user->workvec2));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->workvec));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->workvec3));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->b));
  PetscCall(VecSet(user->x0, 0.));

  //workvec: y_star
  PetscCall(VecSetRandom(user->workvec, rctx));
  PetscCall(VecNorm(user->workvec, NORM_2, &norm));
  PetscCall(VecScale(user->workvec, 1/norm));

  PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD, user->m, user->n, NULL, &user->A));
  PetscCall(MatDenseGetArray(user->A, &array));

  /* Gaussian Matrix, MATLAB equivalent of A = randn(m,n) */
  for (i = 0; i < user->m * user->n; i++) {
    PetscReal a[1] = {0.};

    PetscCall(PetscRandomGetValueReal(rctx, &a[0]));
    PetscCall(PetscPDFSampleGaussian1D(a, NULL, &array[i]));
  }
  PetscCall(MatDenseRestoreArray(user->A, &array));
  PetscCall(MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(user->A, 2.));
  PetscCall(MatShift(user->A, -1.));

  PetscCall(MatMultTranspose(user->A, user->workvec, user->workvec2));
  PetscCall(VecAbs(user->workvec2));
  PetscCall(VecCopy(user->workvec2, user->x)); // using x vec as workvec for sortperm
  PetscCall(PetscMalloc1(user->n, &indices));
  for (i = 0; i < user->n; i++) indices[i] = i;

  /* indices: perm, workvec2: CTy, x0: alpha */
  PetscCall(VecGetArray(user->x, &array));
  PetscCall(VecGetArray(user->workvec2, &array2));
  PetscCall(VecGetArray(user->x0, &array3));
  PetscCall(PetscSortRealWithArrayInt(user->n, array, indices)); //in increasing order

  for (i = 0; i < user->n; i++) {
    temp = indices[i];
    if (i > user->n - p) { //TODO julia 1-indexing crap..
      array3[temp] = user->scale / array2[temp];
    } else {
      temp2 = array2[temp];
      if (temp < 0.1*user->scale) {
        array3[temp] = user->scale;
      } else {
        PetscCall(PetscRandomGetValueReal(rctx, &randreal));
        array3[temp] = user->scale * randreal / temp2;
      }
    }
  }
  PetscCall(VecRestoreArray(user->x, &array));
  PetscCall(VecRestoreArray(user->workvec2, &array2));
  PetscCall(VecRestoreArray(user->x0, &array3));

  PetscCall(MatDiagonalScale(user->A, NULL, user->x0));

  // generate the primal solution
  // x0 is x_star
  PetscCall(VecSet(user->x0, 0.));
  PetscCall(VecGetArray(user->x0, &array));
  for (i = 0; i < user->n; i++) {
    if (i > user->n - p) {
      temp = indices[i];
      PetscCall(PetscRandomGetValueReal(rctx, &randreal));
      PetscCall(MatGetColumnVector(user->A, user->workvec3, temp-1));
      PetscCall(VecDot(user->workvec3, user->workvec, &norm));
      array[temp] = randreal*PetscSqrtReal((int) p) * PetscSign(norm);
    }
  }
  PetscCall(VecRestoreArray(user->x0, &array));

  PetscCall(MatMultAdd(user->A, user->x0, user->workvec, user->b));

  PetscCall(VecNorm(user->workvec, NORM_2, &norm));
  user->optimum = norm/2.;

  PetscCall(VecNorm(user->x0, NORM_1, &norm));
  user->optimum += user->scale*norm;

  PetscCall(PetscFree(indices));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->x0));
  PetscCall(VecDestroy(&user->b));
  PetscCall(VecDestroy(&user->workvec));
  PetscCall(MatDestroy(&user->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *probTypes[2] = {"use_tao", "use_dm"};

  PetscInt probtype;

  PetscFunctionBegin;
  user->k        = 5;
  user->n        = 20;
  user->m        = 10;
  user->scale    = 1.e-4;
  user->probType = USE_TAO;
  PetscOptionsBegin(comm, "", "Forward-backward example", "TAO");
  probtype = user->probType;

  PetscCall(PetscOptionsEList("-problem", "Decide whether to use Tao or DM to setup problem statement.", "fb_example.c", probTypes, 2, probTypes[user->probType], &probtype, NULL));

  user->probType = (ProblemType)probtype;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-k", &user->k, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user->n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &user->m, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-scale", &user->scale, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        fdm, gdm;
  Tao       tao;
  AppCtx    user;
  PetscReal v1, v2, matnorm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCheck(user.n > 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "Problem size needs to be greater than 2.");
  PetscCall(DataCreate(&user));
  MatNorm(user.A, NORM_FROBENIUS, &matnorm);

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetSolution(tao, user.x0));
  PetscCall(TaoSetType(tao, TAOFB));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &fdm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &gdm));
  PetscCall(DMTaoSetType(gdm, DMTAOL1));

  switch (user.probType) {
  case USE_TAO:
    PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *)&user));
    PetscCall(TaoPSSetLipschitz(tao, matnorm));
    break;
  case USE_DM:
    PetscCall(DMTaoSetObjectiveAndGradient(fdm, UserObjGrad_DM, (void *)&user));
    PetscCall(DMTaoSetLipschitz(fdm, matnorm));
    PetscCall(TaoPSSetSmoothTerm(tao, fdm));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
  }

  PetscCall(TaoPSSetNonSmoothTerm(tao, gdm));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  /* compute the error */
  PetscCall(VecAXPY(user.x0, -1, user.x));
  PetscCall(VecNorm(user.x0, NORM_2, &v1));
  PetscCall(VecNorm(user.x, NORM_2, &v2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1 / v2)));

  PetscCall(DataDestroy(&user));
  PetscCall(TaoDestroy(&tao));
  PetscCall(DMDestroy(&fdm));
  PetscCall(DMDestroy(&gdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: use_tao0
      args: -problem use_tao -tao_fb_approx_lip 0
      output_file: output/fb_example.out
      requires: !single

   test:
      suffix: use_tao1
      args: -problem use_tao -tao_fb_approx_lip 1
      output_file: output/fb_example.out
      requires: !single

   test:
      suffix: use_dm0
      args: -problem use_dm -tao_fb_approx_lip 0
      output_file: output/fb_example.out
      requires: !single

   test:
      suffix: use_dm1
      args: -problem use_dm -tao_fb_approx_lip 1
      output_file: output/fb_example.out
      requires: !single

TEST*/
