/* TAOFB example. Solves 0.5 |Ax-b|_2^2 + \lambda |x|_1,
 * with A is Gaussian random with size M*N, and b is a measurement vector of size M. */

#include <petsctao.h>
#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>

static char help[] = "This example demonstrates TaoFB to solve proximal algorithm. \n";

/* https://github.com/tomgoldstein/fasta-matlab/blob/master/test_sparseLeastSquares.m */

typedef enum {
  USE_TAO,
  USE_DM
} ProblemType;

typedef struct {
  ProblemType probType;
  PetscInt    m, n, k; //A : m x n, k : signal sparsity
  Mat         A, ATA;
  Vec         x0, x, workvec, workvec2, workvec3, b;
  PetscReal   scale, optimum, lip;
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

PetscErrorCode manualData(AppCtx *user)
{
  PetscFunctionBegin;
  MatSetValue(user->A, 0, 0, 0.493886, INSERT_VALUES);
  MatSetValue(user->A, 1, 0, 0.493602, INSERT_VALUES);
  MatSetValue(user->A, 2, 0, 0.95334, INSERT_VALUES);
  MatSetValue(user->A, 3, 0, -0.159568, INSERT_VALUES);
  MatSetValue(user->A, 4, 0, 0.440955, INSERT_VALUES);

  MatSetValue(user->A, 0, 1, -0.126165, INSERT_VALUES);
  MatSetValue(user->A, 1, 1, 0.0667351, INSERT_VALUES);
  MatSetValue(user->A, 2, 1, -0.709366, INSERT_VALUES);
  MatSetValue(user->A, 3, 1, -0.806696, INSERT_VALUES);
  MatSetValue(user->A, 4, 1, 0.39602, INSERT_VALUES);

  MatSetValue(user->A, 0, 2, -0.0594853, INSERT_VALUES);
  MatSetValue(user->A, 1, 2, -0.972651, INSERT_VALUES);
  MatSetValue(user->A, 2, 2, 0.690508, INSERT_VALUES);
  MatSetValue(user->A, 3, 2, 0.00638238, INSERT_VALUES);
  MatSetValue(user->A, 4, 2, -0.981031, INSERT_VALUES);

  MatSetValue(user->A, 0, 3, 0.989288, INSERT_VALUES);
  MatSetValue(user->A, 1, 3, 0.672411, INSERT_VALUES);
  MatSetValue(user->A, 2, 3, 0.0625485, INSERT_VALUES);
  MatSetValue(user->A, 3, 3, -0.643244, INSERT_VALUES);
  MatSetValue(user->A, 4, 3, -0.195779, INSERT_VALUES);

  MatSetValue(user->A, 0, 4, -0.207008, INSERT_VALUES);
  MatSetValue(user->A, 1, 4, -0.487988, INSERT_VALUES);
  MatSetValue(user->A, 2, 4, -0.777435, INSERT_VALUES);
  MatSetValue(user->A, 3, 4, 0.950613, INSERT_VALUES);
  MatSetValue(user->A, 4, 4, 0.307498, INSERT_VALUES);

  MatSetValue(user->A, 0, 5, 0.816804, INSERT_VALUES);
  MatSetValue(user->A, 1, 5, 0.140358, INSERT_VALUES);
  MatSetValue(user->A, 2, 5, 0.450445, INSERT_VALUES);
  MatSetValue(user->A, 3, 5, 0.715589, INSERT_VALUES);
  MatSetValue(user->A, 4, 5, -0.474396, INSERT_VALUES);

  MatSetValue(user->A, 0, 6, 0.309691, INSERT_VALUES);
  MatSetValue(user->A, 1, 6, 0.874254, INSERT_VALUES);
  MatSetValue(user->A, 2, 6, 0.201755, INSERT_VALUES);
  MatSetValue(user->A, 3, 6, -0.552453, INSERT_VALUES);
  MatSetValue(user->A, 4, 6, -0.462365, INSERT_VALUES);

  MatSetValue(user->A, 0, 7, 0.259592, INSERT_VALUES);
  MatSetValue(user->A, 1, 7, -0.309828, INSERT_VALUES);
  MatSetValue(user->A, 2, 7, 0.893619, INSERT_VALUES);
  MatSetValue(user->A, 3, 7, 0.639405, INSERT_VALUES);
  MatSetValue(user->A, 4, 7, -0.72335, INSERT_VALUES);

  MatSetValue(user->A, 0, 8, -0.520142, INSERT_VALUES);
  MatSetValue(user->A, 1, 8, 0.53352, INSERT_VALUES);
  MatSetValue(user->A, 2, 8, -0.243375, INSERT_VALUES);
  MatSetValue(user->A, 3, 8, -0.380436, INSERT_VALUES);
  MatSetValue(user->A, 4, 8, 0.532067, INSERT_VALUES);

  MatSetValue(user->A, 0, 9, -0.582464, INSERT_VALUES);
  MatSetValue(user->A, 1, 9, 0.431009, INSERT_VALUES);
  MatSetValue(user->A, 2, 9, -0.361007, INSERT_VALUES);
  MatSetValue(user->A, 3, 9, -0.154164, INSERT_VALUES);
  MatSetValue(user->A, 4, 9, 0.415068, INSERT_VALUES);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscRandom rctx;
  PetscReal   norm, *array, *array2, *array3, temp2;
  PetscInt    i, *indices, p, temp;

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

  VecSetValue(user->workvec, 0, 0.34817899371580235, INSERT_VALUES);
  VecSetValue(user->workvec, 1, 0.05882732877575909, INSERT_VALUES);
  VecSetValue(user->workvec, 2, 0.7399056632723374, INSERT_VALUES);
  VecSetValue(user->workvec, 3, 0.07378183042153355, INSERT_VALUES);
  VecSetValue(user->workvec, 4, 0.5678085810212193, INSERT_VALUES);

  PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD, user->m, user->n, NULL, &user->A));
  PetscCall(MatDenseGetArray(user->A, &array));
  /* Gaussian Matrix, MATLAB equivalent of A = randn(m,n) */
  for (i = 0; i < user->m * user->n; i++) {
    PetscReal a[1] = {0.};

    PetscCall(PetscRandomGetValueReal(rctx, &a[0]));
    PetscCall(PetscPDFSampleGaussian1D(a, NULL, &array[i]));
  }
  PetscCall(MatDenseRestoreArray(user->A, &array));
  PetscCall(MatScale(user->A, 2.));
  PetscCall(MatShift(user->A, -1.));

  manualData(user); //somehow cant load julia mat....
  PetscCall(MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY));

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

  PetscReal randarr[6] = {0.8812782509403484, 0.9318977128510509, 0.8367505346120852,
                       0.9813388392678519, 0.3942976628581718, 0.06878420133755148};
   PetscInt j = 0;
//  for (i = 0; i < user->n; i++) {
  for (i = user->n - 1; i >= 0; i--) {
    temp = indices[i];
    if (i >= user->n - p) { //TODO julia 1-indexing crap..
      array3[temp] = user->scale / array2[temp];
    } else {
      temp2 = array2[temp];
      if (temp2 < 0.1*user->scale) {
        array3[temp] = user->scale;
      } else {
//        PetscCall(PetscRandomGetValueReal(rctx, &randreal));
//        array3[temp] = user->scale * randreal / temp2;
        array3[temp] = user->scale * randarr[j]/ temp2;
        j++;
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

//  PetscReal randarr2[2] = {0.6325038460017051, 0.0438830190403352};
  PetscReal randarr2[2] = {0.8812782509403484, 0.9318977128510509};
  j = 0;

//  for (i = 0; i < user->n; i++) {
  for (i = user->n - 1; i >= 0; i--) {
    if (i >= user->n - p) {
      temp = indices[i];
//      PetscCall(PetscRandomGetValueReal(rctx, &randreal));
      PetscCall(MatGetColumnVector(user->A, user->workvec3, temp));
      PetscCall(VecDot(user->workvec3, user->workvec, &norm));
//      array[temp] = randreal*PetscSqrtReal((int) p) * PetscSign(norm);
      array[temp] = randarr2[j]/PetscSqrtReal((double) p) * PetscSign(norm);
      j++;
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

  PetscCall(MatTransposeMatMult(user->A, user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user->ATA));

  KSP       ksp;
  PetscReal emax, emin;

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetOperators(ksp, user->ATA, user->ATA));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetComputeEigenvalues(ksp, PETSC_TRUE));
  PetscCall(KSPSetUp(ksp));
  PetscCall(VecSetRandom(user->x, rctx));
  PetscCall(KSPSolve(ksp, user->x, user->workvec2));
  PetscCall(KSPComputeExtremeSingularValues(ksp, &emax, &emin));

  user->lip = emax * emax;
  //Note: for 5*10, julia gives (scale=10), 35.65014107630037, petsc 44.923013618611812
  user->lip = 35.65014107630037*35.65014107630037;

  PetscCall(VecSet(user->x, 0.));
  PetscCall(MatDestroy(&user->ATA));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->x0));
  PetscCall(VecDestroy(&user->b));
  PetscCall(VecDestroy(&user->workvec));
  PetscCall(VecDestroy(&user->workvec2));
  PetscCall(VecDestroy(&user->workvec3));
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
  user->scale    = 1.;
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
  PetscReal v1, v2;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCheck(user.n > 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "Problem size needs to be greater than 2.");
  PetscCall(DataCreate(&user));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetSolution(tao, user.x));
  PetscCall(TaoSetType(tao, TAOFB));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &fdm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &gdm));
  PetscCall(DMTaoSetType(gdm, DMTAOL1));

  switch (user.probType) {
  case USE_TAO:
    PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *)&user));
    PetscCall(TaoPSSetLipschitz(tao, user.lip));
    break;
  case USE_DM:
    PetscCall(DMTaoSetObjectiveAndGradient(fdm, UserObjGrad_DM, (void *)&user));
    PetscCall(DMTaoSetLipschitz(fdm, user.lip));
    PetscCall(TaoPSSetSmoothTerm(tao, fdm, 1));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
  }

  PetscCall(TaoPSSetNonSmoothTerm(tao, gdm, user.scale));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  /* compute the error */
  PetscCall(VecNorm(user.x0, NORM_2, &v2));
  PetscCall(VecAXPY(user.x0, -1, user.x));
  PetscCall(VecNorm(user.x0, NORM_2, &v1));
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
