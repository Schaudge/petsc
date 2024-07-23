/* TAOCV example. Solves dual SVM problem, or least absolute deviation (LAD),
 *
 * dual svm:
 * minimize_{a_1,...a_N} 0.5*|\sum a_i g_i d_i |_2^2 - \sum a_i
 * s.t. 0 \leq a_i \leq C,  \sum a_i g_i = 0
 *
 * -> f(x) + g(x) + h(Ax),
 *  where f(x) = 0.5 x^T Q x + x^T q
 *        g(x) : box constraint indicator function
 *        h(x) : zero cone indicator function
 *
 * least absolute deviation:
 *
 * min_x |Ax - b|_1 + scale*|x|_1
 *
 * -> f(x) + g(x) + h(Ax),
 *  where f(x) = Zero(),
 *        g(x) = |x|_1
 *        h(x) = |\cdot - b|_1        */

#include <petsctao.h>
#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>

static char help[] = "This example demonstrates TaoCV to solve proximal primal-dual algorithm. \n";

typedef enum {
  DUAL_SVM,
  LAD, //Least absolute deviation
} ProbType;

typedef enum {
  USE_TAO,
  USE_DM
} FormType;

typedef struct {
  ProbType  probType;
  FormType  formType;
  PetscInt  m, n;
  Mat       Q, A;
  Vec       x0, x, workvec, workvec2, workvec3, q, y_translation;
  PetscReal C, t, lip, matnorm;
} AppCtx;

PetscErrorCode LAD_UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscFunctionBegin;
  f = 0;
  PetscCall(VecSet(G,0.));
  PetscFunctionReturn(PETSC_SUCCESS);
}
//TODO how to handle null obj?
/* Least absolute deviation
 * f(x) = zero()              */
PetscErrorCode LAD_UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscFunctionBegin;
  f = 0;
  PetscCall(VecSet(G,0.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 x^T Q x + x^T b
 * grad f = Qx + b                */
PetscErrorCode SVM_UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx   *user = (AppCtx *)ptr;
  PetscReal temp1, temp2;

  PetscFunctionBegin;
  PetscCall(MatMult(user->Q, X, user->workvec));
  PetscCall(VecTDot(user->workvec, user->workvec, &temp1));
  PetscCall(VecTDot(X, user->q, &temp2));
  PetscCall(VecAXPY(user->workvec, +1., user->q));
  *f = 0.5*temp1 + temp2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 x^T Q x + x^T b
 * grad f = Qx + b                */
PetscErrorCode SVM_UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx   *user = (AppCtx *)ptr;
  PetscReal temp1, temp2;

  PetscFunctionBegin;
  PetscCall(MatMult(user->Q, X, G));
  PetscCall(VecTDot(G, X, &temp1));
  PetscCall(VecTDot(X, user->q, &temp2));
  PetscCall(VecAXPY(G, +1., user->q));
  *f = 0.5*temp1 + temp2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscViewer viewer;
  Mat         X;
  Vec         y;
  PetscReal  *matarr, *vecarr;
  PetscInt    i;

  char mat_data1[] = "matrix-heart-scale.dat";
  char vec_data1[] = "vector-heart-scale.dat";
  char mat_data2[] = "matrix-housing-scale.dat";
  char vec_data2[] = "vector-housing-scale.dat";

  PetscFunctionBegin;

  switch (user->probType) {
  case DUAL_SVM:
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, vec_data1, FILE_MODE_READ, &viewer));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
    PetscCall(VecLoad(y, viewer));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, mat_data1, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
    PetscCall(MatSetType(X, MATSEQAIJ));
    PetscCall(MatLoad(X, viewer));
    PetscCall(MatGetSize(X, &user->m, &user->n));
    PetscCall(MatMatTransposeMult(X, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user->Q));
    PetscCall(MatDiagonalScale(user->Q, y, y));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &user->A));
    PetscCall(MatSetType(user->A, MATSEQDENSE));
    PetscCall(MatSetSizes(user->A, PETSC_DECIDE, PETSC_DECIDE, 1, user->m));
    PetscCall(MatSetFromOptions(user->A));
    PetscCall(MatSetUp(user->A));
    PetscCall(MatDenseGetArray(user->A, &matarr));
    PetscCall(VecGetArray(y, &vecarr));

    for (i = 0; i < user->m; i++) matarr[i] = vecarr[i];

    /* A = Matrix(y'), with size = (1, m) */
    PetscCall(MatDenseRestoreArray(user->A, &matarr));
    PetscCall(VecRestoreArray(y, &vecarr));
    PetscCall(MatNorm(user->A, NORM_FROBENIUS, &user->matnorm));

    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->x));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->x0));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->workvec2));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->workvec));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->workvec3));
    break;
  case LAD:
  {
    Vec x_col, a_col;

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, vec_data2, FILE_MODE_READ, &viewer));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &user->y_translation));
    PetscCall(VecLoad(user->y_translation, viewer));
    PetscCall(VecScale(user->y_translation, -1.));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, mat_data2, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
    PetscCall(MatSetType(X, MATSEQAIJ));
    PetscCall(MatLoad(X, viewer));
    PetscCall(MatGetSize(X, &user->m, &user->n));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &user->A));
    PetscCall(MatSetType(user->A, MATSEQDENSE));
    PetscCall(MatSetSizes(user->A, PETSC_DECIDE, PETSC_DECIDE, user->m, user->n+1));
    PetscCall(MatSetFromOptions(user->A));
    PetscCall(MatSetUp(user->A));
    PetscCall(MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY));

    PetscCall(MatCreateVecs(X, NULL, &x_col));
    PetscCall(VecSetFromOptions(x_col));

    for (i = 0; i < user->n; i++) {
      PetscCall(MatDenseGetColumnVecWrite(user->A, i, &a_col));
      PetscCall(MatGetColumnVector(X, x_col, i));
      PetscCall(VecCopy(x_col, a_col));
      PetscCall(MatDenseRestoreColumnVecWrite(user->A, i, &a_col));
    }
    PetscCall(MatDenseGetColumnVecWrite(user->A, user->n, &a_col));
    PetscCall(VecSet(a_col, 1.));
    PetscCall(MatDenseRestoreColumnVecWrite(user->A, user->n, &a_col));

    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n+1, &user->x));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n+1, &user->x0));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n+1, &user->workvec2));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n+1, &user->workvec));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n+1, &user->workvec3));
  }
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid problem type.");
  }

  //TODO matcreatevecs(A, x, NULL)
  PetscCall(VecSet(user->x0, 0.));
  PetscCall(VecSet(user->x, 0.));

  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->q));
  PetscCall(VecSet(user->q, -1));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->x0));
  PetscCall(VecDestroy(&user->q));
  PetscCall(VecDestroy(&user->workvec));
  PetscCall(VecDestroy(&user->workvec2));
  PetscCall(VecDestroy(&user->workvec3));
  PetscCall(MatDestroy(&user->Q));
  PetscCall(MatDestroy(&user->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *formTypes[2] = {"use_tao", "use_dm"};
  const char *probTypes[2] = {"dual_svm", "lad"};

  PetscInt formtype, probtype;

  PetscFunctionBegin;
  user->t        = 0.1;
  user->C        = 0.1;
  user->formType = USE_TAO;
  user->probType = DUAL_SVM;
  formtype       = user->formType;
  probtype       = user->probType;

  PetscOptionsBegin(comm, "", "Forward-backward example", "TAO");
  PetscCall(PetscOptionsEList("-formation", "Decide whether to use Tao or DM to setup problem statement.", "cv_example.c", formTypes, 2, formTypes[user->formType], &formtype, NULL));
  PetscCall(PetscOptionsEList("-problem", "Decide which problem to solve - dual svm or least absolute deviation.", "cv_example.c", probTypes, 2, probTypes[user->probType], &probtype, NULL));

  user->formType = (FormType)formtype;
  user->probType = (ProbType)probtype;

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-C", &user->C, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-t", &user->t, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        fdm, gdm, hdm;
  Tao       tao;
  AppCtx    user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DataCreate(&user));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetSolution(tao, user.x));
  PetscCall(TaoSetType(tao, TAOCV));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &fdm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &gdm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &hdm));

  switch (user.probType) {
  case DUAL_SVM:
  {

    PetscCall(DMTaoSetType(gdm, DMTAOBOX));
    PetscCall(DMTaoSetType(hdm, DMTAOZERO));
    PetscCall(DMTaoBoxSetContext(gdm, 0, user.C, NULL, NULL));
  }
    break;
  case LAD:
    PetscCall(DMTaoSetType(gdm, DMTAOL1));
    PetscCall(DMTaoSetType(hdm, DMTAOL1));
    PetscCall(DMTaoSetTranslationVector(hdm, user.y_translation));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem type.");
  }


  switch (user.probType) {
  case DUAL_SVM:
  {
    switch (user.formType) {
    case USE_TAO:
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, SVM_UserObjGrad, (void *)&user));
      break;
    case USE_DM:
      PetscCall(DMTaoSetObjectiveAndGradient(fdm, SVM_UserObjGrad_DM, (void *)&user));
      PetscCall(TaoPSSetSmoothTerm(tao, fdm, 1));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
    }
  }
    break;
  case LAD:
  {
    // LAD's f(x) = Zero()  TODO does this work? effectively PDHG....
    switch (user.formType) {
    case USE_TAO:
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, LAD_UserObjGrad, NULL));
      break;
    case USE_DM:
      PetscCall(DMTaoSetObjectiveAndGradient(fdm, LAD_UserObjGrad_DM, (void *)&user));
      PetscCall(TaoPSSetSmoothTerm(tao, fdm, 0));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
    }
  }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem type.");
  }

  PetscCall(TaoPSSetNonSmoothTerm(tao, gdm, 10));
  PetscCall(TaoPSSetNonSmoothTermWithLinearMap(tao, hdm, user.A, user.matnorm, 1.));

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(DataDestroy(&user));
  PetscCall(TaoDestroy(&tao));
  PetscCall(DMDestroy(&fdm));
  PetscCall(DMDestroy(&gdm));
  PetscCall(PetscFinalize());
  return 0;
}
