/* TAOCV example. Solves dual SVM problem,
 *
 * minimize_{a_1,...a_N} 0.5*|\sum a_i g_i d_i |_2^2 - \sum a_i
 * s.t. 0 \leq a_i \leq C,  \sum a_i g_i = 0 */

#include <petsctao.h>
#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>

static char help[] = "This example demonstrates TaoCV to solve proximal primal-dual algorithm. \n";

typedef enum {
  USE_TAO,
  USE_DM
} FormType;

typedef struct {
  FormType  formType;
  PetscInt  m, n;
  Mat       Q, A, At;
  Vec       x0, x, workvec, workvec2, workvec3, q;
  PetscReal C, t, lip, matnorm;
} AppCtx;

/* Objective and Gradient
 *
 * f(x) = 0.5 x^T Q x + x^T b
 * grad f = Qx + b                */
PetscErrorCode UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
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
PetscErrorCode UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
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

  char mat_data[] = "matrix-heart-scale.dat";
  char vec_data[] = "vector-heart-scale.dat";

  PetscFunctionBegin;

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, vec_data, FILE_MODE_READ, &viewer));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecLoad(y, viewer));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, mat_data, FILE_MODE_READ, &viewer));
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
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->q));
  PetscCall(VecSet(user->x0, 0.));
  PetscCall(VecSet(user->x, 0.));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *formTypes[2] = {"use_tao", "use_dm"};

  PetscInt formtype;

  PetscFunctionBegin;
  user->t        = 0.1;
  user->C        = 0.1;
  user->formType = USE_TAO;
  formtype       = user->formType;

  PetscOptionsBegin(comm, "", "Forward-backward example", "TAO");
  PetscCall(PetscOptionsEList("-formation", "Decide whether to use Tao or DM to setup problem statement.", "fb_example.c", formTypes, 2, formTypes[user->formType], &formtype, NULL));

  user->formType = (FormType)formtype;

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
  PetscCall(DMTaoSetType(gdm, DMTAOBOX));
  PetscCall(DMTaoSetType(hdm, DMTAOZERO));

  PetscReal zr = 0.;
  //TODO how to deal with this kind of situation?
  PetscCall(DMTaoBoxSetContext(gdm, &zr, &user.C, NULL, NULL));

  switch (user.formType) {
  case USE_TAO:
    PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *)&user));
//    PetscCall(TaoPSSetLipschitz(tao, user.lip));
    break;
  case USE_DM:
    PetscCall(DMTaoSetObjectiveAndGradient(fdm, UserObjGrad_DM, (void *)&user));
//    PetscCall(DMTaoSetLipschitz(fdm, user.lip));
    PetscCall(TaoPSSetSmoothTerm(tao, fdm, 1));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
  }

  PetscCall(TaoPSSetNonSmoothTerm(tao, gdm, 1));
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
