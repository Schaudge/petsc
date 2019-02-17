
#include <petscfn.h>
#include <../src/fn/utils/fnutils.h>

static PetscErrorCode PetscFnDerivativeVec_Diagonal(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y, PetscErrorCode (*diag)(PetscInt, PetscInt, const PetscScalar *,PetscScalar *))
{
  const Vec *supervecs;
  const PetscScalar *xs;
  PetscScalar       *ys;
  Vec supery;
  PetscInt    i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSuperVectors(fn, der, rangeIdx, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  ierr = VecSet(y,1.);CHKERRQ(ierr);
  for (i = 0; i < der; i++) {
    ierr = VecPointwiseMult(y,y,supervecs[i]);CHKERRQ(ierr);
  }
  ierr = VecGetLocalSize(x, &n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  ierr = (*diag) (der, n, xs, ys);CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = PetscFnRestoreSuperVectors(fn, der, rangeIdx, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeMat_Diagonal(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *A, Mat *Apre, PetscErrorCode (*diag)(PetscInt,PetscInt,const PetscScalar*,PetscScalar*))
{
  const Vec *supervecs;
  const PetscScalar *xs;
  PetscScalar       *ys;
  Vec                y;
  PetscInt    i, n, m, N, M;
  MatReuse       superreuse;
  Mat            *superA, *superApre;
  Mat            *mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSuperVectors(fn, der-1, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  ierr = PetscFnGetSuperMats(fn, rangeIdx-(der-1), subsets ? &subsets[der-1] : NULL, reuse, A, Apre, &superreuse, &superA, &superApre);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &y);CHKERRQ(ierr);
  ierr = VecSet(y,1.);CHKERRQ(ierr);
  for (i = 0; i < der-1; i++) {
    ierr = VecPointwiseMult(y,y,supervecs[i]);CHKERRQ(ierr);
  }
  ierr = VecGetLocalSize(x, &n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  ierr = (*diag) (der, n, xs, ys);CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  mat = superA ? superA : superApre;
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),m,n,M,N,1,NULL,0,NULL,mat);CHKERRQ(ierr);
  }
  ierr = MatDiagonalSet(*mat, y, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  if (superApre && superApre != superA) {
    ierr = MatDuplicateOrCopy(*mat, reuse, superApre);CHKERRQ(ierr);
  }
  ierr = PetscFnRestoreSuperMats(fn, rangeIdx-(der-1), subsets ? &subsets[der-1] : NULL, reuse, A, Apre, &superreuse, &superA, &superApre);CHKERRQ(ierr);
  ierr = PetscFnRestoreSuperVectors(fn, der-1, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Sin(PetscFn fn, Vec x, Vec y)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecCopy(x, y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = PetscSinScalar(xs[i]);
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivative_Sin(PetscInt der, PetscInt n, const PetscScalar *xs, PetscScalar *ys)
{
  PetscScalar    sign = (der & 2) ? -1. : 1.;
  PetscBool      cosine = (der & 1) ? PETSC_TRUE : PETSC_FALSE;
  PetscInt       i;

  PetscFunctionBegin;
  if (cosine) {
    for (i = 0; i < n; i++) {
      ys[i] *= sign * PetscCosScalar(xs[i]);
    }
  }
  else {
    for (i = 0; i < n; i++) {
      ys[i] *= sign * PetscSinScalar(xs[i]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_Sin(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeVec_Diagonal(fn, x, der, rangeIdx, subsets, subvecs, y, PetscFnDerivative_Sin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeMat_Sin(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *A, Mat *Apre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeMat_Diagonal(fn, x, der, rangeIdx, subsets, subvecs, reuse, A, Apre, PetscFnDerivative_Sin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Sin(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_APPLY,         (void (*)(void)) PetscFnApply_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DERIVATIVEVEC, (void (*)(void)) PetscFnDerivativeVec_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DERIVATIVEMAT, (void (*)(void)) PetscFnDerivativeMat_Sin);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "sin");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ERF)
static PetscErrorCode PetscFnApply_Erf(PetscFn fn, Vec x, Vec y)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecCopy(x, y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = erf(xs[i]);CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivative_Erf(PetscInt der, PetscInt n, const PetscScalar *xs, PetscScalar *ys)
{
  PetscInt i;

  PetscFunctionBegin;
  if (!der) {
    for (i = 0; i < n; i++) {
      ys[i] *= erf(xs[i]);
    }
  }
  else {
    PetscReal sign = (der & 1) ? 1. : -1;
    PetscInt  j;

    for (i = 0; i < n; i++) {
      PetscScalar xx = xs[i];
      PetscScalar Hold = 0.;
      PetscScalar H = 2. * PetscExpScalar(-xx*xx) / PetscSqrtScalar(PETSC_PI);

      for (j = 0; j < der - 1; j++) {
        PetscScalar Hnew;

        Hnew = 2. * xx * H - 2 * j * Hold;
        Hold = H;
        H = Hnew;
      }
      ys[i] *= sign * H;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_Erf(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeVec_Diagonal(fn, x, der, rangeIdx, subsets, subvecs, y, PetscFnDerivative_Erf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeMat_Erf(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *A, Mat *Apre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeMat_Diagonal(fn, x, der, rangeIdx, subsets, subvecs, reuse, A, Apre, PetscFnDerivative_Erf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Erf(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Erf);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DERIVATIVEVEC, (void (*)(void)) PetscFnDerivativeVec_Erf);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DERIVATIVEMAT, (void (*)(void)) PetscFnDerivativeMat_Erf);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "erf");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode PetscFnApply_Logistic(PetscFn fn, Vec x, Vec y)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecCopy(x, y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = 1. / (1. + PetscExpScalar(-xs[i]));
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivative_Logistic(PetscInt der, PetscInt n, const PetscScalar *xs, PetscScalar *ys)
{
  PetscInt       i;

  PetscFunctionBegin;
  switch (der) {
  case 0:
    for (i = 0; i < n; i++) {
      ys[i] *= 1. / (1. + PetscExpScalar(-xs[i]));
    }
    break;
  case 1:
    for (i = 0; i < n; i++) {
      PetscScalar l = 1. / (1. + PetscExpScalar(-xs[i]));
      ys[i] *= l * (1. - l);
    }
    break;
  case 2:
    for (i = 0; i < n; i++) {
      PetscScalar l = 1. / (1. + PetscExpScalar(-xs[i]));
      ys[i] *= l * (1. - l) * (1. - 2. * l);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Higher derivatives of logistic function not supported");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_Logistic(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_SUP, "Higher derivatives of logistic function not supported");
  ierr = PetscFnDerivativeVec_Diagonal(fn,x,der,rangeIdx,subsets,subvecs,y,PetscFnDerivative_Logistic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeMat_Logistic(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *A, Mat *Apre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_SUP, "Higher derivatives of logistic function not supported");
  ierr = PetscFnDerivativeMat_Diagonal(fn, x, der, rangeIdx, subsets, subvecs, reuse, A, Apre, PetscFnDerivative_Logistic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Logistic(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Logistic);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DERIVATIVEVEC, (void (*)(void)) PetscFnDerivativeVec_Logistic);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DERIVATIVEMAT, (void (*)(void)) PetscFnDerivativeMat_Logistic);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "logistic");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarApply_Normsquared(PetscFn fn, Vec x, PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot(x, x, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarDerivativeVec_Normsquared(PetscFn fn, Vec x, PetscInt der, const IS subsets[], Vec subvecs[], Vec y)
{
  const Vec *supervecs;
  Vec supery;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der > 2) {
    ierr = VecSet(y, 0.);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscFnGetSuperVectors(fn, der-1, PETSC_DEFAULT, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  if (der == 1) {
    ierr = VecCopy(x, supery);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(supervecs[0], supery);CHKERRQ(ierr);
  }
  ierr = VecScale(y,2.);CHKERRQ(ierr);
  ierr = PetscFnRestoreSuperVectors(fn, der-1, PETSC_DEFAULT, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarDerivativeMat_Normsquared(PetscFn fn, Vec x, PetscInt der, const IS subsets[], Vec subvecs[], MatReuse reuse, Mat *A, Mat *Apre)
{
  const Vec *supervecs;
  PetscInt    n, N;
  MatReuse       superreuse;
  Mat            *superA, *superApre;
  Mat            *mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSuperVectors(fn, der-2, PETSC_DEFAULT, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  ierr = PetscFnGetSuperMats(fn, PETSC_DEFAULT, subsets ? &subsets[der-2] : NULL, reuse, A, Apre, &superreuse, &superA, &superApre);CHKERRQ(ierr);
  mat = superA ? superA : superApre;
  ierr = PetscFnGetSizes(fn, NULL, &n, NULL, &N);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),n,n,N,N,1,NULL,0,NULL,mat);CHKERRQ(ierr);
  }
  if (der == 2) {
    ierr = MatShift(*mat, 2.);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (superApre && superApre != superA) {
    ierr = MatDuplicateOrCopy(*mat, reuse, superApre);CHKERRQ(ierr);
  }
  ierr = PetscFnRestoreSuperMats(fn, PETSC_DEFAULT, subsets ? &subsets[der-2] : NULL, reuse, A, Apre, &superreuse, &superA, &superApre);CHKERRQ(ierr);
  ierr = PetscFnRestoreSuperVectors(fn, der-2, PETSC_DEFAULT, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Normsquared(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_SCALARAPPLY, (void (*)(void)) PetscFnScalarApply_Normsquared);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_SCALARDERIVATIVEVEC, (void (*)(void)) PetscFnScalarDerivativeVec_Normsquared);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_SCALARDERIVATIVEMAT, (void (*)(void)) PetscFnScalarDerivativeMat_Normsquared);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "normsq");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Mat(PetscFn fn, Vec x, Vec y)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = MatMult(mat, x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_Mat(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  const Vec *supervecs;
  Vec supery;
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der > 1) {
    ierr = VecSet(y, 0.);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscFnGetSuperVectors(fn, der, rangeIdx, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  if (der == 0) {
    ierr = MatMult(mat, x, supery);CHKERRQ(ierr);
  } else {
    if (rangeIdx == 0) {
      ierr = MatMultTranspose(mat, supervecs[0], supery);CHKERRQ(ierr);
    } else {
      ierr = MatMult(mat, supervecs[0], supery);CHKERRQ(ierr);
    }
  }
  ierr = PetscFnRestoreSuperVectors(fn, der, rangeIdx, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeMat_Mat(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *A, Mat *Apre)
{
  PetscInt        n, m, N, M;
  Mat            *mat, origmat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &origmat);CHKERRQ(ierr);
  mat = A ? A : Apre;
  if (der == 1) {
    PetscInt notRangeIdx = 1 - rangeIdx;
    if (subsets) {
      ierr = MatCreateSubMatrix(origmat, subsets[rangeIdx], subsets[notRangeIdx], reuse, mat);CHKERRQ(ierr);
    } else {
      ierr = MatDuplicateOrCopy(origmat, reuse, mat);CHKERRQ(ierr);
    }
    if (rangeIdx == 0) {
      ierr = MatTranspose(*mat, MAT_INPLACE_MATRIX,mat);CHKERRQ(ierr);
    }
    if (Apre && Apre != mat) {
      ierr = MatDuplicateOrCopy(*mat, reuse, Apre);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  if (rangeIdx == der) {
    ierr = PetscFnGetSizes(fn, &m, NULL, &M, NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscFnGetSizes(fn, NULL, &m, NULL, &M);CHKERRQ(ierr);
  }
  if (rangeIdx == der - 1) {
    ierr = PetscFnGetSizes(fn, &n, NULL, &N, NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscFnGetSizes(fn, NULL, &n, NULL, &N);CHKERRQ(ierr);
  }
  if (subsets && subsets[0]) {
    ierr = ISGetLocalSize(subsets[0], &n);CHKERRQ(ierr);
    ierr = ISGetSize(subsets[0], &N);CHKERRQ(ierr);
  }
  if (subsets && subsets[1]) {
    ierr = ISGetLocalSize(subsets[1], &m);CHKERRQ(ierr);
    ierr = ISGetSize(subsets[1], &N);CHKERRQ(ierr);
  }
  /* create empty matrix */
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),m,n,M,N,0,NULL,0,NULL,mat);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Apre && Apre != mat) {
    ierr = MatDuplicateOrCopy(*mat, reuse, Apre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnView_Mat(PetscFn fn, PetscViewer viewer)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = MatView(mat, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDestroy_Mat(PetscFn fn)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode PetscFnShellCreate_Mat(PetscFn fn)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DERIVATIVEVEC, (void (*)(void)) PetscFnDerivativeVec_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DERIVATIVEMAT, (void (*)(void)) PetscFnDerivativeMat_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_VIEW, (void (*)(void)) PetscFnView_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DESTROY, (void (*)(void)) PetscFnDestroy_Mat);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "mat");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
