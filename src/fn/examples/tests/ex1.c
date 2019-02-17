#include <petscfn.h>

const char help[] = "Create, view, and test the derivative methods of a shell PetscFn\n";

/* Scalar example f(x) = sin( || x - y || ^2 ) */

static PetscErrorCode PetscFnDestroy_Scalar(PetscFn fn)
{
  Vec            v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &v);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnCreateVecs_Scalar(PetscFn fn, IS domainIS, Vec *domainVec, IS rangeIS, Vec *rangeVec)
{
  PetscInt       m, M, n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  if (rangeVec) {
    ierr = VecCreate(PetscObjectComm((PetscObject) fn), rangeVec);CHKERRQ(ierr);
    ierr = VecSetType(*rangeVec, VECSTANDARD);CHKERRQ(ierr);
    if (rangeIS) {
      ierr = ISGetLocalSize(rangeIS, &m);CHKERRQ(ierr);
      ierr = ISGetSize(rangeIS, &M);CHKERRQ(ierr);
    }
    ierr = VecSetSizes(*rangeVec, m, M);CHKERRQ(ierr);
  }
  if (domainVec) {
    ierr = VecCreate(PetscObjectComm((PetscObject) fn), domainVec);CHKERRQ(ierr);
    ierr = VecSetType(*domainVec, VECSTANDARD);CHKERRQ(ierr);
    if (domainIS) {
      ierr = ISGetLocalSize(domainIS, &n);CHKERRQ(ierr);
      ierr = ISGetSize(domainIS, &N);CHKERRQ(ierr);
    }
    ierr = VecSetSizes(*domainVec, n, N);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarApply_Scalar_Internal(PetscFn fn, Vec x, PetscScalar *z)
{
  Vec y, diff;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &y);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &diff);CHKERRQ(ierr);
  ierr = VecWAXPY(diff, -1., y, x);CHKERRQ(ierr);
  ierr = VecDot(diff, diff, z);CHKERRQ(ierr);
  ierr = VecDestroy(&diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarApply_Scalar(PetscFn fn, Vec x, PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarApply_Scalar_Internal(fn, x, z);CHKERRQ(ierr);
  *z   = PetscSinScalar(*z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Scalar(PetscFn fn, Vec x, Vec y)
{
  PetscScalar z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarApply_Scalar(fn, x, &z);CHKERRQ(ierr);
  ierr = VecSet(y, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarGradient_Scalar_Internal(PetscFn fn, Vec x, Vec g)
{
  Vec            y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &y);CHKERRQ(ierr);
  ierr = VecWAXPY(g, -1., y, x);CHKERRQ(ierr);
  ierr = VecScale(g, 2.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarGradient_Scalar(PetscFn fn, Vec x, Vec g)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarApply_Scalar_Internal(fn, x, &z);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Scalar_Internal(fn, x, g);CHKERRQ(ierr);
  ierr = VecScale(g, PetscCosScalar(z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMult_Scalar(PetscFn fn, Vec x, Vec xhat, Vec Jxhat)
{
  Vec            g;
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Scalar(fn, x, g);CHKERRQ(ierr);
  ierr = VecDot(g, xhat, &z);CHKERRQ(ierr);
  ierr = VecSet(Jxhat, z);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSingletonBcast(Vec v, PetscScalar *zp)
{
  MPI_Comm       comm;
  PetscLayout    map;
  PetscMPIInt    rank;
  PetscInt       broot;
  PetscScalar    z;
  const PetscScalar *zv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)v);
  ierr = VecGetLayout(v, &map);CHKERRQ(ierr);
  ierr = PetscLayoutFindOwner(map, 0, &broot);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v, &zv);CHKERRQ(ierr);
  z    = ((PetscInt) broot == rank) ? zv[0] : 0.;
  ierr = VecRestoreArrayRead(v, &zv);CHKERRQ(ierr);
  ierr = MPI_Bcast(&z, 1, MPIU_REAL, broot, comm);CHKERRQ(ierr);
  *zp  = z;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMultAdjoint_Scalar(PetscFn fn, Vec x, Vec v, Vec Jadjv)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarGradient_Scalar(fn, x, Jadjv);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  ierr = VecScale(Jadjv, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecToMat_Internal(Vec g, PetscBool colVec, MatReuse reuse, Mat *J)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateDenseVecs(PetscObjectComm((PetscObject)g), 1, &g, colVec, J);CHKERRQ(ierr);
  } else {
    ierr = MatSetValuesVec(*J, g, 0, colVec, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuild_Scalar(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  Vec            g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!J && !Jpre) PetscFunctionReturn(0);
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Scalar(fn, x, g);CHKERRQ(ierr);
  if (J) {
    ierr = VecToMat_Internal(g, PETSC_FALSE, reuse, J);CHKERRQ(ierr);
  }
  if (Jpre && Jpre != J) {
    ierr = VecToMat_Internal(g, PETSC_FALSE, reuse, Jpre);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuildAdjoint_Scalar(PetscFn fn, Vec x, MatReuse reuse, Mat *Jadj, Mat *Jadjpre)
{
  Vec            g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!Jadj && !Jadjpre) PetscFunctionReturn(0);
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Scalar(fn, x, g);CHKERRQ(ierr);
  if (Jadj) {
    ierr = VecToMat_Internal(g, PETSC_TRUE, reuse, Jadj);CHKERRQ(ierr);
  }
  if (Jadjpre && Jadjpre != Jadj) {
    ierr = VecToMat_Internal(g, PETSC_TRUE, reuse, Jadjpre);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarHessianMult_Scalar(PetscFn fn, Vec x, Vec xhat, Vec Hxhat)
{
  Vec            g;
  PetscScalar    z, dot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarApply_Scalar_Internal(fn, x, &z);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Scalar_Internal(fn, x, g);CHKERRQ(ierr);
  ierr = VecCopy(xhat, Hxhat);CHKERRQ(ierr);
  ierr = VecDot(xhat, g, &dot);CHKERRQ(ierr);
  ierr = VecScale(Hxhat, 2. * PetscCosScalar(z));CHKERRQ(ierr);
  ierr = VecAXPY(Hxhat, - dot * PetscSinScalar(z), g);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDupOrCopy(Mat orig, MatReuse reuse, Mat *dup)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(orig, MAT_COPY_VALUES, dup);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(orig, *dup, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarHessianBuild_Scalar(PetscFn fn, Vec x, MatReuse reuse, Mat *H, Mat *Hpre)
{
  Vec            g;
  PetscScalar    z;
  Mat            *hes = H ? H : Hpre;
  Mat            Jadj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!H && !Hpre) PetscFunctionReturn(0);
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarApply_Scalar_Internal(fn, x, &z);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Scalar_Internal(fn, x, g);CHKERRQ(ierr);
  ierr = VecToMat_Internal(g, PETSC_TRUE, MAT_INITIAL_MATRIX, &Jadj);CHKERRQ(ierr);
  ierr = MatMatTransposeMult(Jadj,Jadj,reuse,PETSC_DEFAULT,hes);CHKERRQ(ierr);
  ierr = MatScale(*hes, -PetscSinScalar(z));CHKERRQ(ierr);
  ierr = MatShift(*hes, 2. * PetscCosScalar(z));CHKERRQ(ierr);
  ierr = MatDestroy(&Jadj);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  if (Hpre && Hpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMult_Scalar(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec Hxhatxdot)
{
  PetscScalar    z;
  Vec            Hxhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(xhat, &Hxhat);CHKERRQ(ierr);
  ierr = PetscFnScalarHessianMult_Scalar(fn, x, xhat, Hxhat);CHKERRQ(ierr);
  ierr = VecDot(Hxhat, xdot, &z);CHKERRQ(ierr);
  ierr = VecSet(Hxhatxdot, z);CHKERRQ(ierr);
  ierr = VecDestroy(&Hxhat);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMultAdjoint_Scalar(PetscFn fn, Vec x, Vec v, Vec xhat, Vec Hadjvxhat)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarHessianMult_Scalar(fn, x, xhat, Hadjvxhat);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  ierr = VecScale(Hadjvxhat, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuild_Scalar(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *Hxhat, Mat *Hxhatpre)
{
  Vec            hxhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(xhat, &hxhat);CHKERRQ(ierr);
  ierr = PetscFnScalarHessianMult_Scalar(fn, x, xhat, hxhat);CHKERRQ(ierr);
  if (Hxhat) {
    ierr = VecToMat_Internal(hxhat, PETSC_FALSE, reuse, Hxhat);CHKERRQ(ierr);
  }
  if (Hxhatpre && Hxhatpre != Hxhat) {
    ierr = VecToMat_Internal(hxhat, PETSC_FALSE, reuse, Hxhatpre);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&hxhat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuildAdjoint_Scalar(PetscFn fn, Vec x, Vec v, MatReuse reuse, Mat *Hadjv, Mat *Hadjvpre)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarHessianBuild_Scalar(fn, x, reuse, Hadjv, Hadjvpre);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  if (Hadjv) {ierr = MatScale(*Hadjv, z);CHKERRQ(ierr);}
  if (Hadjvpre && Hadjvpre != Hadjv) {ierr = MatScale(*Hadjvpre, z);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuildSwap_Scalar(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *Hswpxhat, Mat *Hswpxhatpre)
{
  Vec            hxhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(xhat, &hxhat);CHKERRQ(ierr);
  ierr = PetscFnScalarHessianMult_Scalar(fn, x, xhat, hxhat);CHKERRQ(ierr);
  if (Hswpxhat) {
    ierr = VecToMat_Internal(hxhat, PETSC_TRUE, reuse, Hswpxhat);CHKERRQ(ierr);
  }
  if (Hswpxhatpre && Hswpxhatpre != Hswpxhat) {
    ierr = VecToMat_Internal(hxhat, PETSC_TRUE, reuse, Hswpxhatpre);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&hxhat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarDerivativeScalar_Scalar(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], PetscScalar *z)
{
  const Vec      *supervecs;
  Vec            outvec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der < 0 || der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der, PETSC_DEFAULT, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  switch (der) {
  case 0:
    ierr = PetscFnScalarApply_Scalar(fn, x, z);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecDuplicate(supervecs[0], &outvec);CHKERRQ(ierr);
    ierr = PetscFnScalarGradient_Scalar(fn, x, outvec);CHKERRQ(ierr);
    ierr = VecDot(outvec, supervecs[0], z);CHKERRQ(ierr);
    ierr = VecDestroy(&outvec);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecDuplicate(supervecs[1], &outvec);CHKERRQ(ierr);
    ierr = PetscFnScalarHessianMult_Scalar(fn, x, supervecs[0], outvec);CHKERRQ(ierr);
    ierr = VecDot(outvec, supervecs[1], z);CHKERRQ(ierr);
    ierr = VecDestroy(&outvec);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  ierr = PetscFnRestoreSuperVectors(fn, der, PETSC_DEFAULT, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarDerivativeVec_Scalar(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], Vec y)
{
  const Vec      *supervecs;
  Vec            supery;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der < 1 || der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der-1, PETSC_DEFAULT, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  switch (der) {
  case 1:
    ierr = PetscFnScalarGradient_Scalar(fn, x, supery);CHKERRQ(ierr);
    break;
  case 2:
    ierr = PetscFnScalarHessianMult_Scalar(fn, x, supervecs[0], supery);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  ierr = PetscFnRestoreSuperVectors(fn, der-1, PETSC_DEFAULT, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarDerivativeMat_Scalar(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *M, Mat *Mpre)
{
  const Vec      *supervecs;
  MatReuse       superreuse;
  Mat            *superM, *superMpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der != 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der-2, PETSC_DEFAULT, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  ierr = PetscFnGetSuperMats(fn, PETSC_DEFAULT, subsets ? &subsets[der-2] : NULL, reuse, M, Mpre, &superreuse, &superM, &superMpre);CHKERRQ(ierr);
  ierr = PetscFnScalarHessianBuild_Scalar(fn, x, superreuse, superM, superMpre);CHKERRQ(ierr);
  ierr = PetscFnRestoreSuperMats(fn, PETSC_DEFAULT, subsets ? &subsets[der-2] : NULL, reuse, M, Mpre, &superreuse, &superM, &superMpre);CHKERRQ(ierr);
  ierr = PetscFnRestoreSuperVectors(fn, der-2, PETSC_DEFAULT, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeScalar_Scalar(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], PetscScalar *z)
{
  const Vec      *supervecs;
  Vec            outvec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der < 0 || der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der+1, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  switch (der) {
  case 0:
    ierr = VecDuplicate(supervecs[0], &outvec);CHKERRQ(ierr);
    ierr = PetscFnApply_Scalar(fn, x, outvec);CHKERRQ(ierr);
    ierr = VecDot(outvec, supervecs[0], z);CHKERRQ(ierr);
    ierr = VecDestroy(&outvec);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecDuplicate(supervecs[1], &outvec);CHKERRQ(ierr);
    if (rangeIdx == 0) {
      ierr = PetscFnJacobianMultAdjoint_Scalar(fn, x, supervecs[0], outvec);CHKERRQ(ierr);
    } else {
      ierr = PetscFnJacobianMult_Scalar(fn, x, supervecs[0], outvec);CHKERRQ(ierr);
    }
    ierr = VecDot(outvec, supervecs[1], z);CHKERRQ(ierr);
    ierr = VecDestroy(&outvec);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecDuplicate(supervecs[2], &outvec);CHKERRQ(ierr);
    if (rangeIdx == 0) {
      ierr = PetscFnHessianMultAdjoint_Scalar(fn, x, supervecs[0], supervecs[1], outvec);CHKERRQ(ierr);
    } else if (rangeIdx == 1) {
      ierr = PetscFnHessianMultAdjoint_Scalar(fn, x, supervecs[1], supervecs[0], outvec);CHKERRQ(ierr);
    } else {
      ierr = PetscFnHessianMult_Scalar(fn, x, supervecs[0], supervecs[1], outvec);CHKERRQ(ierr);
    }
    ierr = VecDot(outvec, supervecs[2], z);CHKERRQ(ierr);
    ierr = VecDestroy(&outvec);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  ierr = PetscFnRestoreSuperVectors(fn, der+1, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_Scalar(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  const Vec      *supervecs;
  Vec            supery;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der < 0 || der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der, rangeIdx, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  switch (der) {
  case 0:
    ierr = PetscFnApply_Scalar(fn, x, supery);CHKERRQ(ierr);
    break;
  case 1:
    if (rangeIdx == 0) {
      ierr = PetscFnJacobianMultAdjoint_Scalar(fn, x, supervecs[0], supery);CHKERRQ(ierr);
    } else {
      ierr = PetscFnJacobianMult_Scalar(fn, x, supervecs[0], supery);CHKERRQ(ierr);
    }
    break;
  case 2:
    if (rangeIdx == 0) {
      ierr = PetscFnHessianMultAdjoint_Scalar(fn, x, supervecs[0], supervecs[1], supery);CHKERRQ(ierr);
    } else if (rangeIdx == 1) {
      ierr = PetscFnHessianMultAdjoint_Scalar(fn, x, supervecs[1], supervecs[0], supery);CHKERRQ(ierr);
    } else {
      ierr = PetscFnHessianMult_Scalar(fn, x, supervecs[0], supervecs[1], supery);CHKERRQ(ierr);
    }
    break;
  default:
    break;
  }
  ierr = PetscFnRestoreSuperVectors(fn, der, rangeIdx, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeMat_Scalar(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *M, Mat *Mpre)
{
  const Vec      *supervecs;
  MatReuse       superreuse;
  Mat            *superM, *superMpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der < 1 || der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der-1, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  ierr = PetscFnGetSuperMats(fn, rangeIdx-(der-1), subsets ? &subsets[der-1] : NULL, reuse, M, Mpre, &superreuse, &superM, &superMpre);CHKERRQ(ierr);
  switch (der) {
  case 1:
    if (rangeIdx == 0) {
      ierr = PetscFnJacobianBuildAdjoint_Scalar(fn, x, superreuse, superM, superMpre);CHKERRQ(ierr);
    } else {
      ierr = PetscFnJacobianBuild_Scalar(fn, x, superreuse, superM, superMpre);CHKERRQ(ierr);
    }
    break;
  case 2:
    if (rangeIdx == 0) {
      ierr = PetscFnHessianBuildAdjoint_Scalar(fn, x, supervecs[0], superreuse, superM, superMpre);CHKERRQ(ierr);
    } else if (rangeIdx == 1) {
      ierr = PetscFnHessianBuildSwap_Scalar(fn, x, supervecs[0], superreuse, superM, superMpre);CHKERRQ(ierr);
    } else {
      ierr = PetscFnHessianBuild_Scalar(fn, x, supervecs[0], superreuse, superM, superMpre);CHKERRQ(ierr);
    }
    break;
  default:
    break;
  }
  ierr = PetscFnRestoreSuperMats(fn, rangeIdx-(der-1), subsets ? &subsets[der-1] : NULL, reuse, M, Mpre, &superreuse, &superM, &superMpre);CHKERRQ(ierr);
  ierr = PetscFnRestoreSuperVectors(fn, der-2, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Vector example f(x) = log. ( A* exp.(x)) */

static PetscErrorCode PetscFnDestroy_Vector(PetscFn fn)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Vector_Internal(PetscFn fn, Vec x, Vec y)
{
  Mat            A;
  Vec            expx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &expx);CHKERRQ(ierr);
  ierr = VecCopy(x, expx);CHKERRQ(ierr);
  ierr = VecExp(expx);CHKERRQ(ierr);
  ierr = MatMult(A, expx, y);CHKERRQ(ierr);
  ierr = VecDestroy(&expx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Vector(PetscFn fn, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnApply_Vector_Internal(fn, x, y);CHKERRQ(ierr);
  ierr = VecLog(y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMult_Vector_Internal(PetscFn fn, Vec x, Vec xhat, Vec Jxhat)
{
  Mat            A;
  Vec            expx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &expx);CHKERRQ(ierr);
  ierr = VecCopy(x, expx);CHKERRQ(ierr);
  ierr = VecExp(expx);CHKERRQ(ierr);
  ierr = VecPointwiseMult(expx, expx, xhat);CHKERRQ(ierr);
  ierr = MatMult(A, expx, Jxhat);CHKERRQ(ierr);
  ierr = VecDestroy(&expx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMult_Vector(PetscFn fn, Vec x, Vec xhat, Vec Jxhat)
{
  Mat            A;
  Vec            gx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = VecDuplicate(Jxhat, &gx);CHKERRQ(ierr);
  ierr = PetscFnApply_Vector_Internal(fn, x, gx);CHKERRQ(ierr);
  ierr = PetscFnJacobianMult_Vector_Internal(fn, x, xhat, Jxhat);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Jxhat, Jxhat, gx);CHKERRQ(ierr);
  ierr = VecDestroy(&gx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMultAdjoint_Vector_Internal(PetscFn fn, Vec x, Vec v, Vec Jadjv)
{
  Mat            A;
  Vec            expx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = MatMultTranspose(A, v, Jadjv);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &expx);CHKERRQ(ierr);
  ierr = VecCopy(x, expx);CHKERRQ(ierr);
  ierr = VecExp(expx);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Jadjv, Jadjv, expx);CHKERRQ(ierr);
  ierr = VecDestroy(&expx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMultAdjoint_Vector(PetscFn fn, Vec x, Vec v, Vec Jadjv)
{
  Mat            A;
  Vec            vdup;
  Vec            gx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = VecDuplicate(v, &vdup);CHKERRQ(ierr);
  ierr = VecDuplicate(v, &gx);CHKERRQ(ierr);
  ierr = PetscFnApply_Vector_Internal(fn, x, gx);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(vdup, v, gx);CHKERRQ(ierr);
  ierr = PetscFnJacobianMultAdjoint_Vector_Internal(fn, x, vdup, Jadjv);CHKERRQ(ierr);
  ierr = VecDestroy(&gx);CHKERRQ(ierr);
  ierr = VecDestroy(&vdup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuild_Vector(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  Mat            A;
  Mat            *jac = J ? J : Jpre;
  Vec            gx, expx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac) PetscFunctionReturn(0);
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &gx);CHKERRQ(ierr);
  ierr = PetscFnApply_Vector_Internal(fn, x, gx);CHKERRQ(ierr);
  ierr = VecReciprocal(gx);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &expx);CHKERRQ(ierr);
  ierr = VecCopy(x, expx);CHKERRQ(ierr);
  ierr = VecExp(expx);CHKERRQ(ierr);

  ierr = MatDupOrCopy(A, reuse, jac);CHKERRQ(ierr);
  ierr = MatDiagonalScale(*jac, gx, expx);CHKERRQ(ierr);

  if (Jpre && Jpre != jac) {
    ierr = MatDupOrCopy(*jac, reuse, Jpre);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&gx);CHKERRQ(ierr);
  ierr = VecDestroy(&expx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuildAdjoint_Vector(PetscFn fn, Vec x, MatReuse reuse, Mat *Jadj, Mat *Jadjpre)
{
  Mat            *jacadj = Jadj ? Jadj : Jadjpre;
  Mat            jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jacadj) PetscFunctionReturn(0);
  ierr = PetscFnJacobianBuild(fn, x, MAT_INITIAL_MATRIX, &jac, NULL);CHKERRQ(ierr);
  ierr = MatTranspose(jac, reuse, jacadj);CHKERRQ(ierr);
  ierr = MatDestroy(&jac);CHKERRQ(ierr);
  if (Jadjpre && Jadjpre != jacadj) {
    ierr = MatDupOrCopy(*jacadj, reuse, Jadjpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMult_Vector_Internal(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec Hxhatxdot)
{
  Mat            A;
  Vec            expx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &expx);CHKERRQ(ierr);
  ierr = VecCopy(x, expx);CHKERRQ(ierr);
  ierr = VecExp(expx);CHKERRQ(ierr);
  ierr = VecPointwiseMult(expx, expx, xhat);CHKERRQ(ierr);
  ierr = VecPointwiseMult(expx, expx, xdot);CHKERRQ(ierr);
  ierr = MatMult(A, expx, Hxhatxdot);CHKERRQ(ierr);
  ierr = VecDestroy(&expx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMult_Vector(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec Hxhatxdot)
{
  Vec            Jxhat, Jxdot;
  Vec            gx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(Hxhatxdot, &gx);CHKERRQ(ierr);
  ierr = PetscFnApply_Vector_Internal(fn, x, gx);CHKERRQ(ierr);
  ierr = PetscFnHessianMult_Vector_Internal(fn, x, xhat, xdot, Hxhatxdot);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Hxhatxdot, Hxhatxdot, gx);CHKERRQ(ierr);
  ierr = VecDuplicate(Hxhatxdot, &Jxhat);CHKERRQ(ierr);
  ierr = VecDuplicate(Hxhatxdot, &Jxdot);CHKERRQ(ierr);
  ierr = PetscFnJacobianMult_Vector(fn, x, xhat, Jxhat);CHKERRQ(ierr);
  ierr = PetscFnJacobianMult_Vector(fn, x, xdot, Jxdot);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Jxhat, Jxhat, Jxdot);CHKERRQ(ierr);
  ierr = VecAXPY(Hxhatxdot, -1., Jxhat);CHKERRQ(ierr);
  ierr = VecDestroy(&Jxhat);CHKERRQ(ierr);
  ierr = VecDestroy(&Jxdot);CHKERRQ(ierr);
  ierr = VecDestroy(&gx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMultAdjoint_Vector_Internal(PetscFn fn, Vec x, Vec v, Vec xhat, Vec Hadjvxhat)
{
  Mat            A;
  Vec            expx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = MatMultTranspose(A, v, Hadjvxhat);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &expx);CHKERRQ(ierr);
  ierr = VecCopy(x, expx);CHKERRQ(ierr);
  ierr = VecExp(expx);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Hadjvxhat, Hadjvxhat, expx);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Hadjvxhat, Hadjvxhat, xhat);CHKERRQ(ierr);
  ierr = VecDestroy(&expx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMultAdjoint_Vector(PetscFn fn, Vec x, Vec v, Vec xhat, Vec Hadjvxhat)
{
  Mat            A;
  Vec            Jxhat;
  Vec            hdup;
  Vec            gx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);

  ierr = VecDuplicate(v, &Jxhat);CHKERRQ(ierr);
  ierr = PetscFnJacobianMult_Vector(fn, x, xhat, Jxhat);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Jxhat, Jxhat, v);CHKERRQ(ierr);
  ierr = VecScale(Jxhat, -1.);CHKERRQ(ierr);
  ierr = PetscFnJacobianMultAdjoint_Vector(fn, x, Jxhat, Hadjvxhat);CHKERRQ(ierr);

  ierr = VecDuplicate(v, &gx);CHKERRQ(ierr);
  ierr = PetscFnApply_Vector_Internal(fn, x, gx);CHKERRQ(ierr);
  ierr = VecDuplicate(Hadjvxhat, &hdup);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Jxhat, v, gx);CHKERRQ(ierr);
  ierr = PetscFnHessianMultAdjoint_Vector_Internal(fn, x, Jxhat, xhat, hdup);CHKERRQ(ierr);
  ierr = VecAXPY(Hadjvxhat, 1., hdup);CHKERRQ(ierr);
  ierr = VecDestroy(&hdup);CHKERRQ(ierr);
  ierr = VecDestroy(&Jxhat);CHKERRQ(ierr);
  ierr = VecDestroy(&gx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuild_Vector(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *Hxhat, Mat *Hxhatpre)
{
  Mat            *hes = Hxhat ? Hxhat : Hxhatpre;
  Mat            hescopy;
  Vec            Jxhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &Jxhat);CHKERRQ(ierr);
  ierr = PetscFnJacobianBuild(fn, x, reuse, hes, NULL);CHKERRQ(ierr);
  ierr = MatDuplicate(*hes, MAT_COPY_VALUES, &hescopy);CHKERRQ(ierr);
  ierr = PetscFnJacobianMult(fn, x, xhat, Jxhat);CHKERRQ(ierr);
  ierr = VecScale(Jxhat, -1.);CHKERRQ(ierr);
  ierr = MatDiagonalScale(*hes, Jxhat, NULL);CHKERRQ(ierr);
  ierr = MatDiagonalScale(hescopy, NULL, xhat);CHKERRQ(ierr);
  ierr = MatAXPY(*hes,1.,hescopy,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&hescopy);CHKERRQ(ierr);
  ierr = VecDestroy(&Jxhat);CHKERRQ(ierr);
  if (Hxhatpre && Hxhatpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hxhatpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuildAdjoint_Vector(PetscFn fn, Vec x, Vec v, MatReuse reuse, Mat *Hadjv, Mat *Hadjvpre)
{
  Mat            J, JT;
  Mat            *hes = Hadjv ? Hadjv : Hadjvpre;
  Mat            A;
  Vec            gx, vgx, Avgx, expx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = VecDuplicate(v, &gx);CHKERRQ(ierr);
  ierr = VecDuplicate(v, &vgx);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &Avgx);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &expx);CHKERRQ(ierr);
  ierr = PetscFnJacobianBuild(fn, x, MAT_INITIAL_MATRIX, &J, NULL);CHKERRQ(ierr);
  ierr = MatTranspose(J, MAT_INITIAL_MATRIX, &JT);CHKERRQ(ierr);
  ierr = MatDiagonalScale(J, v, NULL);CHKERRQ(ierr);
  ierr = PetscFnApply_Vector_Internal(fn, x, gx);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(vgx, v, gx);CHKERRQ(ierr);
  ierr = MatMultTranspose(A, vgx, Avgx);CHKERRQ(ierr);
  ierr = VecCopy(x, expx);CHKERRQ(ierr);
  ierr = VecExp(expx);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Avgx, Avgx, expx);CHKERRQ(ierr);
  ierr = MatMatMult(JT, J, reuse, PETSC_DEFAULT, hes);CHKERRQ(ierr);
  ierr = MatScale(*hes, -1.);CHKERRQ(ierr);
  ierr = MatSetOption(*hes, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDiagonalSet(*hes, Avgx, ADD_VALUES);CHKERRQ(ierr);
  ierr = VecDestroy(&gx);CHKERRQ(ierr);
  ierr = VecDestroy(&vgx);CHKERRQ(ierr);
  ierr = VecDestroy(&Avgx);CHKERRQ(ierr);
  ierr = VecDestroy(&expx);CHKERRQ(ierr);
  ierr = MatDestroy(&JT);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  if (Hadjvpre && Hadjvpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hadjvpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuildSwap_Vector(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *Hswpxhat, Mat *Hswpxhatpre)
{
  Mat            hesorig;
  Mat            *hes = Hswpxhat ? Hswpxhat : Hswpxhatpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnHessianBuild_Vector(fn, x, xhat, MAT_INITIAL_MATRIX, &hesorig, NULL);CHKERRQ(ierr);
  ierr = MatTranspose(hesorig, reuse, hes);CHKERRQ(ierr);
  ierr = MatDestroy(&hesorig);CHKERRQ(ierr);
  if (Hswpxhatpre && Hswpxhatpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hswpxhatpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeScalar_Vector(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], PetscScalar *z)
{
  const Vec      *supervecs;
  Vec            outvec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der < 0 || der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der+1, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  switch (der) {
  case 0:
    ierr = VecDuplicate(supervecs[0], &outvec);CHKERRQ(ierr);
    ierr = PetscFnApply_Vector(fn, x, outvec);CHKERRQ(ierr);
    ierr = VecDot(outvec, supervecs[0], z);CHKERRQ(ierr);
    ierr = VecDestroy(&outvec);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecDuplicate(supervecs[1], &outvec);CHKERRQ(ierr);
    if (rangeIdx == 0) {
      ierr = PetscFnJacobianMultAdjoint_Vector(fn, x, supervecs[0], outvec);CHKERRQ(ierr);
    } else {
      ierr = PetscFnJacobianMult_Vector(fn, x, supervecs[0], outvec);CHKERRQ(ierr);
    }
    ierr = VecDot(outvec, supervecs[1], z);CHKERRQ(ierr);
    ierr = VecDestroy(&outvec);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecDuplicate(supervecs[2], &outvec);CHKERRQ(ierr);
    if (rangeIdx == 0) {
      ierr = PetscFnHessianMultAdjoint_Vector(fn, x, supervecs[0], supervecs[1], outvec);CHKERRQ(ierr);
    } else if (rangeIdx == 1) {
      ierr = PetscFnHessianMultAdjoint_Vector(fn, x, supervecs[1], supervecs[0], outvec);CHKERRQ(ierr);
    } else {
      ierr = PetscFnHessianMult_Vector(fn, x, supervecs[0], supervecs[1], outvec);CHKERRQ(ierr);
    }
    ierr = VecDot(outvec, supervecs[2], z);CHKERRQ(ierr);
    ierr = VecDestroy(&outvec);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  ierr = PetscFnRestoreSuperVectors(fn, der+1, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_Vector(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  const Vec      *supervecs;
  Vec            supery;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der < 0 || der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der, rangeIdx, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  switch (der) {
  case 0:
    ierr = PetscFnApply_Vector(fn, x, supery);CHKERRQ(ierr);
    break;
  case 1:
    if (rangeIdx == 0) {
      ierr = PetscFnJacobianMultAdjoint_Vector(fn, x, supervecs[0], supery);CHKERRQ(ierr);
    } else {
      ierr = PetscFnJacobianMult_Vector(fn, x, supervecs[0], supery);CHKERRQ(ierr);
    }
    break;
  case 2:
    if (rangeIdx == 0) {
      ierr = PetscFnHessianMultAdjoint_Vector(fn, x, supervecs[0], supervecs[1], supery);CHKERRQ(ierr);
    } else if (rangeIdx == 1) {
      ierr = PetscFnHessianMultAdjoint_Vector(fn, x, supervecs[1], supervecs[0], supery);CHKERRQ(ierr);
    } else {
      ierr = PetscFnHessianMult_Vector(fn, x, supervecs[0], supervecs[1], supery);CHKERRQ(ierr);
    }
    break;
  default:
    break;
  }
  ierr = PetscFnRestoreSuperVectors(fn, der, rangeIdx, subsets, subvecs, y, &supervecs, &supery);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeMat_Vector(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *M, Mat *Mpre)
{
  const Vec      *supervecs;
  MatReuse        superreuse;
  Mat            *superM, *superMpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (der < 1 || der > 2) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE, "Higher derivatives not supported");
  ierr = PetscFnGetSuperVectors(fn, der-1, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  ierr = PetscFnGetSuperMats(fn, rangeIdx-(der-1), subsets ? &subsets[der-1] : NULL, reuse, M, Mpre, &superreuse, &superM, &superMpre);CHKERRQ(ierr);
  switch (der) {
  case 1:
    if (rangeIdx == 0) {
      ierr = PetscFnJacobianBuildAdjoint_Vector(fn, x, superreuse, superM, superMpre);CHKERRQ(ierr);
    } else {
      ierr = PetscFnJacobianBuild_Vector(fn, x, superreuse, superM, superMpre);CHKERRQ(ierr);
    }
    break;
  case 2:
    if (rangeIdx == 0) {
      ierr = PetscFnHessianBuildAdjoint_Vector(fn, x, supervecs[0], superreuse, superM, superMpre);CHKERRQ(ierr);
    } else if (rangeIdx == 1) {
      ierr = PetscFnHessianBuildSwap_Vector(fn, x, supervecs[0], superreuse, superM, superMpre);CHKERRQ(ierr);
    } else {
      ierr = PetscFnHessianBuild_Vector(fn, x, supervecs[0], superreuse, superM, superMpre);CHKERRQ(ierr);
    }
    break;
  default:
    break;
  }
  ierr = PetscFnRestoreSuperMats(fn, rangeIdx-(der-1), subsets ? &subsets[der-1] : NULL, reuse, M, Mpre, &superreuse, &superM, &superMpre);CHKERRQ(ierr);
  ierr = PetscFnRestoreSuperVectors(fn, der-2, rangeIdx, subsets, subvecs, NULL, &supervecs, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestBasicOps(PetscFn fn, PetscInt order, PetscRandom rand, PetscBool build_mat, PetscBool build_pre)
{
  Vec            x, f, xhat = NULL, xdot = NULL, Jxhat, v = NULL, Jadjv, Hxhatxdot, Hadjvxhat;
  Vec            g, Hxhat;
  Mat            jac, jacadj, hes, hesadj, hesswp, scalhes;
  Mat            jacPre, jacadjPre, hesPre, hesadjPre, hesswpPre, scalhesPre;
  PetscScalar    z;
  PetscBool      isScalar;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscFnIsScalar(fn, &isScalar);CHKERRQ(ierr);

  ierr = PetscFnCreateVecs(fn, NULL, &x, NULL, &f);CHKERRQ(ierr);
  ierr = VecSetRandom(x, rand);CHKERRQ(ierr);

  ierr = PetscFnApply(fn, x, f);CHKERRQ(ierr);
  if (isScalar) {ierr = PetscFnScalarApply(fn, x, &z);CHKERRQ(ierr);}

  if (order > 0) {
    Vec subvecs[2];
    ierr = PetscFnCreateVecs(fn, NULL, &xhat, NULL, &Jxhat);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn, NULL, &Jadjv,NULL, &v);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn, NULL, &xdot, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn, NULL, &g, NULL, NULL);CHKERRQ(ierr);

    ierr = VecSetRandom(xhat, rand);CHKERRQ(ierr);
    ierr = VecSetRandom(xdot, rand);CHKERRQ(ierr);
    ierr = VecSetRandom(v, rand);CHKERRQ(ierr);

    jac = jacadj = jacPre = jacadjPre = NULL;

    subvecs[0] = xhat;
    subvecs[1] = v;
    ierr = PetscFnDerivativeScalar(fn, x, 1, 1, NULL, subvecs, &z);CHKERRQ(ierr);
    ierr = PetscFnDerivativeVec(fn, x, 1, 1, NULL, &xhat, Jxhat);CHKERRQ(ierr);
    ierr = PetscFnDerivativeMat(fn, x, 1, 1, NULL, NULL, MAT_INITIAL_MATRIX, build_mat ? &jac : NULL, build_pre ? &jacPre : NULL);CHKERRQ(ierr);

    subvecs[0] = v;
    subvecs[1] = xhat;
    ierr = PetscFnDerivativeScalar(fn, x, 1, 0, NULL, subvecs, &z);CHKERRQ(ierr);
    ierr = PetscFnDerivativeVec(fn, x, 1, 0, NULL, &v, Jadjv);CHKERRQ(ierr);
    ierr = PetscFnDerivativeMat(fn, x, 1, 0, NULL, NULL, MAT_INITIAL_MATRIX, build_mat ? &jacadj : NULL, build_pre ? &jacadjPre : NULL);CHKERRQ(ierr);

    ierr = MatDestroy(&jacadj);CHKERRQ(ierr);
    ierr = MatDestroy(&jac);CHKERRQ(ierr);
    ierr = MatDestroy(&jacadjPre);CHKERRQ(ierr);
    ierr = MatDestroy(&jacPre);CHKERRQ(ierr);

    if (isScalar) {
      ierr = PetscFnScalarDerivativeScalar(fn, x, 1, NULL, &xhat, &z);CHKERRQ(ierr);
      ierr = PetscFnScalarDerivativeVec(fn, x, 1, NULL, NULL, g);CHKERRQ(ierr);
    }

    ierr = PetscFnJacobianMult(fn, x, xhat, Jxhat);CHKERRQ(ierr);
    ierr = PetscFnJacobianMultAdjoint(fn, x, v, Jadjv);CHKERRQ(ierr);

    if (isScalar) {ierr = PetscFnScalarGradient(fn, x, g);CHKERRQ(ierr);}

    ierr = PetscFnJacobianBuild(fn, x, MAT_INITIAL_MATRIX, build_mat ? &jac : NULL, build_pre ? &jacPre : NULL);CHKERRQ(ierr);
    ierr = PetscFnJacobianBuildAdjoint(fn, x, MAT_INITIAL_MATRIX, build_mat ? &jacadj : NULL, build_pre ? &jacadjPre: NULL);CHKERRQ(ierr);

    ierr = MatDestroy(&jacadj);CHKERRQ(ierr);
    ierr = MatDestroy(&jac);CHKERRQ(ierr);
    ierr = MatDestroy(&jacadjPre);CHKERRQ(ierr);
    ierr = MatDestroy(&jacPre);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
    ierr = VecDestroy(&Jadjv);CHKERRQ(ierr);
    ierr = VecDestroy(&Jxhat);CHKERRQ(ierr);
  }
  if (order > 1) {
    Vec subvecs[3];

    ierr = PetscFnCreateVecs(fn,NULL,&Hadjvxhat,NULL,&Hxhatxdot);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn,NULL,&Hxhat,NULL,NULL);CHKERRQ(ierr);

    hes    = hesadj    = hesswp    = scalhes    = NULL;
    hesPre = hesadjPre = hesswpPre = scalhesPre = NULL;

    subvecs[0] = xhat;
    subvecs[1] = xdot;
    subvecs[2] = v;
    ierr = PetscFnDerivativeScalar(fn, x, 2, 2, NULL, subvecs, &z);CHKERRQ(ierr);
    ierr = PetscFnDerivativeVec(fn, x, 2, 2, NULL, subvecs, Hxhatxdot);CHKERRQ(ierr);
    ierr = PetscFnDerivativeMat(fn, x, 2, 2, NULL, &xhat, MAT_INITIAL_MATRIX, build_mat ? &hes : NULL, build_pre ? &hesPre : NULL);CHKERRQ(ierr);

    subvecs[0] = xhat;
    subvecs[1] = v;
    subvecs[2] = xdot;
    ierr = PetscFnDerivativeScalar(fn, x, 2, 1, NULL, subvecs, &z);CHKERRQ(ierr);
    ierr = PetscFnDerivativeVec(fn, x, 2, 1, NULL, subvecs, Hadjvxhat);CHKERRQ(ierr);
    ierr = PetscFnDerivativeMat(fn, x, 2, 1, NULL, &xhat, MAT_INITIAL_MATRIX, build_mat ? &hesswp : NULL, build_pre ? &hesswpPre : NULL);CHKERRQ(ierr);

    subvecs[0] = v;
    subvecs[1] = xhat;
    subvecs[2] = xdot;
    ierr = PetscFnDerivativeScalar(fn, x, 2, 0, NULL, subvecs, &z);CHKERRQ(ierr);
    ierr = PetscFnDerivativeVec(fn, x, 2, 0, NULL, subvecs, Hadjvxhat);CHKERRQ(ierr);
    ierr = PetscFnDerivativeMat(fn, x, 2, 0, NULL, &v, MAT_INITIAL_MATRIX, build_mat ? &hesadj : NULL, build_pre ? &hesadjPre : NULL);CHKERRQ(ierr);

    if (isScalar) {
      subvecs[0] = xhat;
      subvecs[1] = xdot;

      ierr = PetscFnScalarDerivativeScalar(fn, x, 2, NULL, subvecs, &z);CHKERRQ(ierr);
      ierr = PetscFnScalarDerivativeVec(fn, x, 2, NULL, &xhat, Hxhat);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&scalhesPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesswpPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesadjPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesPre);CHKERRQ(ierr);
    ierr = MatDestroy(&scalhes);CHKERRQ(ierr);
    ierr = MatDestroy(&hesswp);CHKERRQ(ierr);
    ierr = MatDestroy(&hesadj);CHKERRQ(ierr);
    ierr = MatDestroy(&hes);CHKERRQ(ierr);

    ierr = PetscFnHessianMult(fn, x, xhat, xdot, Hxhatxdot);CHKERRQ(ierr);
    ierr = PetscFnHessianMultAdjoint(fn, x, v, xhat, Hadjvxhat);CHKERRQ(ierr);
    if (isScalar) {ierr = PetscFnScalarHessianMult(fn, x, xhat, Hxhat);CHKERRQ(ierr);}

    ierr = PetscFnHessianBuild(fn, x, xhat, MAT_INITIAL_MATRIX, build_mat ? &hes : NULL, build_pre ? &hesPre : NULL);CHKERRQ(ierr);
    ierr = PetscFnHessianBuildAdjoint(fn, x, v, MAT_INITIAL_MATRIX, build_mat ? &hesadj : NULL, build_pre ? &hesadjPre : NULL);CHKERRQ(ierr);
    ierr = PetscFnHessianBuildSwap(fn, x, xhat, MAT_INITIAL_MATRIX, build_mat ? &hesswp : NULL, build_pre ? &hesswpPre : NULL);CHKERRQ(ierr);
    if (isScalar) {ierr = PetscFnScalarHessianBuild(fn, x, MAT_INITIAL_MATRIX, build_mat ? &scalhes : NULL, build_pre ? &scalhesPre : NULL);CHKERRQ(ierr);}

    ierr = MatDestroy(&scalhesPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesswpPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesadjPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesPre);CHKERRQ(ierr);
    ierr = MatDestroy(&scalhes);CHKERRQ(ierr);
    ierr = MatDestroy(&hesswp);CHKERRQ(ierr);
    ierr = MatDestroy(&hesadj);CHKERRQ(ierr);
    ierr = MatDestroy(&hes);CHKERRQ(ierr);
    ierr = VecDestroy(&Hxhat);CHKERRQ(ierr);
    ierr = VecDestroy(&Hadjvxhat);CHKERRQ(ierr);
    ierr = VecDestroy(&Hxhatxdot);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&xhat);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDerivativeFns(PetscFn fn, PetscRandom rand)
{
  Vec            dotVecs[3];
  PetscBool      isScalar;
  PetscFn        df;
  Vec            x, xhat, xdot, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnCreateVecs(fn, NULL, &x, NULL, &v);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xhat);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xdot);CHKERRQ(ierr);

  ierr = VecSetRandom(x, rand);CHKERRQ(ierr);
  ierr = VecSetRandom(xhat, rand);CHKERRQ(ierr);
  ierr = VecSetRandom(xdot, rand);CHKERRQ(ierr);
  ierr = VecSetRandom(v, rand);CHKERRQ(ierr);

  ierr = PetscFnDerivativeFn(fn, 0, 0, 1, NULL, &v, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 2, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  ierr = PetscFnDerivativeFn(fn, 1, 1, 1, NULL, &xhat, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = xhat;
  dotVecs[1] = v;
  ierr = PetscFnDerivativeFn(fn, 1, 1, 2, NULL, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  ierr = PetscFnDerivativeFn(fn, 1, 0, 1, NULL, &v, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = v;
  dotVecs[1] = xhat;
  ierr = PetscFnDerivativeFn(fn, 1, 0, 2, NULL, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = xhat;
  dotVecs[1] = xdot;
  ierr = PetscFnDerivativeFn(fn, 2, 2, 2, NULL, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = xhat;
  dotVecs[1] = xdot;
  dotVecs[2] = v;
  ierr = PetscFnDerivativeFn(fn, 2, 2, 3, NULL, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = v;
  dotVecs[1] = xhat;
  ierr = PetscFnDerivativeFn(fn, 2, 0, 2, NULL, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = v;
  dotVecs[1] = xhat;
  dotVecs[2] = xdot;
  ierr = PetscFnDerivativeFn(fn, 2, 0, 3, NULL, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  ierr = PetscFnIsScalar(fn, &isScalar);CHKERRQ(ierr);
  if (isScalar) {
    ierr = PetscFnDerivativeFn(fn, 1, PETSC_DEFAULT, 0, NULL, NULL, &df);CHKERRQ(ierr);
    ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
    ierr = PetscFnSetUp(df);CHKERRQ(ierr);
    ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

    ierr = PetscFnDerivativeFn(fn, 1, PETSC_DEFAULT, 1, NULL, &xhat, &df);CHKERRQ(ierr);
    ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
    ierr = PetscFnSetUp(df);CHKERRQ(ierr);
    ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

    ierr = PetscFnDerivativeFn(fn, 2, PETSC_DEFAULT, 1, NULL, &xhat, &df);CHKERRQ(ierr);
    ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
    ierr = PetscFnSetUp(df);CHKERRQ(ierr);
    ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

    dotVecs[0] = xhat;
    dotVecs[1] = xdot;
    ierr = PetscFnDerivativeFn(fn, 2, PETSC_DEFAULT, 2, NULL, dotVecs, &df);CHKERRQ(ierr);
    ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
    ierr = PetscFnSetUp(df);CHKERRQ(ierr);
    ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&df);CHKERRQ(ierr);
  }

  VecDestroy(&x);CHKERRQ(ierr);
  VecDestroy(&xhat);CHKERRQ(ierr);
  VecDestroy(&xdot);CHKERRQ(ierr);
  VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestScalar(PetscBool set_vector, PetscBool set_scalar, PetscBool build_mat, PetscBool build_pre, PetscBool test_ders, PetscRandom rand)
{
  PetscFn        fn;
  PetscBool      isShell;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnCreate(PETSC_COMM_WORLD, &fn);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(fn, PETSC_DECIDE, 1, 1, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(fn);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)fn, PETSCFNSHELL, &isShell);CHKERRQ(ierr);
  comm = PetscObjectComm((PetscObject)fn);
  if (isShell) {
    PetscInt n, N, i, l;
    Vec v;
    void *ctx;
    PetscScalar *a;

    ierr = PetscFnGetSizes(fn, NULL, &n, NULL, &N);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm,n,N,&v);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(v, &l, NULL);CHKERRQ(ierr);
    ierr = VecGetArray(v, &a);CHKERRQ(ierr);
    for (i = 0; i < n; i++) {
      a[i] = i + l;
    }
    ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
    ierr = PetscFnShellSetContext(fn,(void *) v);CHKERRQ(ierr);
    ierr = PetscFnShellGetContext(fn,(void *) &ctx);CHKERRQ(ierr);
    if ((void *) v != ctx) SETERRQ(comm,PETSC_ERR_PLIB, "Shell context mismatch");
    ierr = PetscObjectReference((PetscObject)v);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DESTROY,(void (*)(void))PetscFnDestroy_Scalar);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_CREATEVECS,(void (*)(void))PetscFnCreateVecs_Scalar);CHKERRQ(ierr);

    if (set_vector) {
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_APPLY,(void (*)(void))PetscFnApply_Scalar);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DERIVATIVESCALAR,(void (*)(void))PetscFnDerivativeScalar_Scalar);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DERIVATIVEVEC,(void (*)(void))PetscFnDerivativeVec_Scalar);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DERIVATIVEMAT,(void (*)(void))PetscFnDerivativeMat_Scalar);CHKERRQ(ierr);
    }

    if (set_scalar) {
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARAPPLY,(void (*)(void))PetscFnScalarApply_Scalar);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARDERIVATIVESCALAR,(void (*)(void))PetscFnScalarDerivativeScalar_Scalar);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARDERIVATIVEVEC,(void (*)(void))PetscFnScalarDerivativeVec_Scalar);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARDERIVATIVEMAT,(void (*)(void))PetscFnScalarDerivativeMat_Scalar);CHKERRQ(ierr);
    }
  }
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  ierr = PetscFnViewFromOptions(fn, NULL, "-fn_view");CHKERRQ(ierr);


  ierr = TestBasicOps(fn, 2, rand, build_mat, build_pre);CHKERRQ(ierr);
  if (test_ders) {ierr = TestDerivativeFns(fn, rand);CHKERRQ(ierr);}

  ierr = PetscFnDestroy(&fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestVector(PetscBool build_mat, PetscBool build_pre, PetscBool test_ders, PetscRandom rand)
{
  PetscFn        fn;
  PetscBool      isShell;
  PetscInt       rank;
  MPI_Comm       comm;
  PetscRandom    rand_plus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnCreate(PETSC_COMM_WORLD, &fn);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(fn, rank + 3, rank + 7, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(fn);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)fn, PETSCFNSHELL, &isShell);CHKERRQ(ierr);
  comm = PetscObjectComm((PetscObject)fn);
  if (isShell) {
    PetscInt n, N, m, M;
    Mat  A;
    void *ctx;

    ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
    ierr = MatCreateAIJ(comm, m, n, M, N, 2, NULL, 2, NULL, &A);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PetscObjectComm((PetscObject)fn), &rand_plus);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(rand_plus, 0., 0.5);CHKERRQ(ierr);
    ierr = MatSetRandom(A, rand_plus);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rand_plus);CHKERRQ(ierr);
    ierr = PetscFnShellSetContext(fn,(void *) A);CHKERRQ(ierr);
    ierr = PetscFnShellGetContext(fn,(void *) &ctx);CHKERRQ(ierr);
    if ((void *) A != ctx) SETERRQ(comm,PETSC_ERR_PLIB, "Shell context mismatch");
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DESTROY,(void (*)(void))PetscFnDestroy_Vector);CHKERRQ(ierr);

    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_APPLY,(void (*)(void))PetscFnApply_Vector);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DERIVATIVESCALAR,(void (*)(void))PetscFnDerivativeScalar_Vector);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DERIVATIVEVEC,(void (*)(void))PetscFnDerivativeVec_Vector);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DERIVATIVEMAT,(void (*)(void))PetscFnDerivativeMat_Vector);CHKERRQ(ierr);
  }
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  ierr = PetscFnViewFromOptions(fn, NULL, "-fn_view");CHKERRQ(ierr);

  ierr = TestBasicOps(fn, 2, rand, build_mat, build_pre);CHKERRQ(ierr);
  if (test_ders) {ierr = TestDerivativeFns(fn, rand);CHKERRQ(ierr);}

  ierr = PetscFnDestroy(&fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscBool      set_vector, set_scalar, build_mat, build_pre;
  PetscBool      test_ders;
  PetscRandom    rand;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  set_vector = PETSC_TRUE;
  set_scalar = PETSC_TRUE;
  build_mat  = PETSC_TRUE;
  build_pre  = PETSC_FALSE;
  test_ders  = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "PetscFn Test Options", "PetscFn");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-set_vector", "Set vector callbacks for PetscFnShell",          "ex1.c", set_vector, &set_vector, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-set_scalar", "Set scalar callbacks for PetscFnShell",          "ex1.c", set_scalar, &set_scalar, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-build_mat",  "Build the derivative matrices",                  "ex1.c", build_mat,  &build_mat, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-build_pre",  "Build the derivative preconditioning matrices",  "ex1.c", build_pre,  &build_pre, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_ders",  "Test the construnction of derivative functions", "ex1.c", test_ders,  &test_ders, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = TestScalar(set_vector, set_scalar, build_mat, build_pre, test_ders, rand);CHKERRQ(ierr);
  ierr = TestVector(build_mat, build_pre, test_ders, rand);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -fn_test_scalar -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0

   test:
      suffix: 2
      nsize: 1
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0
      output_file: output/ex1_1.out

   test:
      suffix: 3
      nsize: 2
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0

   test:
      suffix: 4
      nsize: 2
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0
      output_file: output/ex1_3.out

   test:
      suffix: 5
      nsize: 1
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -build_pre -build_mat 0
      output_file: output/ex1_1.out

   test:
      suffix: 6
      nsize: 1
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0 -build_pre -build_mat 0
      output_file: output/ex1_1.out

   test:
      suffix: 7
      nsize: 1
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0 -build_pre -build_mat 0
      output_file: output/ex1_1.out

   test:
      suffix: 8
      nsize: 2
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -build_pre -build_mat 0
      output_file: output/ex1_3.out

   test:
      suffix: 9
      nsize: 2
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0 -build_pre -build_mat 0
      output_file: output/ex1_3.out

   test:
      suffix: 10
      nsize: 2
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0 -build_pre -build_mat 0
      output_file: output/ex1_3.out

   test:
      suffix: 11
      nsize: 1
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0 -build_pre -build_mat

   test:
      suffix: 12
      nsize: 1
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0 -build_pre -build_mat
      output_file: output/ex1_11.out

   test:
      suffix: 13
      nsize: 2
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0 -build_pre -build_mat
      output_file: output/ex1_11.out

   test:
      suffix: 14
      nsize: 2
      args: -fn_test_scalar -fn_test_vec -fn_test_mat -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0 -build_pre -build_mat
      output_file: output/ex1_11.out

   test:
      suffix: 15
      nsize: 1
      args: -test_ders -fn_test_fn -fn_test_derivativefn_view -der_fn_test_vec -der_fn_test_mat -der_fn_test_derivative_view -der_fn_test_derivativemat_view

   test:
      suffix: 16
      nsize: 4
      args: -test_ders -fn_test_fn -fn_test_derivativefn_view -der_fn_test_vec -der_fn_test_mat -der_fn_test_derivative_view -der_fn_test_derivativemat_view

TEST*/
