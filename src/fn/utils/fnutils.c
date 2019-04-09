
#include <petsc/private/vecimpl.h>       /*I "petscvec.h" I*/
#include <petsc/private/matimpl.h>       /*I "petscmat.h" I*/
#include <petsc/private/fnimpl.h>        /*I "petscfn.h" I*/

PetscErrorCode MatDuplicateOrCopy(Mat orig, MatReuse reuse, Mat *dup)
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

PetscErrorCode VecScalarBcast(Vec v, PetscScalar *zp)
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

PetscErrorCode PetscFnGetSuperVector(PetscFn fn, PetscBool isRange, IS subset, Vec subvec, Vec *supervec, PetscBool read)
{
  Vec            newvec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!subset) {
    *supervec = subvec;
  } else {
    PetscInt n;
    const PetscInt *idx;
    const PetscScalar *va;

    if (isRange) {
      ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &newvec);CHKERRQ(ierr);
    } else {
      ierr = PetscFnCreateVecs(fn, NULL, &newvec, NULL, NULL);CHKERRQ(ierr);
    }
    if (read) {
      ierr = ISGetLocalSize(subset, &n);CHKERRQ(ierr);
      ierr = ISGetIndices(subset, &idx);CHKERRQ(ierr);
      ierr = VecGetArrayRead(subvec, &va);CHKERRQ(ierr);
      ierr = VecSetValues(newvec, n, idx, va, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(subvec, &va);CHKERRQ(ierr);
      ierr = ISRestoreIndices(subset, &idx);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(newvec);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(newvec);CHKERRQ(ierr);
    }
    *supervec = newvec;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnRestoreSuperVector(PetscFn fn, PetscBool isRange, IS subset, Vec subvec, Vec *supervec, PetscBool write)
{
  Vec            newvec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!subset) {
    *supervec = NULL;
  } else {
    PetscInt n;
    const PetscInt *idx;
    PetscScalar *va;

    newvec = *supervec;
    if (write) {
      ierr = ISGetLocalSize(subset, &n);CHKERRQ(ierr);
      ierr = ISGetIndices(subset, &idx);CHKERRQ(ierr);
      ierr = VecGetArray(subvec, &va);CHKERRQ(ierr);
      ierr = VecGetValues(newvec, n, idx, va);CHKERRQ(ierr);
      ierr = VecRestoreArray(subvec, &va);CHKERRQ(ierr);
      ierr = ISRestoreIndices(subset, &idx);CHKERRQ(ierr);
    }
    ierr = VecDestroy(supervec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetSuperVectors(PetscFn fn, PetscInt numVecs, PetscInt rangeIdx, const IS subsets[], const Vec subvecs [], Vec outsubvec, const Vec *supervecs[], Vec *outsupervec)
{
  PetscInt       i;
  Vec            *newvecs;
  PetscBool      anySubset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  anySubset = PETSC_TRUE;
  if (!subsets) {
    anySubset = PETSC_FALSE;
  } else {
    for (i = 0; i < numVecs; i++) {
      if (subsets[i]) break;
    }
    if (i < numVecs) {
      anySubset = PETSC_FALSE;
    } else {
      if (outsubvec && subsets[numVecs]) {
        anySubset = PETSC_FALSE;
      }
    }
  }
  if (!anySubset) {
    *supervecs = subvecs;
    if (outsubvec) {
      *outsupervec = outsubvec;
    }
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc1(numVecs, &newvecs);CHKERRQ(ierr);
  for (i = 0; i < numVecs; i++) {
    ierr = PetscFnGetSuperVector(fn, (i == rangeIdx) ? PETSC_TRUE : PETSC_FALSE, subsets[i], subvecs[i], &newvecs[i], PETSC_TRUE);CHKERRQ(ierr);
  }
  if (outsubvec) {
    ierr = PetscFnGetSuperVector(fn, (numVecs == rangeIdx) ? PETSC_TRUE : PETSC_FALSE, subsets[numVecs], outsubvec, outsupervec, PETSC_FALSE);CHKERRQ(ierr);
  }
  *supervecs = newvecs;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnRestoreSuperVectors(PetscFn fn, PetscInt numVecs, PetscInt rangeIdx, const IS subsets[], const Vec subvecs [], Vec outsubvec, const Vec *supervecs[], Vec *outsupervec)
{
  PetscInt       i;
  Vec            *newvecs;
  PetscBool      anySubset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  anySubset = PETSC_TRUE;
  if (!subsets) {
    anySubset = PETSC_FALSE;
  } else {
    for (i = 0; i < numVecs; i++) {
      if (subsets[i]) break;
    }
    if (i < numVecs) {
      anySubset = PETSC_FALSE;
    } else {
      if (outsubvec && subsets[numVecs]) {
        anySubset = PETSC_FALSE;
      }
    }
  }
  if (!anySubset) PetscFunctionReturn(0);
  newvecs = (Vec *) *supervecs;
  if (outsubvec) {
    ierr = PetscFnGetSuperVector(fn, (numVecs == rangeIdx) ? PETSC_TRUE : PETSC_FALSE, subsets[numVecs], outsubvec, outsupervec, PETSC_TRUE);CHKERRQ(ierr);
  }
  for (i = 0; i < numVecs; i++) {
    ierr = PetscFnRestoreSuperVector(fn, (i == rangeIdx) ? PETSC_TRUE : PETSC_FALSE, subsets[i], subvecs[i], &newvecs[i], PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = PetscFree(*newvecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetSuperMats(PetscFn fn, PetscInt numSubsets, PetscInt rangeIdx, const IS subsets[], MatReuse reuse, Mat *J, Mat *Jpre, MatReuse *superReuse, Mat **superJ, Mat **superJpre)
{
  IS rightIS = subsets ? subsets[numSubsets-2] : NULL;
  IS leftIS  = subsets ? subsets[numSubsets-1] : NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!rightIS && !leftIS) {
    *superReuse = reuse;
    *superJ = J;
    *superJpre = Jpre;
    PetscFunctionReturn(0);
  }
  *superReuse = MAT_INITIAL_MATRIX;
  if (J) {
    Mat *mat;

    ierr = PetscMalloc1(1,&mat);CHKERRQ(ierr);
    *mat = NULL;
    *superJ = mat;
  } else {
    *superJ = NULL;
  }
  if (Jpre) {
    if (reuse == MAT_INITIAL_MATRIX || !J || *J != *Jpre) {
      Mat *matPre;

      ierr = PetscMalloc1(1,&matPre);CHKERRQ(ierr);
      *matPre = NULL;
      *superJpre = matPre;
    } else {
      *superJpre = NULL;
    }
  } else {
    *superJpre = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnRestoreSuperMats(PetscFn fn, PetscInt numSubsets, PetscInt rangeIdx, const IS subsets[], MatReuse reuse, Mat *J, Mat *Jpre, MatReuse *superReuse, Mat **superJ, Mat **superJpre)
{
  IS rightIS = subsets ? subsets[numSubsets-2] : NULL;
  IS leftIS  = subsets ? subsets[numSubsets-1] : NULL;
  PetscErrorCode ierr;
  Mat supermat = NULL;
  Mat supermatpre = NULL;
  Mat submat = NULL;
  Mat submatpre = NULL;

  PetscFunctionBegin;
  if (!rightIS && !leftIS) {
    PetscFunctionReturn(0);
  }
  if (*superJ) {
    supermat = **superJ;
  }
  if (*superJpre) {
    supermatpre = **superJpre;
  }
  if (supermat) {
    ierr = MatCreateSubMatrix(supermat, leftIS, rightIS, MAT_INITIAL_MATRIX, &submat);CHKERRQ(ierr);
  }
  if (supermatpre) {
    if (!supermat || supermat != supermatpre) {
      ierr = MatCreateSubMatrix(supermat, leftIS, rightIS, MAT_INITIAL_MATRIX, &submatpre);CHKERRQ(ierr);
    } else {
      submatpre = submat;
      ierr = PetscObjectReference((PetscObject)submat);CHKERRQ(ierr);
    }
  }
  if (J) {
    if (reuse == MAT_INITIAL_MATRIX) {
      *J = submat;
    } else {
      ierr = MatCopy(submat, *J, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&submat);CHKERRQ(ierr);
    }
  }
  if (Jpre) {
    if (reuse == MAT_INITIAL_MATRIX) {
      *Jpre = submatpre;
    } else {
      ierr = MatCopy(submatpre, *Jpre, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&submatpre);CHKERRQ(ierr);
    }
  }
  if (*superJ) {
    ierr = PetscFree(*superJ);CHKERRQ(ierr);
  }
  if (*superJpre) {
    ierr = PetscFree(*superJpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreateDefaultScalarVec(MPI_Comm comm, Vec *vz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPI(comm, PETSC_DETERMINE, 1, vz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
