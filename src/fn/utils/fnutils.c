
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
    if (subsets[i]) {
      Vec newvec;
      PetscInt n;
      const PetscInt *idx;
      const PetscScalar *va;

      if (i == rangeIdx) {
        ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &newvec);CHKERRQ(ierr);
      } else {
        ierr = PetscFnCreateVecs(fn, NULL, &newvec, NULL, NULL);CHKERRQ(ierr);
      }
      ierr = ISGetLocalSize(subsets[i], &n);CHKERRQ(ierr);
      ierr = ISGetIndices(subsets[i], &idx);CHKERRQ(ierr);
      ierr = VecGetArrayRead(subvecs[i], &va);CHKERRQ(ierr);
      ierr = VecSetValues(newvec, n, idx, va, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(subvecs[i], &va);CHKERRQ(ierr);
      ierr = ISRestoreIndices(subsets[i], &idx);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(newvec);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(newvec);CHKERRQ(ierr);
    }
  }
  if (outsubvec && subsets[numVecs]) {
    if (numVecs == rangeIdx) {
      ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, outsupervec);CHKERRQ(ierr);
    } else {
      ierr = PetscFnCreateVecs(fn, NULL, outsupervec, NULL, NULL);CHKERRQ(ierr);
    }
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
  if (outsubvec && subsets[numVecs]) {
    PetscInt n;
    const PetscInt *idx;
    PetscScalar *va;

    ierr = ISGetLocalSize(subsets[numVecs], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(subsets[numVecs], &idx);CHKERRQ(ierr);
    ierr = VecGetArray(outsubvec, &va);CHKERRQ(ierr);
    ierr = VecGetValues(*outsupervec, n, idx, va);CHKERRQ(ierr);
    ierr = VecRestoreArray(outsubvec, &va);CHKERRQ(ierr);
    ierr = ISRestoreIndices(subsets[numVecs], &idx);CHKERRQ(ierr);
    ierr = VecDestroy(outsupervec);CHKERRQ(ierr);
  }
  for (i = 0; i < numVecs; i++) {
    if (subsets[i]) {
      ierr = VecDestroy(&newvecs[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(*newvecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetSuperMats(PetscFn fn, PetscInt numSubsets, PetscInt rangeIdx, const IS subsets[], MatReuse reuse, Mat *J, Mat *Jpre, Mat *superM, Mat *superJpre)
{
  IS rightIS = subsets ? subsets[numSubsets-2] : NULL;
  IS leftIS  = subsets ? subsets[numSubsets-1] : NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!rightIS && !leftIS) {
    *superM = *J;
    *superJpre = *Jpre;
    PetscFunctionReturn(0);
  }
  if (reuse == MAT_INITIAL_MATRIX) {
    *superM = NULL;
    *superJpre = NULL;
  } else {
    PetscInt m, M, n, N;
    PetscInt matm, matM, matn, matN;
    Mat mat = NULL, matPre = NULL;
    MatType type;

    ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
    if (rangeIdx == numSubsets-2) {
      matn = m;
      matN = M;
    } else {
      matn = n;
      matN = N;
    }
    if (rangeIdx == numSubsets-1) {
      matm = m;
      matM = M;
    } else {
      matm = n;
      matM = N;
    }
    if (J) {
      ierr = MatCreate(PetscObjectComm((PetscObject)fn),&mat);CHKERRQ(ierr);
      ierr = MatSetSizes(mat, matm, matn, matM, matN);CHKERRQ(ierr);
      ierr = MatGetType(*J, &type);CHKERRQ(ierr);
      ierr = MatSetType(mat, type);CHKERRQ(ierr);
      *superM = mat;
    }
    if (Jpre) {
      if (J && *Jpre == *J) {
        *superJpre = *superM;
      }
      else {
        ierr = MatCreate(PetscObjectComm((PetscObject)fn),&matPre);CHKERRQ(ierr);
        ierr = MatSetSizes(matPre, matm, matn, matM, matN);CHKERRQ(ierr);
        ierr = MatGetType(*Jpre, &type);CHKERRQ(ierr);
        ierr = MatSetType(matPre, type);CHKERRQ(ierr);
        *superJpre = matPre;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnRestoreSuperMats(PetscFn fn, PetscInt numSubsets, PetscInt rangeIdx, const IS subsets[], MatReuse reuse, Mat *J, Mat *Jpre, Mat *superM, Mat *superJpre)
{
  IS rightIS = subsets ? subsets[numSubsets-2] : NULL;
  IS leftIS  = subsets ? subsets[numSubsets-1] : NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!rightIS && !leftIS) {
    *J = *superM;
    *Jpre = *superJpre;
  }
  if (reuse == MAT_INITIAL_MATRIX) {
    if (superM) {
      ierr = MatCreateSubMatrix(*superM, leftIS, rightIS, MAT_INITIAL_MATRIX, J);CHKERRQ(ierr);
      ierr = MatDestroy(superM);CHKERRQ(ierr);
    }
    if (superJpre) {
      if (superM && *superM == *superJpre) {
        *Jpre = *J;
      } else {
        ierr = MatCreateSubMatrix(*superJpre, leftIS, rightIS, MAT_INITIAL_MATRIX, Jpre);CHKERRQ(ierr);
        ierr = MatDestroy(superJpre);CHKERRQ(ierr);
      }
    }
  } else {
    Mat mat;

    if (superM) {
      ierr = MatCreateSubMatrix(*superM, leftIS, rightIS, MAT_INITIAL_MATRIX, &mat);CHKERRQ(ierr);
      ierr = MatCopy(mat, *J, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&mat);CHKERRQ(ierr);
      ierr = MatDestroy(superM);CHKERRQ(ierr);
    }
    if (superJpre) {
      if (superM && *superM == *superJpre) {
        *Jpre = *J;
      } else {
        ierr = MatCreateSubMatrix(*superJpre, leftIS, rightIS, MAT_INITIAL_MATRIX, &mat);CHKERRQ(ierr);
        ierr = MatCopy(mat, *Jpre, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatDestroy(&mat);CHKERRQ(ierr);
        ierr = MatDestroy(superJpre);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}
