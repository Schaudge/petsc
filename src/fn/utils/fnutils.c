
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

PetscErrorCode PetscFnVecsGetSuperVectors(PetscFn fn, PetscInt numVecs, PetscInt rangeIdx, const IS subsets[], const Vec subvecs [], Vec outsubvec, const Vec *supervecs[], Vec *outsupervec)
{
  PetscInt       i;
  Vec            *newvecs;
  PetscBool      anySubset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(fn,PETSCFN_CLASSID,1);
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
        anySubset = PETSC_TRUE;
      }
    }
  }
  if (!anySubset) {
    *supervecs = subvecs;
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc1(numVecs, &newvecs);CHKERRQ(ierr);
  for (i = 0; i < numVecs; i++) {
    if (subsets[i]) {
      Vec newvec;

      if (i == rangeIdx) {
        ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &newvec);CHKERRQ(ierr);
      } else {
        ierr = PetscFnCreateVecs(fn, NULL, &newvec, NULL, NULL);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}
