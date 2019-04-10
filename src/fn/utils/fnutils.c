
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

/*@
   PetscFnGetSuperVector - Get a full vector from a subvector described by its subset.

   Collective on PetscFn, Vec

   Input Parameter:
+  fn - a PetscFn function
.  isRange - PETSC_TRUE if the subvector is taken from a range-space vector, PETSC_FALSE if it is taken from a full-space vector
.  subset - the subset describing the subvector
.  subvec - the subvector
-  read - PETSC_TRUE if we need to read values from subvec, PETSC_FALSE if we just need a supervector of the appropriate shape

   Output Parameter:
.  supervec - the supervector

   Note:

   Should be restored with PetscFnRestoreSuperVector()

   Level: advanced

.seealso: PetscFnRestoreSuperVector(), PetscFnGetSuperVectors(), PetscFnRestoreSuperVectors()
@*/
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

/*@
   PetscFnRestoreSuperVector - Restore a full vector for a subvector described by its subset that was got from
                               PetscFnGetSuperVector().

   Collective on PetscFn, Vec

   Input Parameter:
+  fn - a PetscFn function
.  isRange - PETSC_TRUE if the subvector is taken from a range-space vector, PETSC_FALSE if it is taken from a full-space vector
.  subset - the subset describing the subvector
.  supervec - the supervector
-  write - PETSC_TRUE if we need to write values back to the subvec, PETSC_FALSE otherwise

   Output Parameter:
.  subvec - the subvector

   Note:

   Level: advanced

.seealso: PetscFnGetSuperVector(), PetscFnGetSuperVectors(), PetscFnRestoreSuperVectors()
@*/
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

/*@
   PetscFnGetSuperVectors - Get full vectors from subvectors described by their subsets: can be used
   on the arguments of various derivative functions (PetscFnDerivativeScalar(), PetscFnDerivativeVec(), PetscFnDerivativeMat(),
   PetscFnScalarDerivativeScalar(), PetscFnScalarDerivativeVec(), PetscFnScalarDerivativeMat()) if the implementation
   does not handle partial derivatives directly.  The supervectors will be padded with zeros outside the subsets.

   Collective on PetscFn, Vec

   Input Parameter:
+  fn - a PetscFn function
.  numVecs - the number of vectors in subvecs
.  rangeIdx - the index of the vector in subvecs that is a range-space vector.  Use rangeIdx = numVecs if outsubvec is a range-space vector.  Use rangeIdx = PETSC_DEFAULT if fn is scalar-valued
.  subsets - the subsets describing the subvector.  Should be of length numVecs if outsubvec is not present, or numVecs + 1 if it is.
.  subvecs - the subvectors whose values must be read (like the variations use in , e.g. PetscFnDerivativeVec())
-  outsubvec - (optional) the subvector whose values must be written (like the output y of, e.g. PetscFnDerivativeVec())

   Output Parameter:
+  supervecs - the supervectors of the vectors in subvecs: values outside the subsets are padded with zeros
-  outsupervec - (optional) the supervectors of the outsubvec: only values written in the subset will be restored to outsubvec in PetscFnRestoreSuperVectors()

   Note:

   Should be restored with PetscFnRestoreSuperVectors()

   Level: advanced

.seealso: PetscFnRestoreSuperVectors(), PetscFnGetSuperVector(), PetscFnRestoreSuperVector()
@*/
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

/*@
   PetscFnRestoreSuperVectors - Restore full vectors that were got with PetscFnGetSuperVectors()

   Collective on PetscFn, Vec

   Input Parameter:
+  fn - a PetscFn function
.  numVecs - the number of vectors in subvecs
.  rangeIdx - the index of the vector in subvecs that is a range-space vector.  Use rangeIdx = numVecs if outsubvec is a range-space vector.  Use rangeIdx = PETSC_DEFAULT if fn is scalar-valued
.  subsets - the subsets describing the subvector.  Should be of length numVecs if outsubvec is not present, or numVecs + 1 if it is.
.  subvecs - the subvectors whose values must be read (like the variations use in , e.g. PetscFnDerivativeVec())
-  outsupervec - (optional) the supervectors of the outsubvec: only values written in the subset will be restored to outsubvec in PetscFnRestoreSuperVectors()

   Output Parameter:
-  supervecs - the supervectors of the vectors in subvecs: no longer accessible after this call
-  outsubvec - (optional) the subvector whose values must be written (like the output y of, e.g. PetscFnDerivativeVec())

   Note:

   Should be restored with PetscFnRestoreSuperVectors()

   Level: advanced

.seealso: PetscFnRestoreSuperVectors(), PetscFnGetSuperVector(), PetscFnRestoreSuperVector()
@*/
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

/*@
   PetscFnGetSuperMats - Get full matrices: can be used
   on the arguments of PetscFnDerivativeMat() and PetscFnScalarDerivativeMat() if the implementation does not handle
   partial derivatives directly.

   Collective on PetscFn

   Input Parameter:
+  fn - a PetscFn function
.  rangeIdx - 0 if the column space of the matrix is the fn range space; 1 if the row space of the matrix is the fn range space; ignored otherwise
.  subsets - (optional) the subsets describing the subvector of the column space and row space, respectively.
.  reuse - the desired reuse behavior of the original submatrices J and Jpre
.  J - the submatrix: will be filled on the matching call to PetscFnRestoreSuperMats()
-  Jpre - (optional) the preconditioner submatrix: will be filled on the matching call to PetscFnRestoreSuperMats()

   Output Parameter:
+  superreuse - the matrix reuse behavior that should be used for the super matrices
.  superJ - the supermatrix for J: values outside the submatrix destribed by the subsets will be ignored
-  superJpre - (optional) the supermatrix for Jpre: values outside the submatrix destribed by the subsets will be ignored

   Note:

   Should be restored with PetscFnRestoreSuperMats()

   Level: advanced

.seealso: PetscFnRestoreSuperMats()
@*/
PetscErrorCode PetscFnGetSuperMats(PetscFn fn, PetscInt rangeIdx, const IS subsets[], MatReuse reuse, Mat *J, Mat *Jpre, MatReuse *superReuse, Mat **superJ, Mat **superJpre)
{
  IS rightIS = subsets ? subsets[0] : NULL;
  IS leftIS  = subsets ? subsets[1] : NULL;
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

/*@
   PetscFnRestoreSuperMats - Restore full matrices that were got with PetscFnGetSuperMats(): can be used
   on the arguments of PetscFnDerivativeMat() and PetscFnScalarDerivativeMat() if the implementation does not handle
   partial derivatives directly.

   Collective on PetscFn

   Input Parameter:
+  fn - a PetscFn function
.  rangeIdx - 0 if the column space of the matrix is the fn range space; 1 if the row space of the matrix is the fn range space; ignored otherwise
.  subsets - (optional) the subsets describing the subvector of the column space and row space, respectively.
.  reuse - the desired reuse behavior of the original submatrices J and Jpre

   Output Parameter:
+  J - the submatrix: will be filled with the values of superJ within the subsets
-  Jpre - (optional) the preconditioner submatrix: will be filled with the values of superJpre within the subsets

   Note:

   Level: advanced

.seealso: PetscFnGetSuperMats()
@*/
PetscErrorCode PetscFnRestoreSuperMats(PetscFn fn, PetscInt rangeIdx, const IS subsets[], MatReuse reuse, Mat *J, Mat *Jpre, MatReuse *superReuse, Mat **superJ, Mat **superJpre)
{
  IS rightIS = subsets ? subsets[0] : NULL;
  IS leftIS  = subsets ? subsets[1] : NULL;
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
