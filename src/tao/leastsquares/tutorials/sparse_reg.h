#ifndef PETSCSPARSEREG_H
#define PETSCSPARSEREG_H

#include <petsctao.h>

typedef struct _p_SparseReg* SparseReg;
PETSC_EXTERN PetscClassId    SPARSEREG_CLASSID;

PETSC_EXTERN PetscErrorCode SparseRegCreate(SparseReg*);
PETSC_EXTERN PetscErrorCode SparseRegSetThreshold(SparseReg, PetscReal);
PETSC_EXTERN PetscErrorCode SparseRegSetMonitor(SparseReg, PetscBool);
PETSC_EXTERN PetscErrorCode SparseRegSetFromOptions(SparseReg);
PETSC_EXTERN PetscErrorCode SparseRegDestroy(SparseReg*);

PETSC_EXTERN PetscErrorCode SparseRegSTLSQR(SparseReg, Mat, Vec, Mat, Vec);
PETSC_EXTERN PetscErrorCode SparseRegLS(SparseReg, Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode SparseRegRLS(SparseReg, Mat, Vec, Mat, Vec);

PETSC_EXTERN PetscErrorCode SparseRegGetTotalIterationNumber(SparseReg, PetscInt*);

#endif
