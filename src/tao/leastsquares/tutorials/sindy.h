#ifndef PETSCSINDY_H
#define PETSCSINDY_H

#include <petsctao.h>

typedef struct _p_Basis* Basis;
typedef struct _p_SparseReg* SparseReg;

PETSC_EXTERN PetscErrorCode SINDyBasisCreate(PetscInt, PetscInt, Basis*);
PETSC_EXTERN PetscErrorCode SINDyBasisDestroy(Basis*);
PETSC_EXTERN PetscErrorCode SINDyBasisSetFromOptions(Basis);
PETSC_EXTERN PetscErrorCode SINDyBasisCreateData(Basis, Vec*, PetscInt);
PETSC_EXTERN PetscErrorCode SINDyBasisDataGetSize(Basis, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode SINDyBasisPrint(Basis, PetscInt, Vec*);

PETSC_EXTERN PetscErrorCode SINDyFindSparseCoefficients(Basis, SparseReg, PetscInt, Vec*, Vec*);
PETSC_EXTERN PetscErrorCode SINDySparseLeastSquares(Mat, Vec, Mat, Vec);
PETSC_EXTERN PetscErrorCode SINDySparseRegCreate(SparseReg*);
PETSC_EXTERN PetscErrorCode SINDySparseRegSetThreshold(SparseReg, PetscReal);
PETSC_EXTERN PetscErrorCode SINDySparseRegSetFromOptions(SparseReg);
PETSC_EXTERN PetscErrorCode SINDySparseRegDestroy(SparseReg*);


#endif
