#ifndef PETSCSINDY_H
#define PETSCSINDY_H

#include <petsctao.h>

typedef struct _p_Basis* Basis;

PETSC_EXTERN PetscErrorCode SINDyCreateBasis(PetscInt, PetscInt, Basis*);
PETSC_EXTERN PetscErrorCode SINDyCreateBasisData(Basis, Vec, PetscInt);
PETSC_EXTERN PetscErrorCode SINDyCreateBasisAndData(Vec, PetscInt, PetscInt, PetscInt, Basis*);
PETSC_EXTERN PetscErrorCode SINDyBasisDestroy(Basis*);
PETSC_EXTERN PetscErrorCode SINDyBasisDataGetSize(Basis, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode SINDyFindSparseCoefficients(Basis, PetscInt, Vec*, Vec*);
PETSC_EXTERN PetscErrorCode SINDySparseLeastSquares(Mat, Vec, Mat, Vec);
PETSC_EXTERN PetscErrorCode SINDyBasisPrint(Basis, PetscInt, Vec*);


#endif
