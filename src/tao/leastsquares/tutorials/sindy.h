#ifndef PETSCSINDY_H
#define PETSCSINDY_H

#include <petsctao.h>
#include <petscts.h>
#include "sparse_reg.h"
#include "variable.h"

typedef struct _p_Basis* Basis;

PETSC_EXTERN PetscErrorCode SINDyBasisCreate(PetscInt, PetscInt, Basis*);
PETSC_EXTERN PetscErrorCode SINDyBasisDestroy(Basis*);
PETSC_EXTERN PetscErrorCode SINDyBasisSetNormalizeColumns(Basis, PetscBool);
PETSC_EXTERN PetscErrorCode SINDyBasisSetCrossTermRange(Basis, PetscInt);
PETSC_EXTERN PetscErrorCode SINDyBasisSetFromOptions(Basis);
PETSC_EXTERN PetscErrorCode SINDyBasisDataGetSize(Basis, PetscInt*, PetscInt*);

PETSC_EXTERN PetscErrorCode SINDyFindSparseCoefficients(Basis, SparseReg, PetscInt, Vec*);
PETSC_EXTERN PetscErrorCode SINDyBasisPrint(Basis, PetscInt, Vec*);

PETSC_EXTERN PetscErrorCode SINDyBasisAddVariables(Basis, PetscInt, Variable*);
PETSC_EXTERN PetscErrorCode SINDyBasisSetOutputVariable(Basis, Variable);

#endif
