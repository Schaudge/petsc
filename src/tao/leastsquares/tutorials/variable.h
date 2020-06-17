#ifndef PETSCVARIABLE_H
#define PETSCVARIABLE_H

#include <petscdmda.h>

typedef struct _p_Variable* Variable;

PETSC_EXTERN PetscErrorCode VariableCreate(const char*, Variable*);
PETSC_EXTERN PetscErrorCode VariableSetScalarData(Variable, PetscInt, PetscScalar*);
PETSC_EXTERN PetscErrorCode VariableSetVecData(Variable, PetscInt, Vec*, DM);
PETSC_EXTERN PetscErrorCode VariableDestroy(Variable*);
PETSC_EXTERN PetscErrorCode VariablePrint(Variable);

PETSC_EXTERN PetscErrorCode VariableDifferentiateSpatial(Variable, PetscInt, PetscInt, const char*, Variable*);
PETSC_EXTERN PetscErrorCode VariableExtractDataByDim(Variable, Vec**);

#endif
