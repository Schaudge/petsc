#ifndef PETSCSINDY_H
#define PETSCSINDY_H

#include <petsctao.h>
#include <petscts.h>
#include <petscdmda.h>

typedef struct _p_Basis* Basis;
typedef struct _p_SparseReg* SparseReg;

PETSC_EXTERN PetscErrorCode SINDyBasisCreate(PetscInt, PetscInt, Basis*);
PETSC_EXTERN PetscErrorCode SINDyBasisDestroy(Basis*);
PETSC_EXTERN PetscErrorCode SINDyBasisSetNormalizeColumns(Basis, PetscBool);
PETSC_EXTERN PetscErrorCode SINDyBasisSetCrossTermRange(Basis, PetscInt);
PETSC_EXTERN PetscErrorCode SINDyBasisSetFromOptions(Basis);
PETSC_EXTERN PetscErrorCode SINDyBasisCreateData(Basis, Vec*, PetscInt);
PETSC_EXTERN PetscErrorCode SINDyBasisDataGetSize(Basis, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode SINDyBasisPrint(Basis, PetscInt, Vec*);


PETSC_EXTERN PetscErrorCode SINDyFindSparseCoefficients(Basis, SparseReg, PetscInt, Vec*, PetscInt, Vec*);
PETSC_EXTERN PetscErrorCode SINDySparseLeastSquares(Mat, Vec, Mat, Vec);
PETSC_EXTERN PetscErrorCode SINDySparseRegCreate(SparseReg*);
PETSC_EXTERN PetscErrorCode SINDySparseRegSetThreshold(SparseReg, PetscReal);
PETSC_EXTERN PetscErrorCode SINDySparseRegSetMonitor(SparseReg, PetscBool);
PETSC_EXTERN PetscErrorCode SINDySparseRegSetFromOptions(SparseReg);
PETSC_EXTERN PetscErrorCode SINDySparseRegDestroy(SparseReg*);


typedef struct _p_Variable* Variable;

PETSC_EXTERN PetscErrorCode SINDyVariableCreate(const char*, Variable*);
PETSC_EXTERN PetscErrorCode SINDyVariableSetScalarData(Variable, PetscInt, PetscScalar*);
PETSC_EXTERN PetscErrorCode SINDyVariableSetVecData(Variable, PetscInt, Vec*, DM);
PETSC_EXTERN PetscErrorCode SINDyVariableDestroy(Variable*);

// PETSC_EXTERN PetscErrorCode SINDyVariableDifferentiate(Variable, PetscInt dim, PetscInt order, const char*, Variable*);

PETSC_EXTERN PetscErrorCode SINDyBasisAddVariables(Basis, PetscInt, Variable*);
PETSC_EXTERN PetscErrorCode SINDyBasisSetOutputVariable(Basis, Variable);

PETSC_EXTERN PetscErrorCode SINDyFindSparseCoefficientsVariable(Basis, SparseReg, PetscInt, Vec*);
PETSC_EXTERN PetscErrorCode SINDyBasisPrintVariable(Basis, PetscInt, Vec*);
PETSC_EXTERN PetscErrorCode SINDySequentialThresholdedLeastSquares(SparseReg, Mat, Vec, Mat, Vec);



#endif
