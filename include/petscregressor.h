#ifndef PETSCREGRESSOR_H
#define PETSCREGRESSOR_H

#include <petsctao.h>

typedef struct _p_PetscRegressor* PetscRegressor;

/*J
    PetscRegressorType - String with the name of a PETSc regression method.

   Level: beginner

.seealso: PetscRegressorSetType(), PetscRegressor, PetscRegressorRegister(), PetscRegressorCreate(), PetscRegressorSetFromOptions()
J*/
typedef const char *PetscRegressorType;
#define PETSCREGRESSORLINEAR "linear"

PETSC_EXTERN PetscFunctionList PetscRegressorList;
PETSC_EXTERN PetscClassId PETSCREGRESSOR_CLASSID;

PETSC_EXTERN PetscErrorCode PetscRegressorInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscRegressorFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscRegressorRegister(const char[],PetscErrorCode (*)(PetscRegressor));
PETSC_EXTERN PetscErrorCode PetscRegressorCreate(MPI_Comm,PetscRegressor*);
PETSC_EXTERN PetscErrorCode PetscRegressorReset(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorDestroy(PetscRegressor*);
PETSC_EXTERN PetscErrorCode PetscRegressorSetType(PetscRegressor,PetscRegressorType);
PETSC_EXTERN PetscErrorCode PetscRegressorSetUp(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorView(PetscRegressor,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscRegressorSetTraining(PetscRegressor,Mat,Vec);
// PETSC_EXTERN PetscErrorCode PetscRegressorFit(PetscRegressor);
// TODO: Decide if PetscRegressorFit() take only an PetscRegressor, or the training data and label (as in Scikit-learn), below.
PETSC_EXTERN PetscErrorCode PetscRegressorFit(PetscRegressor,Mat,Vec);
PETSC_EXTERN PetscErrorCode PetscRegressorPredict(PetscRegressor,Mat,Vec);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearSetFitIntercept(PetscRegressor,PetscBool);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearSetUseKSP(PetscRegressor,PetscBool);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetKSP(PetscRegressor,KSP*);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetTao(PetscRegressor,Tao*);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetCoefficients(PetscRegressor,Vec*);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetIntercept(PetscRegressor,PetscScalar*);
#endif
