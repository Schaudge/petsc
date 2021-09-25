#ifndef PETSCMLREGRESSOR_H
#define PETSCMLREGRESSOR_H

#include <petsctao.h>

typedef struct _p_MLRegressor* MLRegressor;

/*J
    MLRegressorType - String with the name of a PETSc regression method.

   Level: beginner

.seealso: MLRegressorSetType(), MLRegressor, MLRegressorRegister(), MLRegressorCreate(), MLRegressorSetFromOptions()
J*/
typedef const char *MLRegressorType;
#define MLREGRESSORLINEAR "linear"

PETSC_EXTERN PetscFunctionList MLRegressorList;
PETSC_EXTERN PetscClassId MLREGRESSOR_CLASSID;

PETSC_EXTERN PetscErrorCode MLRegressorInitializePackage(void);
PETSC_EXTERN PetscErrorCode MLRegressorFinalizePackage(void);
PETSC_EXTERN PetscErrorCode MLRegressorRegister(const char[],PetscErrorCode (*)(MLRegressor));
PETSC_EXTERN PetscErrorCode MLRegressorCreate(MPI_Comm,MLRegressor*);
PETSC_EXTERN PetscErrorCode MLRegressorReset(MLRegressor);
PETSC_EXTERN PetscErrorCode MLRegressorDestroy(MLRegressor*);
PETSC_EXTERN PetscErrorCode MLRegressorSetType(MLRegressor,MLRegressorType);
PETSC_EXTERN PetscErrorCode MLRegressorSetUp(MLRegressor);
PETSC_EXTERN PetscErrorCode MLRegressorSetFromOptions(MLRegressor);
PETSC_EXTERN PetscErrorCode MLRegressorView(MLRegressor,PetscViewer);
PETSC_EXTERN PetscErrorCode MLRegressorSetTraining(MLRegressor,Mat,Vec);
// PETSC_EXTERN PetscErrorCode MLRegressorFit(MLRegressor);
// TODO: Decide if MLRegressorFit() take only an MLRegressor, or the training data and label (as in Scikit-learn), below.
PETSC_EXTERN PetscErrorCode MLRegressorFit(MLRegressor,Mat,Vec);
PETSC_EXTERN PetscErrorCode MLRegressorPredict(MLRegressor,Mat,Vec);
PETSC_EXTERN PetscErrorCode MLRegressorLinearGetKSP(MLRegressor,KSP*);
PETSC_EXTERN PetscErrorCode MLRegressorLinearGetCoefficients(MLRegressor,Vec*);
PETSC_EXTERN PetscErrorCode MLRegressorLinearGetIntercept(MLRegressor,PetscScalar*);
#endif
