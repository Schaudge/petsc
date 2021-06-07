#include <petsc/private/mlregressorimpl.h>

PETSC_EXTERN PetscErrorCode MLRegressorCreate_Linear(MLRegressor);

PetscErrorCode MLRegressorRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MLRegressorRegisterAllCalled) PetscFunctionReturn(0);
  MLRegressorRegisterAllCalled = PETSC_TRUE;
  // TODO: Register all of the sub-types
  ierr = MLRegressorRegister(MLREGRESSORLINEAR,MLRegressorCreate_Linear);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
