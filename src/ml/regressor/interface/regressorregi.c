#include <petsc/private/regressorimpl.h>

PETSC_EXTERN PetscErrorCode PetscRegressorCreate_Linear(PetscRegressor);

PetscErrorCode PetscRegressorRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscRegressorRegisterAllCalled) PetscFunctionReturn(0);
  PetscRegressorRegisterAllCalled = PETSC_TRUE;
  // TODO: Register all of the sub-types
  ierr = PetscRegressorRegister(PETSCREGRESSORLINEAR,PetscRegressorCreate_Linear);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
