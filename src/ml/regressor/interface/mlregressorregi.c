#include <petsc/private/mlregressorimpl.h>

PetscErrorCode MLRegressorRegisterAll(void)
{
  //PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MLRegressorRegisterAllCalled) PetscFunctionReturn(0);
  MLRegressorRegisterAllCalled = PETSC_TRUE;
  // TODO: Register all of the sub-types
  //ierr = MLRegressorRegister(MLREGRESSORLINEAR,MLRegressorCreate_Linear);CHKERRQ(ierr);  // Uncomment after MLRegressorCreate_Linear() exists!
  PetscFunctionReturn(0);
}
