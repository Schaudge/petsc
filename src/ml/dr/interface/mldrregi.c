#include <petsc/private/mldrimpl.h>

PetscErrorCode MLDRRegisterAll(void)
{
  //PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MLDRRegisterAllCalled) PetscFunctionReturn(0);
  MLDRRegisterAllCalled = PETSC_TRUE;
  // TODO: Register all of the sub-types
  //ierr = MLDRRegister(MLDRPCA,MLDRCreate_PCA);CHKERRQ(ierr);  // Uncomment after MLDRCreate_PCA() exists!
  PetscFunctionReturn(0);
}
