#include <petsc/private/petscmldrimpl.h>

PetscErrorCode MLDRRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MLDRRegisterAllCalled) PetscFunctionReturn(0);
  MLDRRegisterAllCalled = PETSC_TRUE;
  ierr = MLDRRegister(MLDRPCA,MLDRCreate_PCA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


