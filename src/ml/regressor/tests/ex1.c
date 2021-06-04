static char help[] = "Tests basic creation and destruction of MLRegressor objects.\n\n";

#include <petscmlregressor.h>

int main(int argc,char **args)
{
  MLRegressor    mlregressor;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  ierr = MLRegressorCreate(PETSC_COMM_WORLD,&mlregressor);CHKERRQ(ierr);
  ierr = MLRegressorDestroy(&mlregressor);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
