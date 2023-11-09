static char help[] = "Tests basic creation and destruction of MLDR objects.\n\n";

#include <petscmldr.h>

int main(int argc,char **args)
{
  MLDR           mldr;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  ierr = MLDRCreate(PETSC_COMM_WORLD,&mldr);CHKERRQ(ierr);
  ierr = MLDRDestroy(&mldr);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
