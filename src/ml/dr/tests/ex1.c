static char help[] = "Tests basic creation and destruction of MLDR objects.\n\n";

#include <petscmldr.h>

int main(int argc,char **args)
{
  MLDR           mldr;
  PetscErrorCode ierr;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  PetscCall(MLDRCreate(PETSC_COMM_WORLD,&mldr));
  PetscCall(MLDRDestroy(&mldr));

  PetscCall(PetscFinalize());
  return 0;
}
