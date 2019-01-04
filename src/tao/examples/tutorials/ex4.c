static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>

int main (int argc, char** argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args:

TEST*/
