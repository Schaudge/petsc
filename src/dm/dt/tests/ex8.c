static char help[] = "Tests PetscSpace_Sum.\n\n";

#include <petscfe.h>
int main(int argc,char **argv)
{
  PetscSpace P;
  PetscInt Ns = 2;
  PetscInt Nc = 1;
  PetscInt Nv = 1;
  PetscInt order = 2;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscSpaceCreate(PETSC_COMM_WORLD, &P); CHKERRQ(ierr);
  ierr = PetscSpaceSetType(P, PETSCSPACESUM); CHKERRQ(ierr);
  ierr = PetscSpaceSumSetNumSubspaces(P, Ns); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P,Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, order, order);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);

  ierr = PetscSpaceView(P, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  PetscSpaceDestroy(&P);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    args: -malloc_dump
TEST*/
