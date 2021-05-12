#include <petscfe.h>

static char help[] = "Test PetscSpace_Koszul";

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscSpace     domain,koszul;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,PETSC_NULL,help); if (ierr) return ierr;

  /* Create a polynomial space to serve as domain of koszul map */
  ierr = PetscSpaceCreate(PETSC_COMM_WORLD,&domain);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(domain,PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(domain,1);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(domain,1);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(domain,2,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(domain);CHKERRQ(ierr);

  ierr = PetscSpaceView(domain,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Create a koszul space */
  ierr = PetscSpaceCreateKoszul(&koszul,&domain);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*TEST
  test:
    suffix: 0
TEST*/
