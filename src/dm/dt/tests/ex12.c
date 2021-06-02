#include <petscfe.h>

static char help[] = "Test PetscSpace_Koszul";

int main(int argc,char ** argv)
{
  PetscSpace     domain,koszul;
  PetscReal*     B_d, *B_k, points[4]={0,0,1,1};
  PetscInt dim_d,dim_k,npoints=2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsBegin (
    PETSC_COMM_WORLD,"","PetscSpaceKoszul Tests","PetscSpace"
    );CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (ierr) return ierr;

  /* Create a polynomial space to serve as the domain for our koszul space. */
  ierr = PetscSpaceCreate(PETSC_COMM_WORLD,&domain);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(domain,PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(domain,2);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(domain,1);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(domain,2,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(domain);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(domain,&dim_d);CHKERRQ(ierr);

  ierr = PetscCalloc1(dim_d*npoints,&B_d);CHKERRQ(ierr);
  ierr = PetscSpaceEvaluate(domain,npoints,points,B_d,NULL,NULL);CHKERRQ(ierr);

  ierr = PetscSpaceView(domain,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscRealView(dim_d*npoints,B_d,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Create a koszul space */
  /*
  ierr = PetscSpaceCreate(PETSC_COMM_WORLD,&koszul);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(koszul,PETSCSPACEKOSZUL);CHKERRQ(ierr);
  ierr = PetscSpaceKoszulSetDomain(koszul,domain);CHKERRQ(ierr);
  */

  ierr = PetscSpaceCreateKoszul(&domain,1,2,&koszul);CHKERRQ(ierr);

  ierr = PetscSpaceGetDimension(koszul,&dim_k);CHKERRQ(ierr);
  ierr = PetscCalloc1(dim_k*npoints,&B_k);CHKERRQ(ierr);

  ierr = PetscSpaceEvaluate(koszul,npoints,points,B_k,NULL,NULL);CHKERRQ(ierr);


  ierr = PetscSpaceView(koszul,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscRealView(dim_k*npoints,B_k,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



  PetscFree(B_d);
  PetscFree(B_k);
  ierr = PetscSpaceDestroy(&koszul);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&domain);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*TEST
  test:
    suffix: 0
TEST*/
