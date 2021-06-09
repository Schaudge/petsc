#include <petscfe.h>
#include <petscmath.h>

static char help[] = "Test PetscSpace_Koszul";

static PetscErrorCode PetscAssert(PetscBool b){
    return !b;
}
static PetscBool isEqualTol(PetscReal a, PetscReal b, PetscReal tol)
{
    return PetscAbs(b-a) <= tol;
}


int main(int argc,char ** argv)
{
  PetscSpace     domain,koszul,koszul2;
  PetscReal      * B_d,*B_k,*B_kk,points[4]={0,0,1,1};
  PetscInt       dim_d,dim_k,dim_kk,npoints=2,i;
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
  ierr = PetscCalloc1(dim_k*npoints*2,&B_k);CHKERRQ(ierr);

  ierr = PetscSpaceEvaluate(koszul,npoints,points,B_k,NULL,NULL);CHKERRQ(ierr);

  ierr = PetscSpaceView(koszul,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscRealView(dim_k*npoints,B_k,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscSpaceCreateKoszul(&koszul,0,1,&koszul2);CHKERRQ(ierr);

  ierr = PetscSpaceGetDimension(koszul2,&dim_kk);CHKERRQ(ierr);
  ierr = PetscCalloc1(dim_kk*npoints,&B_kk);CHKERRQ(ierr);

  ierr = PetscSpaceEvaluate(koszul2,npoints,points,B_kk,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSpaceView(koszul2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscRealView(dim_kk*npoints,B_kk,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  for (i = 0; i < dim_kk*npoints; ++i){
      ierr = PetscAssert(isEqualTol(B_kk[i],0,PETSC_MACHINE_EPSILON));
  }

  PetscFree(B_kk);
  PetscFree(B_k);
  PetscFree(B_d);
  ierr = PetscSpaceDestroy(&koszul2);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&koszul);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&domain);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*TEST
  test:
    suffix: 0
TEST*/
