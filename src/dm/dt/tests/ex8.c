static char help[] = "Tests PetscSpace_Sum.\n\n";

#include <petscfe.h>
int main(int argc,char **argv)
{
  PetscSpace P,S1,S2,*SpArr,P2,P3;
  PetscInt Ns = 2;
  PetscInt Nc = 1;
  PetscInt Nv = 1;
  PetscInt order = 2;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscSpaceCreate(PETSC_COMM_WORLD, &S1); CHKERRQ(ierr);
  ierr = PetscSpaceSetType(S1, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(S1, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(S1,Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(S1,order,order);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(S1);CHKERRQ(ierr);

  ierr = PetscSpaceCreate(PETSC_COMM_WORLD, &S2); CHKERRQ(ierr);
  ierr = PetscSpaceSetType(S2, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(S2, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(S2,Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(S2,order,order);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(S2);CHKERRQ(ierr);


  ierr = PetscSpaceCreate(PETSC_COMM_WORLD, &P); CHKERRQ(ierr);
  ierr = PetscSpaceSetType(P, PETSCSPACESUM); CHKERRQ(ierr);
  ierr = PetscSpaceSumSetNumSubspaces(P, Ns); CHKERRQ(ierr);
  ierr = PetscSpaceSumSetSubspace(P, 0, S1);CHKERRQ(ierr);
  ierr = PetscSpaceSumSetSubspace(P, 1, S2);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P,Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, order, order);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  
  ierr = PetscCalloc1(2, &SpArr);CHKERRQ(ierr);
  SpArr[0] = S1;
  SpArr[1] = S2;
  ierr = PetscSpaceCreateSum(Ns,SpArr,PETSC_TRUE,&P2);CHKERRQ(ierr);

  ierr = PetscSpaceView(P, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSpaceView(P2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
 
  PetscSpaceDestroy(&P2);
  PetscSpaceDestroy(&SpArr[1]);
  PetscSpaceDestroy(&SpArr[0]);
  PetscFree(SpArr); 
  PetscSpaceDestroy(&P);
  PetscSpaceDestroy(&S2);
  PetscSpaceDestroy(&S1);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    args: 
TEST*/
