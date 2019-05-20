

static char help[] = "Tests FAIJ matrices.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode FillUpMatrix(DM,Mat);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da;
  PetscInt       dof = 2;
  Vec            x,y;
  Mat            A;
  PetscBool      view = PETSC_FALSE;
  PetscLogStage  stage;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-view",&view,NULL);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,7,dof,1,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATSEQFAIJ);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = FillUpMatrix(da,A);CHKERRQ(ierr);
  if (view) {
    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&y);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Large block diagonal with efficient indexing",&stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  if (view) {
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode FillUpMatrix(DM da,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       mx,xs,xm,bs,i,j,col[3];
  PetscReal      h;
  PetscScalar    *v;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,&bs,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  h    = 1.0/(mx-1);

  ierr = PetscMalloc1(3*bs,&v);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    if (i==0 || i==mx-1) {
      for (j=0; j<bs; j++) v[j] = 2.0/h;
      ierr = MatSetValuesBlocked(A,1,&i,1,&i,v,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      for (j=0; j<bs; j++) v[j] = -1.0/h;
      col[0] = i-1;
      for (j=0; j<bs; j++) v[bs+j] = 2.0/h;
      col[1] = i;
      for (j=0; j<bs; j++) v[2*bs+j] = -1.0/h;
      col[2] = i+1;
      ierr  = MatSetValuesBlocked(A,1,&i,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
