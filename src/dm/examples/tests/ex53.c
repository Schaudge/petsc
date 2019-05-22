

static char help[] = "Tests FAIJ matrices.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode FillUpMatrix(DM,Mat);
PetscErrorCode FillUpMatrixSmall(DM,Mat);
PetscErrorCode FillUpMatrixBig(DM,Mat);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da;
  PetscInt       dof = 100,M = 100000, sw = 1;
  Vec            x,y;
  Mat            A;
  PetscBool      view = PETSC_FALSE;
  PetscInt       i,*diags;
  PetscLogStage  stage,stagesmall,stagebig,stagesetup,stagesmallsetup,stagebigsetup;
  PetscClassId   classid;
  PetscLogEvent  setup;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-sw",&sw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-view",&view,NULL);CHKERRQ(ierr);

  ierr = PetscClassIdRegister("Setup",&classid);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Setup",classid,&setup);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Large block diagonal with efficient indexing. Setup",&stagesetup);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stagesetup);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(setup,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,M,dof,sw,NULL,&da);CHKERRQ(ierr);
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
  ierr = PetscLogEventEnd(setup,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Large block diagonal with efficient indexing",&stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  for (i=0; i<10; i++) {
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  if (view) {
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* ------------------------------------------------------------------------*/
  ierr = PetscLogStageRegister("Many small matrices. Setup",&stagesmallsetup);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stagesmallsetup);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(setup,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,M,1,sw,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATSEQAIJ);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = FillUpMatrixSmall(da,A);CHKERRQ(ierr);
  if (view) {
    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&y);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(setup,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Many small matrices",&stagesmall);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stagesmall);CHKERRQ(ierr);
  for (i=0; i<10*dof; i++) {
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  if (view) {
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* ------------------------------------------------------------------------*/
  ierr = PetscLogStageRegister("Large matrix with in-efficient indexing. Setup",&stagebigsetup);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stagebigsetup);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(setup,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,M,dof,sw,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATSEQAIJ);CHKERRQ(ierr);
  /* indicate the nonzero pattern of each block of the matrix to be the diagonal */
  ierr = PetscMalloc1(2*dof+1,&diags);CHKERRQ(ierr);
  for (i=0; i<dof+1; i++) diags[i] = dof+1+i;
  for (i=0; i<dof; i++) diags[dof+1+i] = i;
  ierr = DMDASetBlockFillsSparse(da,diags,diags);CHKERRQ(ierr);
  ierr = PetscFree(diags);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = FillUpMatrixBig(da,A);CHKERRQ(ierr);
  if (view) {
    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&y);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(setup,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Large matrix with in-efficient indexing",&stagebig);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stagebig);CHKERRQ(ierr);
  for (i=0; i<10; i++) {
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
  }
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
  PetscInt       mx,xs,xm,dof,i,j,col[3];
  PetscReal      h;
  PetscScalar    *v;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  h    = 1.0/(mx-1);

  ierr = PetscMalloc1(3*dof,&v);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    if (i==0 || i==mx-1) {
      for (j=0; j<dof; j++) v[j] = 2.0/h;
      ierr = MatSetValuesBlocked(A,1,&i,1,&i,v,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      for (j=0; j<dof; j++) v[j] = -1.0/h;
      col[0] = i-1;
      for (j=0; j<dof; j++) v[dof+j] = 2.0/h;
      col[1] = i;
      for (j=0; j<dof; j++) v[2*dof+j] = -1.0/h;
      col[2] = i+1;
      ierr  = MatSetValuesBlocked(A,1,&i,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FillUpMatrixSmall(DM da,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       mx,xs,xm,i,col[3];
  PetscReal      h;
  PetscScalar    v[3];

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  h    = 1.0/(mx-1);

  for (i=xs; i<xs+xm; i++) {
    if (i==0 || i==mx-1) {
      v[0] = 2.0/h;
      ierr = MatSetValuesBlocked(A,1,&i,1,&i,v,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      v[0] = -1.0/h;
      col[0] = i-1;
      v[1] = 2.0/h;
      col[1] = i;
      v[2] = -1.0/h;
      col[2] = i+1;
      ierr  = MatSetValuesBlocked(A,1,&i,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FillUpMatrixBig(DM da,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       mx,xs,xm,i,col[3],dof,j,row;
  PetscReal      h;
  PetscScalar    v[3];

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  h    = 1.0/(mx-1);

  for (i=xs; i<xs+xm; i++) {
    if (i==0 || i==mx-1) {
      v[0] = 2.0/h;
      for (j=0; j<dof; j++) {
        row = i*dof + j;
        ierr = MatSetValues(A,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    } else {
      v[0] = -1.0/h;
      v[1] = 2.0/h;
      v[2] = -1.0/h;
      for (j=0; j<dof; j++) {
        row = i*dof + j;
        col[0] = (i-1)*dof + j;
        col[1] = i*dof + j;
        col[2] = (i+1)*dof + j;
        ierr  = MatSetValues(A,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
