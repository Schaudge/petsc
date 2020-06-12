#include <petscts.h>
#include "sindy.h"

static char help[] = "Run SINDy on data generated from du/dt = [-sin(u1), cos(u2)].\n";

typedef struct {
  PetscScalar u,v;
} PointValue;

typedef struct {
  PetscInt    runs,steps,N,i;
  PetscReal   dt;
  Vec         *all_x,*all_dx;
  PetscReal   *all_t;
  DM          da;
  PetscScalar t0;
  PetscScalar dx,dy;
  PetscScalar xmin,xmax,ymin,ymax;
} Data;

PetscErrorCode InitialCondition(Data *user, Vec X)
{
  PetscErrorCode ierr;
  DM             cda;
  DMDACoor2d     **coors;
  PointValue     **p;
  Vec            gc;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym;
  PetscScalar    xi,yi;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(user->da,&gc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,X,&p);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  /* Initial condition is the same as coordinates. */
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      xi = coors[j][i].x; yi = coors[j][i].y;
      p[j][i].u = xi;
      p[j][i].v = yi;
    }
  }

  ierr = DMDAVecRestoreArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,X,&p);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscErrorCode    ierr;
  Data              *user=(Data*)ctx;
  DM                cda;
  const DMDACoor2d  **coors;
  const PointValue  **p;
  PointValue        **f;
  PetscInt          i,j;
  PetscInt          xs,ys,xm,ym;
  Vec               localX,gc;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  ierr = DMGetLocalVector(user->da,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(user->da,&gc);CHKERRQ(ierr);

  ierr = DMDAVecGetArrayRead(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(user->da,localX,&p);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,F,&f);CHKERRQ(ierr);
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      f[j][i].u = -PetscSinReal(p[j][i].u);
      f[j][i].v =  -PetscTanReal(p[j][i].v);
      // f[j][i].v =  PetscCosReal(p[j][i].v);
    }
  }
  ierr = DMDAVecRestoreArrayRead(user->da,localX,&p);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda,gc,&coors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat Jpre, void* ctx)
{

  PetscErrorCode    ierr;
  Data              *user=(Data*)ctx;
  DM                cda;
  DMDACoor2d        **coors;
  PetscInt          i,j;
  PetscInt          xs,ys,xm,ym;
  Vec               localX,gc;
  PetscScalar       val[1];
  MatStencil        row,col[1];
  const PointValue  **p;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  ierr = DMGetLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(user->da,&gc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(user->da,localX,&p);CHKERRQ(ierr);
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      PetscInt nc = 0;
      row.i = i; row.j = j; row.c = 0;
      // col[nc].i = i-1; col[nc].j = j;   col[nc].c = 0; val[nc++] = 0;
      // col[nc].i = i+1; col[nc].j = j;   col[nc].c = 0; val[nc++] = 0;
      // col[nc].i = i;   col[nc].j = j-1; col[nc].c = 0; val[nc++] = 0;
      // col[nc].i = i;   col[nc].j = j+1; col[nc].c = 0; val[nc++] = 0;
      col[nc].i = i;   col[nc].j = j;   col[nc].c = 0; val[nc++] = -PetscCosReal(p[j][i].u);
      ierr = MatSetValuesStencil(Jpre,1,&row,nc,col,val,INSERT_VALUES);CHKERRQ(ierr);

      nc = 0;
      row.i = i; row.j = j; row.c = 1;
      // col[nc].i = i-1; col[nc].j = j;   col[nc].c = 1; val[nc++] = 0;
      // col[nc].i = i+1; col[nc].j = j;   col[nc].c = 1; val[nc++] = 0;
      // col[nc].i = i;   col[nc].j = j-1; col[nc].c = 1; val[nc++] = 0;
      // col[nc].i = i;   col[nc].j = j+1; col[nc].c = 1; val[nc++] = 0;
      // col[nc].i = i;   col[nc].j = j;   col[nc].c = 1; val[nc++] = -PetscSinReal(p[j][i].v);
      col[nc].i = i;   col[nc].j = j;   col[nc].c = 1; val[nc++] = -PetscPowRealInt(1.0/PetscCosReal(p[j][i].v), 2);
      ierr = MatSetValuesStencil(Jpre,1,&row,nc,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMDAVecRestoreArrayRead(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(user->da,localX,&p);CHKERRQ(ierr);

  ierr =  MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataInitializeParams(Data* data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  data->steps  = 10;
  data->dt     = 0.001;
  data->t0     = 0;
  data->xmin   = -1.25; data->xmax = 1.25;
  data->ymin   = -1.25; data->ymax = 1.25;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Data generation options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-steps","how many timesteps to simulate in each run","",data->steps,&data->steps,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dt","timestep size","",data->dt,&data->dt,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-xmin","coordinates params","",data->xmin,&data->xmin,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-xmax","coordinates params","",data->xmax,&data->xmax,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-ymin","coordinates params","",data->ymin,&data->ymin,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-ymax","coordinates params","",data->ymax,&data->ymax,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-t0","initial time","",data->t0,&data->t0,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  data->runs = 1;
  data->N = data->runs * data->steps;
  data->i = 0;

  PetscFunctionReturn(0);
}

PetscErrorCode DataSetUp(Data* data, Vec X)
{
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  /* Create Vecs to hold data. */
  ierr = VecDuplicateVecs(X, data->N, &data->all_x);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(X, data->N, &data->all_dx);CHKERRQ(ierr);
  ierr = PetscMalloc1(data->N, &data->all_t);

  /* Calculate grid spacing. */
  ierr = DMDAGetInfo(data->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  data->dx = (data->xmax - data->xmin)/(M-1);
  data->dy = (data->ymax - data->ymin)/(N-1);

  PetscFunctionReturn(0);
}

PetscErrorCode DataPostStep(TS ts)
{
  PetscErrorCode ierr;
  Vec            X;
  Data           *data;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&data);CHKERRQ(ierr);
  if (data->i == data->N) {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Cannot record more than %d vectors.",data->N);
  }
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = VecCopy(X, data->all_x[data->i]);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &data->all_t[data->i]);CHKERRQ(ierr);

  data->i++;
  PetscFunctionReturn(0);
}

PetscErrorCode DataComputeDerivative(Data* data)
{
  PetscErrorCode ierr;
  PetscInt       i,t,r;

  /* Get derivate data using fourth-order central difference. */
  PetscFunctionBegin;
  i = 0;
  for (r = 0; r < data->runs; r++) {
    for (t = 0; t < data->steps; t++) {
      ierr = VecSet(data->all_dx[i], 0);CHKERRQ(ierr);
      if (t >= 2 && t < data->steps - 2) {
        ierr = VecAXPY(data->all_dx[i], -1.0, data->all_x[i+2]);CHKERRQ(ierr);
        ierr = VecAXPY(data->all_dx[i],  8.0, data->all_x[i+1]);CHKERRQ(ierr);
        ierr = VecAXPY(data->all_dx[i], -8.0, data->all_x[i-1]);CHKERRQ(ierr);
        ierr = VecAXPY(data->all_dx[i],  1.0, data->all_x[i-2]);CHKERRQ(ierr);
        ierr = VecScale(data->all_dx[i], 1.0/(12.0*data->dt));CHKERRQ(ierr);
      }
      i++;
    }
    /* Set boundary values to 0. */
    ierr = VecSet(data->all_x[i-1], 0);CHKERRQ(ierr);
    ierr = VecSet(data->all_x[i-2], 0);CHKERRQ(ierr);
    ierr = VecSet(data->all_x[i-data->steps], 0);CHKERRQ(ierr);
    ierr = VecSet(data->all_x[i-data->steps+1], 0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode GetData(PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p, PetscReal** all_t_p, DM* dm_p)
{ 
  PetscErrorCode ierr;
  PetscInt       r;
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;
  Data           data;

  PetscFunctionBegin;
  /* Create a 2D DA with dof = 2 */
  ierr = DataInitializeParams(&data);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&data.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(data.da);CHKERRQ(ierr);
  ierr = DMSetUp(data.da);CHKERRQ(ierr);
  /* Set x and y coordinates */
  ierr = DMDASetUniformCoordinates(data.da,data.xmin,data.xmax,data.ymin,data.ymax,0.0,1.0);CHKERRQ(ierr);

  /* Get global vector x from DM  */
  ierr = DMCreateGlobalVector(data.da,&X);CHKERRQ(ierr);
  ierr = DMSetMatType(data.da,MATAIJ);CHKERRQ(ierr);
  ierr = DataSetUp(&data, X);CHKERRQ(ierr);

  // printf("dm:\n");
  // ierr = DMView(data.da, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);


  // PetscInt numCellsX;
  // PetscInt numCellsY;
  // PetscInt numCellsZ;
  // PetscInt numCells;

  // ierr = DMDAGetNumCells(data.da, &numCellsX, &numCellsY, &numCellsZ, &numCells);CHKERRQ(ierr);
  // printf("%15s: %d\n", "numCellsX",numCellsX);
  // printf("%15s: %d\n", "numCellsY",numCellsY);
  // printf("%15s: %d\n", "numCellsZ",numCellsZ);
  // printf("%15s: %d\n", "numCells",numCells);
  // exit(1);

  /* Get Jacobian matrix structure from the da */
  ierr = DMCreateMatrix(data.da,&J);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(ts, TSRK5DP);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, data.steps);CHKERRQ(ierr);

  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, (void*)&data);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, (void*)&data);CHKERRQ(ierr);

  ierr = TSSetApplicationContext(ts, (void*)&data);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,DataPostStep);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Get x data from running TS. */
  for (r = 0; r < data.runs; r++) {
    ierr = InitialCondition(&data, X);CHKERRQ(ierr);
    ierr = DataPostStep(ts);CHKERRQ(ierr);

    ierr = TSSetTime(ts, data.t0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, data.dt);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts, (r+1)*(data.steps-1));CHKERRQ(ierr);
    ierr = TSSolve(ts, NULL);CHKERRQ(ierr);
  }

  if (data.i != data.N) {
    printf("Uh oh: recorded %d data points but expected %d data points\n", data.i, data.N);
  }

  ierr = DataComputeDerivative(&data);CHKERRQ(ierr);

  /* Write output parameters. */
  *N_p = data.N;
  *all_x_p = data.all_x;
  *all_dx_p = data.all_dx;
  *all_t_p = data.all_t;
  *dm_p = data.da;

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char** argv) {
  PetscErrorCode ierr;
  Basis          basis;
  SparseReg      sparse_reg;
  PetscInt       num_bases;
  PetscInt       n;
  Vec            *x,*dx;
  Vec            Xi[2];
  PetscMPIInt    size;
  PetscReal      *t;
  DM             dm;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if(size != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only");

  /*
    0. Get data X and dXdt, which will be the input data. Or generate dXdt from X using finite difference or TV regularized differentiation.

    1. Generate the matrix Theta using selected basis functions.

    2. Do a sparse linear regression to get Xi ~ Theta \ dXdt.

    3. Compute the approximation of x using dxdt = Theta(x^T) Xi.
  */

  /* Generate data. */
  ierr = GetData(&n, &x, &dx, &t, &dm);CHKERRQ(ierr);

  Variable v_x,v_dx,v_t;
  ierr = SINDyVariableCreate("x", &v_x);CHKERRQ(ierr);
  ierr = SINDyVariableSetVecData(v_x, n, x, dm);CHKERRQ(ierr);
  ierr = SINDyVariableCreate("dx/dt", &v_dx);CHKERRQ(ierr);
  ierr = SINDyVariableSetVecData(v_dx, n, dx, dm);CHKERRQ(ierr);
  ierr = SINDyVariableCreate("t", &v_t);CHKERRQ(ierr);
  ierr = SINDyVariableSetScalarData(v_t, n, t);CHKERRQ(ierr);

  /* Create 5th order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(3, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetNormalizeColumns(basis, PETSC_FALSE);CHKERRQ(ierr);
  ierr = SINDyBasisSetCrossTermRange(basis, 0);CHKERRQ(ierr);
  ierr = SINDyBasisSetFromOptions(basis);CHKERRQ(ierr);

  // Variable vars[] = {v_x, v_t};
  Variable vars[] = {v_x, v_dx};
  ierr = SINDyBasisSetOutputVariable(basis, v_dx);CHKERRQ(ierr);
  // ierr = SINDyBasisAddVariables(basis, 2, vars);CHKERRQ(ierr);
  ierr = SINDyBasisAddVariables(basis, 1, &v_dx);CHKERRQ(ierr);
  // ierr = SINDyBasisCreateData(basis, x, n);CHKERRQ(ierr);

  ierr = SINDySparseRegCreate(&sparse_reg);CHKERRQ(ierr);
  ierr = SINDySparseRegSetThreshold(sparse_reg, 5e-3);CHKERRQ(ierr);
  ierr = SINDySparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SINDySparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

  /* Allocate solution vectors */
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi[0]);CHKERRQ(ierr);
  ierr = VecDuplicate(Xi[0], &Xi[1]);CHKERRQ(ierr);

  /* Run least squares */
  ierr = SINDyFindSparseCoefficientsVariable(basis, sparse_reg, 2, Xi);CHKERRQ(ierr);
  // ierr = SINDyFindSparseCoefficients(basis, sparse_reg, n, dx, 2, Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroyVecs(n, &x);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &dx);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[1]);CHKERRQ(ierr);
  ierr = PetscFree(t);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SINDySparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = SINDyVariableDestroy(&v_x);CHKERRQ(ierr);
  ierr = SINDyVariableDestroy(&v_dx);CHKERRQ(ierr);
  ierr = SINDyVariableDestroy(&v_t);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
