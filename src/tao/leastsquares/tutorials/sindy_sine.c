#include <petscts.h>
#include "sindy.h"

static char help[] = "Run SINDy on data generated from dx/dt = -sin(x).\n";

typedef struct {
  PetscInt  runs,steps,N,i;
  PetscReal dt;
  Vec       *all_x,*all_dx;
} Data;

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscErrorCode ierr;
  PetscReal      x;
  PetscInt       idx = 0;

  PetscFunctionBegin;
  ierr = VecGetValues(X, 1, &idx, &x);CHKERRQ(ierr);
  ierr = VecSet(F, -PetscSinReal(x));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat B, void* ctx)
{
  PetscErrorCode ierr;
  PetscReal      x;
  PetscInt       idx = 0;

  PetscFunctionBegin;
  ierr = VecGetValues(X, 1, &idx, &x);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 0, -PetscCosReal(x), INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DataInitialize(Data* data, Vec X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Data generation options","");CHKERRQ(ierr);
  {
    data->steps = 5000;
    ierr = PetscOptionsInt("-steps","how many timesteps to simulate in each run","",data->steps,&data->steps,NULL);CHKERRQ(ierr);
    data->dt = 0.001;
    ierr = PetscOptionsReal("-dt","timestep size","",data->dt,&data->dt,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  data->runs = 10;
  data->N = data->runs * data->steps;
  data->i = 0;

  /* Create Vecs to hold data. */
  ierr = VecDuplicateVecs(X, data->N, &data->all_x);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(X, data->N, &data->all_dx);CHKERRQ(ierr);
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

PetscErrorCode GetData(PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p)
{ 
  PetscErrorCode ierr;
  PetscInt       r;
  PetscReal      x0;
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;
  Data           data;

  PetscFunctionBegin;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &J);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&X);CHKERRQ(ierr);
  ierr = DataInitialize(&data, X);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(ts, TSRK5DP);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, data.steps);CHKERRQ(ierr);

  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, NULL);CHKERRQ(ierr);

  ierr = TSSetApplicationContext(ts, (void*)&data);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,DataPostStep);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);


  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Get x data from running TS. */
  for (r = 0; r < data.runs; r++) {
    x0 = -1.25 + r * 0.25;
    if (x0 >= -1e-3) {
      x0 = -1.25 + (r+1) * 0.25;
    }
    ierr = VecSet(X, x0);CHKERRQ(ierr);
    ierr = DataPostStep(ts);CHKERRQ(ierr);

    ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, data.dt);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts, (r+1)*(data.steps-1));CHKERRQ(ierr);
    ierr = TSSolve(ts, NULL);CHKERRQ(ierr);
  }

  ierr = DataComputeDerivative(&data);CHKERRQ(ierr);

  /* Write output parameters. */
  *N_p = data.N;
  *all_x_p = data.all_x;
  *all_dx_p = data.all_dx;

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char** argv)
{
  PetscErrorCode ierr;
  Basis          basis;
  SparseReg      sparse_reg;
  PetscInt       num_bases;
  PetscInt       n;
  Vec            *x,*dx;
  Vec            Xi;
  PetscMPIInt    size;

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
  ierr = GetData(&n, &x, &dx);CHKERRQ(ierr);

  /* Create 5th order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(5, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetNormalizeColumns(basis, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SINDyBasisSetFromOptions(basis);CHKERRQ(ierr);
  ierr = SINDyBasisCreateData(basis, x, n);CHKERRQ(ierr);

  ierr = SINDySparseRegCreate(&sparse_reg);CHKERRQ(ierr);
  ierr = SINDySparseRegSetThreshold(sparse_reg, 1e-1);CHKERRQ(ierr);
  ierr = SINDySparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SINDySparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

  /* Allocate solution vector */
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi);CHKERRQ(ierr);

  /* Run least squares */
  ierr = SINDyFindSparseCoefficients(basis, sparse_reg, n, dx, 1, &Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroyVecs(n, &x);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &dx);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SINDySparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
