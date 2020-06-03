#include <petscts.h>
#include "sindy.h"

static char help[] = "Run SINDy on data generated from dx/dt = -sin(x).\n";

typedef struct {
  PetscInt  runs,steps,N,i;
  PetscReal dt;
  Vec       *all_x,*all_dx;
} Data;

typedef struct {
  Data      data;
  PetscReal force;
  PetscInt  N;
} AppCtx;

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx_v)
{
  PetscErrorCode ierr;
  const PetscScalar *x;
  PetscScalar       *f;
  AppCtx            *ctx = (AppCtx*)ctx_v;
  PetscInt          i,N = ctx->N;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  /* The general case. */
  for (i = 0; i < N; i++) {
    f[i] = (x[(i+1+N) % N] - x[(i-2+N) % N]) * x[(i-1+N) % N] - x[i];
  }
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);

  /* Add the forcing term. */
  ierr = VecShift(F, ctx->force);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat B, void* ctx_v)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *j;
  AppCtx            *ctx = (AppCtx*)ctx_v;
  PetscInt          i,N = ctx->N;

  PetscFunctionBegin;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = MatDenseGetArray(J, &j);CHKERRQ(ierr);

  /* Set four entries in each row. */
  for (i = 0; i < N; i++) {
    j[i*N + (i-2+N)%N] = -x[(i-1+N)%N];
    j[i*N + (i-1+N)%N] = x[(i+1+N)%N] - x[(i-2+N)%N];
    j[i*N + (i+N)%N]   = -1;
    j[i*N + (i+1+N)%N] = x[(i-1+N)%N];
  }

  ierr = MatDenseRestoreArray(J, &j);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DataInitialize(Data* data, Vec X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"data_","Data generation options","");CHKERRQ(ierr);
  {
    data->steps = 600;
    ierr = PetscOptionsInt("-steps","how many timesteps to simulate in each run","",data->steps,&data->steps,NULL);CHKERRQ(ierr);
    data->dt = 0.01;
    ierr = PetscOptionsReal("-dt","timestep size","",data->dt,&data->dt,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  data->runs = 1;
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
  PetscReal      t;
  AppCtx         *ctx;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&ctx);CHKERRQ(ierr);

  if (ctx->data.i == ctx->data.N) {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Cannot record more than %d vectors.",ctx->data.N);
  }
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = VecCopy(X, ctx->data.all_x[ctx->data.i]);CHKERRQ(ierr);
  ctx->data.i++;
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
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;
  AppCtx         ctx;
  Data           *data;

  PetscFunctionBegin;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"lorentz_","Lorentz96 options","");CHKERRQ(ierr);
  {
    ctx.force = 8;
    ierr = PetscOptionsReal("-force","forcing term","",ctx.force,&ctx.force,NULL);CHKERRQ(ierr);
    ctx.N = 38;
    ierr = PetscOptionsInt("-N","number of vector components","",ctx.N,&ctx.N,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  data = &ctx.data;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, ctx.N, ctx.N, NULL, &J);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ctx.N,&X);CHKERRQ(ierr);
  ierr = DataInitialize(data, X);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(ts, TSRK5DP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, data->dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, data->steps);CHKERRQ(ierr);

  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, &ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, &ctx);CHKERRQ(ierr);

  ierr = TSSetApplicationContext(ts, (void*)&ctx);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,DataPostStep);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Set initial condition. */
  ierr = VecSet(X, 1);CHKERRQ(ierr);
  ierr = VecSetValue(X,ctx.N/2,1.01,INSERT_VALUES);CHKERRQ(ierr);
  ierr = DataPostStep(ts);CHKERRQ(ierr);


  /* Get x data from running TS. */
  ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, data->dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, data->steps-1);CHKERRQ(ierr);
  ierr = TSSolve(ts, NULL);CHKERRQ(ierr);

  /* Get derivative data. */
  ierr = DataComputeDerivative(data);CHKERRQ(ierr);

  /* Write output parameters. */
  *N_p = data->N;
  *all_x_p = data->all_x;
  *all_dx_p = data->all_dx;

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
  PetscInt       n,dim;
  Vec            *x,*dx;
  Vec            *Xi,Xi0;
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
  printf("Generating data...\n");
  ierr = GetData(&n, &x, &dx);CHKERRQ(ierr);
  ierr = VecGetSize(x[0], &dim);CHKERRQ(ierr);

  /* Create 2nd order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(2, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetNormalizeColumns(basis, PETSC_FALSE);CHKERRQ(ierr);
  ierr = SINDyBasisSetCrossTermRange(basis, 2);CHKERRQ(ierr);
  ierr = SINDyBasisSetFromOptions(basis);CHKERRQ(ierr);
  ierr = SINDyBasisCreateData(basis, x, n);CHKERRQ(ierr);

  ierr = SINDySparseRegCreate(&sparse_reg);CHKERRQ(ierr);
  ierr = SINDySparseRegSetThreshold(sparse_reg, 1e-1);CHKERRQ(ierr);
  ierr = SINDySparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SINDySparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

  /* Allocate solution vector */
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi0);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(Xi0, dim, &Xi);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi0);CHKERRQ(ierr);

  /* Run least squares */
  printf("Running sparse least squares...\n");
  ierr = SINDyFindSparseCoefficients(basis, sparse_reg, n, dx, dim, Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroyVecs(n, &x);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &dx);CHKERRQ(ierr);
  ierr = VecDestroyVecs(dim, &Xi);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SINDySparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
