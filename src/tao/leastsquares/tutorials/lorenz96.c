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

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Data generation options","");CHKERRQ(ierr);
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

static PetscInt get_poly_index(PetscInt n, PetscInt j, PetscInt k)
{
  PetscInt tmp;
  if (k < j) {
    tmp = k;
    k = j;
    j = tmp;
  }
  return j*(n+1) - ((j-1)*j)/2 + k - j;
}

PetscErrorCode GetExactCoefficients(PetscInt cross_term_range, AppCtx* ctx, Vec** Xi)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,l,n;
  PetscInt       N = ctx->N;
  PetscInt       idx[4] = {0, 0, 0, 0};
  PetscReal      val[4] = {ctx->force, -1, 1, -1};
  Vec            tmp;

  PetscFunctionBegin;

  if (cross_term_range == -1) {
    ierr = VecCreateSeq(PETSC_COMM_SELF, SINDyCountBases(N, 2, 0), &tmp);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(tmp, N, Xi);CHKERRQ(ierr);
    for (i = 0; i < N; i++) {
      ierr = VecZeroEntries((*Xi)[i]);CHKERRQ(ierr);
      // -1 at xi, +1 at xj*xk, -1 at xl*xj.
      idx[1] = i;
      j = (i-1+N) % N;
      k = (i+1+N) % N;
      l = (i-2+N) % N;
      idx[2] = get_poly_index(N, j+1, k+1);
      idx[3] = get_poly_index(N, j+1, l+1);
      ierr = VecSetValues((*Xi)[i], 4, idx, val, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin((*Xi)[i]);CHKERRQ(ierr);
      ierr = VecAssemblyEnd((*Xi)[i]);CHKERRQ(ierr);
    }
  } else {
    n = 2* cross_term_range + 1;
    ierr = VecCreateSeq(PETSC_COMM_SELF, SINDyCountBases(n, 2, 0), &tmp);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(tmp, N, Xi);CHKERRQ(ierr);

    j = cross_term_range - 1;
    k = cross_term_range + 1;
    l = cross_term_range - 2;
    idx[1] = 1 + cross_term_range;
    idx[2] = get_poly_index(n, j+1, k+1);
    idx[3] = get_poly_index(n, j+1, l+1);
    for (i = 0; i < N; i++) {
      ierr = VecZeroEntries((*Xi)[i]);CHKERRQ(ierr);
      // -1 at xi, +1 at xj*xk, -1 at xl*xj.
      ierr = VecSetValues((*Xi)[i], 4, idx, val, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin((*Xi)[i]);CHKERRQ(ierr);
      ierr = VecAssemblyEnd((*Xi)[i]);CHKERRQ(ierr);
    }
  }

  ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GetData(PetscInt cross_term_range, Vec** Xi_p, PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p)
{
  PetscErrorCode ierr;
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;
  AppCtx         ctx;
  Data           *data;

  PetscFunctionBegin;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"lorenz_","Lorenz 96 options","");CHKERRQ(ierr);
  {
    ctx.force = 8;
    ierr = PetscOptionsReal("-force","forcing term","",ctx.force,&ctx.force,NULL);CHKERRQ(ierr);
    ctx.N = 36;
    ierr = PetscOptionsInt("-N","number of vector components","",ctx.N,&ctx.N,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = GetExactCoefficients(cross_term_range, &ctx, Xi_p);CHKERRQ(ierr);

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
  PetscInt       num_bases, cross_term_range;
  PetscInt       n,dim;
  Vec            *x,*dx;
  Vec            *Xi,Xi0,*correct_Xi;
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

  PetscPreLoadBegin(PETSC_FALSE,"Setup");

  /* Create 2nd order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(2, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetNormalizeColumns(basis, PETSC_FALSE);CHKERRQ(ierr);
  ierr = SINDyBasisSetCrossTermRange(basis, 2);CHKERRQ(ierr);
  ierr = SINDyBasisSetFromOptions(basis);CHKERRQ(ierr);
  ierr = SINDyBasisGetCrossTermRange(basis, &cross_term_range);CHKERRQ(ierr);

  ierr = SparseRegCreate(&sparse_reg);CHKERRQ(ierr);
  ierr = SparseRegSetThreshold(sparse_reg, 2e-1);CHKERRQ(ierr);
  ierr = SparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

  /* Generate data. */
  PetscPreLoadStage("Data generation");
  printf("Generating data...\n");
  ierr = GetData(cross_term_range, &correct_Xi, &n, &x, &dx);CHKERRQ(ierr);
  ierr = VecGetSize(x[0], &dim);CHKERRQ(ierr);

  Variable v_x,v_dx;
  ierr = VariableCreate("x", &v_x);CHKERRQ(ierr);
  ierr = VariableSetVecData(v_x, n, x, NULL);CHKERRQ(ierr);
  ierr = VariableCreate("dx/dt", &v_dx);CHKERRQ(ierr);
  ierr = VariableSetVecData(v_dx, n, dx, NULL);CHKERRQ(ierr);

  PetscPreLoadStage("AddVariables");
  ierr = SINDyBasisSetOutputVariable(basis, v_dx);CHKERRQ(ierr);
  ierr = SINDyBasisAddVariables(basis, 1, &v_x);CHKERRQ(ierr);


  /* Allocate solution vectors and run least squares. */
  PetscPreLoadStage("Regression");
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi0);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(Xi0, dim, &Xi);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi0);CHKERRQ(ierr);

  printf("Running sparse least squares...\n");
  ierr = SINDyFindSparseCoefficients(basis, sparse_reg, dim, Xi);CHKERRQ(ierr);

  /* Compare to exact coefficients. */
  PetscPreLoadStage("Error calcs");
  {
    Basis             correct_basis;
    PetscScalar       total_error[3], dim_err[3];
    const PetscScalar *data, *correct_data;
    PetscInt          d, i, n, k, correct_n;

    ierr = SINDyBasisCreate(2, 0, &correct_basis);CHKERRQ(ierr);
    ierr = SINDyBasisSetCrossTermRange(correct_basis, cross_term_range);CHKERRQ(ierr);
    ierr = SINDyBasisSetOutputVariable(correct_basis, v_dx);CHKERRQ(ierr);
    ierr = SINDyBasisAddVariables(correct_basis, 1, &v_x);CHKERRQ(ierr);

    // printf("\nCorrect coefficients:\n");
    // ierr = SINDyBasisPrint(correct_basis, dim, correct_Xi);CHKERRQ(ierr);

    ierr = SINDyBasisDataGetSize(correct_basis, NULL, &correct_n);CHKERRQ(ierr);
    ierr = SINDyBasisDataGetSize(basis, NULL, &n);CHKERRQ(ierr);

    for (k = 0; k < 3; k++) total_error[k] = 0;
    for (d = 0; d < dim; d++) {
      for (k = 0; k < 3; k++) dim_err[k] = 0;

      ierr = VecGetArrayRead(Xi[d], &data);CHKERRQ(ierr);
      ierr = VecGetArrayRead(correct_Xi[d], &correct_data);CHKERRQ(ierr);
      for(i = 0; i < PetscMin(n, correct_n); i++) {
        if ((data[i] == 0) != (correct_data[i] == 0)) dim_err[0]++;
        dim_err[1] += PetscAbsScalar(data[i] - correct_data[i]);
        dim_err[2] += PetscSqr(data[i] - correct_data[i]);
      }
      for(; i < correct_n; i++) {
        if (correct_data[i] != 0) dim_err[0]++;
        dim_err[1] += PetscAbsScalar(correct_data[i]);
        dim_err[2] += PetscSqr(correct_data[i]);
      }
      for(; i < n; i++) {
        if (data[i] != 0) dim_err[0]++;
        dim_err[1] += PetscAbsScalar(data[i]);
        dim_err[2] += PetscSqr(data[i]);
      }
      ierr = VecRestoreArrayRead(Xi[d], &data);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(correct_Xi[d], &correct_data);CHKERRQ(ierr);

      for (k = 0; k < 3; k++) total_error[k] += dim_err[k];
      dim_err[2] = PetscSqrtReal(dim_err[2]);
      printf("dim %d mismatches: %g\n", d, dim_err[0]);
      printf("dim %d   l1 error: %g\n", d, dim_err[1]);
      printf("dim %d   l2 error: %g\n", d, dim_err[2]);
    }
    total_error[2] = PetscSqrtReal(total_error[2]);
    printf("total mismatches: %g\n", total_error[0]);
    printf("total   l1 error: %g\n", total_error[1]);
    printf("total   l2 error: %g\n", total_error[2]);
    ierr = SINDyBasisDestroy(&correct_basis);CHKERRQ(ierr);
  }

  PetscInt iterations;
  ierr = SparseRegGetTotalIterationNumber(sparse_reg, &iterations);CHKERRQ(ierr);
  printf("iterations: %d\n", iterations);

  PetscReal res_norm;
  ierr = SINDyGetResidual(basis, &res_norm);CHKERRQ(ierr);
  printf("residual: %g\n", res_norm);

   /* Free PETSc data structures */
  PetscPreLoadStage("Cleanup");
  ierr = VecDestroyVecs(dim, &correct_Xi);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &x);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &dx);CHKERRQ(ierr);
  ierr = VecDestroyVecs(dim, &Xi);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = VariableDestroy(&v_x);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_dx);CHKERRQ(ierr);
  PetscPreLoadEnd();

  ierr = PetscFinalize();
  return ierr;
}
