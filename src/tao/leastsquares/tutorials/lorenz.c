#include <petscts.h>
#include "sindy.h"

static char help[] = "Run SINDy on data generated from the Lorenz attractor system.\n";

typedef struct {
  PetscInt  runs,steps,N,i;
  PetscReal dt;
  Vec       *all_x,*all_dx;
  PetscBool fd_der;
} Data;

typedef struct {
  PetscReal sigma,beta,rho;
  Data      data;
} Lorenz;

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *f;
  Lorenz           *lorenz = (Lorenz*) ctx;

  PetscFunctionBegin;
  
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  f[0] = lorenz->sigma * (x[1] - x[0]);
  f[1] = x[0] * (lorenz->rho - x[2]) - x[1];
  f[2] = x[0] * x[1] - lorenz->beta * x[2];
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat B, void* ctx) {
  PetscErrorCode    ierr;
  const PetscScalar *x;
  Lorenz           *lorenz = (Lorenz*) ctx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 0, -lorenz->sigma, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 1, lorenz->sigma, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 2, 0, INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatSetValue(J, 1, 0, lorenz->rho - x[2], INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 1, 1, -1, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 1, 2, -x[0], INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatSetValue(J, 2, 0, x[1], INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 2, 1, x[0], INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 2, 2, -lorenz->beta, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DataInitialize(Data* data, Vec X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  data->steps  = 1000;
  data->dt     = 0.001;
  data->fd_der = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Data generation options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsBool("-fd_der","use finite-difference to estimate derivative","",data->fd_der,&data->fd_der,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-steps","how many timesteps to simulate in each run","",data->steps,&data->steps,NULL);CHKERRQ(ierr);
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
  Data           *data;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&data);CHKERRQ(ierr);
  if (data->i == data->N) {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Cannot record more than %d vectors.",data->N);
  }
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = VecCopy(X, data->all_x[data->i]);CHKERRQ(ierr);
  if (!data->fd_der) {
    PetscReal     t;
    TSRHSFunction func;
    void          *ctx;
    ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
    ierr = TSGetRHSFunction(ts,NULL,&func,&ctx);CHKERRQ(ierr);
    ierr = func(ts, t, X, data->all_dx[data->i], ctx);CHKERRQ(ierr);
  }
  data->i++;
  PetscFunctionReturn(0);
}

PetscErrorCode DataComputeDerivative_FD(Data* data)
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

PetscErrorCode GetExactCoefficients(Lorenz* lorenz, Vec** Xi)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       idx[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  Vec            tmp;

  PetscFunctionBegin;
  ierr = VecCreateSeq(PETSC_COMM_SELF,10,&tmp);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(tmp, 3, Xi);CHKERRQ(ierr);
  PetscScalar Xi_data[3][10] = {{0, -1.0000e+01,  1.0000e+01,           0,           0,           0,           0, 0, 0, 0},
                                {0,  2.8000e+01, -1.0000e+00,           0,           0,           0, -1.0000e+00, 0, 0, 0},
                                {0,           0,           0, -2.6667e+00,           0,  1.0000e+00,           0, 0, 0, 0}};

  Xi_data[0][1] = -lorenz->sigma;
  Xi_data[0][2] = lorenz->sigma;
  Xi_data[1][1] = lorenz->rho;
  Xi_data[1][2] = -1;
  Xi_data[1][6] = -1;
  Xi_data[2][3] = -lorenz->beta;
  Xi_data[2][5] = 1;

  for (i = 0; i < 3; i++) {
    ierr = VecSetValues((*Xi)[i], 10, idx, Xi_data[i], INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin((*Xi)[i]);CHKERRQ(ierr);
    ierr = VecAssemblyEnd((*Xi)[i]);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GetData(PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p, Vec** Xi_p)
{ 
  PetscErrorCode ierr;
  PetscInt       idx[3] = {0, 1, 2};
  PetscReal      x[3];
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;
  Lorenz         lorenz;

  PetscFunctionBegin;

  lorenz.sigma = 10;
  lorenz.beta = 8.0 / 3.0;
  lorenz.rho = 28;
  ierr = PetscOptionsGetReal(NULL,NULL,"-lorenz_sigma",&lorenz.sigma,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-lorenz_beta",&lorenz.beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-lorenz_rho",&lorenz.rho,NULL);CHKERRQ(ierr);

  ierr = GetExactCoefficients(&lorenz, Xi_p);CHKERRQ(ierr);

  ierr = MatCreateSeqDense(PETSC_COMM_SELF, 3, 3, NULL, &J);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,3,&X);CHKERRQ(ierr);
  ierr = DataInitialize(&lorenz.data, X);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(ts, TSRK5DP);CHKERRQ(ierr);
  ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, lorenz.data.dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, lorenz.data.steps-1);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, (void*)&lorenz);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, (void*)&lorenz);CHKERRQ(ierr);

  ierr = TSSetApplicationContext(ts, (void*)&lorenz.data);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,DataPostStep);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Get x data from running TS. */
  x[0] = -8;
  x[1] = 8;
  x[2] = 27;
  ierr = VecSetValues(X, 2, idx, x, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);

  ierr = DataPostStep(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts, NULL);CHKERRQ(ierr);

  if (lorenz.data.i != lorenz.data.N) {
    printf("Uh oh: recorded %d data points but expected %d data points\n", lorenz.data.i, lorenz.data.N);
  }

  if (lorenz.data.fd_der) {
    ierr = DataComputeDerivative_FD(&lorenz.data);CHKERRQ(ierr);
  }

  /* Write output parameters. */
  *N_p = lorenz.data.N;
  *all_x_p = lorenz.data.all_x;
  *all_dx_p = lorenz.data.all_dx;

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
  Vec            Xi[3];
  Vec            *correct_Xi;
  PetscMPIInt    size;
  PetscBool      flg;
  const int      dim = 3;

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
  PetscPreLoadBegin(PETSC_FALSE,"Data generation");
  printf("Generating data...\n");
  ierr = GetData(&n, &x, &dx, &correct_Xi);CHKERRQ(ierr);

  Variable v_x,v_dx;

  PetscPreLoadStage("Var/Basis setup");
  ierr = VariableCreate("x", &v_x);CHKERRQ(ierr);
  ierr = VariableSetVecData(v_x, n, x, NULL);CHKERRQ(ierr);
  ierr = VariableCreate("dx/dt", &v_dx);CHKERRQ(ierr);
  ierr = VariableSetVecData(v_dx, n, dx, NULL);CHKERRQ(ierr);

  /* Create 3rd order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(3, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetFromOptions(basis);CHKERRQ(ierr);

  PetscPreLoadStage("AddVariables");
  ierr = SINDyBasisSetOutputVariable(basis, v_dx);CHKERRQ(ierr);
  ierr = SINDyBasisAddVariables(basis, 1, &v_x);CHKERRQ(ierr);

  PetscPreLoadStage("Regress setup");
  ierr = SparseRegCreate(&sparse_reg);CHKERRQ(ierr);
  ierr = SparseRegSetThreshold(sparse_reg, 0.025);CHKERRQ(ierr);
  ierr = SparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

  /* Allocate solution vectors */
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi[0]);CHKERRQ(ierr);
  ierr = VecDuplicate(Xi[0], &Xi[1]);CHKERRQ(ierr);
  ierr = VecDuplicate(Xi[0], &Xi[2]);CHKERRQ(ierr);

  /* Use l2prox by default. */
  ierr = PetscOptionsHasName(NULL, NULL, "-tao_brgn_regularization_type", &flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscOptionsSetValue(NULL, "-tao_brgn_regularization_type", "l2prox");CHKERRQ(ierr);
  }

  /* Run least squares */
  PetscPreLoadStage("Regression");
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
    ierr = SINDyBasisSetOutputVariable(correct_basis, v_dx);CHKERRQ(ierr);
    ierr = SINDyBasisAddVariables(correct_basis, 1, &v_x);CHKERRQ(ierr);

    printf("\nCorrect coefficients:\n");
    ierr = SINDyBasisPrint(correct_basis, dim, correct_Xi);CHKERRQ(ierr);

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

  PetscInt  iterations;
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
  ierr = VecDestroy(&Xi[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[2]);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = VariableDestroy(&v_x);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_dx);CHKERRQ(ierr);
  PetscPreLoadEnd();

  ierr = PetscFinalize();
  return ierr;
}
