#include <petscts.h>
#include "sindy.h"

static char help[] = "Run SINDy on data generated from the Lorentz system.\n";

typedef struct {
  PetscReal sigma,beta,rho;
} Lorentz;

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *f;
  Lorentz           *lorentz = (Lorentz*) ctx;

  PetscFunctionBegin;
  
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  f[0] = lorentz->sigma * (x[1] - x[0]);
  f[1] = x[0] * (lorentz->rho - x[2]) - x[1];
  f[2] = x[0] * x[1] - lorentz->beta * x[2];
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat B, void* ctx) {
  PetscErrorCode    ierr;
  const PetscScalar *x;
  Lorentz           *lorentz = (Lorentz*) ctx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 0, -lorentz->sigma, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 1, lorentz->sigma, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 2, 0, INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatSetValue(J, 1, 0, lorentz->rho - x[2], INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 1, 1, -1, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 1, 2, -x[0], INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatSetValue(J, 2, 0, x[1], INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 2, 1, x[0], INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 2, 2, -lorentz->beta, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GetData(PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p)
{ 
  PetscErrorCode ierr;
  PetscInt       steps = 1000;
  PetscReal      dt = 0.001;
  PetscInt       i;
  PetscInt       N = steps;
  PetscInt       idx[3] = {0, 1, 2};
  PetscReal      x[3];
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;
  Vec            *all_x, *all_dx;
  Lorentz        lorentz;

  PetscFunctionBegin;

  lorentz.sigma = 10;
  lorentz.beta = 8.0 / 3.0;
  lorentz.rho = 28;
  ierr = PetscOptionsGetReal(NULL,NULL,"-lorentz_sigma",&lorentz.sigma,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-lorentz_beta",&lorentz.beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-lorentz_rho",&lorentz.rho,NULL);CHKERRQ(ierr);

  ierr = MatCreateSeqDense(PETSC_COMM_SELF, 3, 3, NULL, &J);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(ts, TSRK5DP);CHKERRQ(ierr);
  ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, steps);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, (void*)&lorentz);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, (void*)&lorentz);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,3,&X);CHKERRQ(ierr);

  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Create Vecs to hold data. */
  ierr = VecDuplicateVecs(X, N, &all_x);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(X, N, &all_dx);CHKERRQ(ierr);

  /* Get x data from running TS. */
  x[0] = -8;
  x[1] = 8;
  x[2] = 27;
  ierr = VecSetValues(X, 2, idx, x, INSERT_VALUES);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);

  ierr = VecCopy(X, all_x[0]);CHKERRQ(ierr);
  for (i = 1; i < steps; i++) {
    ierr = TSStep(ts);CHKERRQ(ierr);
    ierr = VecCopy(X, all_x[i]);CHKERRQ(ierr);
  }

  /* Get derivate data using RHS. */
  for (i = 0; i < steps; i++) {
    ierr = RHSFunction(NULL, dt*i, all_x[i], all_dx[i], (void*)&lorentz);CHKERRQ(ierr);
  }

  /* Write output parameters. */
  *N_p = N;
  *all_x_p = all_x;
  *all_dx_p = all_dx;

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
  PetscMPIInt    size;
  PetscBool      flg;

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

  /* Create 3rd order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(3, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetFromOptions(basis);CHKERRQ(ierr);
  ierr = SINDyBasisCreateData(basis, x, n);CHKERRQ(ierr);

  ierr = SINDySparseRegCreate(&sparse_reg);CHKERRQ(ierr);
  ierr = SINDySparseRegSetThreshold(sparse_reg, 0.025);CHKERRQ(ierr);
  ierr = SINDySparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SINDySparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

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
  printf("Running sparse least squares...\n");
  ierr = SINDyFindSparseCoefficients(basis, sparse_reg, n, dx, 3, Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroyVecs(n, &x);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &dx);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[2]);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SINDySparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
