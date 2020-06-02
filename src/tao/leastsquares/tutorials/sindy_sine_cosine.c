#include <petscts.h>
#include "sindy.h"

static char help[] = "Run SINDy on data generated from dx/dt = [-sin(x), cos(x)].\n";

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *f;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  f[0] = -PetscSinReal(x[0]);
  f[1] =  PetscCosReal(x[1]);
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat B, void* ctx) {
  PetscErrorCode    ierr;
  const PetscScalar *x;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 0, -PetscCosReal(x[0]), INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 0, 1, 0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 1, 0, 0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(J, 1, 1, -PetscSinReal(x[1]), INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GetData(PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p)
{ 
  PetscErrorCode ierr;
  PetscInt       runs = 10;
  PetscInt       steps = 5000;
  PetscReal      dt = 0.001;
  PetscInt       i, r, t;
  PetscInt       N = runs * steps;
  PetscInt       idx[2] = {0, 1};
  PetscReal      x[2];
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;
  Vec            *all_x, *all_dx;

  PetscFunctionBegin;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, 2, 2, NULL, &J);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(ts, TSRK5DP);CHKERRQ(ierr);
  ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, steps);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, NULL);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&X);CHKERRQ(ierr);

  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Create Vecs to hold data. */
  ierr = VecDuplicateVecs(X, N, &all_x);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(X, N, &all_dx);CHKERRQ(ierr);

  /* Get x data from running TS. */
  i = 0;
  for (r = 0; r < runs; r++) {
    /* Set the initial x values centered around zero. */
    x[0] = -1.25 + r * 0.25;
    if (x[0] >= -1e-3) {
      x[0] = -1.25 + (r+1) * 0.25;
    }
    /* Choose second component to not be the same as the first component. */
    x[1] = -1.25 + ((r + runs/2 + 1) % runs) * 0.25;
    if (x[1] >= -1e-3) {
      x[1] = -1.25 + ((r + runs/2 + 2) % runs) * 0.25;
    }

    ierr = VecSetValues(X, 2, idx, x, INSERT_VALUES);CHKERRQ(ierr);
    ierr = TSSetSolution(ts, X);CHKERRQ(ierr);

    ierr = VecCopy(X, all_x[i]);CHKERRQ(ierr);
    i++;
    for (t = 1; t < steps; t++) {
      ierr = TSStep(ts);CHKERRQ(ierr);
      ierr = VecCopy(X, all_x[i]);CHKERRQ(ierr);
      i++;
    }
  }

  /* Get derivate data using fourth-order central difference. */

  i = 0;
  for (r = 0; r < runs; r++) {
    for (t = 0; t < steps; t++) {
      ierr = VecSet(all_dx[i], 0);CHKERRQ(ierr);
      if (t >= 2 && t < steps - 2) {
        ierr = VecAXPY(all_dx[i], -1.0, all_x[i+2]);CHKERRQ(ierr);
        ierr = VecAXPY(all_dx[i],  8.0, all_x[i+1]);CHKERRQ(ierr);
        ierr = VecAXPY(all_dx[i], -8.0, all_x[i-1]);CHKERRQ(ierr);
        ierr = VecAXPY(all_dx[i],  1.0, all_x[i-2]);CHKERRQ(ierr);
        ierr = VecScale(all_dx[i], 1.0/(12.0*dt));CHKERRQ(ierr);
      }
      i++;
    }
    /* Set boundary values to 0. */
    ierr = VecSet(all_x[i-1], 0);CHKERRQ(ierr);
    ierr = VecSet(all_x[i-2], 0);CHKERRQ(ierr);
    ierr = VecSet(all_x[i-steps], 0);CHKERRQ(ierr);
    ierr = VecSet(all_x[i-steps+1], 0);CHKERRQ(ierr);
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
  Vec            Xi[2];
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
  ierr = SINDySparseRegSetThreshold(sparse_reg, 5e-1);CHKERRQ(ierr);
  ierr = SINDySparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SINDySparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

  /* Allocate solution vectors */
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi[0]);CHKERRQ(ierr);
  ierr = VecDuplicate(Xi[0], &Xi[1]);CHKERRQ(ierr);

  /* Run least squares */
  ierr = SINDyFindSparseCoefficients(basis, sparse_reg, n, dx, 2, Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroyVecs(n, &x);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &dx);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[1]);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SINDySparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
