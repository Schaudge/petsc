#include <petscts.h>
#include "sindy.h"

static char help[] = "Run SINDy on data generated from dx/dt = -sin(x). Finds the least-squares solution to the under constraint linear model A*Xi = dx/dt, with L1-norm regularizer. \n\
            A is a basis function matrix, Xi is sparse. \n\
            We find the sparse solution by solving 0.5*||A Xi-dx/dt||^2 + lambda*||D*Xi||_1, where lambda (by default 1e-4) is a user specified weight.\n\
            D is the K*N transform matrix so that D*Xi is sparse. By default D is identity matrix, so that D*x = x.\n";

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
  PetscErrorCode ierr;
  PetscReal      x;
  PetscInt       idx = 0;

  PetscFunctionBegin;
  ierr = VecGetValues(X, 1, &idx, &x);CHKERRQ(ierr);
  ierr = VecSet(F, -PetscSinReal(x));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat B, void* ctx) {
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

PetscErrorCode GetData(PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p)
{ 
  PetscErrorCode ierr;
  PetscInt       runs = 10;
  PetscInt       steps = 5000;
  PetscReal      dt = 0.001;
  PetscInt       i, r, t;
  PetscInt       N = runs * steps;
  PetscReal      x0;
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;
  Vec            *all_x, *all_dx;

  PetscFunctionBegin;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &J);CHKERRQ(ierr);

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

  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&X);CHKERRQ(ierr);

  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Create Vecs to hold data. */
  ierr = VecDuplicateVecs(X, N, &all_x);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(X, N, &all_dx);CHKERRQ(ierr);

  /* Get x data from running TS. */
  i = 0;
  for (r = 0; r < runs; r++) {
    x0 = -1.25 + r * 0.25;
    if (x0 >= -1e-3) {
      x0 = -1.25 + (r+1) * 0.25;
    }

    ierr = VecSet(X, x0);CHKERRQ(ierr);
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
  ierr = SINDySparseRegSetThreshold(sparse_reg, 1e-4);CHKERRQ(ierr);
  ierr = SINDySparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

  /* Allocate solution vector */
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi);CHKERRQ(ierr);

  /* Run least squares */
  ierr = SINDyFindSparseCoefficients(basis, sparse_reg, n, dx, 1, &Xi);CHKERRQ(ierr);

  /* View result. */
  ierr = PetscPrintf(PETSC_COMM_SELF, "------ result Xi ------ \n");CHKERRQ(ierr);
  ierr = SINDyBasisPrint(basis, 1, &Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroyVecs(n, &x);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &dx);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SINDySparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
