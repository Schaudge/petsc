#include <petscts.h>
#include "sindy.h"

static char help[] = "Run SINDy on data generated from dx/dt = [-sin(x), cos(x)]. Finds the least-squares solution to the under constraint linear model A*Xi = dx/dt, with L1-norm regularizer. \n\
            A is a basis function matrix, Xi is sparse. \n\
            We find the sparse solution by solving 0.5*||A Xi-dx/dt||^2 + lambda*||D*Xi||_1, where lambda (by default 1e-4) is a user specified weight.\n\
            D is the K*N transform matrix so that D*Xi is sparse. By default D is identity matrix, so that D*x = x.\
            Note that you must pass the option '-tao_brgn_regularization_type l1dict' to use L1 regularization instead of the default l2prox.\n";

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

PetscErrorCode GetData(Vec* data, Vec* der_data)
{ 
  PetscErrorCode ierr;
  PetscInt       runs = 10;
  PetscInt       steps = 5000;
  PetscReal      dt = 0.001;
  PetscInt       i, r, t;
  PetscInt       N = runs * steps;
  PetscInt       idx[2] = {0, 1};
  PetscReal      x[2];
  PetscReal      *data_array;
  PetscReal      *der_data_array;
  Mat            J;
  TS             ts;
  TSAdapt        adapt;
  Vec            X;

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

  ierr = VecCreateSeq(PETSC_COMM_SELF,N*2,data);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&X);CHKERRQ(ierr);
  ierr = VecGetArray(*data, &data_array);CHKERRQ(ierr);

  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Get x data from running TS. */
  i = 0;
  for (r = 0; r < runs; r++) {
    x[0] = -1.25 + r * 0.25;
    if (PetscAbsReal(x[0]) < 1e-3) continue;
    x[1] = -10.25 + r * 2.5 + 1.57;

    ierr = VecSetValues(X, 2, idx, x, INSERT_VALUES);CHKERRQ(ierr);
    ierr = TSSetSolution(ts, X);CHKERRQ(ierr);

    data_array[i] = x[0];
    i++;
    data_array[i] = x[1];
    i++;
    for (t = 1; t < steps; t++) {
      ierr = TSStep(ts);CHKERRQ(ierr);
      ierr = VecGetValues(X, 2, idx, x);CHKERRQ(ierr);
      data_array[i] = x[0];
      i++;
      data_array[i] = x[1];
      i++;
    }
  }

  /* Get derivate data using fourth-order central difference. */

  /* Derivative of first component. */
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&der_data[0]);CHKERRQ(ierr);
  ierr = VecSet(der_data[0], 0);CHKERRQ(ierr);
  ierr = VecGetArray(der_data[0], &der_data_array);CHKERRQ(ierr);
  i = 0;
  for (r = 0; r < runs; r++) {
    for (t = 0; t < steps; t++) {
      if (t >= 2 && t < steps - 2) {
        der_data_array[i] = (1.0/(12.0*dt))*(-data_array[2*(i+2)]+8*data_array[2*(i+1)]-8*data_array[2*(i-1)]+data_array[2*(i-2)]);
      }
      else {
        /* Set boundary values to 0. */
        der_data_array[i] = 0;
      }
      i++;
    }
    /* Set boundary values to 0. */
    data_array[2*(i-1)] = 0;
    data_array[2*(i-2)] = 0;
    data_array[2*(i-steps)] = 0;
    data_array[2*(i-steps+1)] = 0;
  }
  ierr = VecRestoreArray(der_data[0], &der_data_array);CHKERRQ(ierr);

  /* Derivative of second component. */
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&der_data[1]);CHKERRQ(ierr);
  ierr = VecSet(der_data[1], 0);CHKERRQ(ierr);
  ierr = VecGetArray(der_data[1], &der_data_array);CHKERRQ(ierr);
  i = 0;
  for (r = 0; r < runs; r++) {
    for (t = 0; t < steps; t++) {
      if (t >= 2 && t < steps - 2) {
        der_data_array[i] = (1.0/(12.0*dt))*(-data_array[2*(i+2)+1]+8*data_array[2*(i+1)+1]-8*data_array[2*(i-1)+1]+data_array[2*(i-2)+1]);
      }
      else {
        /* Set boundary values to 0. */
        der_data_array[i] = 0;
      }
      i++;
    }
    /* Set boundary values to 0. */
    data_array[2*(i-1)+1] = 0;
    data_array[2*(i-2)+1] = 0;
    data_array[2*(i-steps)+1] = 0;
    data_array[2*(i-steps+1)+1] = 0;
  }
  ierr = VecRestoreArray(der_data[1], &der_data_array);CHKERRQ(ierr);

  ierr = VecRestoreArray(*data, &data_array);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char** argv) {
  PetscErrorCode ierr;
  Basis          basis;
  PetscInt       num_bases;
  Vec            x,dx[2],Xi[2];
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
  ierr = GetData(&x, dx);CHKERRQ(ierr);

  /* Create 5th order polynomial basis, with no sine functions. */
  ierr = SINDyCreateBasisAndData(x, 2, 5, 0, &basis);CHKERRQ(ierr);

  /* Allocate solution vectors */
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi[0]);CHKERRQ(ierr);
  ierr = VecDuplicate(Xi[0], &Xi[1]);CHKERRQ(ierr);

  /* Run least squares */
  ierr = SINDyFindSparseCoefficients(basis, 2, dx, Xi);CHKERRQ(ierr);

  /* View result. */
  ierr = PetscPrintf(PETSC_COMM_SELF, "------------- result Xi ------------- \n");CHKERRQ(ierr);
  ierr = SINDyBasisPrint(basis, 2, Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&dx[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&dx[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi[1]);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
