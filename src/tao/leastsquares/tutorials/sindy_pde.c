#include <petscts.h>
#include "sindy.h"
#include "sindy_pde.h"

static char help[] = "Run SINDy on data generated from a pde.\n";

// PetscErrorCode GetData(PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p)
// {
//   PetscErrorCode ierr;
//   PetscInt       r;
//   Mat            J;
//   TS             ts;
//   TSAdapt        adapt;
//   Vec            X;
//   AppCtx         ctx;
//   Data           *data;

//   PetscFunctionBegin;
//   data = &ctx.data;
//   ierr = MatCreateSeqDense(PETSC_COMM_SELF, ctx.N, ctx.N, NULL, &J);CHKERRQ(ierr);
//   ierr = VecCreateSeq(PETSC_COMM_SELF,ctx.N,&X);CHKERRQ(ierr);
//   ierr = DataInitialize(data, X);CHKERRQ(ierr);

//   ierr = TSCreate(PETSC_COMM_SELF, &ts);CHKERRQ(ierr);
//   ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
//   ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
//   ierr = TSRKSetType(ts, TSRK5DP);CHKERRQ(ierr);
//   ierr = TSSetTimeStep(ts, data->dt);CHKERRQ(ierr);
//   ierr = TSSetMaxSteps(ts, data->steps);CHKERRQ(ierr);

//   ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

//   ierr = TSSetRHSFunction(ts, NULL, RHSFunction, &ctx);CHKERRQ(ierr);
//   ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, &ctx);CHKERRQ(ierr);

//   ierr = TSSetApplicationContext(ts, (void*)&ctx);CHKERRQ(ierr);
//   ierr = TSSetPostStep(ts,DataPostStep);CHKERRQ(ierr);

//   ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
//   ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
//   ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

//   ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
//   ierr = TSSetUp(ts);CHKERRQ(ierr);

//   /* Set initial condition. */
//   ierr = VecSet(X, 1);CHKERRQ(ierr);
//   ierr = VecSetValue(X,ctx.N/2,1.01,INSERT_VALUES);CHKERRQ(ierr);
//   ierr = DataPostStep(ts);CHKERRQ(ierr);


//   /* Get x data from running TS. */
//   ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
//   ierr = TSSetTimeStep(ts, data->dt);CHKERRQ(ierr);
//   ierr = TSSetMaxSteps(ts, data->steps-1);CHKERRQ(ierr);
//   ierr = TSSolve(ts, NULL);CHKERRQ(ierr);

//   /* Get derivative data. */
//   ierr = DataComputeDerivative(data);CHKERRQ(ierr);

//   /* Write output parameters. */
//   *N_p = data->N;
//   *all_x_p = data->all_x;
//   *all_dx_p = data->all_dx;

//   ierr = TSDestroy(&ts);CHKERRQ(ierr);
//   ierr = VecDestroy(&X);CHKERRQ(ierr);
//   ierr = MatDestroy(&J);CHKERRQ(ierr);
//   PetscFunctionReturn(0);
// }

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

  ierr = PetscInitialize(&argc,&argv,"petscopt_ex6",help);if (ierr) return ierr;

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

  // for (PetscInt i = 0; i < n; i++) {
  //   ierr = VecView(x[i], PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  // }

  ierr = VecGetSize(x[0], &dim);CHKERRQ(ierr);

  /* Create 2nd order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(2, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetNormalizeColumns(basis, PETSC_FALSE);CHKERRQ(ierr);
  ierr = SINDyBasisSetCrossTermRange(basis, 1);CHKERRQ(ierr);
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
