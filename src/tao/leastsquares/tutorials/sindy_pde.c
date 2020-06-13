#include "sindy.h"
#include "sindy_pde.h"

static char help[] = "Run SINDy on data generated from a pde.\n";

int main(int argc, char** argv)
{
  PetscErrorCode ierr;
  Basis          basis;
  SparseReg      sparse_reg;
  PetscInt       num_bases;
  PetscInt       n,dim;
  Vec            *u,*du;
  Vec            *Xi,Xi0;
  PetscMPIInt    size;
  PetscReal      *t;
  DM             dm;

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
  ierr = GetData(&n, &u, &du, &t, &dm);CHKERRQ(ierr);

  Variable v_u,v_du,v_t;
  ierr = SINDyVariableCreate("u", &v_u);CHKERRQ(ierr);
  ierr = SINDyVariableSetVecData(v_u, n, u, dm);CHKERRQ(ierr);
  ierr = SINDyVariableCreate("du/dt", &v_du);CHKERRQ(ierr);
  ierr = SINDyVariableSetVecData(v_du, n, du, dm);CHKERRQ(ierr);
  // ierr = SINDyVariableCreate("t", &v_t);CHKERRQ(ierr);
  // ierr = SINDyVariableSetScalarData(v_t, n, t);CHKERRQ(ierr);

  // for (PetscInt i = 0; i < n; i++) {
  //   ierr = VecView(x[i], PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  // }

  /* Create 2nd order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(2, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetNormalizeColumns(basis, PETSC_FALSE);CHKERRQ(ierr);
  // ierr = SINDyBasisSetCrossTermRange(basis, 0);CHKERRQ(ierr);
  ierr = SINDyBasisSetFromOptions(basis);CHKERRQ(ierr);

  ierr = SINDyBasisSetOutputVariable(basis, v_du);CHKERRQ(ierr);
  ierr = SINDyBasisAddVariables(basis, 1, &v_du);CHKERRQ(ierr);

  ierr = SINDySparseRegCreate(&sparse_reg);CHKERRQ(ierr);
  ierr = SINDySparseRegSetThreshold(sparse_reg, 1e-1);CHKERRQ(ierr);
  ierr = SINDySparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SINDySparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

  /* Allocate solution vector */
  if (dm) {
    ierr = DMDAGetDof(dm, &dim);CHKERRQ(ierr);
  } else {
    ierr = VecGetSize(u[0], &dim);CHKERRQ(ierr);
  }
  ierr = SINDyBasisDataGetSize(basis, NULL, &num_bases);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_bases, &Xi0);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(Xi0, dim, &Xi);CHKERRQ(ierr);
  ierr = VecDestroy(&Xi0);CHKERRQ(ierr);

  /* Run least squares */
  printf("Running sparse least squares...\n");
  ierr = SINDyFindSparseCoefficientsVariable(basis, sparse_reg, 1, Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroyVecs(n, &u);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &du);CHKERRQ(ierr);
  ierr = VecDestroyVecs(dim, &Xi);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(t);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SINDySparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = SINDyVariableDestroy(&v_u);CHKERRQ(ierr);
  ierr = SINDyVariableDestroy(&v_du);CHKERRQ(ierr);
  // ierr = SINDyVariableDestroy(&v_t);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
