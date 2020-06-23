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
  PetscReal      *t_exp;
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

  DM       x_dm;
  Vec      *x_vecs;
  Variable v_x;
  {
    Vec        gc;
    DMDACoor2d **coords;
    PetscInt   xs,ys,xm,ym,i,j,k;

    ierr = DMGetCoordinateDM(dm,&x_dm);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm,&gc);CHKERRQ(ierr);

    ierr = VecDuplicateVecs(gc, n, &x_vecs);CHKERRQ(ierr);
    ierr = DMDAGetCorners(x_dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

    ierr = VecCopy(gc, x_vecs[0]);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(x_dm,x_vecs[0],&coords);CHKERRQ(ierr);
    for (i=xs; i < xs+xm; i++) {
      for (j=ys; j < ys+ym; j++) {
        coords[j][i].x = PetscSinReal(coords[j][i].x);
      }
    }
    ierr = DMDAVecRestoreArray(x_dm,x_vecs[0],&coords);CHKERRQ(ierr);
    for (k = 1; k < n; k++) {
      ierr = VecCopy(x_vecs[0], x_vecs[k]);CHKERRQ(ierr);
    }
  }
  ierr = VariableCreate("x", &v_x);CHKERRQ(ierr);
  ierr = VariableSetVecData(v_x, n, x_vecs, x_dm);CHKERRQ(ierr);

  Variable v_u,v_dudt,v_t,v_exp_t;
  ierr = VariableCreate("u", &v_u);CHKERRQ(ierr);
  ierr = VariableSetVecData(v_u, n, u, dm);CHKERRQ(ierr);

  ierr = VariableCreate("du/dt", &v_dudt);CHKERRQ(ierr);
  ierr = VariableSetVecData(v_dudt, n, du, dm);CHKERRQ(ierr);

  ierr = VariableCreate("t", &v_t);CHKERRQ(ierr);
  ierr = VariableSetScalarData(v_t, n, t);CHKERRQ(ierr);

  ierr = VariableCreate("(1 - exp(-t/lambda))", &v_exp_t);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &t_exp);CHKERRQ(ierr);
  for (PetscInt i = 0; i < n; i++) t_exp[i] = 1.0 - PetscExpScalar(-t[i] / 0.1);
  ierr = VariableSetScalarData(v_exp_t, n, t_exp);CHKERRQ(ierr);

  Variable v_dudx,v_dudy,v_dudyy,v_dudxx;
  ierr = VariableDifferentiateSpatial(v_u, 0, 1, "du/dx", &v_dudx);CHKERRQ(ierr);
  ierr = VariableDifferentiateSpatial(v_u, 1, 1, "du/dy", &v_dudy);CHKERRQ(ierr);
  ierr = VariableDifferentiateSpatial(v_u, 0, 2, "d2u/dx2", &v_dudxx);CHKERRQ(ierr);
  ierr = VariableDifferentiateSpatial(v_u, 1, 2, "d2u/dy2", &v_dudyy);CHKERRQ(ierr);

  /* Create 2nd order polynomial basis, with no sine functions. */
  ierr = SINDyBasisCreate(2, 0, &basis);CHKERRQ(ierr);
  ierr = SINDyBasisSetNormalizeColumns(basis, PETSC_FALSE);CHKERRQ(ierr);
  ierr = SINDyBasisSetFromOptions(basis);CHKERRQ(ierr);

  ierr = SINDyBasisSetOutputVariable(basis, v_dudt);CHKERRQ(ierr);

  Variable vars[] = {v_dudx, v_dudy, v_dudyy, v_dudxx, v_t, v_exp_t, v_x};
  printf("Building basis...\n");
  ierr = SINDyBasisAddVariables(basis, sizeof(vars)/sizeof(vars[0]), vars);CHKERRQ(ierr);

  ierr = SparseRegCreate(&sparse_reg);CHKERRQ(ierr);
  ierr = SparseRegSetThreshold(sparse_reg, 1e-1);CHKERRQ(ierr);
  ierr = SparseRegSetMonitor(sparse_reg, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SparseRegSetFromOptions(sparse_reg);CHKERRQ(ierr);

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
  ierr = SINDyFindSparseCoefficients(basis, sparse_reg, 1, Xi);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroyVecs(n, &u);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &du);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n, &x_vecs);CHKERRQ(ierr);
  ierr = VecDestroyVecs(dim, &Xi);CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(t);CHKERRQ(ierr);
  ierr = PetscFree(t_exp);CHKERRQ(ierr);
  ierr = SINDyBasisDestroy(&basis);CHKERRQ(ierr);
  ierr = SparseRegDestroy(&sparse_reg);CHKERRQ(ierr);

  ierr = VariableDestroy(&v_x);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_u);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_dudt);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_dudx);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_dudy);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_dudxx);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_dudyy);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_t);CHKERRQ(ierr);
  ierr = VariableDestroy(&v_exp_t);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
