static char help[] = "Test if EigenBasis works with known eigendecomposition of a fixed matrix.\n\n";
/*
    TODO : CHeck how latex documentation works 

    We do a basic check that the eigen decomposition function callbacks of RiemannSolver 
    work by setting the eigbasis function to be a linearly independent set of 
    eigenvectors of the matrix 
    \[
        A = \frac{1}{10}
        \begin{pmatrix} 
        8  & 6  & 0 \\
        -4 & 22 & 0 \\
        0  & 0  & 30 
        \end{pmatrix}
    ]\

*/

#include <petscriemannsolver.h>

static PetscErrorCode EigBasis_Dummy(void *ctx, const PetscReal *u, Mat eigmat)
{
  PetscInt  m = 3, n = 3, i;
  PetscReal X[m][n];
  PetscInt  idxm[m], idxn[n];

  PetscFunctionBeginUser;
  for (i = 0; i < m; i++) idxm[i] = i;
  for (i = 0; i < n; i++) idxn[i] = i;
  /* Known Eigenvectors of the Matrix A */
  X[0][0] = 3;
  X[1][0] = 1;
  X[2][0] = 0;
  X[0][1] = 2;
  X[1][1] = 4;
  X[2][1] = 0;
  X[0][2] = 0;
  X[1][2] = 0;
  X[2][2] = 1;
  PetscCall(MatSetValues(eigmat, m, idxm, n, idxn, (PetscReal *)X, INSERT_VALUES));
  MatAssemblyBegin(eigmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(eigmat, MAT_FINAL_ASSEMBLY);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscInt       dim = 1, numfields = 3;
  RiemannSolver  rs;
  Mat            A, Eig, AEig;
  PetscReal      a[numfields][numfields];
  PetscReal      aeig[numfields][numfields];
  PetscBool      isequal = PETSC_FALSE;
  PetscViewer    viewer;

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  /* create RiemannSolver */
  PetscCall(RiemannSolverCreate(PETSC_COMM_SELF, &rs));
  PetscCall(RiemannSolverSetFluxDim(rs, dim, numfields));
  PetscCall(RiemannSolverSetEigBasis(rs, EigBasis_Dummy));
  PetscCall(RiemannSolverSetType(rs, "lax"));
  PetscCall(RiemannSolverSetUp(rs));
  {
    a[0][0]    = 8. / 10.;
    aeig[0][0] = 3.0;
    a[0][1]    = -4. / 10.;
    aeig[0][1] = 1.;
    a[0][2]    = 0;
    aeig[0][2] = 0;
    a[1][0]    = 6. / 10.;
    aeig[1][0] = 4.;
    a[1][1]    = 22. / 10.;
    aeig[1][1] = 8.0;
    a[1][2]    = 0;
    aeig[1][2] = 0;
    a[2][0]    = 0;
    aeig[2][0] = 0;
    a[2][1]    = 0;
    aeig[2][1] = 0;
    a[2][2]    = 3.0;
    aeig[2][2] = 3.0;
  }
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, numfields, numfields, (PetscReal *)a, &A));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, numfields, numfields, (PetscReal *)aeig, &AEig));
  PetscCall(RiemannSolverComputeEigBasis(rs, PETSC_NULL, &Eig));
  PetscCall(MatMatMultEqual(A, Eig, AEig, numfields, &isequal));

  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "A matrix\n\n"));
  PetscCall(MatView(A, viewer));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Eig matrix\n\n"));
  PetscCall(MatView(Eig, viewer));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "AEig matrix\n\n"));
  PetscCall(MatView(AEig, viewer));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "EigenBasis is Working Analytically: %s \n", isequal ? "true" : "false"));
  ierr = RiemannSolverDestroy(&rs);
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:

TEST*/
