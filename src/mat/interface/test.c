#include <petscmat.h>

PetscScalar findEPS(PetscScalar eps)
{
  while ((1+eps/2) != 1)
  {
    eps = eps/2;
  }
  return eps;
}

PetscErrorCode shiftedCholeskyQR3(Vec vecs[], PetscInt n, PetscInt m)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscInt		i, j;
  PetscScalar		*colvecs, *vecarrs;
  PetscScalar		shift, u = 0.2, norm;
  Mat			vecMat, Q, R, Rhat;
  MatFactorInfo		info;
  IS			rowperm, colperm;

  ierr = PetscObjectGetComm((PetscObject) vecs[0], &comm);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(comm, n, m, NULL, &vecMat);CHKERRQ(ierr);
  for (i = 0; i < m; i++) {
    ierr = MatDenseGetColumn(vecMat, i, &colvecs);CHKERRQ(ierr);
    ierr = VecGetArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    for (j = 0; j < n; j++) {
      colvecs[j] = vecarrs[j];
    }
    ierr = VecRestoreArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(vecMat, &colvecs);CHKERRQ(ierr);
  }
  ierr = MatNorm(vecMat, NORM_FROBENIUS, &norm);CHKERRQ(ierr);
  ierr = MatDuplicate(vecMat, MAT_COPY_VALUES, &Q);CHKERRQ(ierr);
  ierr = MatMatMult(Q, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Rhat);CHKERRQ(ierr);
  u = findEPS(0.1);

  /*	Initial shift Cholesky	*/
  shift = 11*((m*n)+(n*(n+1)))*u*norm;
  ierr = MatShift(Rhat, shift);CHKERRQ(ierr);
  ierr = MatDuplicate(Rhat, MAT_COPY_VALUES, &R);CHKERRQ(ierr);
  ierr = MatCholeskyFactor(R, NULL, &info);CHKERRQ(ierr);
  ierr = MatSetUnfactored(R);CHKERRQ(ierr);
  ierr = MatSetUnfactored(Q);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  for (i = 0; i < 2; i++) {
    /*	Cholesky QR2/3	*/
    ierr = MatGetOrdering(Q, MATORDERINGNATURAL, &rowperm, &colperm);CHKERRQ(ierr);
    ierr = MatGetFactor(Q, MATSOLVERPETSC, MAT_FACTOR_LU, &Q);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(Q, Q, rowperm, colperm, &info);CHKERRQ(ierr);
    ierr = MatLUFactorNumeric(Q, Q, &info);CHKERRQ(ierr);
    ierr = MatMatSolve(Q, Q, R);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Q, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Q, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetUnfactored(Q);CHKERRQ(ierr);

    ierr = MatMatMult(Q, Q, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Rhat);CHKERRQ(ierr);
    ierr = MatCholeskyFactor(Rhat, NULL, &info);CHKERRQ(ierr);

    ierr = MatGetFactor(Q, MATSOLVERPETSC, MAT_FACTOR_LU, &Q);CHKERRQ(ierr);
    ierr = MatGetOrdering(Q, MATORDERINGNATURAL, &rowperm, &colperm);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(Q, Q, rowperm, colperm, &info);CHKERRQ(ierr);
    ierr = MatLUFactorNumeric(Q, Q, &info);CHKERRQ(ierr);
    ierr = MatMatSolve(Q, Q, Rhat);CHKERRQ(ierr);
    ierr = MatSetUnfactored(Q);CHKERRQ(ierr);
    //ierr = MatSetUnfactored(Rhat);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic(Rhat, R, PETSC_DEFAULT, &R);CHKERRQ(ierr);
    ierr = MatMatMultNumeric(Rhat, R, R);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  MatView(vecMat, 0);
  MatView(Rhat, 0);
  for (i = 0; i < m; i++) {
    ierr = MatDenseGetColumn(R, i, &colvecs);CHKERRQ(ierr);
    ierr = VecGetArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    for (j = 0; j < n; j++) {
      vecarrs[j] = colvecs[j];
    }
    ierr = VecRestoreArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(R, &colvecs);CHKERRQ(ierr);
  }
  ierr = VecView(vecs[0], 0);CHKERRQ(ierr);
  ierr = ISDestroy(&rowperm);CHKERRQ(ierr);
  ierr = ISDestroy(&colperm);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&Rhat);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&vecMat);CHKERRQ(ierr);
  return ierr;
}

int main(int argc, char **argv)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm, self;
  PetscInt		nvecs = 3, i;
  const PetscInt	n = 3;
  const PetscInt	ix[n] = {0, 1, 2};
  const PetscScalar	vecarray1[n] = {0, 1, 2}, vecarray2[n] = {1, 3, 0}, vecarray3[n] = {4, 0, 2};

  Vec                   vecs[nvecs];

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  self = PETSC_COMM_SELF;

  /* Prep	*/
  ierr = VecCreate(self, &vecs[0]);CHKERRQ(ierr);
  ierr = VecSetSizes(vecs[0], n, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(vecs[0]);CHKERRQ(ierr);
  for (i = 1; i < nvecs; i++) { ierr = VecDuplicate(vecs[0], &vecs[i]);CHKERRQ(ierr);}
  ierr = VecSetValues(vecs[0], n, ix, vecarray1, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValues(vecs[1], n, ix, vecarray2, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValues(vecs[2], n, ix, vecarray3, INSERT_VALUES);CHKERRQ(ierr);

  ierr = shiftedCholeskyQR3(vecs, n, nvecs);CHKERRQ(ierr);

  for (i = 0; i < nvecs; i++) {ierr = VecDestroy(&vecs[i]);CHKERRQ(ierr);}
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
