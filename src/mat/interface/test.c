#include <petscmat.h>

PetscErrorCode shiftedCholeskyQR3(Vec vecs[], PetscInt N)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscViewer		viewer;
  PetscInt		i, j, m, M;
  PetscScalar		*colvecs, *vecarrs, *matarray, *matarrayQL, *Rmatarray;
  PetscScalar		shift, norm;
  Mat			X, Q, R, Rhat, QL;
  MatFactorInfo		info;
  IS			rowperm, colperm;

  ierr = PetscObjectGetComm((PetscObject) vecs[0], &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vecs[0], &m);CHKERRQ(ierr);
  PetscPrintf(comm, "N %d m %d\n",N, m);
  ierr = VecGetSize(vecs[0], &M);CHKERRQ(ierr);
  ierr = MatCreateDense(comm, m, PETSC_DETERMINE, PETSC_DETERMINE, N, NULL, &X);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(X, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  for (i = 0; i < N; i++) {
    ierr = MatDenseGetColumn(X, i, &colvecs);CHKERRQ(ierr);
    ierr = VecGetArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    for (j = 0; j < m; j++) {
      colvecs[j] = vecarrs[j];
    }
    ierr = VecRestoreArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(X, &colvecs);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDuplicate(X, MAT_COPY_VALUES, &Q);CHKERRQ(ierr);
  PetscInt qm,qn;
  ierr = MatGetLocalSize(Q, &qm, &qn);CHKERRQ(ierr);
  PetscPrintf(comm, "%d %d\n", qm, qn);
  MatView(X, viewer);

  ierr = MatDenseGetArray(Q, &matarray);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, qm, qn, NULL, &QL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(QL, &matarrayQL);CHKERRQ(ierr);
  for (i = 0; i < N*N; i++) { matarrayQL[i] = matarray[i];}
  ierr = MatDenseRestoreArray(QL, &matarrayQL);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Q, &matarray);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(QL, QL, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Rhat);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Rhat, &Rmatarray);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, Rmatarray, N*N, MPIU_SCALAR, MPI_SUM, comm);CHKERRQ(ierr);
  for (i = 0; i < N*N; i++) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%f\n", Rmatarray[i]);CHKERRQ(ierr);
  }
  //PetscScalarView(N*N, Rmatarray, viewer);
  ierr = MatDenseRestoreArray(Rhat, &Rmatarray);CHKERRQ(ierr);
  MatView(Rhat, 0);
  /*

  ierr = MatNorm(Q, NORM_FROBENIUS, &norm);CHKERRQ(ierr);
  MatView(Q, viewer);
  /*	Initial shift Cholesky
  shift = 11*((M*N)+(N*(N+1)))*PETSC_MACHINE_EPSILON*norm;
  ierr = MatShift(Rhat, shift);CHKERRQ(ierr);
  ierr = MatDuplicate(Rhat, MAT_COPY_VALUES, &R);CHKERRQ(ierr);
  ierr = MatCholeskyFactor(R, NULL, &info);CHKERRQ(ierr);
  ierr = MatSetUnfactored(R);CHKERRQ(ierr);
  for (i = 0; i < 2; i++) {
    /*	Cholesky QR2/3
    ierr = MatGetOrdering(R, MATORDERINGNATURAL, &rowperm, &colperm);CHKERRQ(ierr);
    //ierr = MatLUFactor(, rowperm, colperm, &info);CHKERRQ(ierr);
    ierr = MatMatSolve(Q, Q, R);CHKERRQ(ierr);
    ierr = MatSetUnfactored(Q);CHKERRQ(ierr);

    ierr = MatMatMult(Q, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Rhat);CHKERRQ(ierr);
    ierr = MatGetOrdering(Rhat, MATORDERINGNATURAL, &rowperm, &colperm);CHKERRQ(ierr);
    ierr = MatCholeskyFactor(Rhat, rowperm, &info);CHKERRQ(ierr);

    ierr = MatGetOrdering(Q, MATORDERINGNATURAL, &rowperm, &colperm);CHKERRQ(ierr);
    ierr = MatLUFactor(Q, rowperm, colperm, &info);CHKERRQ(ierr);
    ierr = MatMatSolve(Q, Q, Rhat);CHKERRQ(ierr);
    ierr = MatSetUnfactored(R);CHKERRQ(ierr);
    ierr = MatSetUnfactored(Rhat);CHKERRQ(ierr);
    ierr = MatSetUnfactored(Q);CHKERRQ(ierr);
    ierr = MatMatMult(Rhat, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &R);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  MatView(X, viewer);
  MatView(Q, viewer);
  for (i = 0; i < N; i++) {
    ierr = MatDenseGetColumn(R, i, &colvecs);CHKERRQ(ierr);
    ierr = VecGetArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    for (j = 0; j < m; j++) {
      vecarrs[j] = colvecs[j];
    }
    ierr = VecRestoreArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    ierr = VecNormalize(vecs[i], NULL);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(R, &colvecs);CHKERRQ(ierr);
  }

  ierr = VecView(vecs[0], viewer);CHKERRQ(ierr);
  ierr = ISDestroy(&rowperm);CHKERRQ(ierr);
  ierr = ISDestroy(&colperm);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&Rhat);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
   */
  return ierr;
}

int main(int argc, char **argv)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscInt		nvecs = 3, i;
  const PetscInt	n = 3;
  const PetscInt	ix[n] = {0, 1, 2};
  const PetscScalar	vecarray1[n] = {0, 1, 2}, vecarray2[n] = {1, 3, 0}, vecarray3[n] = {4, 0, 2};
  PetscScalar		vdot;
  Vec                   vecs[nvecs];

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;

  /* Prep	*/
  ierr = VecCreate(comm, &vecs[0]);CHKERRQ(ierr);
  ierr = VecSetType(vecs[0], VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetSizes(vecs[0], PETSC_DETERMINE, n);CHKERRQ(ierr);
  ierr = VecSetUp(vecs[0]);CHKERRQ(ierr);
  for (i = 1; i < nvecs; i++) { ierr = VecDuplicate(vecs[0], &vecs[i]);CHKERRQ(ierr);}
  ierr = VecSetValues(vecs[0], n, ix, vecarray1, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValues(vecs[1], n, ix, vecarray2, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValues(vecs[2], n, ix, vecarray3, INSERT_VALUES);CHKERRQ(ierr);

  ierr = shiftedCholeskyQR3(vecs, nvecs);CHKERRQ(ierr);
  for (i = 0; i < nvecs-1; i++) {
    ierr = VecDot(vecs[i], vecs[i+1], &vdot);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "DOT: %f\n", vdot);CHKERRQ(ierr);
  }
  for (i = 0; i < nvecs; i++) { ierr = VecDestroy(&vecs[i]);CHKERRQ(ierr);}
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
