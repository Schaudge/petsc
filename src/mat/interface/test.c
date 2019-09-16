#include <petscmat.h>

PetscErrorCode PMatView(Mat mat, const char *name, PetscViewer viewer)
{
  PetscErrorCode	ierr;
  PetscScalar		*array;
  PetscInt		m, n;

  ierr = MatGetSize(mat, &m, &n);CHKERRQ(ierr);
  ierr = MatDenseGetArray(mat, &array);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%s\n", name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscScalarView(m*n, array, viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(mat, &array);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode MatNormLocal(Mat mat, PetscInt N, Mat *workmat)
{
  PetscErrorCode	ierr;
  MPI_Comm		comm;
  PetscScalar		*array, *workarray;
  PetscInt		M, i;
  Mat			temp;

  ierr = PetscObjectGetComm((PetscObject) mat, &comm);CHKERRQ(ierr);
  ierr = MatDenseGetArray(mat, &array);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat, &M, NULL);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, M, N, NULL, &temp);CHKERRQ(ierr);
  ierr = MatDenseGetArray(temp, &workarray);CHKERRQ(ierr);
  for (i = 0; i < N*N; i++) { workarray[i] = array[i];}
  ierr = MatDenseRestoreArray(temp, &workarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(mat, &array);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(temp, temp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, workmat);CHKERRQ(ierr);
  ierr = MatDestroy(&temp);CHKERRQ(ierr);
  ierr = MatDenseGetArray(*workmat, &workarray);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, workarray, N*N, MPIU_SCALAR, MPI_SUM, comm);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(*workmat, &workarray);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*workmat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*workmat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return ierr;
}
PetscErrorCode shiftedCholeskyQR3(Vec vecs[], PetscInt N)
{
  PetscErrorCode	ierr;
  MPI_Comm		comm;
  PetscViewer		viewer;
  PetscInt		i, j, m, M;
  PetscScalar		*colvecs, *vecarrs, *matarray, *Qmatarray;
  PetscScalar		shift, norm;
  Mat			X, Q, R, Rinv, QL, QNew, Ident;
  MatFactorInfo		info;
  IS			rows, cols;

  ierr = PetscObjectGetComm((PetscObject) vecs[0], &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vecs[0], &m);CHKERRQ(ierr);
  PetscPrintf(comm, "N %d m %d\n",N, m);
  ierr = VecGetSize(vecs[0], &M);CHKERRQ(ierr);
  ierr = MatCreateDense(comm, m, PETSC_DETERMINE, PETSC_DETERMINE, N, NULL, &X);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, N, N, NULL, &Ident);CHKERRQ(ierr);
  ierr = MatShift(Ident, 1.0);CHKERRQ(ierr);
  ierr = MatDuplicate(Ident, MAT_COPY_VALUES, &Rinv);CHKERRQ(ierr);
  ierr = MatZeroEntries(Rinv);CHKERRQ(ierr);

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
  ierr = MatNorm(X, NORM_FROBENIUS, &norm);CHKERRQ(ierr);
  ierr = MatNormLocal(Q, N, &R);CHKERRQ(ierr);
  PetscPrintf(comm, "\nR MAT\n");
  MatView(R, viewer);
  /*	Initial shift Cholesky	*/
  shift = 11*((M*N)+(N*(N+1)))*PETSC_MACHINE_EPSILON*13.65;
  ierr = MatShift(R, shift);CHKERRQ(ierr);
  for (i = 0; i < 2; i++) {
    /*	Cholesky QR2/3	*/
    ierr = MatGetOwnershipIS(R, &rows, &cols);CHKERRQ(ierr);
    ierr = MatCholeskyFactor(R, cols, &info);CHKERRQ(ierr);
    /*	Invert	*/
    ierr = MatMatSolveTranspose(R, Ident, Rinv);CHKERRQ(ierr);
    ierr = MatSetUnfactored(R);CHKERRQ(ierr);
    PetscPrintf(comm, "\nR MAT INSIDE CHOL\n");
    MatView(R, viewer);
    PetscPrintf(comm, "\nRINV INSIDE CHOL-----------------\n");
    MatView(Rinv, viewer);

    ierr = MatZeroEntries(R);CHKERRQ(ierr);
    /* Update	*/
    ierr = MatDenseGetLocalMatrix(Q, &QL);CHKERRQ(ierr);

    PetscPrintf(comm, "QL, RINV---------------------\n");
    MatView(QL, viewer);
    MatView(Rinv, viewer);

    ierr = MatMatMult(QL, Rinv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &QNew);CHKERRQ(ierr);
    ierr = MatDenseGetArray(QNew, &matarray);CHKERRQ(ierr);
    ierr = MatDenseGetArray(Q, &Qmatarray);CHKERRQ(ierr);
    for (j = 0; j < N*N; j++) {
      Qmatarray[j] = matarray[j];
    }
    ierr = MatDenseRestoreArray(Q, &Qmatarray);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(QNew, &matarray);CHKERRQ(ierr);
    PetscPrintf(comm, "=================QNEW====================\n");
    MatView(QL, viewer);
    PetscPrintf(comm, "-------------\nQ MAT\n");
    MatView(Q, viewer);
    PetscPrintf(comm, "-------------\n");
    ierr = MatNormLocal(Q, N, &R);CHKERRQ(ierr);
  }
  MatView(X, viewer);
  for (i = 0; i < N; i++) {
    ierr = MatDenseGetColumn(Q, i, &colvecs);CHKERRQ(ierr);
    ierr = VecGetArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    for (j = 0; j < m; j++) {
      vecarrs[j] = colvecs[j];
    }
    ierr = VecRestoreArray(vecs[i], &vecarrs);CHKERRQ(ierr);
    ierr = VecNormalize(vecs[i], NULL);CHKERRQ(ierr);
    VecView(vecs[i], viewer);
    ierr = MatDenseRestoreColumn(Q, &colvecs);CHKERRQ(ierr);
  }
  MatView(Q, viewer);
  ierr = MatDestroy(&Ident);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}

int main(int argc, char **argv)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscInt		nvecs = 3, i;
  const PetscInt	n = 3;
  const PetscInt	ix[n] = {0, 1, 2};
  //const PetscScalar	vecarray1[n] = {0, 1, 2}, vecarray2[n] = {1, 3, 0}, vecarray3[n] = {4, 0, 2};
  const PetscScalar	vecarray1[n] = {1, 2, 1}, vecarray2[n] = {0, 2, 0}, vecarray3[n] = {1, 0, 3};
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
  for (i = 0; i < nvecs; i++) {
    ierr = VecAssemblyBegin(vecs[i]);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(vecs[i]);CHKERRQ(ierr);
   }

  ierr = shiftedCholeskyQR3(vecs, nvecs);CHKERRQ(ierr);
  for (i = 0; i < nvecs-1; i++) {
    VecView(vecs[i], 0);
    ierr = VecDot(vecs[i], vecs[i+1], &vdot);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "DOT: %f\n", vdot);CHKERRQ(ierr);
  }
  for (i = 0; i < nvecs; i++) { ierr = VecDestroy(&vecs[i]);CHKERRQ(ierr);}
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
