static char help[] = "Example of using hash table to assemble matrix \n\n";

/*T
   Concepts: Matrix Assembly
   Processors: n
T*/

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             A;
  PetscInt        r,N = 10, start, end, gsize,i;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;
  MPI_Comm        comm;

  ierr = PetscInitialize(&argc, &args, (char*) 0, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MatCreate(comm, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, N, N, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, 3, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL);CHKERRQ(ierr);
  /* Create a linear mesh */
  ierr = MatGetOwnershipRange(A, &start, &end);CHKERRQ(ierr);
  ierr = MatGetSize(A,&gsize,NULL);CHKERRQ(ierr);
  for (r = start; r < end; ++r) {
    if (r == 0) {
      PetscInt    cols[2];
      PetscScalar vals[2];

      cols[0] = r;   cols[1] = r+1;
      vals[0] = 1.0; vals[1] = 1.0;

      ierr = MatSetValues(A, 1, &r, 2, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
    } else if (r == gsize-1) {
      PetscInt    cols[2];
      PetscScalar vals[2];

      cols[0] = r-1; cols[1] = r;
      vals[0] = 1.0; vals[1] = 1.0;

      ierr = MatSetValues(A, 1, &r, 2, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt    cols[3];
      PetscScalar vals[3];

      cols[0] = r-1; cols[1] = r;   cols[2] = r+1;
      vals[0] = 1.0; vals[1] = 1.0; vals[2] = 1.0;

      ierr = MatSetValues(A, 1, &r, 3, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Add to the same location again and again */
  for (i=0; i<10000000; i++) {
    for ( r=0; r<N; r++) {
      PetscInt row = gsize-r-1, col = gsize-r-1;
      PetscScalar val = 10.0;
      ierr = MatSetValues(A, 1, &row, 1, &col, &val, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(A,NULL);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
