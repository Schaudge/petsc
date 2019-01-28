
static char help[] = "Reads a matrix from PETSc binary file, padds zeros on missing diagonals and saves the matrix into a binary file. \n\n";
/*
 Example:
      ./ex16 -A <matrix file>
 */

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat               A;
  PetscViewer       fd;                        /* viewer */
  char              file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscErrorCode    ierr;
  PetscBool         flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* Determine files from which we read the linear systems. */
  ierr = PetscOptionsGetString(NULL,NULL,"-A",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -A option");

  /* Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /* Load the matrix; then destroy the viewer. */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);
  ierr = MatShift(A,0);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = MatView(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

