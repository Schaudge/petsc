static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -A <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscmat.h>
#include <petscksp.h>
#include <petsctime.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       its,m,n,mvec;
  PetscReal      norm,normb;
  Vec            x,b,u;
  Mat            A;
  KSP            ksp;
  char           file[PETSC_MAX_PATH_LEN];
  char           file2[PETSC_MAX_PATH_LEN];
  char           matname[PETSC_MAX_PATH_LEN],str1[PETSC_MAX_PATH_LEN],str2[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      flg;
  PetscLogDouble vstart,vend;
  KSPConvergedReason reason;
  FILE           *f;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* Read matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-matname",matname,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-str1",str1,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-str2",str2,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(file2,sizeof(file2),"%s_%s_%s.txt",matname,str1,str2);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-A",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -A option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-b",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = VecLoad(b,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  } else {
    ierr = VecSetSizes(b,PETSC_DECIDE,m);CHKERRQ(ierr);
    ierr = VecSet(b,1.0);CHKERRQ(ierr);
  }
  /*
     If the load matrix is larger then the vector, due to being padded
     to match the blocksize then create a new padded vector
  */
//  ierr = VecGetSize(b,&mvec);CHKERRQ(ierr);
//  if (m > mvec) {
//    Vec         tmp;
//    PetscScalar *bold,*bnew;
    /* create a new vector b by padding the old one */
//    ierr = VecCreate(PETSC_COMM_WORLD,&tmp);CHKERRQ(ierr);
//    ierr = VecSetSizes(tmp,PETSC_DECIDE,m);CHKERRQ(ierr);
//    ierr = VecSetFromOptions(tmp);CHKERRQ(ierr);
//    ierr = VecGetArray(tmp,&bnew);CHKERRQ(ierr);
//    ierr = VecGetArray(b,&bold);CHKERRQ(ierr);
//    ierr = PetscMemcpy(bnew,bold,mvec*sizeof(PetscScalar));CHKERRQ(ierr);
//    ierr = VecDestroy(&b);CHKERRQ(ierr);
//    b    = tmp;
//  }

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (!rank) {
    ierr = PetscFOpen(PETSC_COMM_SELF,file2,"w",&f);CHKERRQ(ierr);
  }

  /* Set up solution */
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  /* Solve system */
  ierr = PetscTime(&vstart);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = PetscTime(&vend);CHKERRQ(ierr);

  /* Show result */
  ierr = MatMult(A,x,u);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(b,NORM_2,&normb);CHKERRQ(ierr);

  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
  if (!rank) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,f,"%s %s %s ",matname,str1,str2);CHKERRQ(ierr);
    if (reason < 0) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,f,"%d 0 0",reason);CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(PETSC_COMM_SELF,f,"%lf %.6g %.6g",(double)(vend-vstart),norm,norm/normb);CHKERRQ(ierr);
    }
    ierr = PetscFClose(PETSC_COMM_SELF,f);CHKERRQ(ierr);
  }
  //ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm);CHKERRQ(ierr);

  /* Cleanup */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -ksp_gmres_cgs_refinement_type refine_always -f  ${DATAFILESPATH}/matrices/arco1 -ksp_monitor_short
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)

TEST*/
