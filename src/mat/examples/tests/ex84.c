
static char help[] = "Test MatCoarsen.\n\n";

/*T
   Concepts: coarsening
   Processors: 4
T*/

/*
  Include "petscmat.h" so that we can use matrices.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscmat.h>
#include <petsc/private/matimpl.h>

int main(int argc,char **args)
{
  Mat             A;
  PetscErrorCode  ierr;
  PetscMPIInt     rank,size;
  PetscInt        *ia,*ja,nloc = 4,Istart,Iend,i;
  PetscReal       *a;
  PetscInt        *permute;
  PetscBool       *bIndexSet;
  MatCoarsen      crs;
  PetscReal       hashfact;
  PetscRandom     random;
  IS              perm;
  MPI_Comm        comm;
  PetscCoarsenData *agg_lists;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size != 4) SETERRQ(comm,1,"Must run with 4 processors");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscMalloc1(5,&ia);CHKERRQ(ierr);
  ierr = PetscMalloc1(16,&ja);CHKERRQ(ierr);
  ierr = PetscMalloc1(16,&a);CHKERRQ(ierr);
  for (i=0; i<16; i++) a[i] = (PetscReal)rank;
  if (!rank) {
    ja[0] = 1; ja[1] = 4; ja[2] = 0; ja[3] = 2; ja[4] = 5; ja[5] = 1; ja[6] = 3; ja[7] = 6;
    ja[8] = 2; ja[9] = 7;
    ia[0] = 0; ia[1] = 2; ia[2] = 5; ia[3] = 8; ia[4] = 10;
  } else if (rank == 1) {
    ja[0] = 0; ja[1] = 5; ja[2] = 8; ja[3] = 1; ja[4] = 4; ja[5] = 6; ja[6] = 9; ja[7] = 2;
    ja[8] = 5; ja[9] = 7; ja[10] = 10; ja[11] = 3; ja[12] = 6; ja[13] = 11;
    ia[0] = 0; ia[1] = 3; ia[2] = 7; ia[3] = 11; ia[4] = 14;
  } else if (rank == 2) {
    ja[0] = 4; ja[1] = 9; ja[2] = 12; ja[3] = 5; ja[4] = 8; ja[5] = 10; ja[6] = 13; ja[7] = 6;
    ja[8] = 9; ja[9] = 11; ja[10] = 14; ja[11] = 7; ja[12] = 10; ja[13] = 15;
    ia[0] = 0; ia[1] = 3; ia[2] = 7; ia[3] = 11; ia[4] = 14;
  } else {
    ja[0] = 8; ja[1] = 13; ja[2] = 9; ja[3] = 12; ja[4] = 14; ja[5] = 10; ja[6] = 13; ja[7] = 15;
    ja[8] = 11; ja[9] = 14;
    ia[0] = 0; ia[1] = 2; ia[2] = 5; ia[3] = 8; ia[4] = 10;
  }
  ierr = MatCreateMPIAIJWithArrays(comm,4,4,PETSC_DETERMINE,PETSC_DETERMINE,ia,ja,a,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* get MIS aggs - randomize */
  ierr = PetscMalloc1(nloc,&permute);CHKERRQ(ierr);
  ierr = PetscCalloc1(nloc,&bIndexSet);CHKERRQ(ierr);
  for (i=0; i<nloc; i++) {
    permute[i] = i;
  }
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&random);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=0; i<nloc; i++) {
    ierr = PetscRandomGetValueReal(random,&hashfact);CHKERRQ(ierr);
    PetscInt iSwapIndex = (PetscInt) (hashfact*nloc)%nloc;
    if (!bIndexSet[iSwapIndex] && iSwapIndex != i) {
      PetscInt iTemp        = permute[iSwapIndex];
      permute[iSwapIndex]   = permute[i];
      permute[i]            = iTemp;
      bIndexSet[iSwapIndex] = PETSC_TRUE;
    }
  }
  ierr = PetscFree(bIndexSet);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&random);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nloc,permute,PETSC_USE_POINTER,&perm);CHKERRQ(ierr);

  ierr = MatCoarsenCreate(comm,&crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetFromOptions(crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetGreedyOrdering(crs,perm);CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency(crs,A);CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs(crs,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatCoarsenApply(crs);CHKERRQ(ierr);
  ierr = MatCoarsenGetData(crs,&agg_lists);CHKERRQ(ierr);
  ierr = MatCoarsenDestroy(&crs);CHKERRQ(ierr);

  for (i=0; i<nloc; i++) {
    PetscCDIntNd *pos;
    ierr = PetscCDGetHeadPos(agg_lists,i,&pos);CHKERRQ(ierr);
    while (pos) {
      PetscInt gid1;
      ierr = PetscCDIntNdGetID(pos,&gid1);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %D selected: lid %D pos %D\n",rank,i,gid1);CHKERRQ(ierr);
      ierr = PetscCDGetNextPos(agg_lists,i,&pos);CHKERRQ(ierr);
    }
  }

  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = PetscFree(permute);CHKERRQ(ierr);
  ierr = PetscFree(ia);CHKERRQ(ierr);
  ierr = PetscFree(ja);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 4

TEST*/
