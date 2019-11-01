static char help[] = "Generate and execute a PetscSF from an adjacency matrix\n";

#include <petscsys.h>
#include <petscmat.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscMPIInt    rank, size;
  PetscSF        sf;
  PetscInt       nleaves, nroots, globalRows, globalCols;
  PetscInt       *locIdx;
  const PetscInt *cols;
  PetscSFNode    *remIdx;
  Mat            adjMat, adjMatT;
  Vec            locVec;
  PetscScalar    *arr;
  PetscViewer    vwr;
  char           fname[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc,&argv,NULL,help);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscArrayzero(fname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, NULL, "PetscSF generator options", "");CHKERRQ(ierr);
  {
    PetscBool flg = PETSC_FALSE;

    ierr = PetscOptionsString("-filename","File containing adjacency matrix","MatLoad()",fname,fname,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg && !fname[0]) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FILE_READ,"You must indicate an adjacency matrix file to read from");
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = MatCreate(comm,&adjMat);CHKERRQ(ierr);
  ierr = MatSetFromOptions(adjMat);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(comm,fname,FILE_MODE_READ,&vwr);CHKERRQ(ierr);
  ierr = MatLoad(adjMat,vwr);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vwr);CHKERRQ(ierr);
  ierr = MatGetSize(adjMat, &globalRows, &globalCols);CHKERRQ(ierr);
  if (globalRows != globalCols) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Adjacency matrix of size (%D x %D) is not symmetric",globalRows,globalCols);
  if (globalRows != (PetscInt)size) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Adjacency matrix has %D rows/cols but running with %D ranks, must run with the same number of ranks",globalRows,size);
  /*
   Adjacency is encoded by row. There should be sizeof(PETSC_COMM_WORLD) rows in the matrix, such that each rank has a row.

   The index of every nonzero entry per row i is the remote rank that rank i sends to, i.e. the __destination__.

   Then for a given column j, the index of every nonzero row is the rank sending to rank j, i.e. the __source__.
   */

  /* Get number of ranks to send to, i.e. how many root vertices should this process have */
  ierr = MatGetRow(adjMat,rank,&nroots,NULL,NULL);CHKERRQ(ierr);

  /* Create a local vector to serve as this ranks data holder. Make each one sizeof(PETSC_COMM_WORLD) such that there is
enough room to send and receive from every other rank to a unique location */
  ierr = VecCreateSeq(comm,size,&locVec);CHKERRQ(ierr);
  ierr = VecSet(locVec,(PetscScalar)rank);CHKERRQ(ierr);
  ierr = VecSetUp(locVec);CHKERRQ(ierr);

  /* Transpose, now every rows nonzeros tell us who this rank is recieving from, i.e. this ranks leaves */
  ierr = MatTranspose(adjMat,MAT_INITIAL_MATRIX,&adjMatT);CHKERRQ(ierr);
  ierr = MatGetRow(adjMatT,rank,&nleaves,&cols,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc1(nleaves,&remIdx);CHKERRQ(ierr);
  for (PetscInt i = 0; i < nleaves; ++i) {
    /* source rank */
    remIdx[i].rank = cols[i];
    /* location of "data" at the source rank */
    remIdx[i].index = rank;
  }

  ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,remIdx,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sf,NULL,"-sf_view");CHKERRQ(ierr);

  ierr = MatRestoreRow(adjMatT,rank,&nleaves,&cols,NULL);CHKERRQ(ierr);
  ierr = MatRestoreRow(adjMat,rank,&nroots,NULL,NULL);CHKERRQ(ierr);

  ierr = VecViewFromOptions(locVec,NULL,"-vec_view_pre");CHKERRQ(ierr);
  ierr = VecGetArray(locVec,&arr);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_SCALAR,(const void*)arr,(void*)arr);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_SCALAR,(const void*)arr,(void*)arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(locVec,&arr);CHKERRQ(ierr);
  ierr = VecViewFromOptions(locVec,NULL,"-vec_view_post");CHKERRQ(ierr);

  ierr = VecDestroy(&locVec);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = MatDestroy(&adjMat);CHKERRQ(ierr);
  ierr = MatDestroy(&adjMatT);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
