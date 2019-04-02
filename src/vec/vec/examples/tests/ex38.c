static const char help[] = "Test VecGetSubVector()\n\n";

#include <petscvec.h>

int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Vec            X,Y,Z,W;
  PetscMPIInt    rank,size;
  PetscInt       i,rstart,rend,idxs[3];
  PetscScalar    *x;
  PetscViewer    viewer;
  IS             is0,is1,is2;
  PetscBool      test_create, test_nest;
  PetscErrorCode ierr;

  ierr   = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  test_create = PETSC_FALSE;
  ierr   = PetscOptionsGetBool(NULL,NULL,"-test_create",&test_create,NULL);CHKERRQ(ierr);
  test_nest = PETSC_FALSE;
  ierr   = PetscOptionsGetBool(NULL,NULL,"-test_nest",&test_nest,NULL);CHKERRQ(ierr);
  comm   = PETSC_COMM_WORLD;
  viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr   = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = VecCreate(comm,&X);CHKERRQ(ierr);
  ierr = VecSetSizes(X,10,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(X);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&rstart,&rend);CHKERRQ(ierr);

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (i=0; i<rend-rstart; i++) x[i] = rstart+i;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  idxs[0] = (size - rank - 1)*10 + 5;
  idxs[1] = (size - rank - 1)*10 + 2;
  idxs[2] = (size - rank - 1)*10 + 3;

  ierr = ISCreateStride(comm,(rend-rstart)/3+3*(rank>size/2),rstart,1,&is0);CHKERRQ(ierr);
  ierr = ISComplement(is0,rstart,rend,&is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,3,idxs,PETSC_USE_POINTER,&is2);CHKERRQ(ierr);

  ierr = ISView(is0,viewer);CHKERRQ(ierr);
  ierr = ISView(is1,viewer);CHKERRQ(ierr);
  ierr = ISView(is2,viewer);CHKERRQ(ierr);

  if (test_nest) {
    Vec subvecs[2];
    IS  subsets[2];
    Vec Xnest;

    ierr = VecCreateSubVector(X,is0,&subvecs[0]);CHKERRQ(ierr);
    ierr = VecCreateSubVector(X,is1,&subvecs[1]);CHKERRQ(ierr);
    subsets[0] = is0;
    subsets[1] = is1;
    ierr = VecCreateNest(comm, 2, subsets, subvecs, &Xnest);CHKERRQ(ierr);
    ierr = VecDestroy(&subvecs[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&subvecs[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
    X = Xnest;
  }

  if (!test_create) {
    ierr = VecGetSubVector(X,is0,&Y);CHKERRQ(ierr);
    ierr = VecGetSubVector(X,is1,&Z);CHKERRQ(ierr);
    ierr = VecGetSubVector(X,is2,&W);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSubVector(X,is0,&Y);CHKERRQ(ierr);
    ierr = VecCreateSubVector(X,is1,&Z);CHKERRQ(ierr);
    ierr = VecCreateSubVector(X,is2,&W);CHKERRQ(ierr);
  }
  ierr = VecView(Y,viewer);CHKERRQ(ierr);
  ierr = VecView(Z,viewer);CHKERRQ(ierr);
  ierr = VecView(W,viewer);CHKERRQ(ierr);
  if (!test_create) {
    ierr = VecRestoreSubVector(X,is0,&Y);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(X,is1,&Z);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(X,is2,&W);CHKERRQ(ierr);
  } else {
    ierr = VecDestroy(&Y);CHKERRQ(ierr);
    ierr = VecDestroy(&Z);CHKERRQ(ierr);
    ierr = VecDestroy(&W);CHKERRQ(ierr);
  }


  ierr = ISDestroy(&is0);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   testset:
      nsize: 3
      output_file: output/ex38_1.out
      filter: grep -v "  type:"
      test:
        suffix: standard
        args: -vec_type standard
      test:
        requires: cuda
        suffix: cuda
        args: -vec_type cuda
      test:
        requires: viennacl
        suffix:  viennacl
        args: -vec_type viennacl
      test:
        suffix: create
        args: -vec_type standard -test_create
      test:
        suffix: nest
        args: -vec_type standard -test_nest
      test:
        suffix: nest_create
        args: -vec_type standard -test_nest -test_create

TEST*/
