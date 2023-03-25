<<<<<<< HEAD
const char help[] = "Test VecCreateMatDense()\n\n";

=======
>>>>>>> 059356d8ed6 (WIP: gets type:seqdensecuda)
#include <petscdevice_cuda.h>
#include <petscmat.h>
#include <petscconf.h>
#include <assert.h>

int main(int argc, char **args)
{
<<<<<<< HEAD
  Mat      A;
  Vec      X;
  PetscInt N = 20;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Creating Mat from Vec example", NULL);
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscOptionsEnd();

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecSetUp(X));

  PetscCall(VecCreateMatDense(X, PETSC_DECIDE, PETSC_DECIDE, N, N, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  MPI_Comm    X_comm = PetscObjectComm((PetscObject)X);
  MPI_Comm    A_comm = PetscObjectComm((PetscObject)X);
  PetscMPIInt comp;
  PetscCallMPI(MPI_Comm_compare(X_comm, A_comm, &comp));
  PetscAssert(comp == MPI_IDENT || comp == MPI_CONGRUENT, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Failed communicator guarantee in MatCreateDenseMatchingVec()");

  PetscMemType       X_memtype, A_memtype;
  const PetscScalar *array;
  PetscCall(VecGetArrayReadAndMemType(X, &array, &X_memtype));
  PetscCall(VecRestoreArrayReadAndMemType(X, &array));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &array, &A_memtype));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &array));
  PetscAssert(A_memtype == X_memtype, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Failed memtype guarantee in MatCreateDenseMatchingVec()");

  /* test */
  PetscCall(MatViewFromOptions(A, NULL, "-ex19_mat_view"));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
=======
  Mat            A;
  Vec            X;
  PetscInt       N = 20, num_threads = 128;
  PetscLogEvent  event;
  PetscMPIInt    rank, size;
  PetscBool      iscuda = PETSC_FALSE, iship = PETSC_FALSE;
  PetscBool      optionflag, compareflag;
  char           vectypename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_threads", &num_threads, NULL));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-vectype", vectypename, sizeof(vectypename), &optionflag));
  PetscCall(PetscLogEventRegister("GPU operator", MAT_CLASSID, &event));

  if (optionflag) {
    PetscCall(PetscStrncmp(vectypename, "cuda", (size_t)4, &compareflag));
    if (compareflag) iscuda = PETSC_TRUE;
    PetscCall(PetscStrncmp(vectypename, "hip", (size_t)3, &compareflag));
    if (compareflag) iship = PETSC_TRUE;
  }

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, PETSC_DECIDE, N));
  if (iscuda) {
    PetscCall(VecSetType(X, VECCUDA));
  } else if (iship) {
    PetscCall(VecSetType(X, VECHIP));
  } else {
    PetscCall(VecSetType(X, VECSTANDARD));
  }
  PetscCall(VecSetUp(X));
  PetscCall(PetscObjectSetName((PetscObject)X, "X_commworld"));

  PetscCall(MatCreateDenseFromVecType(X, PETSC_DECIDE, PETSC_DECIDE, N, N, NULL, &A));

  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* test */
  PetscCall(MatViewFromOptions(A, NULL, "-ex19_mat_view"));


  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&X));
>>>>>>> 059356d8ed6 (WIP: gets type:seqdensecuda)
  return 0;
}

/*TEST

   test:
      suffix: cuda
      requires: cuda
<<<<<<< HEAD
      args: -vec_type cuda -ex19_mat_view

   test:
      suffix: mpicuda
      requires: cuda
      args: -vec_type mpicuda -ex19_mat_view

   test:
      suffix: hip
      requires: hip
      args: -vec_type hip -ex19_mat_view

   test:
      suffix: standard
      args: -vec_type standard -ex19_mat_view

   test:
      suffix: kokkos_cuda
      requires: kokkos kokkos_kernels cuda
      args: -vec_type kokkos -ex19_mat_view

   test:
      suffix: kokkos_hip
      requires: kokkos kokkos_kernels hip
      args: -vec_type kokkos -ex19_mat_view

   test:
      suffix: kokkos
      requires: kokkos kokkos_kernels !cuda !hip
      args: -vec_type kokkos -ex19_mat_view
=======
      args: -vectype cuda 

>>>>>>> 059356d8ed6 (WIP: gets type:seqdensecuda)
TEST*/
