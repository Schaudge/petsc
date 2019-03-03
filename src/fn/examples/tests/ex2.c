#include <petscfn.h>

const char help[] = "Create, view and test a DAG implementation of PetscFn\n";

/* Scalar example f(x) = sin( || x - y || ^2 ) */

static PetscErrorCode TestVector(PetscRandom rand)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TestScalar(PetscRandom rand)
{
  PetscFn        fn;
  PetscBool      isDag;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnCreate(PETSC_COMM_WORLD, &fn);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(fn, PETSC_DECIDE, 10, 1, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscFnSetType(fn, PETSCFNDAG);CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(fn);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)fn, PETSCFNDAG, &isDag);CHKERRQ(ierr);
  comm = PetscObjectComm((PetscObject)fn);
  if (isDag) {
    PetscInt m, M, n, N, rank;
    PetscFn  fnSin, fnNormSq, fnMat;
    Mat      A;

    ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
    ierr = PetscFnShellCreate(comm, PETSCFNSIN, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, &fnSin);CHKERRQ(ierr);
    ierr = PetscFnSetOptionsPrefix(fnSin, "sin_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(fnSin);CHKERRQ(ierr);
    ierr = PetscFnSetUp(fnSin);CHKERRQ(ierr);
    ierr = PetscFnViewFromOptions(fnSin, NULL, "-fn_view");CHKERRQ(ierr);

    ierr = PetscFnShellCreate(comm, PETSCFNNORMSQUARED, m, n, M, N, NULL, &fnNormSq);CHKERRQ(ierr);
    ierr = PetscFnSetOptionsPrefix(fnNormSq, "normsq_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(fnNormSq);CHKERRQ(ierr);
    ierr = PetscFnSetUp(fnNormSq);CHKERRQ(ierr);
    ierr = PetscFnViewFromOptions(fnNormSq, NULL, "-fn_view");CHKERRQ(ierr);

    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = MatCreateAIJ(comm, rank + 7, rank + 11, PETSC_DETERMINE, PETSC_DETERMINE, 2, NULL, 2, NULL, &A);CHKERRQ(ierr);
    ierr = MatSetRandom(A, rand);CHKERRQ(ierr);
    ierr = PetscFnShellCreate(comm, PETSCFNMAT, rank + 7, rank + 11, PETSC_DETERMINE, PETSC_DETERMINE, (void *) A, &fnMat);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = PetscFnSetOptionsPrefix(fnMat, "mat_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(fnMat);CHKERRQ(ierr);
    ierr = PetscFnSetUp(fnMat);CHKERRQ(ierr);
    ierr = PetscFnViewFromOptions(fnMat, NULL, "-fn_view");CHKERRQ(ierr);

    ierr = PetscFnDestroy(&fnMat);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&fnNormSq);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&fnSin);CHKERRQ(ierr);
  }
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  ierr = PetscFnViewFromOptions(fn, NULL, "-fn_view");CHKERRQ(ierr);
  ierr = PetscFnDestroy(&fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscRandom    rand;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "PetscFn Test Options", "PetscFn");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = TestScalar(rand);CHKERRQ(ierr);
  ierr = TestVector(rand);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr); ierr = PetscFinalize();
  return ierr;
}
