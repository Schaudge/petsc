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
    PetscInt m, M, n, N;
    PetscFn  fnSin, fnNormSq;

    ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
    ierr = PetscFnShellCreate(comm, PETSCSIN, PETSC_DECIDE, 1, PETSC_DECIDE, 1, NULL, &fnSin);CHKERRQ(ierr);
    ierr = PetscFnSetOptionsPrefix(fnSin, "sin_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(fnSin);CHKERRQ(ierr);
    ierr = PetscFnSetUp(fnSin);CHKERRQ(ierr);
    ierr = PetscFnViewFromOptions(fnSin, NULL, "-fn_view");CHKERRQ(ierr);

    ierr = PetscFnShellCreate(comm, PETSCNORMSQUARED, m, M, n, N, NULL, &fnNormSq);CHKERRQ(ierr);
    ierr = PetscFnSetOptionsPrefix(fnNormSq, "normsq_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(fnNormSq);CHKERRQ(ierr);
    ierr = PetscFnSetUp(fnNormSq);CHKERRQ(ierr);
    ierr = PetscFnViewFromOptions(fnNormSq, NULL, "-fn_view");CHKERRQ(ierr);

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
