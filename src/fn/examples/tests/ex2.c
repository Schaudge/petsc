#include <petscfn.h>

const char help[] = "Create, view and test a DAG implementation of PetscFn\n";

static PetscErrorCode PetscFnCreateVecs_Ax(PetscFn fn, Vec *rangeVec, Vec *domainVec)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  if (rangeVec) {
    ierr = MatCreateVecs(A, NULL, rangeVec);CHKERRQ(ierr);
  }
  if (domainVec) {
    Vec x, Avar;
    Vec vecs[2];
    PetscInt m, M, n, N;

    ierr = MatCreateVecs(A, &x, NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)fn), m*N, M*N, &Avar);CHKERRQ(ierr);
    vecs[0] = x;
    vecs[1] = Avar;
    ierr = VecCreateNest(PetscObjectComm((PetscObject)fn), 2, NULL, vecs, domainVec);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&Avar);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Ax(PetscFn fn, Vec ax, Vec y)
{
  Vec *vecs;
  Vec Avar, x;
  PetscInt nvecs = 2;
  const PetscScalar *a;
  Mat A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestGetSubVecs(ax, &nvecs, &vecs);CHKERRQ(ierr);
  x = vecs[0];
  Avar = vecs[1];
  ierr = VecGetArrayRead(Avar, &a);CHKERRQ(ierr);
  ierr = PetscFnShellGetContext(fn, (void *) &A);CHKERRQ(ierr);
  ierr = MatDensePlaceArray(A, a);CHKERRQ(ierr);
  ierr = MatMult(A, x, y);CHKERRQ(ierr);
  ierr = MatDenseResetArray(A);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Avar, &a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDAG(PetscRandom rand)
{
  PetscFn        dag, Ax, soft, misfit, reg;
  PetscBool      isDag;
  MPI_Comm       comm;
  Vec            x, Axvec, Avec, b, y, *vecs;
  PetscInt       Nx, nx, Mb, mb, Nc, Mc, nvecs;
  Mat            A, C;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)rand);

  /* create an input vector x */
  Nx = 37;
  ierr = VecCreateMPI(comm, PETSC_DECIDE, Nx, &x);CHKERRQ(ierr);
  ierr = VecSetRandom(x, rand);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &nx);CHKERRQ(ierr);

  /* create an output vector b */
  Mb = 23;
  ierr = VecCreateMPI(comm, PETSC_DECIDE, Mb, &b);CHKERRQ(ierr);
  ierr = VecSetRandom(b, rand);CHKERRQ(ierr);
  ierr = VecGetLocalSize(b, &mb);CHKERRQ(ierr);

  /* create a dense matrix A */
  ierr = MatCreateDense(comm, mb, nx, Mb, Nx, NULL, &A);CHKERRQ(ierr);

  /* create a fn that multiplies A*x, but takes A and x as inputs, and so
   * uses a VecNest as an input type */
  ierr = PetscFnCreate(comm, &Ax);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(Ax, mb, nx + Nx * mb, Mb, Nx * (1 + Mb));CHKERRQ(ierr);
  ierr = PetscFnSetType(Ax, PETSCFNSHELL);CHKERRQ(ierr);
  ierr = PetscFnShellSetContext(Ax, (void *) A);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(Ax, PETSCFNOP_CREATEVECS, (void (*)(void)) PetscFnCreateVecs_Ax);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(Ax, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Ax);CHKERRQ(ierr);

  ierr = PetscFnCreateVecs(Ax, &y, &Axvec);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(Axvec, &nvecs, &vecs);CHKERRQ(ierr);
  ierr = VecCopy(x, vecs[0]);CHKERRQ(ierr);
  ierr = VecSetRandom(vecs[1], rand);CHKERRQ(ierr);

  ierr = PetscFnApply(Ax, Axvec, y);CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&Axvec);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&Ax);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
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
  ierr = TestDAG(rand);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr); ierr = PetscFinalize();
  return ierr;
}
