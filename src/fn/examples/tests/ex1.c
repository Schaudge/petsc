#include <petscfn.h>

const char help[] = "Create and view a PetscFn\n";

static PetscErrorCode PetscFnDestroy_Vec(PetscFn fn)
{
  Vec            v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &v);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnCreateVecs_Vec(PetscFn fn, Vec *rangeVec, Vec *domainVec)
{
  PetscInt       m, M, n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSize(fn, &N, &M);CHKERRQ(ierr);
  ierr = PetscFnGetLocalSize(fn, &n, &m);CHKERRQ(ierr);
  if (rangeVec) {
    ierr = VecCreate(PetscObjectComm((PetscObject) fn), rangeVec);CHKERRQ(ierr);
    ierr = VecSetType(*rangeVec, VECSTANDARD);CHKERRQ(ierr);
    ierr = VecSetSizes(*rangeVec, n, N);CHKERRQ(ierr);
  }
  if (domainVec) {
    ierr = VecCreate(PetscObjectComm((PetscObject) fn), domainVec);CHKERRQ(ierr);
    ierr = VecSetType(*domainVec, VECSTANDARD);CHKERRQ(ierr);
    ierr = VecSetSizes(*domainVec, m, M);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnCreateMats_Vec(PetscFn fn, Mat *jac, Mat *jacPre, Mat *adj, Mat *adjPre,
                                            Mat *hes, Mat *hesPre)
{
  PetscInt       m, M, n, N;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSize(fn, &N, &M);CHKERRQ(ierr);
  ierr = PetscFnGetLocalSize(fn, &n, &m);CHKERRQ(ierr);
  comm = PetscObjectComm((PetscObject)fn);
  if (jac || jacPre) {
    Mat J;
    ierr = MatCreate(comm, &J);CHKERRQ(ierr);
    ierr = MatSetType(J, MATAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(J, n, N, m, M);CHKERRQ(ierr);

    if (jac) {
      ierr = PetscObjectReference((PetscObject) J);CHKERRQ(ierr);
      *jac = J;
    }
    if (jacPre) {
      ierr = PetscObjectReference((PetscObject) J);CHKERRQ(ierr);
      *jacPre = J;
    }
    ierr = MatDestroy(&J);CHKERRQ(ierr);
  }
  if (adj || adjPre) {
    Mat A;
    ierr = MatCreate(comm, &A);CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(A, n, N, m, M);CHKERRQ(ierr);

    if (adj) {
      ierr = PetscObjectReference((PetscObject) A);CHKERRQ(ierr);
      *adj = A;
    }
    if (adjPre) {
      ierr = PetscObjectReference((PetscObject) A);CHKERRQ(ierr);
      *adjPre = A;
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  if (hes || hesPre) {
    Mat H;
    ierr = MatCreate(comm, &H);CHKERRQ(ierr);
    ierr = MatSetType(H, MATAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(H, n, N, m, M);CHKERRQ(ierr);

    if (hes) {
      ierr = PetscObjectReference((PetscObject) H);CHKERRQ(ierr);
      *hes = H;
    }
    if (hesPre) {
      ierr = PetscObjectReference((PetscObject) H);CHKERRQ(ierr);
      *hesPre = H;
    }
    ierr = MatDestroy(&H);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscFn        fn;
  PetscBool      isShell;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscFnCreate(PETSC_COMM_WORLD, &fn);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(fn, PETSC_DECIDE, 1, 1, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(fn);CHKERRQ(ierr);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  ierr = PetscFnViewFromOptions(fn, NULL, "-fn_view");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)fn, PETSCFNSHELL, &isShell);CHKERRQ(ierr);
  if (isShell) {
    PetscInt n, N;
    Vec v;
    void *ctx;
    MPI_Comm comm;

    comm = PetscObjectComm((PetscObject)fn);
    ierr = PetscFnGetSize(fn, NULL, &N);CHKERRQ(ierr);
    ierr = PetscFnGetLocalSize(fn, NULL, &n);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm,n,N,&v);CHKERRQ(ierr);
    ierr = PetscFnShellSetContext(fn,(void *) v);CHKERRQ(ierr);
    ierr = PetscFnShellGetContext(fn,(void *) &ctx);CHKERRQ(ierr);
    if ((void *) v != ctx) SETERRQ(comm,PETSC_ERR_PLIB, "Shell context mismatch");
    ierr = PetscObjectReference((PetscObject)v);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DESTROY,(void (*)(void))PetscFnDestroy_Vec);CHKERRQ(ierr);

    {
      Vec d, r;
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_CREATEVECS,(void (*)(void))PetscFnCreateVecs_Vec);CHKERRQ(ierr);
      ierr = PetscFnCreateVecs(fn,&d,&r);CHKERRQ(ierr);
      ierr = VecDestroy(&d);CHKERRQ(ierr);
      ierr = VecDestroy(&r);CHKERRQ(ierr);
    }
    {
      Mat jac, adj, hes;

      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_CREATEMATS,(void (*)(void))PetscFnCreateMats_Vec);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, &jac, NULL, &adj, NULL, &hes, NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&hes);CHKERRQ(ierr);
      ierr = MatDestroy(&adj);CHKERRQ(ierr);
      ierr = MatDestroy(&jac);CHKERRQ(ierr);
    }
  }
  ierr = PetscFnDestroy(&fn);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
