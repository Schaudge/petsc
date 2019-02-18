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
    ierr = MatSetSizes(J, n, m, N, M);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, m, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, m, NULL, M - m, NULL);CHKERRQ(ierr);

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
    ierr = MatSetSizes(A, m, n, M, N);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A, n, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A, n, NULL, N - n, NULL);CHKERRQ(ierr);

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
    ierr = MatSetSizes(H, m, m, M, M);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(H, 1, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(H, 1, NULL, 0, NULL);CHKERRQ(ierr);

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

static PetscErrorCode PetscFnScalarApply_Vec(PetscFn fn, Vec x, PetscReal *z)
{
  Vec y, diff;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &y);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &diff);CHKERRQ(ierr);
  ierr = VecWAXPY(diff, -1., y, x);CHKERRQ(ierr);
  ierr = VecDot(diff, diff, z);CHKERRQ(ierr);
  ierr = VecDestroy(&diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Vec(PetscFn fn, Vec x, Vec y)
{
  PetscReal z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarApply_Vec(fn, x, &z);CHKERRQ(ierr);
  ierr = VecSet(y, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarGradient_Vec(PetscFn fn, Vec x, Vec g)
{
  Vec y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &y);CHKERRQ(ierr);
  ierr = VecWAXPY(g, -1., y, x);CHKERRQ(ierr);
  ierr = VecScale(g, 2.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMult_Vec(PetscFn fn, Vec x, Vec xhat, Vec Jxhat)
{
  Vec            g;
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Vec(fn, x, g);CHKERRQ(ierr);
  ierr = VecDot(g, xhat, &z);CHKERRQ(ierr);
  ierr = VecSet(Jxhat, z);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSingletonBcast(Vec v, PetscScalar *zp)
{
  MPI_Comm       comm;
  PetscLayout    map;
  PetscMPIInt    rank;
  PetscInt       broot;
  PetscScalar    z;
  const PetscScalar *zv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)v);
  ierr = VecGetLayout(v, &map);CHKERRQ(ierr);
  ierr = PetscLayoutFindOwner(map, 0, &broot);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v, &zv);CHKERRQ(ierr);
  z    = ((PetscInt) broot == rank) ? zv[0] : 0.;
  ierr = VecRestoreArrayRead(v, &zv);CHKERRQ(ierr);
  ierr = MPI_Bcast(&z, 1, MPIU_REAL, broot, comm);CHKERRQ(ierr);
  *zp  = z;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMultAdjoint_Vec(PetscFn fn, Vec x, Vec v, Vec JTv)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarGradient_Vec(fn, x, JTv);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  ierr = VecScale(JTv, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianCreate_Vec(PetscFn fn, Vec x, Mat J, Mat Jpre)
{
  Vec            g;
  Mat            jac = J ? J : Jpre;
  PetscInt       i, iStart, iEnd;
  const PetscScalar *garray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac) PetscFunctionReturn(0);
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Vec(fn, x, g);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(g, &iStart, &iEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(g,&garray);CHKERRQ(ierr);
  for (i = iStart; i < iEnd; i++) {ierr = MatSetValue(jac, 0, i, garray[i], INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecRestoreArrayRead(g,&garray);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J && J != jac) {ierr = MatCopy(jac, J, SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  if (Jpre && Jpre != jac) {ierr = MatCopy(jac, Jpre, SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianCreateAdjoint_Vec(PetscFn fn, Vec x, Mat J, Mat Jpre)
{
  Vec            g;
  Mat            jac = J ? J : Jpre;
  PetscInt       i, iStart, iEnd;
  const PetscScalar *garray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac) PetscFunctionReturn(0);
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Vec(fn, x, g);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(g, &iStart, &iEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(g,&garray);CHKERRQ(ierr);
  for (i = iStart; i < iEnd; i++) {ierr = MatSetValue(jac, i, 0, garray[i], INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecRestoreArrayRead(g,&garray);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J && J != jac) {ierr = MatCopy(jac, J, SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  if (Jpre && Jpre != jac) {ierr = MatCopy(jac, Jpre, SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarHessianMult_Vec(PetscFn fn, Vec x, Vec xhat, Vec Hxhat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(xhat, Hxhat);CHKERRQ(ierr);
  ierr = VecScale(Hxhat, 2.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarHessianCreate_Vec(PetscFn fn, Vec x, Mat H, Mat Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (H) {ierr = MatShift(H, 2.);CHKERRQ(ierr);}
  if (Hpre && Hpre != H) {ierr = MatShift(Hpre, 2.);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMult_Vec(PetscFn fn, Vec x, Vec v, Vec xhat, Vec vHxhat)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(xhat, vHxhat);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  ierr = VecScale(vHxhat, 2. * z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianCreate_Vec(PetscFn fn, Vec x, Vec v, Mat vH, Mat vHpre)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarHessianCreate_Vec(fn, x, vH, vHpre);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  if (vH) {ierr = MatScale(vH, z);CHKERRQ(ierr);}
  if (vHpre && vHpre != vH) {ierr = MatScale(vHpre, z);CHKERRQ(ierr);}
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
    PetscInt n, N, i, l;
    Vec v;
    void *ctx;
    MPI_Comm comm;
    PetscScalar *a;
    Vec d, r;
      Mat jac, adj, hes;

    comm = PetscObjectComm((PetscObject)fn);
    ierr = PetscFnGetSize(fn, NULL, &N);CHKERRQ(ierr);
    ierr = PetscFnGetLocalSize(fn, NULL, &n);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm,n,N,&v);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(v, &l, NULL);CHKERRQ(ierr);
    ierr = VecGetArray(v, &a);CHKERRQ(ierr);
    for (i = 0; i < n; i++) {
      a[i] = i + l;
    }
    ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
    ierr = PetscFnShellSetContext(fn,(void *) v);CHKERRQ(ierr);
    ierr = PetscFnShellGetContext(fn,(void *) &ctx);CHKERRQ(ierr);
    if ((void *) v != ctx) SETERRQ(comm,PETSC_ERR_PLIB, "Shell context mismatch");
    ierr = PetscObjectReference((PetscObject)v);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_DESTROY,(void (*)(void))PetscFnDestroy_Vec);CHKERRQ(ierr);

    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_CREATEVECS,(void (*)(void))PetscFnCreateVecs_Vec);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn,&d,&r);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_CREATEMATS,(void (*)(void))PetscFnCreateMats_Vec);CHKERRQ(ierr);
    ierr = PetscFnCreateMats(fn, &jac, NULL, &adj, NULL, &hes, NULL);CHKERRQ(ierr);

    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_APPLY,(void (*)(void))PetscFnApply_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_JACOBIANMULT,(void (*)(void))PetscFnJacobianMult_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_JACOBIANMULTADJOINT,(void (*)(void))PetscFnJacobianMultAdjoint_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_JACOBIANCREATE,(void (*)(void))PetscFnJacobianCreate_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_JACOBIANCREATEADJOINT,(void (*)(void))PetscFnJacobianCreateAdjoint_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_HESSIANMULT,(void (*)(void))PetscFnHessianMult_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_HESSIANCREATE,(void (*)(void))PetscFnHessianCreate_Vec);CHKERRQ(ierr);

    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARAPPLY,(void (*)(void))PetscFnScalarApply_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARGRADIENT,(void (*)(void))PetscFnScalarGradient_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARHESSIANMULT,(void (*)(void))PetscFnScalarHessianMult_Vec);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARHESSIANCREATE,(void (*)(void))PetscFnScalarHessianCreate_Vec);CHKERRQ(ierr);

    ierr = MatDestroy(&hes);CHKERRQ(ierr);
    ierr = MatDestroy(&adj);CHKERRQ(ierr);
    ierr = MatDestroy(&jac);CHKERRQ(ierr);
    ierr = VecDestroy(&d);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
  }
  ierr = PetscFnDestroy(&fn);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
