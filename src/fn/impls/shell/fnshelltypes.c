
#include <petscfn.h>

static PetscErrorCode PetscFnCreateMats_Componentwise(PetscFn fn, PetscFnOperation op, Mat *A, Mat *Apre)
{
  PetscInt       m, M, n, N;
  MPI_Comm       comm;
  Mat            J;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSize(fn, &M, &N);CHKERRQ(ierr);
  ierr = PetscFnGetLocalSize(fn, &m, &n);CHKERRQ(ierr);
  if (m != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "input and output layouts must be the same");
  comm = PetscObjectComm((PetscObject)fn);
  if (!A && !Apre) PetscFunctionReturn(0);
  ierr = MatCreate(comm, &J);CHKERRQ(ierr);
  ierr = MatSetType(J, MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(J, m, n, M, N);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J, 1, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J, 1, NULL, 0, NULL);CHKERRQ(ierr);
  if (A) {
    ierr = PetscObjectReference((PetscObject) J);CHKERRQ(ierr);
    *A = J;
  }
  if (Apre) {
    ierr = PetscObjectReference((PetscObject) J);CHKERRQ(ierr);
    *Apre = J;
  }
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Sin(PetscFn fn, Vec x, Vec y)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetLocalSize(fn, &n, NULL);CHKERRQ(ierr);
  ierr = VecCopy(x, y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = PetscSinScalar(xs[i]);
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMult_Sin(PetscFn fn, Vec x, Vec xhat, Vec y)
{
  const PetscScalar *xs;
  const PetscScalar *xhs;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetLocalSize(fn, &n, NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = PetscCosScalar(xs[i]) * xhs[i];
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuild_Sin(PetscFn fn, Vec x, Mat J, Mat Jpre)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  Vec                diag;
  PetscInt           i, n;
  Mat                jac = J ? J : Jpre;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!jac) PetscFunctionReturn(0);
  ierr = PetscFnGetLocalSize(fn, &n, NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &diag);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(diag, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = PetscCosScalar(xs[i]);
  ierr = VecRestoreArray(diag, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = MatDiagonalSet(jac, diag, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Jpre && Jpre != jac) {
    ierr = MatDiagonalSet(Jpre, diag, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMult_Sin(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec y)
{
  const PetscScalar *xs;
  const PetscScalar *xhs;
  const PetscScalar *xds;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetLocalSize(fn, &n, NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xdot, &xds);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = -PetscSinScalar(xs[i]) * xhs[i] * xds[i];
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xdot, &xds);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuild_Sin(PetscFn fn, Vec x, Vec xhat, Mat H, Mat Hpre)
{
  const PetscScalar *xs;
  const PetscScalar *xhs;
  PetscScalar       *ys;
  Vec                diag;
  PetscInt           i, n;
  Mat                hes = H ? H : Hpre;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = PetscFnGetLocalSize(fn, &n, NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &diag);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecGetArray(diag, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = -PetscSinScalar(xs[i]) * xhs[i];
  ierr = VecRestoreArray(diag, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = MatDiagonalSet(hes, diag, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(hes, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(hes, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Hpre && Hpre != hes) {
    ierr = MatDiagonalSet(Hpre, diag, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Sin(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_CREATEMATS, (void (*)(void)) PetscFnCreateMats_Componentwise);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_JACOBIANMULT, (void (*)(void)) PetscFnJacobianMult_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_JACOBIANMULTADJOINT, (void (*)(void)) PetscFnJacobianMult_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_JACOBIANBUILD, (void (*)(void)) PetscFnJacobianBuild_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_JACOBIANBUILDADJOINT, (void (*)(void)) PetscFnJacobianBuild_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANMULT, (void (*)(void)) PetscFnHessianMult_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANMULTADJOINT, (void (*)(void)) PetscFnHessianMult_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANBUILD, (void (*)(void)) PetscFnHessianBuild_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANBUILDADJOINT, (void (*)(void)) PetscFnHessianBuild_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANBUILDSWAP, (void (*)(void)) PetscFnHessianBuild_Sin);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "sin()");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnCreateMats_Normsquared(PetscFn fn, PetscFnOperation op, Mat *A, Mat *Apre)
{
  PetscInt       m, M, n, N;
  MPI_Comm       comm;
  Mat            J;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSize(fn, &M, &N);CHKERRQ(ierr);
  ierr = PetscFnGetLocalSize(fn, &m, &n);CHKERRQ(ierr);
  if (M != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "output must be scalar");
  comm = PetscObjectComm((PetscObject)fn);
  if (!A && !Apre) PetscFunctionReturn(0);
  ierr = MatCreate(comm, &J);CHKERRQ(ierr);
  ierr = MatSetType(J, MATAIJ);CHKERRQ(ierr);
  if (op == PETSCFNOP_JACOBIANBUILD || op == PETSCFNOP_HESSIANBUILD) {
    ierr = MatSetSizes(J, m, n, M, N);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, n, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, n, NULL, N - n, NULL);CHKERRQ(ierr);
  }
  else if (op == PETSCFNOP_JACOBIANBUILDADJOINT || op == PETSCFNOP_HESSIANBUILDSWAP) {
    ierr = MatSetSizes(J, n, m, N, M);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, 1, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, 1, NULL, 0, NULL);CHKERRQ(ierr);
  }
  else if (op == PETSCFNOP_SCALARHESSIANBUILD || op == PETSCFNOP_HESSIANBUILDADJOINT)  {
    ierr = MatSetSizes(J, n, n, N, N);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, 1, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, 1, NULL, 0, NULL);CHKERRQ(ierr);
  }
  if (A) {
    ierr = PetscObjectReference((PetscObject) J);CHKERRQ(ierr);
    *A = J;
  }
  if (Apre) {
    ierr = PetscObjectReference((PetscObject) J);CHKERRQ(ierr);
    *Apre = J;
  }
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarApply_Normsquared(PetscFn fn, Vec x, PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot(x, x, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarGradient_Normsquared(PetscFn fn, Vec x, Vec g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(x, g);CHKERRQ(ierr);
  ierr = VecScale(g, 2.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarHessianMult_Normsquared(PetscFn fn, Vec x, Vec xhat, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(xhat, y);CHKERRQ(ierr);
  ierr = VecScale(y, 2.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarHessianBuild_Normsquared(PetscFn fn, Vec x, Mat H, Mat Hpre)
{
  Mat            hes = H ? H : Hpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = MatShift(hes, 2.);CHKERRQ(ierr);
  if (Hpre && Hpre != H) {
    ierr = MatShift(Hpre, 2.);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Normsquared(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_CREATEMATS, (void (*)(void)) PetscFnCreateMats_Normsquared);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_SCALARAPPLY, (void (*)(void)) PetscFnScalarApply_Normsquared);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_SCALARGRADIENT, (void (*)(void)) PetscFnScalarGradient_Normsquared);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_SCALARHESSIANMULT, (void (*)(void)) PetscFnScalarHessianMult_Normsquared);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_SCALARHESSIANBUILD, (void (*)(void)) PetscFnScalarHessianBuild_Normsquared);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "norm()^2");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
