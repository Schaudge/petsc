
#include <petscfn.h>

static PetscErrorCode MatDupOrCopy(Mat orig, MatReuse reuse, Mat *dup)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(orig, MAT_COPY_VALUES, dup);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(orig, *dup, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Sin(PetscFn fn, Vec x, Vec y)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
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
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = PetscCosScalar(xs[i]) * xhs[i];
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuild_Sin(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  Vec                diag;
  PetscInt           i, n, m, N, M;
  Mat                *jac = J ? J : Jpre;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!jac) PetscFunctionReturn(0);
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &diag);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(diag, &ys);CHKERRQ(ierr);
  for (i = 0; i < m; i++) ys[i] = PetscCosScalar(xs[i]);
  ierr = VecRestoreArray(diag, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),m,n,M,N,1,NULL,0,NULL,jac);CHKERRQ(ierr);
  }
  ierr = MatDiagonalSet(*jac, diag, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Jpre && Jpre != jac) {
    ierr = MatDupOrCopy(*jac, reuse, Jpre);CHKERRQ(ierr);
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
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
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

static PetscErrorCode PetscFnHessianBuild_Sin(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *H, Mat *Hpre)
{
  const PetscScalar *xs;
  const PetscScalar *xhs;
  PetscScalar       *ys;
  Vec                diag;
  PetscInt           i, m, n, M, N;
  Mat                *hes = H ? H : Hpre;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &diag);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecGetArray(diag, &ys);CHKERRQ(ierr);
  for (i = 0; i < m; i++) ys[i] = -PetscSinScalar(xs[i]) * xhs[i];
  ierr = VecRestoreArray(diag, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),m,n,M,N,1,NULL,0,NULL,hes);CHKERRQ(ierr);
  }
  ierr = MatDiagonalSet(*hes, diag, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*hes, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*hes, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Hpre && Hpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hpre);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Sin(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Sin);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "sin");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Logistic(PetscFn fn, Vec x, Vec y)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecCopy(x, y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) ys[i] = 1. / (1. + PetscExpScalar(-xs[i]));
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMult_Logistic(PetscFn fn, Vec x, Vec xhat, Vec y)
{
  const PetscScalar *xs;
  const PetscScalar *xhs;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    PetscScalar z = PetscExpScalar(xs[i]);

    ys[i] = xhs[i] * (z / ((1.+z)*(1.+z)));
  }
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuild_Logistic(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  const PetscScalar *xs;
  PetscScalar       *ys;
  Vec                diag;
  PetscInt           i, n, m, N, M;
  Mat                *jac = J ? J : Jpre;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!jac) PetscFunctionReturn(0);
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &diag);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArray(diag, &ys);CHKERRQ(ierr);
  for (i = 0; i < m; i++) {
    PetscScalar z = PetscExpScalar(xs[i]);

    ys[i] = (z / ((1.+z)*(1.+z)));
  }
  ierr = VecRestoreArray(diag, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),m,n,M,N,1,NULL,0,NULL,jac);CHKERRQ(ierr);
  }
  ierr = MatDiagonalSet(*jac, diag, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Jpre && Jpre != jac) {
    ierr = MatDupOrCopy(*jac, reuse, Jpre);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMult_Logistic(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec y)
{
  const PetscScalar *xs;
  const PetscScalar *xhs;
  const PetscScalar *xds;
  PetscScalar       *ys;
  PetscInt          i, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSizes(fn, &n, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xdot, &xds);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    PetscScalar z = PetscExpScalar(xs[i]);

    ys[i] = xhs[i] * xds[i] * z*(1.-z)/((1.+z)*(1.+z)*(1.+z));
  }
  ierr = VecRestoreArray(y, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xdot, &xds);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuild_Logistic(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *H, Mat *Hpre)
{
  const PetscScalar *xs;
  const PetscScalar *xhs;
  PetscScalar       *ys;
  Vec                diag;
  PetscInt           i, m, n, M, N;
  Mat                *hes = H ? H : Hpre;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &diag);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecGetArray(diag, &ys);CHKERRQ(ierr);
  for (i = 0; i < m; i++) {
    PetscScalar z = PetscExpScalar(xs[i]);

    ys[i] = xhs[i] * z*(1.-z)/((1.+z)*(1.+z)*(1.+z));
  }
  ierr = VecRestoreArray(diag, &ys);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xhat, &xhs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xs);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),m,n,M,N,1,NULL,0,NULL,hes);CHKERRQ(ierr);
  }
  ierr = MatDiagonalSet(*hes, diag, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*hes, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*hes, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Hpre && Hpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hpre);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Logistic(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Logistic);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "logistic");CHKERRQ(ierr);
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

static PetscErrorCode PetscFnScalarHessianBuild_Normsquared(PetscFn fn, Vec x, MatReuse reuse, Mat *H, Mat *Hpre)
{
  Mat            *hes = H ? H : Hpre;
  PetscInt       m, n, M, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn), m, n, M, N, 1, NULL, 0, NULL, hes);CHKERRQ(ierr);
  }
  ierr = MatShift(*hes, 2.);CHKERRQ(ierr);
  if (Hpre && Hpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate_Normsquared(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_SCALARAPPLY, (void (*)(void)) PetscFnScalarApply_Normsquared);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "normsq");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnApply_Mat(PetscFn fn, Vec x, Vec y)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = MatMult(mat, x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMult_Mat(PetscFn fn, Vec x, Vec xhat, Vec y)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = MatMult(mat, xhat, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianMultAdjoint_Mat(PetscFn fn, Vec x, Vec v, Vec y)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = MatMultTranspose(mat, v, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuild_Mat(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  if (J) {
    ierr = MatDupOrCopy(mat, reuse, J);CHKERRQ(ierr);
  }
  if (Jpre && Jpre != J) {
    ierr = MatDupOrCopy(mat, reuse, Jpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuildAdjoint_Mat(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  if (J) {
    ierr = MatTranspose(mat, reuse, J);CHKERRQ(ierr);
  }
  if (Jpre && Jpre != J) {
    ierr = MatTranspose(mat, reuse, Jpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMult_Mat(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(y, 0.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMultAdjoint_Mat(PetscFn fn, Vec x, Vec v, Vec xhat, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(y, 0.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuild_Mat(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *H, Mat *Hpre)
{
  PetscInt       m, n, M, N;
  Mat            *hes = H ? H : Hpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),m,n,M,N,0,NULL,0,NULL,hes);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(*hes);CHKERRQ(ierr);
  if (Hpre && Hpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuildSwap_Mat(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *H, Mat *Hpre)
{
  PetscInt       m, n, M, N;
  Mat            *hes = H ? H : Hpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),n,m,N,M,0,NULL,0,NULL,hes);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(*hes);CHKERRQ(ierr);
  if (Hpre && Hpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuildAdjoint_Mat(PetscFn fn, Vec x, Vec v, MatReuse reuse, Mat *H, Mat *Hpre)
{
  PetscInt       m, n, M, N;
  Mat            *hes = H ? H : Hpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = PetscFnGetSizes(fn, &m, &n, &M, &N);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)fn),n,n,N,N,0,NULL,0,NULL,hes);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(*hes);CHKERRQ(ierr);
  if (Hpre && Hpre != hes) {
    ierr = MatDupOrCopy(*hes, reuse, Hpre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnView_Mat(PetscFn fn, PetscViewer viewer)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = MatView(mat, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDestroy_Mat(PetscFn fn)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode PetscFnShellCreate_Mat(PetscFn fn)
{
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &mat);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_JACOBIANMULT, (void (*)(void)) PetscFnJacobianMult_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_JACOBIANMULTADJOINT, (void (*)(void)) PetscFnJacobianMultAdjoint_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_JACOBIANBUILD, (void (*)(void)) PetscFnJacobianBuild_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_JACOBIANBUILDADJOINT, (void (*)(void)) PetscFnJacobianBuildAdjoint_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANMULT, (void (*)(void)) PetscFnHessianMult_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANMULTADJOINT, (void (*)(void)) PetscFnHessianMultAdjoint_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANBUILD, (void (*)(void)) PetscFnHessianBuild_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANBUILDADJOINT, (void (*)(void)) PetscFnHessianBuildAdjoint_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_HESSIANBUILDSWAP, (void (*)(void)) PetscFnHessianBuildSwap_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_VIEW, (void (*)(void)) PetscFnView_Mat);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(fn, PETSCFNOP_DESTROY, (void (*)(void)) PetscFnDestroy_Mat);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fn, "mat");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
