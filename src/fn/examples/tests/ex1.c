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

static PetscErrorCode PetscFnCreateMats_Vec(PetscFn fn, PetscFnOperation op,Mat *A, Mat *Apre)
{
  PetscInt       m, M, n, N;
  MPI_Comm       comm;
  Mat            J;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnGetSize(fn, &N, &M);CHKERRQ(ierr);
  ierr = PetscFnGetLocalSize(fn, &n, &m);CHKERRQ(ierr);
  comm = PetscObjectComm((PetscObject)fn);
  if (!A && !Apre) PetscFunctionReturn(0);
  ierr = MatCreate(comm, &J);CHKERRQ(ierr);
  ierr = MatSetType(J, MATAIJ);CHKERRQ(ierr);
  if (op == PETSCFNOP_JACOBIANBUILD) {
    ierr = MatSetSizes(J, n, m, N, M);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, m, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, m, NULL, M - m, NULL);CHKERRQ(ierr);
  }
  if (op == PETSCFNOP_JACOBIANBUILDADJOINT) {
    ierr = MatSetSizes(J, m, n, M, N);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, n, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, n, NULL, N - n, NULL);CHKERRQ(ierr);
  }
  if (op == PETSCFNOP_HESSIANBUILD) {
    ierr = MatSetSizes(J, n, m, N, M);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, m, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, m, NULL, M - m, NULL);CHKERRQ(ierr);
  }
  if (op == PETSCFNOP_HESSIANBUILDSWAP) {
    ierr = MatSetSizes(J, m, n, M, N);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, n, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, n, NULL, N - n, NULL);CHKERRQ(ierr);
  }
  if (op == PETSCFNOP_HESSIANBUILDADJOINT || op == PETSCFNOP_SCALARHESSIANBUILD) {
    ierr = MatSetSizes(J, m, m, M, M);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, m, NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(J, m, NULL, M - m, NULL);CHKERRQ(ierr);
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

static PetscErrorCode PetscFnScalarApply_Vec_Internal(PetscFn fn, Vec x, PetscReal *z)
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

static PetscErrorCode PetscFnScalarApply_Vec(PetscFn fn, Vec x, PetscReal *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarApply_Vec_Internal(fn, x, z);CHKERRQ(ierr);
  *z   = PetscSinScalar(*z);CHKERRQ(ierr);
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

static PetscErrorCode PetscFnScalarGradient_Vec_Internal(PetscFn fn, Vec x, Vec g)
{
  Vec            y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &y);CHKERRQ(ierr);
  ierr = VecWAXPY(g, -1., y, x);CHKERRQ(ierr);
  ierr = VecScale(g, 2.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarGradient_Vec(PetscFn fn, Vec x, Vec g)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarApply_Vec_Internal(fn, x, &z);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Vec_Internal(fn, x, g);CHKERRQ(ierr);
  ierr = VecScale(g, PetscCosScalar(z));CHKERRQ(ierr);
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

static PetscErrorCode PetscFnJacobianMultAdjoint_Vec(PetscFn fn, Vec x, Vec v, Vec Jadjv)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarGradient_Vec(fn, x, Jadjv);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  ierr = VecScale(Jadjv, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecToMat_Internal(Vec g, Mat jac, PetscBool col)
{
  PetscInt           i, iStart, iEnd;
  const PetscScalar *garray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(g, &iStart, &iEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(g,&garray);CHKERRQ(ierr);
  ierr = MatSetUp(jac);CHKERRQ(ierr);
  if (col) {
    for (i = iStart; i < iEnd; i++) {ierr = MatSetValue(jac, i, 0, garray[i-iStart], INSERT_VALUES);CHKERRQ(ierr);}
  } else {
    for (i = iStart; i < iEnd; i++) {ierr = MatSetValue(jac, 0, i, garray[i-iStart], INSERT_VALUES);CHKERRQ(ierr);}
  }
  ierr = VecRestoreArrayRead(g,&garray);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuild_Vec(PetscFn fn, Vec x, Mat J, Mat Jpre)
{
  Vec            g;
  Mat            jac = J ? J : Jpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac) PetscFunctionReturn(0);
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Vec(fn, x, g);CHKERRQ(ierr);
  ierr = VecToMat_Internal(g, jac, PETSC_FALSE);CHKERRQ(ierr);
  if (J && J != jac) {ierr = MatCopy(jac, J, SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  if (Jpre && Jpre != jac) {ierr = MatCopy(jac, Jpre, SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnJacobianBuildAdjoint_Vec(PetscFn fn, Vec x, Mat Jadj, Mat Jadjpre)
{
  Vec            g;
  Mat            jacadj = Jadj ? Jadj : Jadjpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jacadj) PetscFunctionReturn(0);
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Vec(fn, x, g);CHKERRQ(ierr);
  ierr = VecToMat_Internal(g, jacadj, PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  if (Jadj && Jadj != jacadj) {ierr = MatCopy(jacadj, Jadj, SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  if (Jadjpre && Jadjpre != jacadj) {ierr = MatCopy(jacadj, Jadjpre, SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarHessianMult_Vec(PetscFn fn, Vec x, Vec xhat, Vec Hxhat)
{
  Vec            g;
  PetscScalar    z, dot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarApply_Vec_Internal(fn, x, &z);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Vec_Internal(fn, x, g);CHKERRQ(ierr);
  ierr = VecCopy(xhat, Hxhat);CHKERRQ(ierr);
  ierr = VecDot(xhat, g, &dot);CHKERRQ(ierr);
  ierr = VecScale(Hxhat, 2. * PetscCosScalar(z));CHKERRQ(ierr);
  ierr = VecAXPY(Hxhat, - dot * PetscSinScalar(z), g);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarHessianBuild_Vec(PetscFn fn, Vec x, Mat H, Mat Hpre)
{
  Vec            g;
  PetscScalar    z;
  Mat            hes = H ? H : Hpre;
  Mat            Jadj, J, JadjJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!hes) PetscFunctionReturn(0);
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = PetscFnScalarApply_Vec_Internal(fn, x, &z);CHKERRQ(ierr);
  ierr = PetscFnScalarGradient_Vec_Internal(fn, x, g);CHKERRQ(ierr);
  ierr = PetscFnCreateMats(fn, PETSCFNOP_JACOBIANBUILDADJOINT, &Jadj, NULL);CHKERRQ(ierr);
  ierr = VecToMat_Internal(g, Jadj, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatTranspose(Jadj, MAT_INITIAL_MATRIX, &J);CHKERRQ(ierr);
  ierr = MatMatMult(Jadj,J,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&JadjJ);CHKERRQ(ierr);
  ierr = MatSetUp(hes);CHKERRQ(ierr);
  ierr = MatCopy(JadjJ, hes, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatScale(hes, -PetscSinScalar(z));CHKERRQ(ierr);
  ierr = MatShift(hes, 2. * PetscCosScalar(z));CHKERRQ(ierr);
  ierr = MatDestroy(&JadjJ);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&Jadj);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  if (H && H != hes) {
    ierr = MatSetUp(H);CHKERRQ(ierr);
    ierr = MatCopy(hes, H, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  if (Hpre && Hpre != hes) {
    ierr = MatSetUp(Hpre);CHKERRQ(ierr);
    ierr = MatCopy(hes, Hpre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMult_Vec(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec Hxhatxdot)
{
  PetscScalar    z;
  Vec            Hxhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(xhat, &Hxhat);CHKERRQ(ierr);
  ierr = PetscFnScalarHessianMult_Vec(fn, x, xhat, Hxhat);CHKERRQ(ierr);
  ierr = VecDot(Hxhat, xdot, &z);CHKERRQ(ierr);
  ierr = VecSet(Hxhatxdot, z);CHKERRQ(ierr);
  ierr = VecDestroy(&Hxhat);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianMultAdjoint_Vec(PetscFn fn, Vec x, Vec v, Vec xhat, Vec Hadjvxhat)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarHessianMult_Vec(fn, x, xhat, Hadjvxhat);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  ierr = VecScale(Hadjvxhat, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuild_Vec(PetscFn fn, Vec x, Vec xhat, Mat Hxhat, Mat Hxhatpre)
{
  Mat            hes = Hxhat ? Hxhat : Hxhatpre;
  Vec            hxhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(xhat, &hxhat);CHKERRQ(ierr);
  ierr = PetscFnScalarHessianMult_Vec(fn, x, xhat, hxhat);CHKERRQ(ierr);
  ierr = VecToMat_Internal(hxhat, hes, PETSC_FALSE);CHKERRQ(ierr);
  if (Hxhat && Hxhat != hes) {
    ierr = MatSetUp(Hxhat);CHKERRQ(ierr);
    ierr = MatCopy(hes, Hxhat, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  if (Hxhatpre && Hxhatpre != hes) {
    ierr = MatSetUp(Hxhatpre);CHKERRQ(ierr);
    ierr = MatCopy(hes, Hxhatpre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&hxhat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuildAdjoint_Vec(PetscFn fn, Vec x, Vec v, Mat Hadjv, Mat Hadjvpre)
{
  PetscScalar    z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarHessianBuild_Vec(fn, x, Hadjv, Hadjvpre);CHKERRQ(ierr);
  ierr = VecSingletonBcast(v, &z);CHKERRQ(ierr);
  if (Hadjv) {ierr = MatScale(Hadjv, z);CHKERRQ(ierr);}
  if (Hadjvpre && Hadjvpre != Hadjv) {ierr = MatScale(Hadjvpre, z);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnHessianBuildSwap_Vec(PetscFn fn, Vec x, Vec xhat, Mat Hswpxhat, Mat Hswpxhatpre)
{
  Mat            hes = Hswpxhat ? Hswpxhat : Hswpxhatpre;
  Vec            hxhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(xhat, &hxhat);CHKERRQ(ierr);
  ierr = PetscFnScalarHessianMult_Vec(fn, x, xhat, hxhat);CHKERRQ(ierr);
  ierr = VecToMat_Internal(hxhat, hes, PETSC_TRUE);CHKERRQ(ierr);
  if (Hswpxhat && Hswpxhat != hes) {
    ierr = MatSetUp(Hswpxhat);CHKERRQ(ierr);
    ierr = MatCopy(hes, Hswpxhat, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  if (Hswpxhatpre && Hswpxhatpre != hes) {
    ierr = MatSetUp(Hswpxhatpre);CHKERRQ(ierr);
    ierr = MatCopy(hes, Hswpxhatpre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&hxhat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestBasicOps(PetscFn fn, PetscInt order, PetscRandom rand, PetscBool build_mat, PetscBool build_pre)
{
  Vec            x, f, xhat = NULL, xdot = NULL, Jxhat, v = NULL, Jadjv, Hxhatxdot, Hadjvxhat;
  Vec            g, Hxhat;
  Mat            jac, jacadj, hes, hesadj, hesswp, scalhes;
  Mat            jacPre, jacadjPre, hesPre, hesadjPre, hesswpPre, scalhesPre;
  PetscReal      z;
  PetscBool      isScalar;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscFnIsScalar(fn, &isScalar);CHKERRQ(ierr);

  ierr = PetscFnCreateVecs(fn,&f,&x);CHKERRQ(ierr);
  ierr = VecSetRandom(x, rand);CHKERRQ(ierr);

  ierr = PetscFnApply(fn, x, f);CHKERRQ(ierr);
  if (isScalar) {ierr = PetscFnScalarApply(fn, x, &z);CHKERRQ(ierr);}

  if (order > 0) {
    ierr = PetscFnCreateVecs(fn,&Jxhat,&xhat);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn,&v,&Jadjv);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn,NULL,&xdot);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn,NULL,&g);CHKERRQ(ierr);

    ierr = VecSetRandom(xhat, rand);CHKERRQ(ierr);
    ierr = VecSetRandom(xdot, rand);CHKERRQ(ierr);
    ierr = VecSetRandom(v, rand);CHKERRQ(ierr);

    ierr = PetscFnJacobianMult(fn, x, xhat, Jxhat);CHKERRQ(ierr);
    ierr = PetscFnJacobianMultAdjoint(fn, x, v, Jadjv);CHKERRQ(ierr);

    if (isScalar) {ierr = PetscFnScalarGradient(fn, x, g);CHKERRQ(ierr);}

    jac    = jacadj    = NULL;
    jacPre = jacadjPre = NULL;

    if (build_mat && !build_pre) {
      ierr = PetscFnCreateMats(fn, PETSCFNOP_JACOBIANBUILD,        &jac,     NULL);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_JACOBIANBUILDADJOINT, &jacadj,  NULL);CHKERRQ(ierr);
    } else if (!build_mat && build_pre) {
      ierr = PetscFnCreateMats(fn, PETSCFNOP_JACOBIANBUILD,        NULL,     &jacPre);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_JACOBIANBUILDADJOINT, NULL,     &jacadjPre);CHKERRQ(ierr);
    } else if (build_mat && build_pre) {
      ierr = PetscFnCreateMats(fn, PETSCFNOP_JACOBIANBUILD,        &jac,     &jacPre);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_JACOBIANBUILDADJOINT, &jacadj,  &jacadjPre);CHKERRQ(ierr);
    }

    ierr = PetscFnJacobianBuild(fn, x, jac, jacPre);CHKERRQ(ierr);
    ierr = PetscFnJacobianBuildAdjoint(fn, x, jacadj, jacadjPre);CHKERRQ(ierr);

    ierr = MatDestroy(&jacadj);CHKERRQ(ierr);
    ierr = MatDestroy(&jac);CHKERRQ(ierr);
    ierr = MatDestroy(&jacadjPre);CHKERRQ(ierr);
    ierr = MatDestroy(&jacPre);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
    ierr = VecDestroy(&Jadjv);CHKERRQ(ierr);
    ierr = VecDestroy(&Jxhat);CHKERRQ(ierr);
  }
  if (order > 1) {
    ierr = PetscFnCreateVecs(fn,&Hxhatxdot,&Hadjvxhat);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn,NULL,&Hxhat);CHKERRQ(ierr);

    ierr = PetscFnHessianMult(fn, x, xhat, xdot, Hxhatxdot);CHKERRQ(ierr);
    ierr = PetscFnHessianMultAdjoint(fn, x, v, xhat, Hadjvxhat);CHKERRQ(ierr);
    if (isScalar) {ierr = PetscFnScalarHessianMult(fn, x, xhat, Hxhat);CHKERRQ(ierr);}

    hes    = hesadj    = hesswp    = scalhes    = NULL;
    hesPre = hesadjPre = hesswpPre = scalhesPre = NULL;
    if (build_mat && !build_pre) {
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILD,         &hes,     NULL);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILDADJOINT,  &hesadj,  NULL);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILDSWAP,     &hesswp,  NULL);CHKERRQ(ierr);
      if (isScalar) {ierr = PetscFnCreateMats(fn, PETSCFNOP_SCALARHESSIANBUILD,   &scalhes, NULL);CHKERRQ(ierr);}
    } else if (!build_mat && build_pre) {
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILD,         NULL,     &hesPre);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILDADJOINT,  NULL,     &hesadjPre);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILDSWAP,     NULL,     &hesswpPre);CHKERRQ(ierr);
      if (isScalar) {ierr = PetscFnCreateMats(fn, PETSCFNOP_SCALARHESSIANBUILD,   NULL,     &scalhesPre);CHKERRQ(ierr);}
    } else if (build_mat && build_pre) {
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILD,         &hes,     &hesPre);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILDADJOINT,  &hesadj,  &hesadjPre);CHKERRQ(ierr);
      ierr = PetscFnCreateMats(fn, PETSCFNOP_HESSIANBUILDSWAP,     &hesswp,  &hesswpPre);CHKERRQ(ierr);
      if (isScalar) {ierr = PetscFnCreateMats(fn, PETSCFNOP_SCALARHESSIANBUILD,   &scalhes, &scalhesPre);CHKERRQ(ierr);}
    }

    ierr = PetscFnHessianBuild(fn, x, xhat, hes, hesPre);CHKERRQ(ierr);
    ierr = PetscFnHessianBuildAdjoint(fn, x, v, hesadj, hesadjPre);CHKERRQ(ierr);
    ierr = PetscFnHessianBuildSwap(fn, x, xhat, hesswp, hesswpPre);CHKERRQ(ierr);
    if (isScalar) {ierr = PetscFnScalarHessianBuild(fn, x, scalhes, scalhesPre);CHKERRQ(ierr);}

    ierr = MatDestroy(&scalhesPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesswpPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesadjPre);CHKERRQ(ierr);
    ierr = MatDestroy(&hesPre);CHKERRQ(ierr);
    ierr = MatDestroy(&scalhes);CHKERRQ(ierr);
    ierr = MatDestroy(&hesswp);CHKERRQ(ierr);
    ierr = MatDestroy(&hesadj);CHKERRQ(ierr);
    ierr = MatDestroy(&hes);CHKERRQ(ierr);
    ierr = VecDestroy(&Hxhat);CHKERRQ(ierr);
    ierr = VecDestroy(&Hadjvxhat);CHKERRQ(ierr);
    ierr = VecDestroy(&Hxhatxdot);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&xhat);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDerivativeFns(PetscFn fn, PetscRandom rand)
{
  Vec            dotVecs[3];
  PetscBool      isScalar;
  PetscFn        df;
  Vec            x, xhat, xdot, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnCreateVecs(fn, &v, &x);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xhat);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xdot);CHKERRQ(ierr);

  ierr = VecSetRandom(x, rand);CHKERRQ(ierr);
  ierr = VecSetRandom(xhat, rand);CHKERRQ(ierr);
  ierr = VecSetRandom(xdot, rand);CHKERRQ(ierr);
  ierr = VecSetRandom(v, rand);CHKERRQ(ierr);

  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_APPLY, 1, &v, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 2, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_JACOBIANMULT, 1, &xhat, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = xhat;
  dotVecs[1] = v;
  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_JACOBIANMULT, 2, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_JACOBIANMULTADJOINT, 1, &v, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = v;
  dotVecs[1] = xhat;
  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_JACOBIANMULTADJOINT, 2, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = xhat;
  dotVecs[1] = xdot;
  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_HESSIANMULT, 2, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = xhat;
  dotVecs[1] = xdot;
  dotVecs[2] = v;
  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_HESSIANMULT, 3, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = v;
  dotVecs[1] = xhat;
  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_HESSIANMULTADJOINT, 2, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  dotVecs[0] = v;
  dotVecs[1] = xhat;
  dotVecs[2] = xdot;
  ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_HESSIANMULTADJOINT, 3, dotVecs, &df);CHKERRQ(ierr);
  ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
  ierr = PetscFnSetUp(df);CHKERRQ(ierr);
  ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

  ierr = PetscFnIsScalar(fn, &isScalar);CHKERRQ(ierr);
  if (isScalar) {
    ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_SCALARGRADIENT, 0, NULL, &df);CHKERRQ(ierr);
    ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
    ierr = PetscFnSetUp(df);CHKERRQ(ierr);
    ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

    ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_SCALARGRADIENT, 1, &xhat, &df);CHKERRQ(ierr);
    ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
    ierr = PetscFnSetUp(df);CHKERRQ(ierr);
    ierr = TestBasicOps(df, 1, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

    ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_SCALARHESSIANMULT, 1, &xhat, &df);CHKERRQ(ierr);
    ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
    ierr = PetscFnSetUp(df);CHKERRQ(ierr);
    ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&df);CHKERRQ(ierr);

    dotVecs[0] = xhat;
    dotVecs[1] = xdot;
    ierr = PetscFnCreateDerivativeFn(fn, PETSCFNOP_SCALARHESSIANMULT, 2, dotVecs, &df);CHKERRQ(ierr);
    ierr = PetscFnAppendOptionsPrefix(df, "der_");CHKERRQ(ierr);
    ierr = PetscFnSetFromOptions(df);CHKERRQ(ierr);
    ierr = PetscFnSetUp(df);CHKERRQ(ierr);
    ierr = TestBasicOps(df, 0, rand, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFnDestroy(&df);CHKERRQ(ierr);
  }

  VecDestroy(&x);CHKERRQ(ierr);
  VecDestroy(&xhat);CHKERRQ(ierr);
  VecDestroy(&xdot);CHKERRQ(ierr);
  VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscFn        fn;
  PetscBool      isShell;
  PetscBool      set_vector, set_scalar, build_mat, build_pre;
  PetscBool      test_ders;
  PetscRandom    rand;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  set_vector = PETSC_TRUE;
  set_scalar = PETSC_TRUE;
  build_mat  = PETSC_TRUE;
  build_pre  = PETSC_FALSE;
  test_ders  = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "PetscFn Test Options", "PetscFn");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-set_vector", "Set vector callbacks for PetscFnShell",          "ex1.c", set_vector, &set_vector, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-set_scalar", "Set scalar callbacks for PetscFnShell",          "ex1.c", set_scalar, &set_scalar, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-build_mat",  "Build the derivative matrices",                  "ex1.c", build_mat,  &build_mat, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-build_pre",  "Build the derivative preconditioning matrices",  "ex1.c", build_pre,  &build_pre, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_ders",  "Test the construnction of derivative functions", "ex1.c", test_ders,  &test_ders, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscFnCreate(PETSC_COMM_WORLD, &fn);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(fn, PETSC_DECIDE, 1, 1, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscFnSetFromOptions(fn);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)fn, PETSCFNSHELL, &isShell);CHKERRQ(ierr);
  comm = PetscObjectComm((PetscObject)fn);
  if (isShell) {
    PetscInt n, N, i, l;
    Vec v;
    void *ctx;
    PetscScalar *a;

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
    ierr = PetscFnShellSetOperation(fn,PETSCFNOP_CREATEMATS,(void (*)(void))PetscFnCreateMats_Vec);CHKERRQ(ierr);

    if (set_vector) {
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_APPLY,(void (*)(void))PetscFnApply_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_JACOBIANMULT,(void (*)(void))PetscFnJacobianMult_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_JACOBIANMULTADJOINT,(void (*)(void))PetscFnJacobianMultAdjoint_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_JACOBIANBUILD,(void (*)(void))PetscFnJacobianBuild_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_JACOBIANBUILDADJOINT,(void (*)(void))PetscFnJacobianBuildAdjoint_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_HESSIANMULT,(void (*)(void))PetscFnHessianMult_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_HESSIANMULTADJOINT,(void (*)(void))PetscFnHessianMultAdjoint_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_HESSIANBUILD,(void (*)(void))PetscFnHessianBuild_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_HESSIANBUILDADJOINT,(void (*)(void))PetscFnHessianBuildAdjoint_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_HESSIANBUILDSWAP,(void (*)(void))PetscFnHessianBuildSwap_Vec);CHKERRQ(ierr);
    }

    if (set_scalar) {
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARAPPLY,(void (*)(void))PetscFnScalarApply_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARGRADIENT,(void (*)(void))PetscFnScalarGradient_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARHESSIANMULT,(void (*)(void))PetscFnScalarHessianMult_Vec);CHKERRQ(ierr);
      ierr = PetscFnShellSetOperation(fn,PETSCFNOP_SCALARHESSIANBUILD,(void (*)(void))PetscFnScalarHessianBuild_Vec);CHKERRQ(ierr);
    }
  }
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  ierr = PetscFnViewFromOptions(fn, NULL, "-fn_view");CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  ierr = TestBasicOps(fn, 2, rand, build_mat, build_pre);CHKERRQ(ierr);
  if (test_ders) {ierr = TestDerivativeFns(fn, rand);CHKERRQ(ierr);}

  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&fn);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -fn_test_jacobianmult -fn_test_jacobianmultadjoint -fn_test_hessianmult -fn_test_hessianmultadjoint -fn_test_scalargradient -fn_test_scalarhessianmult -fn_test_jacobianbuild -fn_test_jacobianbuildadjoint -fn_test_hessianbuild -fn_test_hessianbuildswap -fn_test_hessianbuildadjoint -fn_test_scalarhessianbuild -fn_test_derivative_view -fn_test_derivativemat_view

   test:
      suffix: 2
      nsize: 1
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0
      output_file: output/ex1_1.out

   test:
      suffix: 3
      nsize: 1
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0
      output_file: output/ex1_1.out

   test:
      suffix: 4
      nsize: 2
      args: -fn_test_jacobianmult -fn_test_jacobianmultadjoint -fn_test_hessianmult -fn_test_hessianmultadjoint -fn_test_scalargradient -fn_test_scalarhessianmult -fn_test_jacobianbuild -fn_test_jacobianbuildadjoint -fn_test_hessianbuild -fn_test_hessianbuildswap -fn_test_hessianbuildadjoint -fn_test_scalarhessianbuild -fn_test_derivative_view -fn_test_derivativemat_view

   test:
      suffix: 5
      nsize: 2
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0
      output_file: output/ex1_4.out

   test:
      suffix: 6
      nsize: 2
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0
      output_file: output/ex1_4.out

   test:
      suffix: 7
      nsize: 1
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -build_pre -build_mat 0
      output_file: output/ex1_1.out

   test:
      suffix: 8
      nsize: 1
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0 -build_pre -build_mat 0
      output_file: output/ex1_1.out

   test:
      suffix: 9
      nsize: 1
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0 -build_pre -build_mat 0
      output_file: output/ex1_1.out

   test:
      suffix: 10
      nsize: 2
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -build_pre -build_mat 0
      output_file: output/ex1_4.out

   test:
      suffix: 11
      nsize: 2
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0 -build_pre -build_mat 0
      output_file: output/ex1_4.out

   test:
      suffix: 12
      nsize: 2
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0 -build_pre -build_mat 0
      output_file: output/ex1_4.out

   test:
      suffix: 13
      nsize: 1
      args: -fn_test_jacobianmult -fn_test_jacobianmultadjoint -fn_test_hessianmult -fn_test_hessianmultadjoint -fn_test_scalargradient -fn_test_scalarhessianmult -fn_test_jacobianbuild -fn_test_jacobianbuildadjoint -fn_test_hessianbuild -fn_test_hessianbuildswap -fn_test_hessianbuildadjoint -fn_test_scalarhessianbuild -fn_test_derivative_view -fn_test_derivativemat_view -build_pre -build_mat

   test:
      suffix: 14
      nsize: 1
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0 -build_pre -build_mat
      output_file: output/ex1_13.out

   test:
      suffix: 15
      nsize: 1
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0 -build_pre -build_mat
      output_file: output/ex1_13.out

   test:
      suffix: 16
      nsize: 2
      args: -fn_test_jacobianmult -fn_test_jacobianmultadjoint -fn_test_hessianmult -fn_test_hessianmultadjoint -fn_test_scalargradient -fn_test_scalarhessianmult -fn_test_jacobianbuild -fn_test_jacobianbuildadjoint -fn_test_hessianbuild -fn_test_hessianbuildswap -fn_test_hessianbuildadjoint -fn_test_scalarhessianbuild -fn_test_derivative_view -fn_test_derivativemat_view -build_pre -build_mat

   test:
      suffix: 17
      nsize: 2
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_vector 0 -build_pre -build_mat
      output_file: output/ex1_16.out

   test:
      suffix: 18
      nsize: 2
      args: -fn_test_allmult -fn_test_allbuild -fn_test_derivative_view -fn_test_derivativemat_view -set_scalar 0 -build_pre -build_mat
      output_file: output/ex1_16.out

   test:
      suffix: 19
      nsize: 1
      args: -test_ders
      output_file: output/ex1_19.out

   test:
      suffix: 20
      nsize: 4
      args: -test_ders
      output_file: output/ex1_19.out

TEST*/
