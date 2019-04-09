#include <petscfn.h>

const char help[] = "Create, view and test a DAG implementation of PetscFn\n";

typedef struct _n_Ax Ax;

struct _n_Ax
{
  Mat A;
  IS  Ais;
  IS  xis;
  Vec superu;
};

static PetscErrorCode AxGetSubVecs(PetscFn fn, const IS *subsets, PetscInt k, Vec u, Vec *uA, Vec *uX, PetscBool read)
{
  IS             subset;
  PetscBool      conforms;
  PetscBool      isEqual;
  Ax             *axo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &axo);CHKERRQ(ierr);
  subset = subsets ? subset[k] : NULL;
  if (!subset) {
    ierr = VecGetSubVector(u, axo->Ais, uA);CHKERRQ(ierr);
    ierr = VecGetSubVector(u, axo->xis, uX);CHKERRQ(ierr);
  } else if (axo->Ais == subset) {
    *uA = u;
    *uX = NULL;
  } else if (axo->xis == subset) {
    *uX = u;
    *uA = NULL;
  } else {
    ierr = PetscFnGetSuperVector(fn, PETSC_FALSE, subset, u, &axo->superu, read);CHKERRQ(ierr);
    ierr = VecGetSubVector(axo->superu, axo->Ais, uA);CHKERRQ(ierr);
    ierr = VecGetSubVector(axo->superu, axo->xis, uX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AxRestoreSubVecs(PetscFn fn, const IS *subsets, PetscInt k, Vec u, Vec *uA, Vec *uX, PetscBool write)
{
  IS             subset;
  PetscBool      conforms;
  PetscBool      isEqual;
  Ax             *axo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &axo);CHKERRQ(ierr);
  subset = subsets ? subset[k] : NULL;
  if (!subset) {
    ierr = VecRestoreSubVector(u, axo->xis, uX);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(u, axo->Ais, uA);CHKERRQ(ierr);
  } else if (axo->Ais == subset) {
    *uA = NULL;
  } else if (axo->xis == subset) {
    *uX = NULL;
  } else {
    ierr = VecRestoreSubVector(axo->superu, axo->xis, uX);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(axo->superu, axo->Ais, uA);CHKERRQ(ierr);
    ierr = PetscFnRestoreSuperVectors(fn, PETSC_FALSE, subset, u, &axo->superu, write);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AxOuterProduct(Ax *axo, Vec x, Vec y, Vec xy)
{
  PetscScalar *xya;
  const PetscScalar *xa;
  const PetscScalar *ya;
  PetscInt       i, j, m, N;
  PetscMPIInt    size;
  Vec            xred;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(axo->A, &m, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(axo->A, NULL, N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)x),&size);CHKERRQ(ierr);
  ierr = VecCreateRedundantVector(x,size,PETSC_COMM_SELF,&xred);CHKERRQ(ierr);
  ierr = VecGetArray(xred, &xa);CHKERRQ(ierr);
  ierr = VecGetArray(y, &ya);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xy, &xya);CHKERRQ(ierr);
  /* MatDense uses column major */
  for (j = 0, l = 0; j < N; j++) {
    for (i = 0; i < m; i++, l++) {
      xya[l] = xa[j] * ya[i];
    }
  }
  ierr = VecRestoreArrayRead(xy, &xya);CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &ya);CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &xa);CHKERRQ(ierr);
  ierr = VecDestroy(&xred);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AxCompareIS(Ax *axo, const IS *subsets, PetscInt k, PetscBool *conforms, PetscBool *containsAis, PetscBool *containsXis)
{
  IS             subset;
  PetscBool      isEqual;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  subset = subsets ? subset[k] : NULL;
  if (!subset) {
    *conforms = PETSC_TRUE;
    *containsAis = PETSC_TRUE;
    *containsXis = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  *conforms = PETSC_FALSE;
  *containsAis = PETSC_FALSE;
  *containsXis = PETSC_FALSE;
  if (axo->Ais == subset) {
    *conforms = PETSC_TRUE;
    *containsAis = PETSC_TRUE;
  }
  if (axo->xis == subset) {
    *conforms = PETSC_TRUE;
    *containsxis = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnCreateVecs_Ax(PetscFn fn, IS domainIS, Vec *domainVec, IS rangeIS, Vec *rangeVec)
{
  Ax             *ax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &ax);CHKERRQ(ierr);
  if (rangeVec) {
    if (rangeIS) {
      ierr = MatCreateVecs(ax->A, NULL, rangeVec);CHKERRQ(ierr);
    } else {
      PetscInt n, N;

      ierr = ISGetLocalSize(rangeIS, &n);CHKERRQ(ierr);
      ierr = ISGetSize(rangeIS, &N);CHKERRQ(ierr);
      ierr = VecCreateMPI(PetscObjectComm((PetscObject)fn), n, N, &rangeVec);CHKERRQ(ierr);
    }
  }
  if (domainVec) {
    Vec x, Avar;
    Vec vecs[2];
    PetscInt m, M, nx, Nx;
    PetscInt nA, NA;

    ierr = MatGetSize(A, &M, &Nx);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &m, &nx);CHKERRQ(ierr);

    nA = Nx * m;
    NA = Nx * M;
    if (domainIS) {
      IS domA, domx;

      ierr = ISIntersect(domainIS, ax->Ais, &domA);CHKERRQ(ierr);
      ierr = ISIntersect(domainIS, ax->xis, &domx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(domA, &nA);CHKERRQ(ierr);
      ierr = ISGetSize(domA, &NA);CHKERRQ(ierr);
      ierr = ISGetLocalSize(domx, &nX);CHKERRQ(ierr);
      ierr = ISGetSize(domx, &NX);CHKERRQ(ierr);
      ierr = VecCreateMPI(PetscObjectComm((PetscObject)fn), nx, Nx, &x);CHKERRQ(ierr);
      ierr = VecCreateMPI(PetscObjectComm((PetscObject)fn), nA, NA, &Avar);CHKERRQ(ierr);
      ierr = ISDestroy(&domA);CHKERRQ(ierr);
      ierr = ISDestroy(&domx);CHKERRQ(ierr);
    } else {
      ierr = MatCreateVecs(A, &x, NULL);CHKERRQ(ierr);
      ierr = VecCreateMPI(PetscObjectComm((PetscObject)fn), nA, NA, &Avar);CHKERRQ(ierr);
    }
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
  Ax  *axo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestGetSubVecs(ax, &nvecs, &vecs);CHKERRQ(ierr);
  x = vecs[0];
  Avar = vecs[1];
  ierr = VecGetArrayRead(Avar, &a);CHKERRQ(ierr);
  ierr = PetscFnShellGetContext(fn, (void *) &axo);CHKERRQ(ierr);
  ierr = MatDensePlaceArray(axo->A, a);CHKERRQ(ierr);
  ierr = MatMult(axo->A, x, y);CHKERRQ(ierr);
  ierr = MatDenseResetArray(axo->A);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Avar, &a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_Ax(PetscFn fn, Vec ax, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  Ax  *axo;
  PetscInt nvecs = 2;
  Vec *vecs;
  Vec Avar, x;
  const PetscScalar *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &axo);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(ax, &nvecs, &vecs);CHKERRQ(ierr);
  x = vecs[0];
  Avar = vecs[1];
  ierr = VecGetArrayRead(Avar, &a);CHKERRQ(ierr);
  switch (der) {
  case 0:
    Vec supery;

    ierr = PetscFnGetSuperVector(fn, PETSC_TRUE, subsets ? subsets[0] : NULL, y, &supery, PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscFnApply_Ax(fn, ax, supery);CHKERRQ(ierr);
    ierr = PetscFnGetSuperVector(fn, PETSC_TRUE, subsets ? subsets[0] : NULL, y, &supery, PETSC_TRUE);CHKERRQ(ierr);
    break;
  case 1:
    if (rangeIdx == 0) { /* adjoint */
      Vec v = subvecs[0], superv;
      Vec Ahat, xhat;

      ierr = PetscFnGetSuperVector(fn, PETSC_TRUE, subsets ? subsets[0] : NULL, v, &superv, PETSC_TRUE);CHKERRQ(ierr);
      ierr = AxGetSubVecs(fn, subsets, 1, y, &Ahat, &xhat, PETSC_FALSE);CHKERRQ(ierr);
      if (Ahat) { /* Ahat is the outer product of superv and x */
        ierr = AxOuterProduct(axo, xhat, superv, Ahat);CHKERRQ(ierr);
      }
      if (xhat) { /* xhat is A^T superv */
        ierr = MatDensePlaceArray(axo->A, a);CHKERRQ(ierr);
        ierr = MatMultTranspose(axo->A, superv, xhat);CHKERRQ(ierr);
        ierr = MatDenseResetArray(axo->A);CHKERRQ(ierr);
      }
      ierr = AxRetoreSubVecs(fn, subsets, 1, y, &Ahat, &xhat, PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscFnGetSuperVector(fn, PETSC_TRUE, subsets ? subsets[0] : NULL, v, &superv, PETSC_FALSE);CHKERRQ(ierr);
    } else { /* tlm */
      Vec supery;
      Vec Ahat, xhat;

      ierr = PetscFnGetSuperVector(fn, PETSC_TRUE, subsets ? subsets[1] : NULL, y, &supery, PETSC_FALSE);CHKERRQ(ierr);
      ierr = AxGetSubVecs(fn, subsets, 0, subvecs[0], &Ahat, &xhat, PETSC_TRUE);CHKERRQ(ierr);
      ierr = VecSet(supery, 0.);CHKERRQ(ierr);
      if (Ahat) { /* Ahat is the outer product of superv and x */
        const PetscScalar *ahat;

        ierr = VecGetArrayRead(Ahat, &ahat);CHKERRQ(ierr);
        ierr = MatDensePlaceArray(axo->A, ahat);CHKERRQ(ierr);
        ierr = MatMult(axo->A,ahat,x,supery);CHKERRQ(ierr);
        ierr = MatDenseResetArray(axo->A);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(Ahat, &ahat);CHKERRQ(ierr);
      }
      if (xhat) { /* xhat is A^T superv */
        ierr = MatDensePlaceArray(axo->A, a);CHKERRQ(ierr);
        ierr = MatMultAdd(axo->A, xhat, supery);CHKERRQ(ierr);
        ierr = MatDenseResetArray(axo->A);CHKERRQ(ierr);
      }
      ierr = AxRetoreSubVecs(fn, subsets, 0, subvecs[0], &Ahat, &xhat, PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscFnRestoreSuperVector(fn, PETSC_TRUE, subsets ? subsets[1] : NULL, y, &supery, PETSC_TRUE);CHKERRQ(ierr);
    }
    break;
  case 2:
    if (rangeIdx == 0 || rangeIdx == 1) { /* Hessian adjoint */
      Vec v = subvecs[0], superv;
      Vec Ahat, xhat, Atilde, xtilde;

      ierr = PetscFnGetSuperVector(fn, PETSC_TRUE, subsets ? subsets[rangeIdx] : NULL, v, &superv, PETSC_TRUE);CHKERRQ(ierr);
      ierr = AxGetSubVecs(fn, subsets, 1, subvecs[1-rangeIdx], &Ahat, &xhat, PETSC_TRUE);CHKERRQ(ierr);
      ierr = AxGetSubVecs(fn, subsets, 2, y, &Atilde, &xtilde, PETSC_FALSE);CHKERRQ(ierr);
      if (Atilde && xhat) {
        ierr = AxOuterProduct(axo, xhat, superv, Atilde);CHKERRQ(ierr);
      }
      if (xtilde && Ahat) {
        const PetscScalar *ahat;

        ierr = VecGetArrayRead(Ahat, &ahat);CHKERRQ(ierr);
        ierr = MatDensePlaceArray(axo->A, ahat);CHKERRQ(ierr);
        ierr = MatMultTranspose(axo->A, superv, xtilde);CHKERRQ(ierr);
        ierr = MatDenseResetArray(axo->A);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(Ahat, &ahat);CHKERRQ(ierr);
      }
      ierr = AxRetoreSubVecs(fn, subsets, 2, y, &Atilde, &xtilde, PETSC_TRUE);CHKERRQ(ierr);
      ierr = AxRetoreSubVecs(fn, subsets, 1, subvecs[1-rangeIdx], &Ahat, &xhat, PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscFnGetSuperVector(fn, PETSC_TRUE, subsets ? subsets[rangeIdx] : NULL, v, &superv, PETSC_FALSE);CHKERRQ(ierr);
    } else { /* Hessian */
      Vec supery;
      Vec Ahat, xhat, Atilde, xtilde;

      ierr = PetscFnGetSuperVector(fn, PETSC_TRUE, subsets ? subsets[2] : NULL, y, &supery, PETSC_FALSE);CHKERRQ(ierr);
      ierr = AxGetSubVecs(fn, subsets, 0, subvecs[0], &Ahat, &xhat, PETSC_TRUE);CHKERRQ(ierr);
      ierr = AxGetSubVecs(fn, subsets, 1, subvecs[1], &Atilde, &xtilde, PETSC_TRUE);CHKERRQ(ierr);
      ierr = VecSet(supery, 0.);CHKERRQ(ierr);
      if (Ahat && xtilde) {
        const PetscScalar *ahat;

        ierr = VecGetArrayRead(Ahat, &ahat);CHKERRQ(ierr);
        ierr = MatDensePlaceArray(axo->A, ahat);CHKERRQ(ierr);
        ierr = MatMult(axo->A,ahat,xtilde,supery);CHKERRQ(ierr);
        ierr = MatDenseResetArray(axo->A);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(Ahat, &ahat);CHKERRQ(ierr);
      }
      if (Atilde && xhat) {
        const PetscScalar *atilde;

        ierr = VecGetArrayRead(Atilde, &atilde);CHKERRQ(ierr);
        ierr = MatDensePlaceArray(axo->A, atilde);CHKERRQ(ierr);
        ierr = MatMult(axo->A,ahat,xhat,supery);CHKERRQ(ierr);
        ierr = MatDenseResetArray(axo->A);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(Atilde, &atilde);CHKERRQ(ierr);
      }
      ierr = AxRetoreSubVecs(fn, subsets, 1, subvecs[1], &Atilde, &xtilde, PETSC_FALSE);CHKERRQ(ierr);
      ierr = AxRetoreSubVecs(fn, subsets, 0, subvecs[0], &Ahat, &xhat, PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscFnRestoreSuperVector(fn, PETSC_TRUE, subsets ? subsets[1] : NULL, y, &supery, PETSC_TRUE);CHKERRQ(ierr);
    }
    break;
  default:
    ierr = VecSet(y, 0.);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(Avar, &a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDestroy_Ax(PetscFn *fn)
{
  Ax             *axo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void *) &axo);CHKERRQ(ierr);
  ierr = MatDestroy(&(axo->A));CHKERRQ(ierr);
  ierr = VecDestroy(&(axo->x));CHKERRQ(ierr);
  ierr = ISDestroy(&(axo->Ais));CHKERRQ(ierr);
  ierr = ISDestroy(&(axo->xis));CHKERRQ(ierr);
  ierr = VecDestroy(&(axo->superu));CHKERRQ(ierr);
  ierr = PetscFree(axo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarApply_L1(PetscFn fn, Vec x, PetscScalar *z)
{
  PetscReal      r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNorm(x,NORM_1,&r);CHKERRQ(ierr);
  *z = r;
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDAG(PetscRandom rand)
{
  PetscFn        dag, subdag, Ax, logistic, misfit, reg;
  PetscBool      isDag;
  MPI_Comm       comm;
  Vec            x, Axvec, Avec, b, y, *vecs, vz;
  Vec            Avec;
  Vec            axnest;
  Vec            vecs[2];
  IS             iss[2];
  PetscInt       Nx, nx, Mb, mb, Nc, Mc, nvecs;
  PetscInt       node_A, node_x, node_b, node_Ax, node_logistic, node_misfit, node_reg, node_loss;
  Mat            A, C;
  Ax             *axo;
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

  /* create the degrees of freedom of the matrix */
  ierr = VecCreateMPI(comm,mb * Nx, Mb * Nx, &Avec);CHKERRQ(ierr);
  ierr = VecSetRandom(Avec, rand);CHKERRQ(ierr);


  /* create a fn that multiplies A*x, but takes A and x as inputs, and so
   * uses a VecNest as an input type */
  ierr = PetscNew(&axo);CHKERRQ(ierr);

  /* get the shape of a nested vector that contains all of the variables in A and all of the variables in x */
  vecs[0] = x;
  vecs[1] = Avec;
  ierr = VecCreateNest(comm,2,NULL,vecs,&axnest);CHKERRQ(ierr);
  ierr = VecNestGetISs(axnest,iss);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)iss[0]);CHKERRQ(ierr);
  axo->xis = iss[0];
  ierr = PetscObjectReference((PetscObject)iss[1]);CHKERRQ(ierr);
  axo->Ais = iss[1];
  ierr = VecDestroy(&axnest);CHKERRQ(ierr);

  ierr = MatCreateDense(comm, mb, nx, Mb, Nx, NULL, &A);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  axo->A = A;
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFnCreate(comm, &Ax);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(Ax, mb, nx + Nx * mb, Mb, Nx * (1 + Mb));CHKERRQ(ierr);
  ierr = PetscFnSetType(Ax, PETSCFNSHELL);CHKERRQ(ierr);
  ierr = PetscFnShellSetContext(Ax, (void *) axo);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(Ax, PETSCFNOP_CREATEVECS, (void (*)(void)) PetscFnCreateVecs_Ax);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(Ax, PETSCFNOP_APPLY, (void (*)(void)) PetscFnApply_Ax);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(Ax, PETSCFNOP_DERIVATIVEVEC, (void (*)(void)) PetscFnDerivativeVec_Ax);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(Ax, PETSCFNOP_DESTROY, (void (*)(void)) PetscFnDestroy_Ax);CHKERRQ(ierr);

  ierr = PetscFnCreateVecs(Ax, NULL, &y, NULL, &Axvec);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(Axvec, &nvecs, &vecs);CHKERRQ(ierr);
  ierr = VecCopy(x, vecs[0]);CHKERRQ(ierr);
  ierr = VecCopy(Avec, vecs[1]);CHKERRQ(ierr);

  ierr = PetscFnShellCreate(comm,PETSCFNLOGISTIC, mb, mb, Mb, Mb, NULL, &logistic);CHKERRQ(ierr);
  ierr = PetscFnShellCreate(comm,PETSCFNNORMSQUARED, PETSC_DETERMINE, mb, 1, MB, NULL, &misfit);CHKERRQ(ierr);

  ierr = PetscFnCreate(comm, &reg);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(reg, PETSC_DETERMINE, mb * Nx, 1, Mb * Nx);CHKERRQ(ierr);
  ierr = PetscFnSetType(reg, PETSCFNSHELL);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(reg, PETSCFNOP_SCALARAPPLY, (void (*)(void)) PetscFnApply_L1);CHKERRQ(ierr);

  ierr = PetscFnCreate(comm, &dag);CHKERRQ(ierr);
  ierr = PetscFnSetType(dag, PETSCFNDAG);CHKERRQ(ierr);
  ierr = PetscFnCreateDefaultScalarVec(comm, &vz);CHKERRQ(ierr);
  ierr = PetscFnDAGAddNode(dag, NULL,     x,    "x",        &node_x);CHKERRQ(ierr);
  ierr = PetscFnDAGAddNode(dag, NULL,     b,    "b",        &node_b);CHKERRQ(ierr);
  ierr = PetscFnDAGAddNode(dag, NULL,     Avec, "A",        &node_A);CHKERRQ(ierr);
  ierr = PetscFnDAGAddNode(dag, Ax,       y,    "Ax",       &node_Ax);CHKERRQ(ierr);
  ierr = PetscFnDAGAddNode(dag, logistic, NULL, "logistic", &node_logistic);CHKERRQ(ierr);
  ierr = PetscFnDAGAddNode(dag, misfit,   NULL, "misfit",   &node_misfit);CHKERRQ(ierr);
  ierr = PetscFnDAGAddNode(dag, reg,      NULL, "reg",      &node_reg);CHKERRQ(ierr);
  ierr = PetscFnDAGAddNode(dag, NULL,     vz,   "loss",     &node_loss);CHKERRQ(ierr);

  ierr = PetscFnDAGAddEdge(dag, node_x,        node_Ax,       NULL, axo->xis, 1.,   NULL);CHKERRQ(ierr);
  ierr = PetscFnDAGAddEdge(dag, node_A,        node_Ax,       NULL, axo->Ais, 1.,   NULL);CHKERRQ(ierr);
  ierr = PetscFnDAGAddEdge(dag, node_Ax,       node_logistic, NULL, NULL,     1.,   NULL);CHKERRQ(ierr);
  ierr = PetscFnDAGAddEdge(dag, node_logistic, node_misfit,   NULL, NULL,     0.5,  NULL);CHKERRQ(ierr);
  ierr = PetscFnDAGAddEdge(dag, node_b,        node_misfit,   NULL, NULL,     -0.5, NULL);CHKERRQ(ierr);
  ierr = PetscFnDAGAddEdge(dag, node_A,        node_reg,      NULL, NULL,     1.,   NULL);CHKERRQ(ierr);
  ierr = PetscFnDAGAddEdge(dag, node_misfit,   node_loss,     NULL, NULL,     1.,   NULL);CHKERRQ(ierr);
  ierr = PetscFnDAGAddEdge(dag, node_reg,      node_loss,     NULL, NULL,     1.,   NULL);CHKERRQ(ierr);

  ierr = PetscFnDAGSetInputNode(dag, node_A);CHKERRQ(ierr);
  ierr = PetscFnDAGSetOutputNode(dag, node_loss);CHKERRQ(ierr);

  ierr = PetscFnSetUp(dag);CHKERRQ(ierr);

  ierr = PetscFnDAGCreateSubDAG(dag, node_x, node_misfit, PETSC_TRUE, &subdag);CHKERRQ(ierr);

  ierr = PetscFnApply(Ax, Axvec, y);CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&subdag);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&dag);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&reg);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&misfit);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&soft);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&Avec);CHKERRQ(ierr);
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
