#include <petsc/private/netrsimpl.h> /*I "petscnetrs.h"  I*/
#include <petscmat.h>
#include <petscvec.h>
#include <petsc/private/riemannsolverimpl.h> /* should not be here */

/*
    Implementation of exact nonlinear network solver for the SWE
*/

typedef struct {
  SNES snes;
  Vec  x, b;
} NRS_ExactSWE;

typedef struct {
  NetRS                netrs;
  const EdgeDirection *dir;
  const PetscReal     *u;
} ExactSWE_Wrapper;

/* Rework to seperate the Algebaric Coupling Conditions and the lax curve stuff */
static PetscErrorCode ExactSWE_LaxCurveFun(SNES snes, Vec x, Vec f, void *ctx)
{
  PetscInt           i, n, dof = 2, wavenum;
  ExactSWE_Wrapper  *wrapper = (ExactSWE_Wrapper *)ctx;
  const PetscScalar *ustar, *u = wrapper->u, ubar[2];
  PetscScalar       *F;
  NetRS              netrs = wrapper->netrs;

  PetscFunctionBeginUser;
  PetscCall(VecGetSize(x, &n));
  PetscCall(VecGetArrayRead(x, &ustar));
  PetscCall(VecGetArray(f, &F));

  /* Algebraic Coupling Condition */
  F[n - 2] = (wrapper->dir[netrs->numedges - 1] == EDGEIN) ? ustar[n - 1] : -ustar[n - 1];

  for (i = 0; i < netrs->numedges - 1; i++) {
    /* algebraic coupling */
    F[dof * i] = ustar[dof * i] - ustar[dof * (i + 1)];
    F[n - 2] += (wrapper->dir[i] == EDGEIN) ? ustar[dof * i + 1] : -ustar[dof * i + 1];
    /* physics based coupling */
    wavenum = (wrapper->dir[i] == EDGEIN) ? 1 : 2;
    PetscCall(RiemannSolverEvalLaxCurve(netrs->rs, u + dof * i, ustar[dof * i], wavenum, (PetscReal *)ubar));
    F[dof * i + 1] = ustar[dof * i + 1] - ubar[1];
  }
  wavenum = (wrapper->dir[netrs->numedges - 1] == EDGEIN) ? 1 : 2;
  PetscCall(RiemannSolverEvalLaxCurve(netrs->rs, u + dof * (netrs->numedges - 1), ustar[dof * (netrs->numedges - 1)], wavenum, (PetscReal *)ubar));
  F[n - 1] = ustar[n - 1] - ubar[1];
  PetscCall(VecRestoreArrayRead(x, &ustar));
  PetscCall(VecRestoreArray(f, &F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NRSEvaluate_ExactSWE(NetRS netrs, const PetscScalar *u, const EdgeDirection *dir, PetscScalar *flux, PetscReal *error)
{
  PetscErrorCode   ierr;
  ExactSWE_Wrapper wrapper;
  PetscInt         i, n, dof = netrs->numfields;
  PetscScalar     *x;
  NRS_ExactSWE    *exactswe = (NRS_ExactSWE *)netrs->data;
  void            *ctx;

  PetscFunctionBeginUser;
  wrapper.netrs = netrs;
  wrapper.dir   = dir;
  wrapper.u     = u;
  ierr          = SNESSetFunction(exactswe->snes, exactswe->b, ExactSWE_LaxCurveFun, &wrapper);
  /* Set initial condition as the reconstructed h,v values*/
  PetscCall(VecGetArray(exactswe->x, &x));
  PetscCall(VecGetSize(exactswe->x, &n));
  for (i = 0; i < netrs->numedges; i++) {
    x[i * dof]     = u[i * dof];
    x[i * dof + 1] = u[i * dof + 1];
  }
  PetscCall(VecRestoreArray(exactswe->x, &x));
  PetscCall(SNESSolve(exactswe->snes, NULL, exactswe->x));
  PetscCall(VecGetArray(exactswe->x, &x));

  if (netrs->estimate) {
    for (i = 0; i < netrs->numedges; i++) { PetscCall(NetRSErrorEstimate(netrs, dir[i], u + dof * i, x + dof * i, &error[i])); /* compute error esimate on star state */ }
  }
  PetscCall(NetRSGetApplicationContext(netrs, &ctx));
  for (i = 0; i < netrs->numedges * dof; i++) { flux[i] = x[i]; }
  PetscCall(VecRestoreArray(exactswe->x, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NRSSetUp_ExactSWE(NetRS rs)
{
  NRS_ExactSWE *exactswe = (NRS_ExactSWE *)rs->data;

  PetscFunctionBegin;
  PetscCall(VecCreateSeq(MPI_COMM_SELF, rs->numfields * rs->numedges, &exactswe->x)); /* Specific to the SWE with equal height coupling. To be adjusted */
  PetscCall(VecDuplicate(exactswe->x, &exactswe->b));
  PetscCall(SNESCreate(PETSC_COMM_SELF, &exactswe->snes));
  PetscCall(SNESSetFromOptions(exactswe->snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NRSReset_ExactSWE(NetRS rs)
{
  NRS_ExactSWE *exactswe = (NRS_ExactSWE *)rs->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&exactswe->x));
  PetscCall(VecDestroy(&exactswe->b));
  PetscCall(SNESDestroy(&exactswe->snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NRSDestroy_ExactSWE(NetRS rs)
{
  PetscFunctionBegin;
  PetscCall(NRSReset_ExactSWE(rs));
  PetscCall(PetscFree(rs->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NRSSetFromOptions_ExactSWE(PetscOptionItems *PetscOptionsObject, NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NRSView_ExactSWE(NetRS rs, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NRSCreate_ExactSWEStar(NetRS rs)
{
  NRS_ExactSWE *exactswe;

  PetscFunctionBegin;
  PetscCall(PetscNew(&exactswe));
  rs->data = (void *)exactswe;
  if (rs->numfields > -1) {
    if (rs->numfields != 2) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The Exact SWE solver requires numfields to be 2. (and should be solving the SWE)"); }
  }
  rs->ops->setup          = NRSSetUp_ExactSWE;
  rs->ops->reset          = NRSReset_ExactSWE;
  rs->ops->destroy        = NRSDestroy_ExactSWE;
  rs->ops->setfromoptions = NRSSetFromOptions_ExactSWE;
  rs->ops->view           = NRSView_ExactSWE;
  rs->ops->evaluate       = NRSEvaluate_ExactSWE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
