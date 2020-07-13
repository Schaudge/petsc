#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc/private/tsimpl.h>     /*I "petscts.h" I*/
#include <petsc/private/snesimpl.h>
#include <petscds.h>
#include <petscfv.h>

static PetscErrorCode DMTSConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  if (isPlex) {
    *plex = dm;
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex);CHKERRQ(ierr);
    if (!*plex) {
      ierr = DMConvert(dm,DMPLEX,plex);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex);CHKERRQ(ierr);
      if (copy) {
        PetscInt    i;
        PetscObject obj;
        const char *comps[3] = {"A","dmAux","dmCh"};

        ierr = DMCopyDMTS(dm, *plex);CHKERRQ(ierr);
        ierr = DMCopyDMSNES(dm, *plex);CHKERRQ(ierr);
        for (i = 0; i < 3; i++) {
          ierr = PetscObjectQuery((PetscObject) dm, comps[i], &obj);CHKERRQ(ierr);
          ierr = PetscObjectCompose((PetscObject) *plex, comps[i], obj);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = PetscObjectReference((PetscObject) *plex);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeRHSFunctionFVM - Form the local forcing F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
- user - The user context

  Output Parameter:
. F  - Global output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *user)
{
  Vec            locF;
  IS             cellIS;
  DM             plex;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
  if (!cellIS) {
    ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);
  }
  ierr = DMGetLocalVector(plex, &locF);CHKERRQ(ierr);
  ierr = VecZeroEntries(locF);CHKERRQ(ierr);
  ierr = DMPlexComputeResidual_Internal(plex, cellIS, time, locX, NULL, time, locF, user);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(plex, locF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(plex, locF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(plex, &locF);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeBoundary - Insert the essential boundary values for the local input X and/or its time derivative X_t using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
- user - The user context

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeBoundary(DM dm, PetscReal time, Vec locX, Vec locX_t, void *user)
{
  DM             plex;
  Vec            faceGeometryFVM = NULL;
  PetscInt       Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSConvertPlex(dm, &plex, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMGetNumFields(plex, &Nf);CHKERRQ(ierr);
  if (!locX_t) {
    /* This is the RHS part */
    for (f = 0; f < Nf; f++) {
      PetscObject  obj;
      PetscClassId id;

      ierr = DMGetField(plex, f, NULL, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFV_CLASSID) {
        ierr = DMPlexGetGeometryFVM(plex, &faceGeometryFVM, NULL, NULL);CHKERRQ(ierr);
        break;
      }
    }
  }
  ierr = DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, time, faceGeometryFVM, NULL, NULL);CHKERRQ(ierr);
  /* TODO: locX_t */
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeIFunctionFEM - Form the local residual F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
- user - The user context

  Output Parameter:
. locF  - Local output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeIFunctionFEM(DM dm, PetscReal time, Vec locX, Vec locX_t, Vec locF, void *user)
{
  DM             plex;
  IS             cellIS;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
  if (!cellIS) {
    ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);
  }
  ierr = DMPlexComputeResidual_Internal(plex, cellIS, time, locX, locX_t, time, locF, user);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeIJacobianFEM - Form the local Jacobian J from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
. X_tshift - The multiplicative parameter for dF/du_t
- user - The user context

  Output Parameter:
. locF  - Local output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeIJacobianFEM(DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  DM             plex;
  PetscDS        prob;
  PetscBool      hasJac, hasPrec;
  IS             cellIS;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(prob, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(prob, &hasPrec);CHKERRQ(ierr);
  if (hasJac && hasPrec) {ierr = MatZeroEntries(Jac);CHKERRQ(ierr);}
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(plex,&depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
  if (!cellIS) {ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);}
  ierr = DMPlexComputeJacobian_Internal(plex, cellIS, time, X_tShift, locX, locX_t, Jac, JacP, user);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMTSCheckFromOptions - Check the residual and Jacobian functions using the exact solution by outputting some diagnostic information

  Input Parameters:
+ ts - the TS object
- u  - representative TS vector

  Note: The user must call PetscDSSetExactSolution() beforehand

  Level: developer
@*/
PetscErrorCode DMTSCheckFromOptions(TS ts, Vec u)
{
  DM             dm;
  SNES           snes;
  Vec            sol;
  PetscBool      check;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(((PetscObject)ts)->options,((PetscObject)ts)->prefix, "-dmts_check", &check);CHKERRQ(ierr);
  if (!check) PetscFunctionReturn(0);
  ierr = VecDuplicate(u, &sol);CHKERRQ(ierr);
  ierr = VecCopy(u, sol);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, u);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes, u);CHKERRQ(ierr);
  ierr = DMSNESCheck_Internal(snes, dm, sol);CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMPlexLandFormLandau_Internal(Vec a_X, Mat JacP, const PetscInt dim, void *a_ctx);
/*@
  DMPlexLandIFunction
@*/
PetscErrorCode DMPlexLandIFunction(TS ts,PetscReal time_dummy,Vec X,Vec X_t,Vec F,void *actx)
{
  PetscErrorCode ierr;
  LandCtx        *ctx=(LandCtx*)actx;
  PetscReal      unorm;
  PetscInt       dim;
  PetscFunctionBeginUser;
  if (PETSC_TRUE) {
    DM dm;
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
    if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  }
  ierr = VecNorm(X,NORM_2,&unorm);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr); // remove in real application
  ierr = PetscLogEventBegin(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMGetDimension(ctx->dmv, &dim);CHKERRQ(ierr);
  if (ctx->normJ!=unorm) {
    ctx->normJ = unorm;
    ierr = DMPlexLandFormLandau_Internal(X,ctx->J,dim,(void*)ctx);CHKERRQ(ierr);
    ctx->aux_bool = PETSC_TRUE; /* debug: set flag that we made a new Jacobian */
  } else ctx->aux_bool = PETSC_FALSE;
  /* mat vec for op */
  ierr = MatMult(ctx->J,X,F);CHKERRQ(ierr);CHKERRQ(ierr); /* C*f */
  /* add time term */
  if (X_t) {
    ierr = MatMultAdd(ctx->M,X_t,F,F);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@
  DMPlexLandIJacobian
@*/
PetscErrorCode DMPlexLandIJacobian(TS ts,PetscReal time_dummy,Vec X,Vec U_tdummy,PetscReal shift,Mat Amat,Mat Pmat,void *actx)
{
  PetscErrorCode ierr;
  LandCtx        *ctx=NULL;
  PetscReal      unorm;
  PetscInt       dim;
  PetscFunctionBeginUser;
  if (1) {
    DM dm;
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
    if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  }
  if (Amat!=Pmat || Amat!=ctx->J) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Amat!=Pmat || Amat!=ctx->J");
  ierr = DMGetDimension(ctx->dmv, &dim);CHKERRQ(ierr);
  /* get collision Jacobian into A */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = VecNorm(X,NORM_2,&unorm);CHKERRQ(ierr);
  if (ctx->normJ!=unorm) {
    ierr = DMPlexLandFormLandau_Internal(X,ctx->J,dim,(void*)ctx); CHKERRQ(ierr);
    ctx->normJ = unorm;
    ctx->aux_bool = PETSC_TRUE; /* debug: set flag that we made a new Jacobian */
  } else ctx->aux_bool = PETSC_FALSE;
  /* add C */
  ierr = MatCopy(ctx->J,Pmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  /* add mass */
  ierr = MatAXPY(Pmat,shift,ctx->M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
