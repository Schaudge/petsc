#include <petsc/private/dmimpl.h>
#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  PetscErrorCode (*boundarylocal)(DM, PetscReal, Vec, Vec, void *);
  PetscErrorCode (*ifunctionlocal)(DM, PetscReal, Vec, Vec, Vec, void *);
  PetscErrorCode (*ijacobianlocal)(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
  PetscErrorCode (*rhsfunctionlocal)(DM, PetscReal, Vec, Vec, void *);
  PetscErrorCode (*rhsjacobianlocal)(DM, PetscReal, Vec, Mat, Mat, void *);
  void *boundarylocalctx;
  void *ifunctionlocalctx;
  void *ijacobianlocalctx;
  void *rhsfunctionlocalctx;
  void *rhsjacobianlocalctx;
  Vec   lumpedmassinv;
  Mat   mass;
  KSP   kspmass;
} DMTS_Local;

static PetscErrorCode DMTSDestroy_DMLocal(DMTS tdm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(tdm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTSDuplicate_DMLocal(DMTS oldtdm, DMTS tdm)
{
  PetscFunctionBegin;
  PetscCall(PetscNew((DMTS_Local **)&tdm->data));
  if (oldtdm->data) PetscCall(PetscMemcpy(tdm->data, oldtdm->data, sizeof(DMTS_Local)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLocalTSGetContext(DM dm, DMTS tdm, DMTS_Local **dmlocalts)
{
  PetscFunctionBegin;
  *dmlocalts = NULL;
  if (!tdm->data) {
    PetscCall(PetscNew((DMTS_Local **)&tdm->data));

    tdm->ops->destroy   = DMTSDestroy_DMLocal;
    tdm->ops->duplicate = DMTSDuplicate_DMLocal;
  }
  *dmlocalts = (DMTS_Local *)tdm->data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSComputeIFunction_DMLocal(TS ts, PetscReal time, Vec X, Vec X_t, Vec F, void *ctx)
{
  DM          dm;
  Vec         U, U_t, locX, locX_t, locF;
  DMTS_Local *dmlocalts = (DMTS_Local *)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(X_t, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 5);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 1024, &U));
  if (U) {
    DM sdm;
    IS subis;

    PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 1025, &U_t));
    PetscCall(DMGetSubdofIS(dm, &subis));
    PetscCall(VecISCopy(U, subis, SCATTER_FORWARD, X));
    PetscCall(VecISCopy(U_t, subis, SCATTER_FORWARD, X_t));
    PetscCall(VecGetDM(U, &sdm));
    PetscCall(DMGetLocalVector(sdm, &locX));
    PetscCall(DMGetLocalVector(sdm, &locX_t));
    PetscCall(VecZeroEntries(locX));
    PetscCall(VecZeroEntries(locX_t));
    if (dmlocalts->boundarylocal) PetscCall((*dmlocalts->boundarylocal)(sdm, time, locX, locX_t, dmlocalts->boundarylocalctx));
    PetscCall(DMGlobalToLocalBegin(sdm, U, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalEnd(sdm, U, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalBegin(sdm, U_t, INSERT_VALUES, locX_t));
    PetscCall(DMGlobalToLocalEnd(sdm, U_t, INSERT_VALUES, locX_t));
  } else {
    PetscCall(DMGetLocalVector(dm, &locX));
    PetscCall(VecZeroEntries(locX));
    PetscCall(DMGetLocalVector(dm, &locX_t));
    PetscCall(VecZeroEntries(locX_t));
    if (dmlocalts->boundarylocal) PetscCall((*dmlocalts->boundarylocal)(dm, time, locX, locX_t, dmlocalts->boundarylocalctx));
    PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalBegin(dm, X_t, INSERT_VALUES, locX_t));
    PetscCall(DMGlobalToLocalEnd(dm, X_t, INSERT_VALUES, locX_t));
  }
  PetscCall(DMGetLocalVector(dm, &locF));
  PetscCall(VecZeroEntries(locF));
  PetscCall((*dmlocalts->ifunctionlocal)(dm, time, locX, locX_t, locF, dmlocalts->ifunctionlocalctx));
  PetscCall(VecZeroEntries(F));
  PetscCall(DMLocalToGlobalBegin(dm, locF, ADD_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(dm, locF, ADD_VALUES, F));
  if (U) {
    DM sdm;

    PetscCall(VecGetDM(U, &sdm));
    PetscCall(DMRestoreLocalVector(sdm, &locX));
    PetscCall(DMRestoreLocalVector(sdm, &locX_t));
  } else {
    PetscCall(DMRestoreLocalVector(dm, &locX));
    PetscCall(DMRestoreLocalVector(dm, &locX_t));
  }
  PetscCall(DMRestoreLocalVector(dm, &locF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSComputeRHSFunction_DMLocal(TS ts, PetscReal time, Vec X, Vec F, void *ctx)
{
  DM          dm;
  Vec         U, locX, locF;
  DMTS_Local *dmlocalts = (DMTS_Local *)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 4);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 1024, &U));
  if (U) {
    DM sdm;
    IS subis;

    PetscCall(DMGetSubdofIS(dm, &subis));
    PetscCall(VecISCopy(U, subis, SCATTER_FORWARD, X));
    PetscCall(VecGetDM(U, &sdm));
    PetscCall(DMGetLocalVector(sdm, &locX));
    PetscCall(VecZeroEntries(locX));
    if (dmlocalts->boundarylocal) PetscCall((*dmlocalts->boundarylocal)(sdm, time, locX, NULL, dmlocalts->boundarylocalctx));
    PetscCall(DMGlobalToLocalBegin(sdm, U, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalEnd(sdm, U, INSERT_VALUES, locX));
  } else {
    PetscCall(DMGetLocalVector(dm, &locX));
    PetscCall(VecZeroEntries(locX));
    if (dmlocalts->boundarylocal) PetscCall((*dmlocalts->boundarylocal)(dm, time, locX, NULL, dmlocalts->boundarylocalctx));
    PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
  }
  PetscCall(DMGetLocalVector(dm, &locF));
  PetscCall(VecZeroEntries(locF));
  PetscCall((*dmlocalts->rhsfunctionlocal)(dm, time, locX, locF, dmlocalts->rhsfunctionlocalctx));
  PetscCall(VecZeroEntries(F));
  PetscCall(DMLocalToGlobalBegin(dm, locF, ADD_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(dm, locF, ADD_VALUES, F));
  if (dmlocalts->lumpedmassinv) {
    PetscCall(VecPointwiseMult(F, dmlocalts->lumpedmassinv, F));
  } else if (dmlocalts->kspmass) {
    Vec tmp;

    PetscCall(DMGetGlobalVector(dm, &tmp));
    PetscCall(KSPSolve(dmlocalts->kspmass, F, tmp));
    PetscCall(VecCopy(tmp, F));
    PetscCall(DMRestoreGlobalVector(dm, &tmp));
  }
  if (U) {
    DM sdm;

    PetscCall(VecGetDM(U, &sdm));
    PetscCall(DMRestoreLocalVector(sdm, &locX));
  } else {
    PetscCall(DMRestoreLocalVector(dm, &locX));
  }
  PetscCall(DMRestoreLocalVector(dm, &locF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSComputeIJacobian_DMLocal(TS ts, PetscReal time, Vec X, Vec X_t, PetscReal a, Mat A, Mat B, void *ctx)
{
  DM          dm;
  Vec         U, U_t, locX, locX_t;
  DMTS_Local *dmlocalts = (DMTS_Local *)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(X_t, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 6);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 7);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 1024, &U));
  if (dmlocalts->ijacobianlocal) {
    if (U) {
      DM sdm;
      IS subis;

      PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 1025, &U_t));
      PetscCall(DMGetSubdofIS(dm, &subis));
      PetscCall(VecISCopy(U, subis, SCATTER_FORWARD, X));
      PetscCall(VecISCopy(U_t, subis, SCATTER_FORWARD, X_t));
      PetscCall(VecGetDM(U, &sdm));
      PetscCall(DMGetLocalVector(sdm, &locX));
      PetscCall(DMGetLocalVector(sdm, &locX_t));
      PetscCall(VecZeroEntries(locX));
      PetscCall(VecZeroEntries(locX_t));
      if (dmlocalts->boundarylocal) PetscCall((*dmlocalts->boundarylocal)(sdm, time, locX, locX_t, dmlocalts->boundarylocalctx));
      PetscCall(DMGlobalToLocalBegin(sdm, U, INSERT_VALUES, locX));
      PetscCall(DMGlobalToLocalEnd(sdm, U, INSERT_VALUES, locX));
      PetscCall(DMGlobalToLocalBegin(sdm, U_t, INSERT_VALUES, locX_t));
      PetscCall(DMGlobalToLocalEnd(sdm, U_t, INSERT_VALUES, locX_t));
    } else {
      PetscCall(DMGetLocalVector(dm, &locX));
      PetscCall(DMGetLocalVector(dm, &locX_t));
      PetscCall(VecZeroEntries(locX));
      PetscCall(VecZeroEntries(locX_t));
      if (dmlocalts->boundarylocal) PetscCall((*dmlocalts->boundarylocal)(dm, time, locX, locX_t, dmlocalts->boundarylocalctx));
      PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
      PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
      PetscCall(DMGlobalToLocalBegin(dm, X_t, INSERT_VALUES, locX_t));
      PetscCall(DMGlobalToLocalEnd(dm, X_t, INSERT_VALUES, locX_t));
    }
    PetscCall((*dmlocalts->ijacobianlocal)(dm, time, locX, locX_t, a, A, B, dmlocalts->ijacobianlocalctx));
    if (U) {
      DM sdm;

      PetscCall(VecGetDM(U, &sdm));
      PetscCall(DMRestoreLocalVector(sdm, &locX));
      PetscCall(DMRestoreLocalVector(sdm, &locX_t));
    } else {
      PetscCall(DMRestoreLocalVector(dm, &locX));
      PetscCall(DMRestoreLocalVector(dm, &locX_t));
    }
  } else {
    MatFDColoring fdcoloring;
    PetscCall(PetscObjectQuery((PetscObject)dm, "DMDASNES_FDCOLORING", (PetscObject *)&fdcoloring));
    if (!fdcoloring) {
      ISColoring coloring;

      PetscCall(DMCreateColoring(dm, dm->coloringtype, &coloring));
      PetscCall(MatFDColoringCreate(B, coloring, &fdcoloring));
      PetscCall(ISColoringDestroy(&coloring));
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        PetscCall(MatFDColoringSetFunction(fdcoloring, (PetscErrorCode(*)(void))TSComputeIFunction_DMLocal, dmlocalts));
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "No support for coloring type '%s'", ISColoringTypes[dm->coloringtype]);
      }
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fdcoloring, ((PetscObject)dm)->prefix));
      PetscCall(MatFDColoringSetFromOptions(fdcoloring));
      PetscCall(MatFDColoringSetUp(B, coloring, fdcoloring));
      PetscCall(PetscObjectCompose((PetscObject)dm, "DMDASNES_FDCOLORING", (PetscObject)fdcoloring));
      PetscCall(PetscObjectDereference((PetscObject)fdcoloring));

      /* The following breaks an ugly reference counting loop that deserves a paragraph. MatFDColoringApply() will call
       * VecDuplicate() with the state Vec and store inside the MatFDColoring. This Vec will duplicate the Vec, but the
       * MatFDColoring is composed with the DM. We dereference the DM here so that the reference count will eventually
       * drop to 0. Note the code in DMDestroy() that exits early for a negative reference count. That code path will be
       * taken when the PetscObjectList for the Vec inside MatFDColoring is destroyed.
       */
      PetscCall(PetscObjectDereference((PetscObject)dm));
    }
    PetscCall(MatFDColoringApply(B, fdcoloring, X, ts));
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSComputeRHSJacobian_DMLocal(TS ts, PetscReal time, Vec X, Mat A, Mat B, void *ctx)
{
  DM          dm;
  Vec         U, locX;
  DMTS_Local *dmlocalts = (DMTS_Local *)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 4);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 5);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 1024, &U));
  if (dmlocalts->rhsjacobianlocal) {
    if (U) {
      DM sdm;
      IS subis;

      PetscCall(DMGetSubdofIS(dm, &subis));
      PetscCall(VecISCopy(U, subis, SCATTER_FORWARD, X));
      PetscCall(VecGetDM(U, &sdm));
      PetscCall(DMGetLocalVector(sdm, &locX));
      PetscCall(VecZeroEntries(locX));
      if (dmlocalts->boundarylocal) PetscCall((*dmlocalts->boundarylocal)(sdm, time, locX, NULL, dmlocalts->boundarylocalctx));
      PetscCall(DMGlobalToLocalBegin(sdm, U, INSERT_VALUES, locX));
      PetscCall(DMGlobalToLocalEnd(sdm, U, INSERT_VALUES, locX));
    } else {
      PetscCall(DMGetLocalVector(dm, &locX));
      PetscCall(VecZeroEntries(locX));
      if (dmlocalts->boundarylocal) PetscCall((*dmlocalts->boundarylocal)(dm, time, locX, NULL, dmlocalts->boundarylocalctx));
      PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
      PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
    }
    PetscCall((*dmlocalts->rhsjacobianlocal)(dm, time, locX, A, B, dmlocalts->rhsjacobianlocalctx));
    if (U) {
      DM sdm;

      PetscCall(VecGetDM(U, &sdm));
      PetscCall(DMRestoreLocalVector(sdm, &locX));
    } else {
      PetscCall(DMRestoreLocalVector(dm, &locX));
    }
  } else {
    MatFDColoring fdcoloring;
    PetscCall(PetscObjectQuery((PetscObject)dm, "DMDASNES_FDCOLORING", (PetscObject *)&fdcoloring));
    if (!fdcoloring) {
      ISColoring coloring;

      PetscCall(DMCreateColoring(dm, dm->coloringtype, &coloring));
      PetscCall(MatFDColoringCreate(B, coloring, &fdcoloring));
      PetscCall(ISColoringDestroy(&coloring));
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        PetscCall(MatFDColoringSetFunction(fdcoloring, (PetscErrorCode(*)(void))TSComputeRHSFunction_DMLocal, dmlocalts));
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "No support for coloring type '%s'", ISColoringTypes[dm->coloringtype]);
      }
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fdcoloring, ((PetscObject)dm)->prefix));
      PetscCall(MatFDColoringSetFromOptions(fdcoloring));
      PetscCall(MatFDColoringSetUp(B, coloring, fdcoloring));
      PetscCall(PetscObjectCompose((PetscObject)dm, "DMDASNES_FDCOLORING", (PetscObject)fdcoloring));
      PetscCall(PetscObjectDereference((PetscObject)fdcoloring));

      /* The following breaks an ugly reference counting loop that deserves a paragraph. MatFDColoringApply() will call
       * VecDuplicate() with the state Vec and store inside the MatFDColoring. This Vec will duplicate the Vec, but the
       * MatFDColoring is composed with the DM. We dereference the DM here so that the reference count will eventually
       * drop to 0. Note the code in DMDestroy() that exits early for a negative reference count. That code path will be
       * taken when the PetscObjectList for the Vec inside MatFDColoring is destroyed.
       */
      PetscCall(PetscObjectDereference((PetscObject)dm));
    }
    PetscCall(MatFDColoringApply(B, fdcoloring, X, ts));
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetBoundaryLocal - set the function for essential boundary data for a local implicit function evaluation.

  Logically Collective

  Input Parameters:
+ dm   - `DM` to associate callback with
. func - local function evaluation
- ctx  - context for function evaluation

  Level: intermediate

  Notes:
  `func` should set the essential boundary data for the local portion of the solution, as
  well its time derivative (if it is not `NULL`).

  Vectors are initialized to zero before this function, so it is only needed for non
  homogeneous data.

  This function is somewhat optional: boundary data could potentially be inserted by a function
  passed to `DMTSSetIFunctionLocal()`. The use case for this function is for discretizations
  with constraints (see `DMGetDefaultConstraints()`): this function inserts boundary values
  before constraint interpolation.

.seealso: [](ch_ts), `DM`, `TS`, `DMTSSetIFunction()`, `DMTSSetIJacobianLocal()`
@*/
PetscErrorCode DMTSSetBoundaryLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Vec, void *), void *ctx)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));

  dmlocalts->boundarylocal    = func;
  dmlocalts->boundarylocalctx = ctx;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetIFunctionLocal - get the local implicit function evaluation function. This function is called with local vector
  containing the local vector information PLUS ghost point information. It should compute a result for all local
  elements and `DM` will automatically accumulate the overlapping values.

  Logically Collective

  Input Parameter:
. dm - `DM` to associate callback with

  Output Parameters:
+ func - local function evaluation
- ctx  - context for function evaluation

  Level: beginner

.seealso: [](ch_ts), `DM`, `DMTSSetIFunctionLocal()`, `DMTSSetIFunction()`, `DMTSSetIJacobianLocal()`
@*/
PetscErrorCode DMTSGetIFunctionLocal(DM dm, PetscErrorCode (**func)(DM, PetscReal, Vec, Vec, Vec, void *), void **ctx)
{
  DMTS           tdm;
  DMTS_Local    *dmlocalts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDMTS(dm, &tdm);
  CHKERRQ(ierr);
  ierr = DMLocalTSGetContext(dm, tdm, &dmlocalts);
  CHKERRQ(ierr);
  if (func) {
    PetscAssertPointer(func, 2);
    *func = dmlocalts->ifunctionlocal;
  }
  if (ctx) {
    PetscAssertPointer(ctx, 3);
    *ctx = dmlocalts->ifunctionlocalctx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetIFunctionLocal - set a local implicit function evaluation function. This function is called with local vector
  containing the local vector information PLUS ghost point information. It should compute a result for all local
  elements and `DM` will automatically accumulate the overlapping values.

  Logically Collective

  Input Parameters:
+ dm   - `DM` to associate callback with
. func - local function evaluation
- ctx  - context for function evaluation

  Level: beginner

.seealso: [](ch_ts), `DM`, `DMTSGetIFunctionLocal()`, `DMTSSetIFunction()`, `DMTSSetIJacobianLocal()`
@*/
PetscErrorCode DMTSSetIFunctionLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Vec, Vec, void *), void *ctx)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));

  dmlocalts->ifunctionlocal    = func;
  dmlocalts->ifunctionlocalctx = ctx;

  if (func) {
    PetscCall(DMTSSetIFunction(dm, TSComputeIFunction_DMLocal, dmlocalts));
    if (!tdm->ops->ijacobian) { /* Call us for the Jacobian too, can be overridden by the user. */
      PetscCall(DMTSSetIJacobian(dm, TSComputeIJacobian_DMLocal, dmlocalts));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetIJacobianLocal - get a local Jacobian evaluation function

  Logically Collective

  Input Parameter:
. dm - `DM` to associate callback with

  Output Parameters:
+ func - local Jacobian evaluation
- ctx  - optional context for local Jacobian evaluation

  Level: beginner

.seealso: [](ch_ts), `DM`, `DMTSSetIJacobianLocal()`, `DMTSSetIFunctionLocal()`, `DMTSSetIJacobian()`, `DMTSSetIFunction()`
@*/
PetscErrorCode DMTSGetIJacobianLocal(DM dm, PetscErrorCode (**func)(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *), void **ctx)
{
  DMTS           tdm;
  DMTS_Local    *dmlocalts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDMTS(dm, &tdm);
  CHKERRQ(ierr);
  ierr = DMLocalTSGetContext(dm, tdm, &dmlocalts);
  CHKERRQ(ierr);
  if (func) {
    PetscAssertPointer(func, 2);
    *func = dmlocalts->ijacobianlocal;
  }
  if (ctx) {
    PetscAssertPointer(ctx, 3);
    *ctx = dmlocalts->ijacobianlocalctx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetIJacobianLocal - set a local Jacobian evaluation function

  Logically Collective

  Input Parameters:
+ dm   - `DM` to associate callback with
. func - local Jacobian evaluation
- ctx  - optional context for local Jacobian evaluation

  Level: beginner

.seealso: [](ch_ts), `DM`, `DMTSGetIJacobianLocal()`, `DMTSSetIFunctionLocal()`, `DMTSSetIJacobian()`, `DMTSSetIFunction()`
@*/
PetscErrorCode DMTSSetIJacobianLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *), void *ctx)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));

  dmlocalts->ijacobianlocal    = func;
  dmlocalts->ijacobianlocalctx = ctx;

  if (func) PetscCall(DMTSSetIJacobian(dm, TSComputeIJacobian_DMLocal, dmlocalts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetRHSFunctionLocal - get a local rhs function evaluation function. This function is called with local vector
  containing the local vector information PLUS ghost point information. It should compute a result for all local
  elements and `DM` will automatically accumulate the overlapping values.

  Logically Collective

  Input Parameter:
. dm - `DM` to associate callback with

  Output Parameters:
+ func - local function evaluation
- ctx  - context for function evaluation

  Level: beginner

.seealso: [](ch_ts), `DM`, `DMTSSetRHSFunctionLocal()`, `DMTSSetRHSFunction()`, `DMTSSetIFunction()`, `DMTSSetIJacobianLocal()`
@*/
PetscErrorCode DMTSGetRHSFunctionLocal(DM dm, PetscErrorCode (**func)(DM, PetscReal, Vec, Vec, void *), void **ctx)
{
  DMTS           tdm;
  DMTS_Local    *dmlocalts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDMTS(dm, &tdm);
  CHKERRQ(ierr);
  ierr = DMLocalTSGetContext(dm, tdm, &dmlocalts);
  CHKERRQ(ierr);
  if (func) {
    PetscAssertPointer(func, 2);
    *func = dmlocalts->rhsfunctionlocal;
  }
  if (ctx) {
    PetscAssertPointer(ctx, 3);
    *ctx = dmlocalts->rhsfunctionlocalctx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetRHSFunctionLocal - set a local rhs function evaluation function. This function is called with local vector
  containing the local vector information PLUS ghost point information. It should compute a result for all local
  elements and `DM` will automatically accumulate the overlapping values.

  Logically Collective

  Input Parameters:
+ dm   - `DM` to associate callback with
. func - local function evaluation
- ctx  - context for function evaluation

  Level: beginner

.seealso: [](ch_ts), `DM`, `DMTSGetRHSFunctionLocal()`, `DMTSSetRHSFunction()`, `DMTSSetIFunction()`, `DMTSSetIJacobianLocal()`
@*/
PetscErrorCode DMTSSetRHSFunctionLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Vec, void *), void *ctx)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));

  dmlocalts->rhsfunctionlocal    = func;
  dmlocalts->rhsfunctionlocalctx = ctx;

  if (func) PetscCall(DMTSSetRHSFunction(dm, TSComputeRHSFunction_DMLocal, dmlocalts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetRHSJacobianLocal - get a local RHS Jacobian evaluation function

  Logically Collective

  Input Parameter:
. dm - `DM` to associate callback with

  Output Parameters:
+ func - local Jacobian evaluation
- ctx - optional context for local RHS Jacobian evaluation

  Level: beginner

.seealso: [](ch_ts), `DM`, `DMTSSetRHSJacobianLocal()`, `DMTSSetRHSFunctionLocal()`, `DMTSSetRHSJacobian()`, `DMTSSetRHSFunction()`
@*/
PetscErrorCode DMTSGetRHSJacobianLocal(DM dm, PetscErrorCode (**func)(DM, PetscReal, Vec, Mat, Mat, void *), void **ctx)
{
  DMTS           tdm;
  DMTS_Local    *dmlocalts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDMTS(dm, &tdm);
  CHKERRQ(ierr);
  ierr = DMLocalTSGetContext(dm, tdm, &dmlocalts);
  CHKERRQ(ierr);
  if (func) {
    PetscAssertPointer(func, 2);
    *func = dmlocalts->rhsjacobianlocal;
  }
  if (ctx) {
    PetscAssertPointer(ctx, 3);
    *ctx = dmlocalts->rhsjacobianlocalctx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetRHSJacobianLocal - set a local RHS Jacobian evaluation function

  Logically Collective

  Input Parameters:
+ dm - `DM` to associate callback with
. func - local Jacobian evaluation
- ctx - optional context for local RHS Jacobian evaluation

  Level: beginner

.seealso: [](ch_ts), `DM`, `DMTSGetRHSJacobianLocal()`, `DMTSSetRHSFunctionLocal()`, `DMTSSetRHSJacobian()`, `DMTSSetRHSFunction()`
@*/
PetscErrorCode DMTSSetRHSJacobianLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Mat, Mat, void *), void *ctx)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));

  dmlocalts->rhsjacobianlocal    = func;
  dmlocalts->rhsjacobianlocalctx = ctx;

  if (func) PetscCall(DMTSSetRHSJacobian(dm, TSComputeRHSJacobian_DMLocal, dmlocalts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSCreateRHSMassMatrix - This creates the mass matrix associated with the given `DM`, and a solver to invert it, and stores them in the `DM` context.

  Collective

  Input Parameter:
. dm - `DM` providing the mass matrix

  Level: developer

  Note:
  The idea here is that an explicit system can be given a mass matrix, based on the `DM`, which is inverted on the RHS at each step.

.seealso: [](ch_ts), `DM`, `DMTSCreateRHSMassMatrixLumped()`, `DMTSDestroyRHSMassMatrix()`, `DMCreateMassMatrix()`, `DMTS`
@*/
PetscErrorCode DMTSCreateRHSMassMatrix(DM dm)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;
  const char *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));
  PetscCall(DMCreateMassMatrix(dm, dm, &dmlocalts->mass));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &dmlocalts->kspmass));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
  PetscCall(KSPSetOptionsPrefix(dmlocalts->kspmass, prefix));
  PetscCall(KSPAppendOptionsPrefix(dmlocalts->kspmass, "mass_"));
  PetscCall(KSPSetFromOptions(dmlocalts->kspmass));
  PetscCall(KSPSetOperators(dmlocalts->kspmass, dmlocalts->mass, dmlocalts->mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSCreateRHSMassMatrixLumped - This creates the lumped mass matrix associated with the given `DM`, and a solver to invert it, and stores them in the `DM` context.

  Collective

  Input Parameter:
. dm - `DM` providing the mass matrix

  Level: developer

  Note:
  The idea here is that an explicit system can be given a mass matrix, based on the `DM`, which is inverted on the RHS at each step.
  Since the matrix is lumped, inversion is trivial.

.seealso: [](ch_ts), `DM`, `DMTSCreateRHSMassMatrix()`, `DMTSDestroyRHSMassMatrix()`, `DMCreateMassMatrix()`, `DMTS`
@*/
PetscErrorCode DMTSCreateRHSMassMatrixLumped(DM dm)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));
  PetscCall(DMCreateMassMatrixLumped(dm, &dmlocalts->lumpedmassinv));
  PetscCall(VecReciprocal(dmlocalts->lumpedmassinv));
  PetscCall(VecViewFromOptions(dmlocalts->lumpedmassinv, NULL, "-lumped_mass_inv_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTSGetRHSMassMatrixLumped(DM dm, Vec *massinv)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(massinv, 2);
  PetscCall(DMGetDMTS(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));
  *massinv = dmlocalts->lumpedmassinv;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTSSetRHSMassMatrixLumped(DM dm, Vec massinv)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(massinv, 2);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));
  dmlocalts->lumpedmassinv = massinv;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSDestroyRHSMassMatrix - Destroys the mass matrix and solver stored in the `DM` context, if they exist.

  Logically Collective

  Input Parameter:
. dm - `DM` providing the mass matrix

  Level: developer

.seealso: [](ch_ts), `DM`, `DMTSCreateRHSMassMatrixLumped()`, `DMCreateMassMatrix()`, `DMTS`
@*/
PetscErrorCode DMTSDestroyRHSMassMatrix(DM dm)
{
  DMTS        tdm;
  DMTS_Local *dmlocalts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tdm));
  PetscCall(DMLocalTSGetContext(dm, tdm, &dmlocalts));
  PetscCall(VecDestroy(&dmlocalts->lumpedmassinv));
  PetscCall(MatDestroy(&dmlocalts->mass));
  PetscCall(KSPDestroy(&dmlocalts->kspmass));
  PetscFunctionReturn(PETSC_SUCCESS);
}
